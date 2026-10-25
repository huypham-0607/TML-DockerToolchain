"""
    Section L0 - Docker Client

    The one place that creates the Docker SDK client and maps SDK exceptions to
    LifecycleError codes. Every other lifecycle module goes through this module
    (client.get_client()), so tests can replace it with a fake.

    Plain Python helper, not an agent tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import docker
import requests
from docker import errors as docker_errors

from .models import LifecycleError


# Seconds to wait for one daemon request
DEFAULT_TIMEOUT = 30

# How a daemon says that it cannot give a container a GPU: no NVIDIA runtime (engines before 28),
# a toolkit without a working driver, or no / stale CDI devices (newer engines)
GPU_UNAVAILABLE_RE = re.compile(
    r"could not select device driver|nvidia-container-cli|nvml error|failed to discover GPU vendor"
    r"|unresolvable CDI devices|failed to inject CDI devices",
    re.I,
)

# The daemon socket of a rootful engine is open to root and the docker group only
SOCKET_PERMISSION_HINT = (
    "this user may not use the Docker socket: add it to the docker group "
    "(sudo usermod -aG docker $USER), then log in again"
)

# A daemon can list a GPU that it cannot attach (Docker Desktop registers the NVIDIA runtime even
# without an NVIDIA GPU). The outcome of a real attach is remembered per daemon in this file.
CACHE_DIR_ENV = "TML_CACHE_DIR"
GPU_STATE_FILE = "gpu_attach.json"
GPU_STATE_TTL = 7 * 24 * 3600   # seconds; afterwards a GPU is tried again

_client: docker.DockerClient | None = None


def get_client(timeout: int = DEFAULT_TIMEOUT) -> docker.DockerClient:
    """Returns the shared SDK client (from DOCKER_HOST and the usual environment).

    Raises LifecycleError("docker_unavailable") when the daemon cannot be reached.
    """
    global _client
    if _client is None:
        try:
            _client = docker.from_env(timeout=timeout)
        except (docker_errors.DockerException, requests.exceptions.RequestException) as e:
            raise translate(e, "connect to the Docker daemon") from e
    return _client


def reset_client() -> None:
    """Drops the shared client, so the next get_client() reads the environment again."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None


def docker_available() -> bool:
    """True if the daemon answers a ping."""
    try:
        return bool(get_client().ping())
    except (LifecycleError, docker_errors.DockerException, requests.exceptions.RequestException):
        return False


def has_nvidia_runtime(info: dict | None = None) -> bool:
    """True if the daemon has an NVIDIA runtime (nvidia-container-toolkit in runtime mode)."""
    if info is None:
        with translated("read the daemon info"):
            info = get_client().info()
    return "nvidia" in (info.get("Runtimes") or {})


def _cdi_gpus(info: dict) -> list[str]:
    devices = info.get("DiscoveredDevices") or []
    return sorted(str(d.get("ID", "")) for d in devices if isinstance(d, dict) and str(d.get("ID", "")).startswith("nvidia.com/gpu"))


def _info(info: dict | None) -> dict:
    if info is None:
        with translated("read the daemon info"):
            info = get_client().info()
    return info


def gpu_claimed(info: dict | None = None) -> bool:
    """True if the daemon lists an NVIDIA GPU: the NVIDIA runtime, or a CDI device (newer engines list
    CDI devices under DiscoveredDevices). A listed GPU does not have to work: see gpu_available()."""
    info = _info(info)
    return has_nvidia_runtime(info) or bool(_cdi_gpus(info))


def gpu_available(info: dict | None = None) -> bool:
    """True if containers can get an NVIDIA GPU: the daemon lists one, and an earlier attach on this
    daemon did not fail (in the last GPU_STATE_TTL seconds)."""
    info = _info(info)
    return gpu_claimed(info) and not gpu_attach_failed(info)


# ---------------------------------------------------------------- remembered GPU attach outcome

def cache_dir() -> Path:
    """$TML_CACHE_DIR, else $XDG_CACHE_HOME/tml-dockertoolchain, else ~/.cache/tml-dockertoolchain."""
    if custom := os.environ.get(CACHE_DIR_ENV):
        return Path(custom).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tml-dockertoolchain"


def gpu_state_path() -> Path:
    return cache_dir() / GPU_STATE_FILE


def gpu_daemon_key(info: dict) -> str:
    """Identifies a daemon and its GPU setup: a new runtime or CDI device gives a new key, so a
    remembered failure does not outlive a change of the setup."""
    parts = {
        "id": info.get("ID", ""), "version": info.get("ServerVersion", ""),
        "runtimes": sorted(info.get("Runtimes") or {}), "cdi": _cdi_gpus(info),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _read_gpu_state() -> dict:
    try:
        state = json.loads(gpu_state_path().read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_gpu_state(state: dict) -> None:
    """Best effort: a cache that cannot be written only costs one more failed GPU attach."""
    path = gpu_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(path)
    except OSError:
        pass


def gpu_attach_failed(info: dict) -> bool:
    """True if a GPU attach failed on this daemon (same GPU setup) in the last GPU_STATE_TTL seconds."""
    entry = _read_gpu_state().get(gpu_daemon_key(info))
    if not isinstance(entry, dict):
        return False
    try:
        return time.time() - float(entry.get("failed_at", 0)) < GPU_STATE_TTL
    except (TypeError, ValueError):
        return False


def gpu_attach_known(info: dict) -> bool:
    """True if a GPU attach on this daemon (same GPU setup) worked or failed in the last GPU_STATE_TTL seconds."""
    entry = _read_gpu_state().get(gpu_daemon_key(info))
    if not isinstance(entry, dict):
        return False
    try:
        at = float(entry.get("failed_at") or entry.get("worked_at") or 0)
    except (TypeError, ValueError):
        return False
    return time.time() - at < GPU_STATE_TTL


def remember_gpu_attach(info: dict, worked: bool, detail: str = "") -> None:
    """Records the outcome of a real GPU attach on this daemon."""
    state = _read_gpu_state()
    entry = {"worked_at": time.time()} if worked else {"failed_at": time.time(), "detail": detail[:300]}
    state[gpu_daemon_key(info)] = entry
    _write_gpu_state(state)


def probe_gpu_attach(image: str, info: dict | None = None) -> bool:
    """Tries once to start a container with a GPU, and remembers the outcome.

    The container comes from `image` (already on this machine: nothing is pulled), runs nothing
    of the image (`true`), and is always removed. Only the attach counts: a start that passes the
    GPU step is a success even if the image has no `true`.

    Raises LifecycleError for problems that are not about the GPU (missing image, daemon down).
    """
    from docker.types import DeviceRequest

    info = _info(info)
    sdk = get_client()
    with translated(f"create a GPU probe container from {image!r}"):
        probe = sdk.containers.create(
            image, entrypoint=["true"], labels={"tml.probe": "gpu"}, init=True,   # init: the start passes even without `true`
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        )
    try:
        try:
            with translated("start the GPU probe container"):
                probe.start()
        except LifecycleError as e:
            if e.code != "gpu_unavailable":
                raise
            remember_gpu_attach(info, worked=False, detail=e.message)
            return False
        remember_gpu_attach(info, worked=True)
        return True
    finally:
        try:
            probe.remove(force=True, v=True)
        except Exception:
            pass


def forget_gpu_attach() -> None:
    """Forgets every remembered outcome: the next gpu="auto" tries the GPU again."""
    try:
        gpu_state_path().unlink()
    except OSError:
        pass


def translate(exc: BaseException, what: str = "") -> LifecycleError:
    """Maps an SDK or transport exception to a LifecycleError.

    Args:
        exc: the exception from the Docker SDK or from requests.
        what: the operation that failed, used as the message prefix.
    """
    if isinstance(exc, LifecycleError):
        return exc
    prefix = f"could not {what}: " if what else ""
    detail = str(getattr(exc, "explanation", None) or exc).strip()
    low = detail.lower()

    def err(code: str, hint: str = "") -> LifecycleError:
        return LifecycleError(code, prefix + detail, hint)

    if isinstance(exc, docker_errors.ImageNotFound):
        return err("image_not_found", "build the image first, or pass a dockerfile; lifecycle does not pull images")
    if isinstance(exc, docker_errors.NotFound):
        return err("not_found", "list the managed containers with container_status()")
    if isinstance(exc, docker_errors.APIError):
        if GPU_UNAVAILABLE_RE.search(detail):
            return err("gpu_unavailable", "this daemon cannot give the container a GPU; provision again with gpu='none'")
        if "mounts denied" in low:
            return err(
                "invalid_argument",
                "Docker Desktop shares only some host directories: use one under your home directory, "
                "or add this one in Docker Desktop > Settings > Resources > File Sharing",
            )
        if "port is already allocated" in low or "address already in use" in low:
            return err("port_conflict", "choose a different host port")
        if exc.status_code == 409:
            if "already in use by container" in low:
                return err("name_conflict", "use another name, or pass replace=True")
            if "is not running" in low:
                return err("not_running", "start it again with provision_container, or read diagnose_container")
        return err("docker_error")
    if isinstance(exc, requests.exceptions.Timeout):
        return err("timeout", "the daemon did not answer in time")
    if "permission denied" in low:
        # not an APIError (handled above): the socket itself refused this user
        return err("docker_unavailable", SOCKET_PERMISSION_HINT)
    if isinstance(exc, (requests.exceptions.ConnectionError, docker_errors.DockerException)):
        return err("docker_unavailable", "check that the Docker daemon is running and that this user can reach it")
    return err("docker_error")


@contextmanager
def translated(what: str = ""):
    """Runs SDK calls and re-raises their exceptions as LifecycleError."""
    try:
        yield
    except LifecycleError:
        raise
    except (docker_errors.DockerException, requests.exceptions.RequestException) as e:
        raise translate(e, what) from e
