"""
    Section L0 - Docker Client

    The one place that creates the Docker SDK client and maps SDK exceptions to
    LifecycleError codes. Every other lifecycle module goes through this module
    (client.get_client()), so tests can replace it with a fake.

    Plain Python helper, not an agent tool.
"""

from __future__ import annotations

from contextlib import contextmanager

import docker
import requests
from docker import errors as docker_errors

from .models import LifecycleError


# Seconds to wait for one daemon request
DEFAULT_TIMEOUT = 30

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
    """True if the daemon has an NVIDIA runtime, i.e. containers can get a GPU."""
    if info is None:
        with translated("read the daemon info"):
            info = get_client().info()
    return "nvidia" in (info.get("Runtimes") or {})


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
        if "could not select device driver" in low or "nvidia-container-cli" in low:
            return err("gpu_unavailable", "this daemon cannot give the container a GPU; provision again with gpu='none'")
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
