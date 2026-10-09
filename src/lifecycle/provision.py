"""
    Section L1 - Provision

    Starts a container from an image (or from a Dockerfile, built once by lifecycle.image)
    and waits until it is ready.

    Two start modes:
      idle    (default) the entrypoint becomes `sleep infinity`, so the container stays up
              for the execution toolchain whatever the image's CMD is
      native  the image's own ENTRYPOINT / CMD (or `command`) runs

    Guardrails, because an LLM chooses the arguments:
      - images are never pulled; a missing image is an error
      - never privileged, no host network, no Docker socket in the container
      - mounts only from the allowed roots (runs/ of this repository, plus TML_MOUNT_ROOTS)
      - ports are published on 127.0.0.1 only
      - memory and process limits by default

    A container with the same name is reused when its image and settings are the same.

    Plain Python helper, not an agent tool. Use provision(ContainerSpec(image=...)).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import uuid
from pathlib import Path

from docker import errors as docker_errors
from docker.types import DeviceRequest
from docker.utils import parse_bytes

from . import client
from .image import ensure_image, resolve_image
from .models import (
    LABEL_IMAGE, LABEL_MANAGED, LABEL_MODE, LABEL_SPEC,
    ContainerSpec, LifecycleError, ProvisionResult,
)
from .status import is_managed, wait_ready


# Mounts must come from here (repository root / runs) or from TML_MOUNT_ROOTS
DEFAULT_MOUNT_ROOT = Path(__file__).resolve().parents[2] / "runs"
MOUNT_ROOTS_ENV = "TML_MOUNT_ROOTS"

# Default memory limit: this share of the host memory, with no swap on top
DEFAULT_MEMORY_FRACTION = 0.75

# Processes and threads in one container (stops a fork bomb, leaves room for DataLoader workers)
PIDS_LIMIT = 8192

# Published ports are reachable from this machine only
PUBLISH_ADDRESS = "127.0.0.1"

# Seconds to wait for the container to be ready
READY_TIMEOUT = 30.0
HEALTHCHECK_READY_TIMEOUT = 120.0   # an image HEALTHCHECK needs time for its first runs

IDLE_ENTRYPOINT = ["sleep"]
IDLE_COMMAND = ["infinity"]

_PORT_RE = re.compile(r"^(?:(?P<host>\d+):)?(?P<container>\d+)(?:/(?P<proto>tcp|udp))?$")


# ---------------------------------------------------------------- pure helpers

def _invalid(message: str, hint: str = "") -> LifecycleError:
    return LifecycleError("invalid_argument", message, hint)


def allowed_mount_roots() -> list[Path]:
    """Directories a mount source may come from."""
    extra = [p for p in os.environ.get(MOUNT_ROOTS_ENV, "").split(os.pathsep) if p]
    return [DEFAULT_MOUNT_ROOT.resolve()] + [Path(p).expanduser().resolve() for p in extra]


def parse_mounts(mounts: list[str]) -> dict[str, dict]:
    """Checks "host_path:container_path[:ro|rw]" items and returns the SDK `volumes` mapping.

    Raises LifecycleError("invalid_argument") for a mount outside the allowed roots.
    """
    roots = allowed_mount_roots()
    volumes: dict[str, dict] = {}
    for item in mounts:
        parts = item.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("ro", "rw")):
            raise _invalid(f"invalid mount {item!r}", "use host_path:container_path or host_path:container_path:ro")
        host, target = Path(parts[0]).expanduser(), parts[1]
        if not host.is_absolute():
            raise _invalid(f"mount source must be an absolute host path: {item!r}")
        if not target.startswith("/"):
            raise _invalid(f"mount target must be an absolute container path: {item!r}")
        source = host.resolve()  # follows symlinks, so a link cannot lead out of a root
        if source.name == "docker.sock":
            raise _invalid(f"refusing to mount the Docker socket: {item!r}", "a container with the socket controls the host")
        if not any(source == root or source.is_relative_to(root) for root in roots):
            raise _invalid(
                f"mount source {str(source)!r} is outside the allowed roots",
                f"allowed roots: {[str(r) for r in roots]}; add one with {MOUNT_ROOTS_ENV}",
            )
        if not source.exists():
            raise _invalid(f"mount source does not exist: {str(source)!r}", "create the directory first")
        volumes[str(source)] = {"bind": target, "mode": parts[2] if len(parts) == 3 else "rw"}
    return volumes


def parse_ports(ports: list[str]) -> dict[str, tuple]:
    """Checks "host:container" / "container" items and returns the SDK `ports` mapping.

    A lone container port gets a free host port. Every port is published on 127.0.0.1.
    """
    out: dict[str, tuple] = {}
    for item in ports:
        m = _PORT_RE.match(str(item).strip())
        if not m:
            raise _invalid(f"invalid port {item!r}", "use host:container or container, e.g. 8888:8888")
        numbers = [int(n) for n in (m.group("host"), m.group("container")) if n]
        if not all(1 <= n <= 65535 for n in numbers):
            raise _invalid(f"port out of range in {item!r}")
        host_port = int(m.group("host")) if m.group("host") else None
        out[f"{m.group('container')}/{m.group('proto') or 'tcp'}"] = (PUBLISH_ADDRESS, host_port)
    return out


def _size(value: str, what: str) -> int:
    try:
        size = parse_bytes(value)
    except (docker_errors.DockerException, ValueError, TypeError):
        size = 0
    if not isinstance(size, int) or size <= 0:
        raise _invalid(f"invalid {what} {value!r}", "use a number with a unit, e.g. 512m or 8g")
    return size


def default_memory(info: dict) -> int | None:
    """Default memory limit in bytes from `docker info`; None if the host size is unknown."""
    total = info.get("MemTotal") or 0
    return int(total * DEFAULT_MEMORY_FRACTION) or None


def generate_name(image: str) -> str:
    """tml-<image name>-<6 hex digits>"""
    base = image.split("@")[0].rsplit("/", 1)[-1].split(":")[0]
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-._") or "container"
    return f"tml-{base}-{uuid.uuid4().hex[:6]}"


def build_create_kwargs(
    spec: ContainerSpec,
    image: str,
    name: str,
    use_gpu: bool = False,
    memory_default: int | None = None,
) -> dict:
    """Arguments for the SDK's containers.create(). No Docker calls.

    Args:
        spec: what to start; its mounts and ports are checked here.
        image: image reference that exists on this machine.
        name: container name.
        use_gpu: give the container all GPUs (spec.gpu already resolved against the daemon).
        memory_default: memory limit in bytes when spec.memory is empty; None = no limit.
    """
    kwargs: dict = {
        "image": image,
        "name": name,
        # toolchain labels last: a spec label cannot switch them off
        "labels": {**spec.labels, LABEL_MANAGED: "true", LABEL_MODE: spec.mode, LABEL_IMAGE: image},
        "init": True,            # PID 1 forwards signals and reaps zombies, so stop is fast
        "privileged": False,
        "network_mode": spec.network,
        "pids_limit": PIDS_LIMIT,
        "shm_size": _size(spec.shm_size, "shm_size"),
    }
    if spec.mode == "idle":
        kwargs["entrypoint"] = IDLE_ENTRYPOINT
        kwargs["command"] = IDLE_COMMAND
        # the image's service is not running in idle mode, so its HEALTHCHECK would always fail
        kwargs["healthcheck"] = {"test": ["NONE"]}
    elif spec.command:
        try:
            kwargs["command"] = shlex.split(spec.command)
        except ValueError as e:
            raise _invalid(f"cannot parse command {spec.command!r}: {e}") from e

    memory = _size(spec.memory, "memory") if spec.memory else memory_default
    if memory:
        kwargs["mem_limit"] = memory
        kwargs["memswap_limit"] = memory   # same value = no swap: a leak is killed, not swapped
    if spec.cpus:
        kwargs["nano_cpus"] = int(spec.cpus * 1_000_000_000)
    if use_gpu:
        kwargs["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    if spec.env:
        kwargs["environment"] = {str(k): str(v) for k, v in spec.env.items()}
    if spec.mounts:
        kwargs["volumes"] = parse_mounts(spec.mounts)
    if spec.ports:
        kwargs["ports"] = parse_ports(spec.ports)
    if spec.workdir:
        kwargs["working_dir"] = spec.workdir
    if spec.user:
        kwargs["user"] = spec.user
    return kwargs


def spec_fingerprint(kwargs: dict, image_id: str) -> str:
    """Identifies the image and settings of a container (not its name or labels)."""
    settings = {k: v for k, v in kwargs.items() if k not in ("name", "labels")}
    payload = json.dumps({"image_id": image_id, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- Docker access

def _find_by_name(name: str):
    """The container with exactly this name, or None."""
    with client.translated(f"inspect container {name!r}"):
        try:
            obj = client.get_client().containers.get(name)
        except docker_errors.NotFound:
            return None
    # containers.get also matches id prefixes; only an exact name is a conflict
    return obj if obj.name == name else None


def _discard(obj) -> None:
    """Removes a container that could not be started. Best effort."""
    try:
        obj.remove(force=True, v=True)
    except Exception:
        pass


def _has_healthcheck(obj) -> bool:
    test = ((obj.attrs.get("Config") or {}).get("Healthcheck") or {}).get("Test") or []
    return bool(test) and test != ["NONE"]


def provision(spec: ContainerSpec, replace: bool = False, ready_timeout: float | None = None) -> ProvisionResult:
    """Starts a container and waits until it is ready.

    A container that starts and then fails is NOT an exception: the result carries its
    status (health "failed", exit code) and the container is kept for diagnosis.
    Check ProvisionResult.ok.

    If a managed container with the same name exists:
      - same image and settings: it is reused (and started again if it had stopped)
      - different image or settings: name_conflict, unless replace=True removes it first

    Args:
        spec: what to start.
        replace: remove an existing managed container with the same name and different settings.
        ready_timeout: seconds to wait for readiness; default 30, or 120 if the image has a HEALTHCHECK.

    Raises:
        LifecycleError: invalid_argument, image_not_found, build_failed, name_conflict,
            gpu_unavailable, port_conflict, docker_unavailable, docker_error.
    """
    spec.validate()
    # cheap argument checks first, before a build that can take minutes
    parse_mounts(spec.mounts)
    parse_ports(spec.ports)

    sdk = client.get_client()
    if spec.dockerfile:
        image = ensure_image(spec.dockerfile, spec.context, labels=spec.labels)
    else:
        image = resolve_image(spec.image)

    with client.translated("read the daemon info"):
        info = sdk.info()
    use_gpu = spec.gpu == "all" or (spec.gpu == "auto" and client.has_nvidia_runtime(info))

    name = spec.name or generate_name(image.tag)
    kwargs = build_create_kwargs(spec, image.tag, name, use_gpu, default_memory(info))
    fingerprint = spec_fingerprint(kwargs, image.id)
    kwargs["labels"][LABEL_SPEC] = fingerprint

    action = "created"
    existing = _find_by_name(name)
    if existing is not None:
        if not is_managed(existing.attrs):
            raise LifecycleError(
                "name_conflict",
                f"the name {name!r} belongs to a container this toolchain did not create",
                "choose another name",
            )
        if existing.labels.get(LABEL_SPEC) == fingerprint:
            action = "reused"
            if existing.status != "running":
                action = "restarted"
                with client.translated(f"start container {name!r}"):
                    existing.start()
            timeout = ready_timeout or (HEALTHCHECK_READY_TIMEOUT if _has_healthcheck(existing) else READY_TIMEOUT)
            return ProvisionResult(status=wait_ready(name, timeout=timeout), image=image, action=action)
        if not replace:
            raise LifecycleError(
                "name_conflict",
                f"container {name!r} exists with a different image or different settings",
                "pass replace=True to remove it and start a new one, or use another name",
            )
        with client.translated(f"remove container {name!r}"):
            existing.remove(force=True, v=True)
        action = "replaced"

    with client.translated(f"create container {name!r}"):
        obj = sdk.containers.create(**kwargs)
    try:
        with client.translated(f"start container {name!r}"):
            obj.start()
    except LifecycleError:
        _discard(obj)   # never started: nothing to diagnose, and it would block the name
        raise

    timeout = ready_timeout or (HEALTHCHECK_READY_TIMEOUT if _has_healthcheck(obj) else READY_TIMEOUT)
    return ProvisionResult(status=wait_ready(name, timeout=timeout), image=image, action=action)
