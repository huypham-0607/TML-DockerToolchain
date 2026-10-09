"""
    Section L2 - Status and Health

    Reads the state of containers this toolchain created and derives one health verdict:
      starting | healthy | unhealthy | completed | stopped | failed | missing

    The verdict combines evidence, because research images rarely define a HEALTHCHECK:
      - Docker state, exit code, OOMKilled, restart count
      - the image HEALTHCHECK, if it has one
      - a readiness probe (`true` run in the container) when there is no HEALTHCHECK

    Only containers with the label tml.managed=true are visible. The execution toolchain
    calls require_running(container) before it runs a command.

    Plain Python helper, not an agent tool. Use get_status(container).
"""

from __future__ import annotations

import time

from . import client
from .models import (
    LABEL_IMAGE, LABEL_MANAGED, LABEL_MODE, LABEL_PREFIX,
    ContainerHandle, ContainerStatus, LifecycleError,
)


# Docker's zero timestamp: "never started" / "not finished"
_ZERO_TIME = "0001-01-01T00:00:00Z"

MAX_HEALTHCHECK_CHARS = 500

# Fixed, no-op command of the readiness probe. Not an execution interface.
PROBE_COMMAND = ["true"]


# ---------------------------------------------------------------- pure helpers

def is_managed(attrs: dict) -> bool:
    return ((attrs.get("Config") or {}).get("Labels") or {}).get(LABEL_MANAGED) == "true"


def _time(value: str | None) -> str | None:
    return None if not value or value == _ZERO_TIME else value


def handle_from_attrs(attrs: dict) -> ContainerHandle:
    """Builds the hand-off object from `docker inspect` output."""
    config = attrs.get("Config") or {}
    host = attrs.get("HostConfig") or {}
    labels = config.get("Labels") or {}

    # live port map when running (shows daemon-assigned ports), configured bindings otherwise
    port_map = (attrs.get("NetworkSettings") or {}).get("Ports") or host.get("PortBindings") or {}
    ports = {port: binds[0].get("HostPort", "") for port, binds in port_map.items() if binds}

    mounts = [
        f"{m.get('Source', '')}:{m.get('Destination', '')}" + ("" if m.get("RW", True) else ":ro")
        for m in attrs.get("Mounts") or []
    ]
    return ContainerHandle(
        id=attrs.get("Id", "")[:12],
        name=attrs.get("Name", "").lstrip("/"),
        image=labels.get(LABEL_IMAGE) or config.get("Image", ""),
        image_id=attrs.get("Image", "").removeprefix("sha256:")[:12],
        mode=labels.get(LABEL_MODE, "native"),
        gpu=bool(host.get("DeviceRequests")),
        workdir=config.get("WorkingDir") or "",
        ports=ports,
        mounts=mounts,
        created_at=attrs.get("Created", ""),
        labels={k: v for k, v in labels.items() if k.startswith(LABEL_PREFIX)},
    )


def _healthcheck_output(state: dict) -> str:
    log = (state.get("Health") or {}).get("Log") or []
    return (log[-1].get("Output") or "").strip()[-MAX_HEALTHCHECK_CHARS:] if log else ""


def derive_health(attrs: dict, probe_ok: bool | None = None) -> tuple[str, str]:
    """Returns (health verdict, one-line reason) for `docker inspect` output.

    Args:
        attrs: the inspect dictionary of the container.
        probe_ok: result of the readiness probe; None if the probe did not run.
    """
    state = attrs.get("State") or {}
    status = state.get("Status", "")
    code = state.get("ExitCode", 0)
    error = state.get("Error") or ""

    if status == "running":
        health = state.get("Health")
        if health and health.get("Status") in ("starting", "healthy", "unhealthy"):
            # the image HEALTHCHECK is the authority when there is one
            verdict = health["Status"]
            if verdict == "starting":
                return "starting", "running; the image HEALTHCHECK has not passed yet"
            if verdict == "healthy":
                return "healthy", "running; the image HEALTHCHECK passes"
            out = _healthcheck_output(state)
            return "unhealthy", "running; the image HEALTHCHECK fails" + (f": {out}" if out else "")
        if probe_ok is False:
            return "unhealthy", "running, but the readiness probe failed (cannot run `true` in the container)"
        if probe_ok is None:
            return "healthy", "running (no HEALTHCHECK; readiness not probed)"
        return "healthy", "running; the readiness probe passes"

    if status == "restarting":
        return "unhealthy", f"restarting (crash loop); restart count {attrs.get('RestartCount', 0)}, last exit code {code}"
    if status == "paused":
        return "unhealthy", "paused; commands cannot run until it is unpaused"
    if status == "created":
        if error or code:
            return "failed", f"did not start (exit code {code})" + (f": {error}" if error else "")
        return "stopped", "created but never started"
    if status == "removing":
        return "missing", "being removed"
    if status == "dead":
        return "failed", "dead: the daemon could not stop or remove it" + (f": {error}" if error else "")

    # exited
    if state.get("OOMKilled"):
        return "failed", f"killed by the kernel: out of memory (exit code {code})"
    if code == 0:
        return "completed", "exited with code 0"
    if code in (137, 143):
        signal = "SIGKILL" if code == 137 else "SIGTERM"
        return "stopped", f"exited with code {code} ({signal}): stopped or killed from outside"
    return "failed", f"exited with code {code}" + (f": {error}" if error else "")


def status_from_attrs(attrs: dict, probe_ok: bool | None = None, asked: str = "") -> ContainerStatus:
    """Builds a ContainerStatus from `docker inspect` output."""
    state = attrs.get("State") or {}
    status = state.get("Status", "")
    code = state.get("ExitCode", 0)
    health, reason = derive_health(attrs, probe_ok)
    handle = handle_from_attrs(attrs)
    # ExitCode and FinishedAt of a live container belong to an earlier run
    live = status in ("running", "paused")
    never_ran = status == "created" and not code
    return ContainerStatus(
        container=asked or handle.name,
        state=status,
        health=health,
        reason=reason,
        handle=handle,
        exit_code=None if live or never_ran else code,
        oom_killed=bool(state.get("OOMKilled")),
        restart_count=attrs.get("RestartCount", 0),
        started_at=_time(state.get("StartedAt")),
        finished_at=None if live else _time(state.get("FinishedAt")),
        error=state.get("Error") or "",
        healthcheck_output=_healthcheck_output(state),
    )


def _missing(container: str) -> ContainerStatus:
    return ContainerStatus(container=container, state="missing", health="missing", reason="no such container")


# ---------------------------------------------------------------- Docker access

def require_managed(container: str):
    """Returns the SDK container object, freshly inspected.

    Raises LifecycleError("not_found") if there is no such container, and
    LifecycleError("not_managed") if this toolchain did not create it.
    """
    if not container:
        raise LifecycleError("invalid_argument", "container name or id is empty")
    with client.translated(f"inspect container {container!r}"):
        obj = client.get_client().containers.get(container)
    if not is_managed(obj.attrs):
        raise LifecycleError(
            "not_managed",
            f"container {container!r} was not created by this toolchain",
            f"lifecycle only acts on containers with the label {LABEL_MANAGED}=true",
        )
    return obj


def _probe(obj) -> bool:
    """Readiness probe: can a process be started in the container?"""
    try:
        return obj.exec_run(PROBE_COMMAND).exit_code == 0
    except Exception:  # stopped in the meantime, daemon error, ...: not ready either way
        return False


def _status_of(obj, probe: bool, asked: str = "") -> ContainerStatus:
    state = obj.attrs.get("State") or {}
    # the probe only decides when the container runs and the image has no HEALTHCHECK
    needs_probe = probe and state.get("Status") == "running" and not state.get("Health")
    return status_from_attrs(obj.attrs, _probe(obj) if needs_probe else None, asked)


def get_status(container: str, probe: bool = True) -> ContainerStatus:
    """Returns the state and health of one managed container.

    A container that does not exist gives health "missing" (no exception).

    Args:
        container: container name or id.
        probe: run the readiness probe when the image has no HEALTHCHECK.
    """
    try:
        obj = require_managed(container)
    except LifecycleError as e:
        if e.code == "not_found":
            return _missing(container)
        raise
    return _status_of(obj, probe, container)


def list_managed(all: bool = True, labels: dict[str, str] | None = None, probe: bool = True) -> list[ContainerStatus]:
    """Returns the status of every container this toolchain created, oldest first.

    Args:
        all: include containers that are not running.
        labels: extra label filter (label -> value).
        probe: run the readiness probe on running containers without a HEALTHCHECK.
    """
    wanted = [f"{LABEL_MANAGED}=true"] + [f"{k}={v}" for k, v in (labels or {}).items()]
    with client.translated("list containers"):
        found = client.get_client().containers.list(all=all, filters={"label": wanted})
    statuses = [_status_of(obj, probe) for obj in found]
    statuses.sort(key=lambda s: s.handle.created_at if s.handle else "")
    return statuses


def wait_ready(container: str, timeout: float = 30.0, interval: float = 0.25, settle: float = 1.0) -> ContainerStatus:
    """Polls until the container leaves "starting", then returns its status.

    A container that turns "healthy" must stay healthy for `settle` seconds, so a
    process that crashes just after its start is not reported as ready.
    If `timeout` runs out first, the last status is returned (health "starting").

    Args:
        container: container name or id.
        timeout: seconds to wait for a verdict other than "starting".
        interval: seconds between polls.
        settle: seconds a healthy container must stay healthy.
    """
    deadline = time.monotonic() + timeout
    while True:
        status = get_status(container)
        if status.health == "healthy":
            if settle > 0:
                time.sleep(settle)
                status = get_status(container)
            return status
        if status.health != "starting":
            return status
        if time.monotonic() >= deadline:
            status.reason += f" (still starting after {timeout:g} s)"
            return status
        time.sleep(interval)


def require_running(container: str) -> ContainerHandle:
    """Returns the handle of a running managed container. For the execution toolchain.

    Raises LifecycleError: not_found, not_managed, or not_running.
    """
    obj = require_managed(container)
    status = status_from_attrs(obj.attrs, asked=container)
    if not status.running:
        raise LifecycleError(
            "not_running",
            f"container {container!r} is {status.state}: {status.reason}",
            "start it again with provision_container, or read diagnose_container",
        )
    return status.handle
