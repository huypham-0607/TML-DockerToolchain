"""
    Section L5 - Agent Tools

    The six LLM-facing tools of the container lifecycle:
      provision_container   start a container (from an image or a Dockerfile) and wait until it is ready
      container_status      state and health of one container, or of all of them
      container_logs        runtime logs, stdout and stderr apart
      diagnose_container    status, log tail, hints, processes, resource use, events
      stop_container        graceful stop; the container is kept
      destroy_container     stop if needed, then remove

    Each one is a plain function that returns a JSON string and never raises:
      success:          {"ok": true, ...}
      failed request:   {"ok": false, "error": {"code", "message", "hint"}}
    provision_container also answers {"ok": false, "status": ..., "diagnosis": ...} when the
    container started but is not usable.

    The output is compact: empty fields are left out, so the LLM reads less. The Python API
    (src.lifecycle) returns the full objects.

    TOOLS holds the LangChain versions for the agent; the CLI (python -m src.lifecycle) calls the
    plain functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from langchain_core.tools import tool

from .diagnostics import DEFAULT_TAIL, diagnose, get_logs
from .models import ContainerSpec, ContainerStatus, Diagnosis, LifecycleError, TeardownResult
from .provision import provision
from .status import get_status, list_managed
from .teardown import STOP_TIMEOUT, destroy, stop

# Labels added to every container that provision_container makes (for example a session id).
# Not an argument of the tool: the LLM does not choose them.
SESSION_LABELS: dict[str, str] = {}

# Handle fields the LLM needs; the rest (labels, created_at, image_id) stays in the Python API
_HANDLE_FIELDS = ("name", "id", "image", "mode", "gpu", "workdir", "ports", "mounts")


# ---------------------------------------------------------------- compact output

def compact(value):
    """Drops None, empty strings, empty lists and empty dicts, at every level. False and 0 stay."""
    if isinstance(value, dict):
        out = {k: compact(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def status_view(status: ContainerStatus) -> dict:
    view = status.to_dict()
    view.pop("running", None)        # state already says it
    if status.handle is not None:
        view["handle"] = {k: view["handle"][k] for k in _HANDLE_FIELDS}
    if not status.oom_killed:
        view.pop("oom_killed")
    if not status.restart_count:
        view.pop("restart_count")
    return compact(view)


def diagnosis_view(diagnosis: Diagnosis, with_status: bool = True) -> dict:
    view = {
        "logs": asdict(diagnosis.logs) if diagnosis.logs else None,
        "hints": [asdict(h) for h in diagnosis.hints],
        "processes": diagnosis.processes,
        "resources": diagnosis.resources,
        "events": diagnosis.events,
        "notes": diagnosis.notes,
    }
    if view["logs"]:
        view["logs"].pop("container")
        view["logs"]["truncated"] = view["logs"]["truncated"] or None   # shown only when true
    if with_status:
        view = {"status": status_view(diagnosis.status), **view}
    return compact(view)


def teardown_view(result: TeardownResult) -> dict:
    view = {
        "container": result.container,
        "stopped": result.stopped,
        "killed": result.killed,
        "removed": result.removed,
        "exit_code": result.exit_code,
        "status": status_view(result.status),
    }
    return compact(view)


def _reply(build) -> str:
    """Runs one tool body and turns every outcome into the JSON envelope."""
    try:
        reply = build()
    except LifecycleError as e:
        reply = e.envelope()
    except Exception as e:   # a bug in the toolchain must not end the agent's run
        reply = {"ok": False, "error": {
            "code": "internal_error",
            "message": f"{type(e).__name__}: {e}",
            "hint": "this is a fault in the lifecycle toolchain, not in the container; report it",
        }}
    return json.dumps(reply)


# ---------------------------------------------------------------- the tools

def provision_container(
    image: str = "",
    dockerfile: str = "",
    context: str = "",
    name: str = "",
    mode: str = "idle",
    command: str = "",
    gpu: str = "auto",
    env: dict[str, str] | None = None,
    mounts: list[str] | None = None,
    ports: list[str] | None = None,
    memory: str = "",
    shm_size: str = "2g",
    user: str = "",
    ready_timeout: float = 0,
    replace: bool = False,
) -> str:
    """Starts a Docker container and waits until it is ready. Call this before any command runs in a container.

    Give exactly one of `image` or `dockerfile`. Images are never pulled; a Dockerfile is built once.
    The reply has "ok": true when the container is ready, and its "status.handle.name" is the name for the
    other tools. "ok": false with "diagnosis" means it started but failed: read "diagnosis.hints", where each
    hint names an owner (lifecycle: provision again with other options; dockerization: the image must change;
    execution: the command must change; host: report to the user). A container with the same name and the same
    settings is reused; different settings give error "name_conflict" unless replace is true.

    Args:
        image: tag or id of an image that is already on this machine.
        dockerfile: path to a Dockerfile to build and run instead of an image.
        context: build context directory for the Dockerfile; default is the Dockerfile's directory.
        name: container name; one is generated when empty.
        mode: "idle" keeps the container running for later commands (default); "native" runs the image's own command.
        command: native mode only: the command to run instead of the image's CMD.
        gpu: "auto" (use a GPU if this machine has one), "all", or "none".
        env: environment variables, for example {"SEED": "1"}.
        mounts: host directories, "host_path:container_path" or "host_path:container_path:ro"; only from allowed directories.
        ports: published ports, "host_port:container_port" or "container_port"; reachable from this machine only.
        memory: memory limit, for example "8g"; empty uses the default limit.
        shm_size: size of /dev/shm, for example "8g"; raise it when DataLoader workers fail with a Bus error.
        user: user to run as, for example "1000:1000"; use it when a mounted directory is not writable.
        ready_timeout: seconds to wait for readiness; 0 uses the default (30, or 120 with an image HEALTHCHECK).
        replace: remove an existing container of this name that has different settings.
    """
    def build() -> dict:
        spec = ContainerSpec(
            image=image, dockerfile=dockerfile, context=context, name=name, mode=mode, command=command,
            gpu=gpu, env=dict(env or {}), mounts=list(mounts or []), ports=list(ports or []),
            memory=memory, shm_size=shm_size or "2g", user=user, labels=dict(SESSION_LABELS),
        )
        result = provision(spec, replace=replace, ready_timeout=ready_timeout or None)
        reply = {
            "ok": result.ok,
            "action": result.action,
            "image": asdict(result.image),
            "status": status_view(result.status),
        }
        if result.diagnosis is not None:
            reply["diagnosis"] = diagnosis_view(result.diagnosis, with_status=False)
        return compact(reply)
    return _reply(build)


def container_status(container: str = "") -> str:
    """Reports the state and health of one container, or lists every container this toolchain made.

    Health is one of: starting, healthy, unhealthy, completed (ended with code 0), stopped, failed, missing.
    "usable" is true when commands can run in the container. A container that does not exist has health
    "missing" (not an error).

    Args:
        container: container name or id; empty lists all containers made by this toolchain.
    """
    def build() -> dict:
        if not container:
            return {"ok": True, "containers": [status_view(s) for s in list_managed()]}
        return {"ok": True, "status": status_view(get_status(container))}
    return _reply(build)


def container_logs(container: str, tail: int = DEFAULT_TAIL, since: str = "", grep: str = "") -> str:
    """Reads the runtime output of a container (not build output), stdout and stderr apart. Newest lines are kept.

    Args:
        container: container name or id; it may be stopped.
        tail: lines of each stream, 1 to 2000.
        since: only output after this point: a duration back from now such as "10m", or an ISO timestamp.
        grep: regular expression; only matching lines are returned, for example "Error|Traceback".
    """
    def build() -> dict:
        logs = asdict(get_logs(container, tail=tail, since=since, grep=grep))
        logs["truncated"] = logs["truncated"] or None
        return {"ok": True, "logs": compact(logs)}
    return _reply(build)


def diagnose_container(container: str) -> str:
    """Explains the state of a container: status, log tail, hints, processes, CPU and memory use, recent events.

    Use it when a container is not healthy, or when a command in it behaved strangely. Each hint has a code,
    the log line or state that shows it, a suggestion, and an owner: lifecycle (provision again with other
    options), dockerization (the image must change), execution (the command must change), or host (report to
    the user). No hints means no known pattern: read the logs.

    Args:
        container: container name or id.
    """
    return _reply(lambda: {"ok": True, "diagnosis": diagnosis_view(diagnose(container))})


def stop_container(container: str, timeout: int = STOP_TIMEOUT) -> str:
    """Stops a container and keeps it, with its files and logs. provision_container with the same settings starts it again.

    "killed": true in the reply means the process did not end in time after SIGTERM and was killed.
    Stopping a container that is not running changes nothing.

    Args:
        container: container name or id.
        timeout: seconds to wait after SIGTERM before SIGKILL, 0 to 600.
    """
    return _reply(lambda: {"ok": True, "result": teardown_view(stop(container, timeout=timeout))})


def destroy_container(container: str, force: bool = False) -> str:
    """Stops a container if it runs, then removes it and its anonymous volumes. The image and mounted host directories stay.

    Destroying a container that does not exist is not an error ("removed": false).

    Args:
        container: container name or id.
        force: kill at once instead of a graceful stop.
    """
    return _reply(lambda: {"ok": True, "result": teardown_view(destroy(container, force=force))})


TOOL_FUNCTIONS = (
    provision_container, container_status, container_logs,
    diagnose_container, stop_container, destroy_container,
)

# LangChain tools for the agent; the docstrings above become the descriptions the LLM reads
TOOLS = [tool(parse_docstring=True)(fn) for fn in TOOL_FUNCTIONS]
