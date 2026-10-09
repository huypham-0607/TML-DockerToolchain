"""
    Section L4 - Teardown

    Stops and removes the containers of this toolchain:
      - stop(container)       graceful stop; the container, its files and its logs are kept
      - remove(container)     removes a stopped container and its anonymous volumes
      - destroy(container)    stop if needed, then remove; a missing container is not an error
      - destroy_all_managed() the same for every managed container (end of a session, test cleanup)
      - remove_built_images() removes images that lifecycle.image built

    Only containers with the label tml.managed=true are touched; any other container gives
    LifecycleError("not_managed"). Images are removed only if lifecycle built them
    (label tml.built-by=lifecycle): every other image belongs to the Dockerization toolchain.

    Plain Python helper, not an agent tool. Use destroy(container).
"""

from __future__ import annotations

from docker import errors as docker_errors

from . import client
from .models import LABEL_BUILT_BY, ContainerStatus, LifecycleError, TeardownResult
from .status import list_managed, require_managed, status_from_attrs


# Seconds between SIGTERM and SIGKILL of a stop
STOP_TIMEOUT = 10
MAX_STOP_TIMEOUT = 600

# Docker states in which a container still has a process
_ACTIVE = ("running", "paused", "restarting")

# Exit code of a process that was ended with SIGKILL
_SIGKILL_EXIT = 137


def _check_timeout(timeout: int) -> int:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= MAX_STOP_TIMEOUT:
        raise LifecycleError(
            "invalid_argument", f"invalid stop timeout {timeout!r}",
            f"use a number of seconds from 0 to {MAX_STOP_TIMEOUT}",
        )
    return int(timeout)


def _gone(container: str, reason: str) -> ContainerStatus:
    return ContainerStatus(container=container, state="missing", health="missing", reason=reason)


def _stop(obj, container: str, timeout: int) -> TeardownResult:
    before = status_from_attrs(obj.attrs, asked=container)
    if before.state not in _ACTIVE:
        return TeardownResult(container=container, status=before, exit_code=before.exit_code)
    with client.translated(f"stop container {container!r}"):
        obj.stop(timeout=timeout)   # SIGTERM, then SIGKILL after `timeout` seconds
        obj.reload()
    after = status_from_attrs(obj.attrs, asked=container)
    killed = after.exit_code == _SIGKILL_EXIT and not after.oom_killed
    if killed:
        after.reason += f"; it did not end within {timeout} s of SIGTERM"
    return TeardownResult(container=container, status=after, stopped=True, killed=killed, exit_code=after.exit_code)


def _remove(obj, container: str, force: bool, volumes: bool) -> TeardownResult:
    before = status_from_attrs(obj.attrs, asked=container)
    active = before.state in _ACTIVE
    if active and not force:
        raise LifecycleError(
            "still_running",
            f"container {container!r} is {before.state}",
            "stop it first, use destroy, or pass force=True",
        )
    with client.translated(f"remove container {container!r}"):
        obj.remove(force=force, v=volumes)
    return TeardownResult(
        container=container,
        status=_gone(container, "removed"),
        stopped=active, killed=active,   # a forced removal ends the process with SIGKILL
        removed=True,
        exit_code=before.exit_code,
    )


def stop(container: str, timeout: int = STOP_TIMEOUT) -> TeardownResult:
    """Stops a managed container and keeps it (files and logs stay for diagnosis).

    The process gets SIGTERM, and SIGKILL if it has not ended after `timeout` seconds;
    TeardownResult.killed reports that. A container that is not running is left as it is.
    provision() with the same name and settings starts a stopped container again.

    Args:
        container: container name or id.
        timeout: seconds between SIGTERM and SIGKILL.

    Raises:
        LifecycleError: not_found, not_managed, invalid_argument.
    """
    timeout = _check_timeout(timeout)
    return _stop(require_managed(container), container, timeout)


def remove(container: str, force: bool = False, volumes: bool = True) -> TeardownResult:
    """Removes a managed container. Its image is not touched.

    Args:
        container: container name or id.
        force: also remove a running container (it is killed, not stopped).
        volumes: remove the anonymous volumes of the container. Mounted host directories are never removed.

    Raises:
        LifecycleError: not_found, not_managed, still_running (running and force is False).
    """
    return _remove(require_managed(container), container, force, volumes)


def destroy(container: str, force: bool = False, timeout: int = STOP_TIMEOUT) -> TeardownResult:
    """Stops a managed container if it runs, then removes it and its anonymous volumes.

    A container that does not exist is not an error (TeardownResult.removed is False),
    so destroy can be called again after a failure. The image is not removed.

    Args:
        container: container name or id.
        force: kill the container at once instead of a graceful stop.
        timeout: seconds between SIGTERM and SIGKILL of the graceful stop.

    Raises:
        LifecycleError: not_managed, invalid_argument.
    """
    timeout = _check_timeout(timeout)
    try:
        obj = require_managed(container)
    except LifecycleError as e:
        if e.code == "not_found":
            return TeardownResult(container=container, status=_gone(container, "no such container"))
        raise

    stopped = None if force else _stop(obj, container, timeout)
    try:
        removed = _remove(obj, container, force=force, volumes=True)
    except LifecycleError as e:
        if e.code != "not_found":
            raise
        # something else removed it between the stop and the removal
        removed = TeardownResult(container=container, status=_gone(container, "no such container"))
    if stopped is not None:
        removed.stopped, removed.killed, removed.exit_code = stopped.stopped, stopped.killed, stopped.exit_code
    return removed


def destroy_all_managed(
    labels: dict[str, str] | None = None,
    force: bool = False,
    timeout: int = STOP_TIMEOUT,
) -> list[TeardownResult]:
    """Destroys every container this toolchain created.

    Every container is tried even if one fails; the failures are raised together at the end.

    Args:
        labels: only containers that also have these labels (label -> value).
        force: kill the containers at once instead of a graceful stop.
        timeout: seconds between SIGTERM and SIGKILL of each graceful stop.

    Raises:
        LifecycleError: docker_error with the list of containers that could not be destroyed.
    """
    timeout = _check_timeout(timeout)
    results, failures = [], []
    for status in list_managed(all=True, labels=labels, probe=False):
        try:
            results.append(destroy(status.container, force=force, timeout=timeout))
        except LifecycleError as e:
            failures.append(f"{status.container}: {e.message}")
    if failures:
        raise LifecycleError(
            "docker_error",
            f"could not destroy {len(failures)} of {len(failures) + len(results)} containers: " + "; ".join(failures),
        )
    return results


def remove_built_images(labels: dict[str, str] | None = None, force: bool = False) -> list[str]:
    """Removes the images that lifecycle.image built and returns their tags.

    An image that a container still uses is skipped (unless `force`), so destroy the
    containers first. Images that lifecycle did not build are never removed.

    Args:
        labels: only images that also have these labels (label -> value).
        force: also remove an image that a stopped container still uses.
    """
    wanted = [f"{LABEL_BUILT_BY}=lifecycle"] + [f"{k}={v}" for k, v in (labels or {}).items()]
    sdk = client.get_client()
    with client.translated("list images"):
        images = sdk.images.list(filters={"label": wanted})

    removed: list[str] = []
    for image in images:
        for tag in image.tags or [image.id]:
            with client.translated(f"remove image {tag!r}"):
                try:
                    sdk.images.remove(tag, force=force)
                except docker_errors.ImageNotFound:
                    continue   # removed in the meantime (several tags can share one image)
                except docker_errors.APIError as e:
                    if e.status_code != 409:
                        raise
                    continue   # still used by a container
            removed.append(tag)
    return removed
