"""
    Helpers for the lifecycle suites that use a real Docker daemon.

    Every container made here carries the label tml.test=<RUN_ID>, and cleanup removes
    only containers with that label. Containers of a real agent session and containers
    that have nothing to do with this toolchain are never touched.
"""

import os
import unittest
import uuid

from src.lifecycle import client
from src.lifecycle.models import LABEL_IMAGE, LABEL_MANAGED, LABEL_MODE, LABEL_TEST

# Already on most development machines of this project; pulled once if it is not
BASE_IMAGE = os.environ.get("TML_TEST_BASE_IMAGE", "python:3.14-slim")

RUN_ID = uuid.uuid4().hex[:8]
TEST_LABEL = {LABEL_TEST: RUN_ID}

DOCKER_READY = client.docker_available()
requires_docker = unittest.skipUnless(DOCKER_READY, "Docker daemon is not reachable")

SECOND = 1_000_000_000  # healthcheck times are in nanoseconds


def container_name(suffix: str) -> str:
    return f"tml-test-{RUN_ID}-{suffix}"


def ensure_base_image() -> str:
    """Makes sure BASE_IMAGE is on this machine (one pull if it is not)."""
    sdk = client.get_client()
    try:
        sdk.images.get(BASE_IMAGE)
    except Exception:
        sdk.images.pull(BASE_IMAGE)
    return BASE_IMAGE


def healthcheck(shell_command: str, retries: int = 1) -> dict:
    """A fast HEALTHCHECK (1 s interval) for the SDK's `healthcheck=` argument."""
    return {
        "test": ["CMD-SHELL", shell_command],
        "interval": SECOND, "timeout": SECOND, "retries": retries, "start_period": 0,
    }


def start_raw(suffix: str, command=None, *, managed: bool = True, mode: str = "native", start: bool = True, **kwargs):
    """Creates a container directly through the SDK (no lifecycle.provision).

    Args:
        suffix: unique part of the container name.
        command: command list; None uses the image's CMD.
        managed: add the tml.managed label. False makes a container the toolchain must refuse.
        mode: value of the tml.mode label.
        start: start the container after it is created.
        kwargs: passed to containers.create (entrypoint, healthcheck, mem_limit, ...).
    """
    labels = dict(TEST_LABEL)
    if managed:
        labels |= {LABEL_MANAGED: "true", LABEL_MODE: mode, LABEL_IMAGE: BASE_IMAGE}
    kwargs.setdefault("init", True)
    obj = client.get_client().containers.create(
        ensure_base_image(), command, name=container_name(suffix), labels=labels, **kwargs,
    )
    if start:
        obj.start()
    return obj


def start_idle(suffix: str, **kwargs):
    """A container that only stays alive, the way lifecycle's idle mode will start one."""
    return start_raw(suffix, ["infinity"], entrypoint=["sleep"], mode="idle", **kwargs)


def remove_test_containers() -> None:
    """Removes every container of this test run, managed or not."""
    if not DOCKER_READY:
        return
    sdk = client.get_client()
    for obj in sdk.containers.list(all=True, filters={"label": f"{LABEL_TEST}={RUN_ID}"}):
        obj.remove(force=True, v=True)
