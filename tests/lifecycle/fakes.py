"""
    Fakes for the lifecycle suites that need no Docker daemon.

    make_attrs() builds `docker inspect` output; FakeClient stands in for the SDK client.
    use_fake(test, fake) makes src.lifecycle.client.get_client() return the fake for one test.
"""

from collections import namedtuple
from unittest import mock

import requests
from docker import errors as docker_errors

from src.lifecycle import client
from src.lifecycle.models import LABEL_IMAGE, LABEL_MANAGED, LABEL_MODE

ExecResult = namedtuple("ExecResult", "exit_code output")

ZERO_TIME = "0001-01-01T00:00:00Z"


def make_attrs(
    name: str = "tml-demo",
    status: str = "running",
    exit_code: int = 0,
    oom: bool = False,
    error: str = "",
    health: str | None = None,       # starting | healthy | unhealthy; None = image has no HEALTHCHECK
    health_output: str = "",
    managed: bool = True,
    mode: str = "idle",
    image: str = "demo:latest",
    restart_count: int = 0,
    labels: dict | None = None,
    created: str = "2026-10-09T10:00:00Z",
    started: str | None = None,
    finished: str | None = None,
    **overrides,
) -> dict:
    """`docker inspect` output for one container, in the shape Engine 29 returns."""
    all_labels = dict(labels or {})
    if managed:
        all_labels |= {LABEL_MANAGED: "true", LABEL_MODE: mode, LABEL_IMAGE: image}
    ran = status not in ("created",)
    done = status in ("exited", "dead")
    state = {
        "Status": status,
        "Running": status == "running",
        "Paused": status == "paused",
        "Restarting": status == "restarting",
        "OOMKilled": oom,
        "Dead": status == "dead",
        "Pid": 4242 if status == "running" else 0,
        "ExitCode": exit_code,
        "Error": error,
        "StartedAt": started or ("2026-10-09T10:00:01Z" if ran else ZERO_TIME),
        "FinishedAt": finished or ("2026-10-09T10:00:05Z" if done else ZERO_TIME),
    }
    if health:
        state["Health"] = {
            "Status": health,
            "FailingStreak": 1 if health == "unhealthy" else 0,
            "Log": [{"ExitCode": 0 if health == "healthy" else 1, "Output": health_output}] if health != "starting" else [],
        }
    attrs = {
        "Id": (name.encode().hex() + "0" * 64)[:64],
        "Name": "/" + name,
        "Created": created,
        "Image": "sha256:" + "51dafde81dbd" + "0" * 52,
        "RestartCount": restart_count,
        "State": state,
        "Config": {"Image": image, "WorkingDir": "", "Labels": all_labels},
        "HostConfig": {"DeviceRequests": None, "PortBindings": {}, "Init": True},
        "NetworkSettings": {"Ports": {}},
        "Mounts": [],
    }
    attrs.update(overrides)
    return attrs


class FakeContainer:
    """SDK Container stand-in. `probe` is the exit code of exec_run, or an exception to raise."""

    def __init__(self, attrs: dict, probe: int | Exception = 0):
        self.attrs = attrs
        self.probe = probe
        self.exec_calls: list = []

    @property
    def id(self) -> str:
        return self.attrs["Id"]

    @property
    def name(self) -> str:
        return self.attrs["Name"].lstrip("/")

    @property
    def labels(self) -> dict:
        return self.attrs["Config"]["Labels"]

    def reload(self):
        pass

    def exec_run(self, cmd, **kwargs):
        self.exec_calls.append(cmd)
        if isinstance(self.probe, Exception):
            raise self.probe
        return ExecResult(self.probe, b"")


class FakeContainers:
    def __init__(self, containers, error: Exception | None = None):
        self._containers = list(containers)
        self._error = error

    def get(self, key: str):
        if self._error:
            raise self._error
        for c in self._containers:
            if key == c.name or (key and c.id.startswith(key)):
                return c
        raise docker_errors.NotFound(f"404 Client Error: Not Found", explanation=f"No such container: {key}")

    def list(self, all: bool = False, filters: dict | None = None):
        if self._error:
            raise self._error
        wanted = (filters or {}).get("label", [])
        wanted = [wanted] if isinstance(wanted, str) else wanted
        out = []
        for c in self._containers:
            if not all and c.attrs["State"]["Status"] != "running":
                continue
            if all_labels_match(c.labels, wanted):
                out.append(c)
        return out


def all_labels_match(labels: dict, wanted: list[str]) -> bool:
    for item in wanted:
        key, _, value = item.partition("=")
        if labels.get(key) != value:
            return False
    return True


class FakeClient:
    """SDK DockerClient stand-in. `error` makes every container call raise it."""

    def __init__(self, *containers: FakeContainer, info: dict | None = None, error: Exception | None = None):
        self.containers = FakeContainers(containers, error)
        self._info = info if info is not None else {"Runtimes": {"runc": {"path": "runc"}}}

    def ping(self) -> bool:
        return True

    def info(self) -> dict:
        return self._info


def use_fake(test, fake: FakeClient) -> FakeClient:
    """Makes client.get_client() return `fake` until the test ends."""
    patcher = mock.patch.object(client, "get_client", return_value=fake)
    patcher.start()
    test.addCleanup(patcher.stop)
    return fake


def api_error(status_code: int, explanation: str) -> docker_errors.APIError:
    """An SDK APIError with a real HTTP status code."""
    response = requests.Response()
    response.status_code = status_code
    return docker_errors.APIError(f"{status_code} Client Error", response=response, explanation=explanation)
