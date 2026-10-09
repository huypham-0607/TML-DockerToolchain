"""
    Fakes for the lifecycle suites that need no Docker daemon.

    make_attrs() builds `docker inspect` output; FakeClient stands in for the SDK client
    (containers.get / list, container stop / remove / reload / exec_run, images.get / list / remove,
    ping, info).
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
    """SDK Container stand-in.

    Args:
        attrs: `docker inspect` output (see make_attrs).
        probe: exit code of exec_run, or an exception to raise; a list gives one entry
            to each call in turn (the last one repeats).
        after_reload: attrs that reload() switches to, as if the state changed meanwhile;
            an exception makes reload() raise it.
        stop_exit: exit code the container has after stop() (143 = ended on SIGTERM,
            137 = needed SIGKILL); an exception makes stop() raise it.
        stop_oom: stop() also sets OOMKilled.
        remove_error: an exception that remove() raises.
    """

    def __init__(
        self, attrs: dict, probe=0, after_reload: dict | Exception | None = None,
        stop_exit: int | Exception = 143, stop_oom: bool = False, remove_error: Exception | None = None,
    ):
        self.attrs = attrs
        self.probe = probe
        self.after_reload = after_reload
        self.stop_exit = stop_exit
        self.stop_oom = stop_oom
        self.remove_error = remove_error
        self.exec_calls: list = []
        self.calls: list[tuple] = []        # ("stop", timeout) / ("remove", force, v), in order
        self.owner: "FakeContainers | None" = None

    @property
    def id(self) -> str:
        return self.attrs["Id"]

    @property
    def name(self) -> str:
        return self.attrs["Name"].lstrip("/")

    @property
    def labels(self) -> dict:
        return self.attrs["Config"]["Labels"]

    @property
    def status(self) -> str:
        return self.attrs["State"]["Status"]

    def reload(self):
        if isinstance(self.after_reload, Exception):
            raise self.after_reload
        if self.after_reload is not None:
            self.attrs = self.after_reload

    def exec_run(self, cmd, **kwargs):
        self.exec_calls.append(cmd)
        outcome = self.probe
        if isinstance(outcome, list):
            outcome = outcome[min(len(self.exec_calls), len(outcome)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return ExecResult(outcome, b"")

    def stop(self, timeout: int = 10):
        self.calls.append(("stop", timeout))
        if isinstance(self.stop_exit, Exception):
            raise self.stop_exit
        state = {
            **self.attrs["State"], "Status": "exited", "Running": False, "Paused": False, "Restarting": False,
            "Pid": 0, "ExitCode": self.stop_exit, "OOMKilled": self.stop_oom, "FinishedAt": "2026-10-09T10:30:00Z",
        }
        self.attrs = {**self.attrs, "State": state}

    def remove(self, force: bool = False, v: bool = False):
        self.calls.append(("remove", force, v))
        if self.remove_error:
            raise self.remove_error
        if self.owner is not None:
            self.owner.discard(self)


class FakeContainers:
    def __init__(self, containers, error: Exception | None = None):
        self._containers = list(containers)
        self._error = error
        for c in self._containers:
            c.owner = self

    def discard(self, container) -> None:
        if container in self._containers:
            self._containers.remove(container)

    def names(self) -> list[str]:
        return [c.name for c in self._containers]

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


class FakeImage:
    def __init__(self, tag: str, image_id: str = "sha256:" + "51dafde81dbd" + "0" * 52, labels: dict | None = None):
        self.id = image_id
        self.tags = [tag]
        self.labels = labels or {}


class FakeImages:
    """Only the images given to it exist; like lifecycle, it never pulls.

    `remove_errors` maps a tag (or id) to the exception that remove() raises for it.
    """

    def __init__(self, images, remove_errors: dict | None = None):
        self._images = list(images)
        self.remove_errors = remove_errors or {}
        self.get_calls: list[str] = []
        self.remove_calls: list[tuple] = []     # (tag or id, force)

    def tags(self) -> list[str]:
        return [tag for image in self._images for tag in image.tags]

    def list(self, filters: dict | None = None):
        wanted = (filters or {}).get("label", [])
        wanted = [wanted] if isinstance(wanted, str) else wanted
        return [image for image in self._images if all_labels_match(image.labels, wanted)]

    def remove(self, name: str, force: bool = False):
        self.remove_calls.append((name, force))
        if name in self.remove_errors:
            raise self.remove_errors[name]
        for image in self._images:
            if name in image.tags or name == image.id:
                image.tags = [t for t in image.tags if t != name]
                if not image.tags:
                    self._images.remove(image)
                return
        raise docker_errors.ImageNotFound("404 Client Error: Not Found", explanation=f"No such image: {name}")

    def get(self, name: str):
        self.get_calls.append(name)
        for image in self._images:
            if name in image.tags or image.id.startswith(name) or image.id.removeprefix("sha256:").startswith(name):
                return image
        raise docker_errors.ImageNotFound("404 Client Error: Not Found", explanation=f"No such image: {name}")


class FakeClient:
    """SDK DockerClient stand-in. `error` makes every container call raise it."""

    def __init__(
        self, *containers: FakeContainer, images=(), info: dict | None = None,
        error: Exception | None = None, image_remove_errors: dict | None = None,
    ):
        self.containers = FakeContainers(containers, error)
        self.images = FakeImages(images, image_remove_errors)
        self._info = info if info is not None else {"Runtimes": {"runc": {"path": "runc"}}, "MemTotal": 16 * 1024**3}

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
