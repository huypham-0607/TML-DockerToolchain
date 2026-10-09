"""
    Container Lifecycle - Structured Objects

    Dataclasses shared by every lifecycle section, plus the one exception type:
      - ContainerSpec     what to start (input of provision)
      - ImageRef          an image that is ready to run
      - ContainerHandle   the hand-off object for the execution toolchain
      - ContainerStatus   Docker state plus the derived health verdict
      - ProvisionResult   what provision returns: status, image, and what it did
      - ContainerLogs, Hint, Diagnosis   results of the diagnostics section
      - LifecycleError    raised by the Python API; agent tools turn it into {"ok": false, ...}

    Plain Python, no Docker calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict


# Labels: Docker is the source of truth, so ownership and mode live on the container itself
LABEL_MANAGED = "tml.managed"     # "true" on every container this toolchain created
LABEL_MODE = "tml.mode"           # idle | native
LABEL_IMAGE = "tml.image"         # image reference the container was started from
LABEL_SPEC = "tml.spec"           # fingerprint of the image and settings, to decide if a container can be reused
LABEL_BUILT_BY = "tml.built-by"   # "lifecycle" on images built by lifecycle.image
LABEL_TEST = "tml.test"           # test run id, so test cleanup never touches a real session
LABEL_PREFIX = "tml."

MODES = ("idle", "native")
GPU_OPTIONS = ("auto", "none", "all")

# Health verdicts, derived in status.derive_health
HEALTH = ("starting", "healthy", "unhealthy", "completed", "stopped", "failed", "missing")

# Which toolchain acts on a hint
OWNERS = ("lifecycle", "dockerization", "execution", "host")

ERROR_CODES = (
    "docker_unavailable",  # daemon not reachable
    "image_not_found",     # image is not on this machine (lifecycle never pulls implicitly)
    "build_failed",        # the one plain build of a Dockerfile failed
    "not_found",           # no such container
    "not_managed",         # container exists but was not created by this toolchain
    "not_running",         # operation needs a running container
    "name_conflict",       # name is taken by a container of a different image
    "invalid_argument",
    "gpu_unavailable",
    "port_conflict",
    "timeout",
    "docker_error",        # anything else the daemon rejected
)

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]+$")


class LifecycleError(Exception):
    """Every failure of the lifecycle Python API. `code` is one of ERROR_CODES."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}

    def envelope(self) -> dict:
        """The failure shape every agent tool returns."""
        return {"ok": False, "error": self.to_dict()}


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class ContainerSpec(_Serializable):
    image: str = ""                 # image tag or id; exactly one of image / dockerfile
    dockerfile: str = ""            # path to a Dockerfile, built once by lifecycle.image
    context: str = ""               # build context (default: the Dockerfile's directory)
    name: str = ""                  # generated when empty
    mode: str = "idle"              # idle: keep alive for exec | native: the image's own ENTRYPOINT/CMD
    command: str = ""               # native mode only: overrides the image's CMD
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[str] = field(default_factory=list)   # "host_path:container_path[:ro]"
    ports: list[str] = field(default_factory=list)    # "host:container" or "container"
    workdir: str = ""
    gpu: str = "auto"               # auto: use the GPU if the daemon has an NVIDIA runtime
    memory: str = ""                # e.g. "8g"; empty = default limit (a share of the host memory)
    cpus: float = 0.0               # 0 = no limit
    shm_size: str = "2g"            # PyTorch DataLoader workers need more than Docker's 64 MB
    network: str = "bridge"         # bridge | none
    user: str = ""
    labels: dict[str, str] = field(default_factory=dict)  # extra labels for the container (and a built image)

    def validate(self) -> "ContainerSpec":
        """Checks the shape of the spec. Raises LifecycleError("invalid_argument")."""
        def bad(message: str, hint: str = ""):
            raise LifecycleError("invalid_argument", message, hint)

        if bool(self.image) == bool(self.dockerfile):
            bad("give exactly one of image / dockerfile", "image: a tag or id that already exists; dockerfile: a path to build")
        if self.context and not self.dockerfile:
            bad("context is only used together with dockerfile")
        if self.mode not in MODES:
            bad(f"unknown mode {self.mode!r}", f"choose one of {list(MODES)}")
        if self.gpu not in GPU_OPTIONS:
            bad(f"unknown gpu option {self.gpu!r}", f"choose one of {list(GPU_OPTIONS)}")
        if self.command and self.mode != "native":
            bad("command needs mode='native'", "idle mode only keeps the container alive; run commands with the execution tools")
        if self.name and not _NAME_RE.match(self.name):
            bad(f"invalid container name {self.name!r}", "use letters, digits, '_', '.', '-'; at least 2 characters")
        if self.network not in ("bridge", "none"):
            bad(f"unknown network {self.network!r}", "choose 'bridge' or 'none'")
        if self.cpus < 0:
            bad("cpus must not be negative")
        return self


@dataclass
class ImageRef(_Serializable):
    id: str                   # short image id
    tag: str                  # reference to run it with
    built: bool = False       # True if this call built it (False: it already existed)


@dataclass
class ContainerHandle(_Serializable):
    id: str                   # short container id
    name: str
    image: str                # reference the container was created from
    image_id: str             # short image id
    mode: str                 # idle | native
    gpu: bool = False
    workdir: str = ""
    ports: dict[str, str] = field(default_factory=dict)   # "8888/tcp" -> host port
    mounts: list[str] = field(default_factory=list)       # "source:destination[:ro]"
    created_at: str = ""
    labels: dict[str, str] = field(default_factory=dict)  # tml.* labels only


@dataclass
class ContainerStatus(_Serializable):
    container: str                        # the name or id that was asked for
    state: str                            # Docker's: created | running | paused | restarting | removing | exited | dead | missing
    health: str                           # one of HEALTH
    reason: str = ""                      # one line: why this verdict
    handle: ContainerHandle | None = None  # None when the container is missing
    exit_code: int | None = None          # None while it has not exited
    oom_killed: bool = False
    restart_count: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    error: str = ""                       # Docker's own start error, if any
    healthcheck_output: str = ""          # last output of the image HEALTHCHECK, if it has one

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def usable(self) -> bool:
        """True when the execution toolchain can run commands in it."""
        return self.health == "healthy"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["running"] = self.running
        d["usable"] = self.usable
        return d


@dataclass
class ProvisionResult(_Serializable):
    status: ContainerStatus
    image: ImageRef
    action: str                # created | replaced | reused | restarted
    diagnosis: Diagnosis | None = None     # filled when the container did not become usable

    @property
    def handle(self) -> ContainerHandle | None:
        return self.status.handle

    @property
    def ok(self) -> bool:
        """True if the container is ready, or if a native-mode command ran to a clean end."""
        mode = self.status.handle.mode if self.status.handle else ""
        return self.status.usable or (mode == "native" and self.status.health == "completed")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "action": self.action,
            "image": self.image.to_dict(),
            "status": self.status.to_dict(),
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
        }


@dataclass
class ContainerLogs(_Serializable):
    container: str
    stdout: str = ""
    stderr: str = ""
    lines: int = 0             # lines returned (stdout + stderr)
    truncated: bool = False    # True if older output was cut to fit the character budget
    since: str = ""


@dataclass
class Hint(_Serializable):
    code: str                  # e.g. oom, missing_python_module, gpu_unavailable
    evidence: str              # the log line or state field that triggered it
    suggestion: str
    owner: str                 # one of OWNERS: which toolchain acts on it


@dataclass
class Diagnosis(_Serializable):
    status: ContainerStatus
    logs: ContainerLogs | None = None
    hints: list[Hint] = field(default_factory=list)
    processes: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.to_dict()
        return d
