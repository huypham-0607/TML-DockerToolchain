"""
    Sample Dockerfiles and helpers for the lifecycle suites that use a real Docker daemon.

    No image is shipped with the repository: each sample below is a small Dockerfile that
    is written to a temporary directory and built on first use (lifecycle.image).
    Containers are started with lifecycle.provision, the same way the agent starts them.

    Every container and image made here carries the label tml.test=<RUN_ID>, and cleanup()
    removes only what has that label. Containers of a real agent session and containers
    that have nothing to do with this toolchain are never touched.
"""

import atexit
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from src.lifecycle import client
from src.lifecycle.image import ensure_image
from src.lifecycle.models import LABEL_TEST, ContainerSpec, ImageRef, ProvisionResult
from src.lifecycle.provision import DEFAULT_MOUNT_ROOT, provision

# Already on most development machines of this project; `docker build` pulls it once if not
BASE_IMAGE = os.environ.get("TML_TEST_BASE_IMAGE", "python:3.14-slim")

RUN_ID = uuid.uuid4().hex[:8]
TEST_LABEL = {LABEL_TEST: RUN_ID}

DOCKER_READY = client.docker_available()
requires_docker = unittest.skipUnless(DOCKER_READY, "Docker daemon is not reachable")

# Touches memory in 8 MB steps until the container's memory limit kills it
_MEMORY_HOG = r"chunks = []\nwhile True:\n    chunks.append(bytearray(b'x' * 8388608))"

# name -> Dockerfile text; what each one is for is in the comment above it
DOCKERFILES = {
    # CMD prints one line and exits: idle mode must keep it alive, native mode "completes"
    "idle": f"""\
FROM {BASE_IMAGE}
WORKDIR /workspace
CMD ["python", "-c", "print('done')"]
""",
    # has an ENTRYPOINT: idle mode must override it
    "entrypoint": f"""\
FROM {BASE_IMAGE}
ENTRYPOINT ["python"]
CMD ["--version"]
""",
    # long-running, prints "tick"; its HEALTHCHECK passes about one second after the start
    "service": f"""\
FROM {BASE_IMAGE}
HEALTHCHECK --interval=1s --timeout=1s --retries=5 CMD test -f /tmp/ready
CMD ["sh", "-c", "sleep 1; touch /tmp/ready; while true; do echo tick; sleep 1; done"]
""",
    # long-running, but its HEALTHCHECK always fails
    "unhealthy": f"""\
FROM {BASE_IMAGE}
HEALTHCHECK --interval=1s --timeout=1s --retries=1 CMD echo db down; exit 1
CMD ["sleep", "300"]
""",
    # writes to stderr and exits with code 3
    "crash": f"""\
FROM {BASE_IMAGE}
CMD ["python", "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]
""",
    # a real ModuleNotFoundError: the image lacks a package the code needs
    "missing-module": f"""\
FROM {BASE_IMAGE}
CMD ["python", "-c", "import torch"]
""",
    # allocates memory without end: killed by the kernel under a memory limit
    "oom": f"""\
FROM {BASE_IMAGE}
CMD ["python", "-c", "{_MEMORY_HOG}"]
""",
    # no shell, no `sleep`, nothing to run: idle mode cannot start in it (needs no base image)
    "empty": """\
FROM scratch
COPY Dockerfile /Dockerfile
""",
    # the build itself fails
    "broken-build": f"""\
FROM {BASE_IMAGE}
RUN echo about to fail && exit 7
""",
}

_workdir: Path | None = None       # holds one directory per sample
_mount_dirs: list[Path] = []
_made_mount_root = False           # True if this run created runs/, so cleanup may remove it


def dockerfile(sample: str) -> str:
    """Path of the sample's Dockerfile, written on first use. Its directory is the build context."""
    global _workdir
    if _workdir is None:
        _workdir = Path(tempfile.mkdtemp(prefix="tml-samples-")).resolve()
        atexit.register(shutil.rmtree, _workdir, ignore_errors=True)
    path = _workdir / sample / "Dockerfile"
    if not path.exists():
        path.parent.mkdir(parents=True)
        path.write_text(DOCKERFILES[sample])
    return str(path)


def build(sample: str, **kwargs) -> ImageRef:
    """Builds the sample (or finds the image of an earlier build in this run)."""
    return ensure_image(dockerfile(sample), labels=TEST_LABEL, **kwargs)


def container_name(suffix: str) -> str:
    return f"tml-test-{RUN_ID}-{suffix}"


def spec(sample: str, suffix: str, **fields) -> ContainerSpec:
    """A ContainerSpec for the sample, named and labelled for this test run."""
    fields.setdefault("labels", {})
    fields["labels"] = {**fields["labels"], **TEST_LABEL}
    return ContainerSpec(dockerfile=dockerfile(sample), name=container_name(suffix), **fields)


def start(sample: str, suffix: str, replace: bool = False, ready_timeout: float | None = None, **fields) -> ProvisionResult:
    """Provisions a container from the sample. `fields` are ContainerSpec fields (mode, command, memory, ...)."""
    return provision(spec(sample, suffix, **fields), replace=replace, ready_timeout=ready_timeout)


def sdk_container(name: str):
    """The SDK object of a container, for assertions on what Docker really has."""
    return client.get_client().containers.get(name)


def start_unmanaged(suffix: str):
    """A running container WITHOUT the tml.managed label: one the toolchain must refuse."""
    obj = client.get_client().containers.create(
        build("idle").tag, ["infinity"], entrypoint=["sleep"], init=True,
        name=container_name(suffix), labels=dict(TEST_LABEL),
    )
    obj.start()
    return obj


def mount_dir() -> Path:
    """A fresh directory that may be mounted: it is inside the default mount root (runs/).

    A system temporary directory would not do: Docker Desktop does not share /tmp.
    """
    global _made_mount_root
    if not DEFAULT_MOUNT_ROOT.exists():
        DEFAULT_MOUNT_ROOT.mkdir()
        _made_mount_root = True
    path = Path(tempfile.mkdtemp(prefix=f"test-{RUN_ID}-", dir=DEFAULT_MOUNT_ROOT)).resolve()
    _mount_dirs.append(path)
    return path


def remove_containers() -> None:
    """Removes every container of this test run, managed or not."""
    if not DOCKER_READY:
        return
    for obj in client.get_client().containers.list(all=True, filters={"label": f"{LABEL_TEST}={RUN_ID}"}):
        obj.remove(force=True, v=True)


def cleanup() -> None:
    """Removes every container, image and mount directory of this test run."""
    global _made_mount_root
    while _mount_dirs:
        shutil.rmtree(_mount_dirs.pop(), ignore_errors=True)
    if _made_mount_root:
        _made_mount_root = False
        try:
            DEFAULT_MOUNT_ROOT.rmdir()   # only if nothing else was put there meanwhile
        except OSError:
            pass
    if not DOCKER_READY:
        return
    remove_containers()
    sdk = client.get_client()
    for image in sdk.images.list(filters={"label": f"{LABEL_TEST}={RUN_ID}"}):
        for tag in image.tags or [image.id]:
            sdk.images.remove(tag, force=True)


atexit.register(cleanup)
