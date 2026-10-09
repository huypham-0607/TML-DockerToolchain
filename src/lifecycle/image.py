"""
    Section L1a - Image

    Makes sure there is an image to start a container from:
      - resolve_image(image)        an image that is already on this machine (never pulled)
      - ensure_image(dockerfile)    one plain `docker build` of a Dockerfile

    A built image is tagged tml-local/<name>:<hash> and labelled tml.built-by=lifecycle.
    The hash covers the Dockerfile text and the context path, so an unchanged Dockerfile
    reuses its image and skips the build. A change to other files of the context does
    NOT change the hash; pass rebuild=True for that.

    This is the only place lifecycle touches a build. There is no retry, no log analysis
    and no Dockerfile repair: those belong to the Dockerization toolchain. The build goes
    through the `docker build` command because the Docker SDK cannot use BuildKit.

    Plain Python helper, not an agent tool. Use ensure_image(dockerfile).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from docker import errors as docker_errors

from . import client
from .models import LABEL_BUILT_BY, ImageRef, LifecycleError


# Repository of every image this module builds
LOCAL_REPOSITORY = "tml-local"

# Seconds for one build (research images install large frameworks)
BUILD_TIMEOUT = 1800

# How much of a failed build's log goes into the error message
MAX_LOG_LINES = 40
MAX_LOG_CHARS = 3000


# ---------------------------------------------------------------- pure helpers

def _slug(name: str) -> str:
    """A valid image repository component: lowercase letters, digits, single separators."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "image"


def image_tag(dockerfile_text: str, context: str | Path, name: str = "") -> str:
    """The tag ensure_image gives a build: tml-local/<name>:<12 hex digits>.

    Args:
        dockerfile_text: content of the Dockerfile.
        context: build context directory; part of the hash so that two repositories
            with the same Dockerfile text do not share an image.
        name: image name; default is the name of the context directory.
    """
    context = Path(context).resolve()
    digest = hashlib.sha256(f"{dockerfile_text}\0{context}".encode()).hexdigest()[:12]
    return f"{LOCAL_REPOSITORY}/{_slug(name or context.name)}:{digest}"


def build_command(dockerfile: Path, context: Path, tag: str, labels: dict[str, str] | None = None) -> list[str]:
    """The `docker build` command line for one plain build."""
    command = ["docker", "build", "--file", str(dockerfile), "--tag", tag, "--progress", "plain"]
    for key, value in ({**(labels or {}), LABEL_BUILT_BY: "lifecycle"}).items():
        command += ["--label", f"{key}={value}"]
    return command + [str(context)]


def _log_tail(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-MAX_LOG_LINES:])[-MAX_LOG_CHARS:]


def _ref(image, tag: str, built: bool) -> ImageRef:
    return ImageRef(id=image.id.removeprefix("sha256:")[:12], tag=tag, built=built)


# ---------------------------------------------------------------- Docker access

def find_image(image: str):
    """Returns the SDK image object, or None if the image is not on this machine."""
    with client.translated(f"inspect image {image!r}"):
        try:
            return client.get_client().images.get(image)
        except docker_errors.ImageNotFound:
            return None


def resolve_image(image: str) -> ImageRef:
    """Returns the reference of an image that is already on this machine.

    Raises LifecycleError("image_not_found") otherwise. The image is never pulled.
    """
    found = find_image(image)
    if found is None:
        raise LifecycleError(
            "image_not_found",
            f"image {image!r} is not on this machine",
            "build the image first, or pass a dockerfile; lifecycle does not pull images",
        )
    return _ref(found, image, built=False)


def _check_paths(dockerfile: str | Path, context: str | Path) -> tuple[Path, Path]:
    dockerfile = Path(dockerfile).expanduser().resolve()
    if not dockerfile.is_file():
        raise LifecycleError("invalid_argument", f"Dockerfile not found: {dockerfile}")
    context = Path(context).expanduser().resolve() if context else dockerfile.parent
    if not context.is_dir():
        raise LifecycleError("invalid_argument", f"build context is not a directory: {context}")
    # the whole context is sent to the daemon and a Dockerfile can COPY any of it into the image
    if context in (Path("/"), Path.home()):
        raise LifecycleError(
            "invalid_argument",
            f"build context is too broad: {context}",
            "use the repository directory as the context",
        )
    return dockerfile, context


def ensure_image(
    dockerfile: str | Path,
    context: str | Path = "",
    name: str = "",
    labels: dict[str, str] | None = None,
    rebuild: bool = False,
    timeout: float = BUILD_TIMEOUT,
) -> ImageRef:
    """Builds the Dockerfile once and returns the image reference.

    If an image with the same tag (same Dockerfile text and context path) exists, the
    build is skipped and ImageRef.built is False.

    Args:
        dockerfile: path to the Dockerfile.
        context: build context directory; default is the directory of the Dockerfile.
        name: image name; default is the name of the context directory.
        labels: extra labels for the image.
        rebuild: build even if the image exists (other files of the context changed).
        timeout: seconds before the build is stopped.

    Raises:
        LifecycleError: invalid_argument, build_failed, timeout, or docker_unavailable.
    """
    dockerfile, context = _check_paths(dockerfile, context)
    try:
        text = dockerfile.read_text(errors="replace")
    except OSError as e:
        raise LifecycleError("invalid_argument", f"cannot read {dockerfile}: {e}") from e
    tag = image_tag(text, context, name)

    if not rebuild and (existing := find_image(tag)) is not None:
        return _ref(existing, tag, built=False)

    command = build_command(dockerfile, context, tag, labels)
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
    except FileNotFoundError as e:
        raise LifecycleError("docker_unavailable", "the `docker` command is not installed", "install the Docker CLI") from e
    except subprocess.TimeoutExpired as e:
        raise LifecycleError("timeout", f"docker build of {dockerfile} did not finish in {timeout:g} s") from e

    if result.returncode != 0:
        log = _log_tail(result.stderr + "\n" + result.stdout)
        if "cannot connect to the docker daemon" in log.lower():
            raise LifecycleError(
                "docker_unavailable", f"could not build {dockerfile}: {log}",
                "check that the Docker daemon is running and that this user can reach it",
            )
        raise LifecycleError(
            "build_failed",
            f"docker build of {dockerfile} failed (exit code {result.returncode}):\n{log}",
            "the build belongs to the Dockerization toolchain: fix the Dockerfile, then provision again",
        )

    built = find_image(tag)
    if built is None:  # the build reported success, so this should not happen
        raise LifecycleError("docker_error", f"docker build succeeded but image {tag!r} is not there")
    return _ref(built, tag, built=True)
