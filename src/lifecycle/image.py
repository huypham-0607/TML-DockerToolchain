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
    and no Dockerfile repair: those belong to the Dockerization toolchain.

    Two builders, chosen by TML_BUILDER (auto | buildkit | legacy; default auto):
      buildkit  `docker build` with the buildx plugin: BuildKit features and cache. Used when
                `docker buildx version` works.
      legacy    the daemon's own classic builder, through the Docker SDK: needs neither the
                `docker` command nor buildx. BuildKit-only Dockerfile syntax (RUN --mount,
                here-documents, # syntax=) fails there with a hint to install buildx.

    Plain Python helper, not an agent tool. Use ensure_image(dockerfile).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from docker import errors as docker_errors
from docker.credentials.errors import StoreError

from . import client
from .models import LABEL_BUILT_BY, ImageRef, LifecycleError


# Repository of every image this module builds
LOCAL_REPOSITORY = "tml-local"

# Seconds for one build (research images install large frameworks)
BUILD_TIMEOUT = 1800

# How much of a failed build's log goes into the error message
MAX_LOG_LINES = 40

BUILDER_ENV = "TML_BUILDER"
BUILDERS = ("auto", "buildkit", "legacy")

BUILDX_HINT = (
    "install the buildx plugin of the docker command (package docker-buildx on Arch and Fedora, "
    "docker-buildx-plugin from Docker's own Debian/Ubuntu repository)"
)

# Dockerfile syntax that only BuildKit understands
_BUILDKIT_ONLY_RE = re.compile(
    r"^\s*#\s*syntax\s*=|--mount=|--network=|--security=|--link\b|--chmod=|--exclude=|<<-?\s*[\"']?\w+",
    re.I | re.M,
)

_buildkit_cache: bool | None = None
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


def _ref(image, tag: str, built: bool, builder: str = "") -> ImageRef:
    return ImageRef(id=image.id.removeprefix("sha256:")[:12], tag=tag, built=built, builder=builder)


def needs_buildkit(dockerfile_text: str) -> bool:
    """True if the Dockerfile uses syntax that the classic builder does not understand."""
    return bool(_BUILDKIT_ONLY_RE.search(dockerfile_text))


# ---------------------------------------------------------------- builder choice

def buildkit_available() -> bool:
    """True if `docker buildx` works. Asked once per process."""
    global _buildkit_cache
    if _buildkit_cache is None:
        try:
            done = subprocess.run(["docker", "buildx", "version"], capture_output=True, timeout=30)
            _buildkit_cache = done.returncode == 0
        except (OSError, subprocess.TimeoutExpired):   # no docker command at all
            _buildkit_cache = False
    return _buildkit_cache


def choose_builder() -> str:
    """buildkit or legacy, from TML_BUILDER and what is installed."""
    wanted = (os.environ.get(BUILDER_ENV) or "auto").strip().lower()
    if wanted not in BUILDERS:
        raise LifecycleError("invalid_argument", f"{BUILDER_ENV}={wanted!r} is not a builder", f"choose one of {list(BUILDERS)}")
    if wanted == "auto":
        return "buildkit" if buildkit_available() else "legacy"
    if wanted == "buildkit" and not buildkit_available():
        raise LifecycleError("docker_unavailable", f"{BUILDER_ENV}=buildkit, but `docker buildx` does not work", BUILDX_HINT)
    return wanted


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

    builder = choose_builder()
    if builder == "buildkit" and not _buildkit_build(dockerfile, context, tag, labels, timeout):
        builder = "legacy"   # buildx turned out to be broken: the classic builder still works
    if builder == "legacy":
        _legacy_build(dockerfile, context, tag, labels, timeout, text)

    built = find_image(tag)
    if built is None:  # the build reported success, so this should not happen
        raise LifecycleError("docker_error", f"docker build succeeded but image {tag!r} is not there")
    return _ref(built, tag, built=True, builder=builder)


def _build_failed(dockerfile: Path, detail: str, log: str, needs_bk: bool) -> LifecycleError:
    if needs_bk:
        return LifecycleError(
            "build_failed",
            f"docker build of {dockerfile} failed ({detail}): this Dockerfile needs BuildKit, "
            f"and buildx is not installed:\n{log}",
            BUILDX_HINT,
        )
    return LifecycleError(
        "build_failed",
        f"docker build of {dockerfile} failed ({detail}):\n{log}",
        "the build belongs to the Dockerization toolchain: fix the Dockerfile, then provision again",
    )


def _buildkit_build(dockerfile: Path, context: Path, tag: str, labels: dict | None, timeout: float) -> bool:
    """One `docker build` with BuildKit. False if buildx turned out not to work (nothing was built)."""
    global _buildkit_cache
    try:
        result = subprocess.run(
            build_command(dockerfile, context, tag, labels),
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
    except FileNotFoundError:
        _buildkit_cache = False
        return False
    except subprocess.TimeoutExpired as e:
        raise LifecycleError("timeout", f"docker build of {dockerfile} did not finish in {timeout:g} s") from e
    if result.returncode == 0:
        return True

    log = _log_tail(result.stderr + "\n" + result.stdout)
    low = log.lower()
    if "buildx component is missing" in low:
        _buildkit_cache = False
        return False
    if "permission denied while trying to connect" in low:
        raise LifecycleError("docker_unavailable", f"could not build {dockerfile}: {log}", client.SOCKET_PERMISSION_HINT)
    if "cannot connect to the docker daemon" in low or "failed to connect to the docker api" in low:
        raise LifecycleError(
            "docker_unavailable", f"could not build {dockerfile}: {log}",
            "check that the Docker daemon is running and that this user can reach it",
        )
    raise _build_failed(dockerfile, f"exit code {result.returncode}", log, needs_bk=False)


@contextmanager
def _without_registry_auth(api):
    """Builds without registry credentials. The SDK reads them for every build, and fails if
    ~/.docker/config.json names a credential helper that is not installed (Docker Desktop leaves
    "credsStore": "desktop" behind). Public base images need no credentials."""
    api._set_auth_headers = lambda headers: None   # shadows the SDK method on this client only
    try:
        yield
    finally:
        del api._set_auth_headers


def _consume(stream, deadline: float) -> tuple[str, str]:
    """Reads a classic-builder stream. Returns (log, error message or "")."""
    log: list[str] = []
    try:
        for chunk in stream:
            if time.monotonic() > deadline:
                raise LifecycleError("timeout", "the build did not finish in time")
            if "error" in chunk:
                return "".join(log), str(chunk.get("error") or "")
            text = chunk.get("stream") or chunk.get("status")
            if text:
                log.append(text if text.endswith("\n") else text + "\n")
    finally:
        if hasattr(stream, "close"):
            stream.close()   # an abandoned build stops when its connection closes
    return "".join(log), ""


def _legacy_build(dockerfile: Path, context: Path, tag: str, labels: dict | None, timeout: float, text: str) -> None:
    """One build with the daemon's classic builder, through the SDK."""
    api = client.get_client().api
    inside = os.path.relpath(dockerfile, context)
    options = {
        # a Dockerfile inside the context is named relative to it; one outside is sent along by the SDK
        "path": str(context), "dockerfile": str(dockerfile) if inside.startswith("..") else inside, "tag": tag,
        "labels": {**(labels or {}), LABEL_BUILT_BY: "lifecycle"},
        "rm": True, "forcerm": True, "decode": True, "timeout": timeout,
    }
    deadline = time.monotonic() + timeout
    with client.translated(f"build {dockerfile}"):
        try:
            try:
                log, error = _consume(api.build(**options), deadline)
            except StoreError:
                with _without_registry_auth(api):
                    log, error = _consume(api.build(**options), deadline)
        except LifecycleError as e:
            if e.code == "timeout":
                raise LifecycleError("timeout", f"docker build of {dockerfile} did not finish in {timeout:g} s") from e
            raise
    if error:
        needs_bk = needs_buildkit(text) or "requires buildkit" in error.lower()
        raise _build_failed(dockerfile, "classic builder", _log_tail(log + error), needs_bk)
