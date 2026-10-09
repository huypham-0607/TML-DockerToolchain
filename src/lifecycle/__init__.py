"""
    Container Lifecycle

    Starts, checks, diagnoses, stops and removes the containers of this toolchain.

      L0   client     Docker SDK client, exception -> error-code mapping
      L1a  image      an existing image, or one plain build of a Dockerfile
      L1   provision  start a container and wait until it is ready
      L2   status     state and health of managed containers
      L3   diagnostics  logs, and a diagnosis with hints: what failed, who can act on it
      L4   teardown   stop and remove containers; remove images that lifecycle built

    Section L5 (agent tools) is not written yet.
"""

from .client import docker_available, has_nvidia_runtime
from .diagnostics import classify_failure, diagnose, get_logs
from .image import ensure_image, resolve_image
from .models import (
    ContainerHandle, ContainerLogs, ContainerSpec, ContainerStatus,
    Diagnosis, Hint, ImageRef, LifecycleError, ProvisionResult, TeardownResult,
)
from .provision import provision
from .status import get_status, list_managed, require_managed, require_running, wait_ready
from .teardown import destroy, destroy_all_managed, remove, remove_built_images, stop

__all__ = [
    "ContainerHandle", "ContainerLogs", "ContainerSpec", "ContainerStatus",
    "Diagnosis", "Hint", "ImageRef", "LifecycleError", "ProvisionResult", "TeardownResult",
    "docker_available", "has_nvidia_runtime",
    "ensure_image", "resolve_image",
    "get_logs", "diagnose", "classify_failure",
    "provision",
    "get_status", "list_managed", "require_managed", "require_running", "wait_ready",
    "stop", "remove", "destroy", "destroy_all_managed", "remove_built_images",
]
