"""
    Container Lifecycle

    Starts, checks, diagnoses, stops and removes the containers of this toolchain.

      L0   client     Docker SDK client, exception -> error-code mapping
      L1a  image      an existing image, or one plain build of a Dockerfile
      L1   provision  start a container and wait until it is ready
      L2   status     state and health of managed containers

    Sections L3 (diagnostics), L4 (teardown) and L5 (agent tools) are not written yet.
"""

from .client import docker_available, has_nvidia_runtime
from .image import ensure_image, resolve_image
from .models import (
    ContainerHandle, ContainerLogs, ContainerSpec, ContainerStatus,
    Diagnosis, Hint, ImageRef, LifecycleError, ProvisionResult,
)
from .provision import provision
from .status import get_status, list_managed, require_managed, require_running, wait_ready

__all__ = [
    "ContainerHandle", "ContainerLogs", "ContainerSpec", "ContainerStatus",
    "Diagnosis", "Hint", "ImageRef", "LifecycleError", "ProvisionResult",
    "docker_available", "has_nvidia_runtime",
    "ensure_image", "resolve_image",
    "provision",
    "get_status", "list_managed", "require_managed", "require_running", "wait_ready",
]
