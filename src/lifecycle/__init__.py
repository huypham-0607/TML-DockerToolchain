"""
    Container Lifecycle

    Starts, checks, diagnoses, stops and removes the containers of this toolchain.

      L0  client   Docker SDK client, exception -> error-code mapping
      L2  status   state and health of managed containers

    Sections L1 (image, provision), L3 (diagnostics), L4 (teardown) and L5 (agent tools)
    are not written yet. See container_lifecycle_plan.md.
"""

from .client import docker_available, has_nvidia_runtime
from .models import (
    ContainerHandle, ContainerLogs, ContainerSpec, ContainerStatus,
    Diagnosis, Hint, ImageRef, LifecycleError,
)
from .status import get_status, list_managed, require_managed, require_running, wait_ready

__all__ = [
    "ContainerHandle", "ContainerLogs", "ContainerSpec", "ContainerStatus",
    "Diagnosis", "Hint", "ImageRef", "LifecycleError",
    "docker_available", "has_nvidia_runtime",
    "get_status", "list_managed", "require_managed", "require_running", "wait_ready",
]
