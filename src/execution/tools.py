"""
src/execution/tools.py
LangChain agent tool wrapper for container execution.
"""

from __future__ import annotations
import json
from typing import Union, List, Optional
from langchain_core.tools import tool
from .exec import execute_in_container, DEFAULT_TIMEOUT


def stale_image_warning(container: str) -> str | None:
    """A warning if `container` runs an older build of its image tag, else None.

    build_image reuses one tag (tml/<repo>:latest), and a container keeps the image it was
    created from. A command run in a container created before the latest build tests the old
    image, not the fix. Best effort: any problem reading Docker means no warning.
    """
    try:
        from ..lifecycle import client

        sdk = client.get_client()
        attrs = sdk.containers.get(container).attrs
        tag = (attrs.get("Config") or {}).get("Image", "")
        running = attrs.get("Image", "")
        if not tag or not running or tag.startswith("sha256:"):
            return None
        current = sdk.images.get(tag).id
    except Exception:  # noqa: BLE001 - a missing container or daemon is reported by the command itself
        return None
    if current == running:
        return None
    return (f"this container runs an OLDER build of {tag}: the image was rebuilt after the container "
            f"started, so this command tested the old image. Destroy this container and provision a new "
            f"one from {tag} before testing the fix.")


@tool
def run_command_in_container(
    container: str,
    command: str,
    cwd: str = "/workspace",
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Execute an arbitrary bash command inside a running container.

    Use this tool to run tests, check files, verify environments, or execute
    experiments inside a provisioned container.

    Args:
        container: Name or ID of the running container (from provision_container).
        command: The shell command to run (e.g. 'python --version' or 'pytest').
        cwd: Working directory inside the container (default: /workspace).
        timeout: Maximum seconds to wait before terminating (default: 300).

    Returns:
        JSON string containing:
        - ok (bool): True if exit_code == 0 and not timed out
        - warning (str, only when it applies): the container runs an older build of its image
        - exit_code (int): 0 for success, non-zero for failure
        - stdout (str): Standard output text
        - stderr (str): Standard error text
        - duration (float): Execution time in seconds
        - timed_out (bool): Whether execution hit the timeout limit
    """
    res = execute_in_container(container=container, command=command, cwd=cwd, timeout=timeout)
    reply = res.to_dict()
    if warning := stale_image_warning(container):
        reply = {"ok": reply["ok"], "warning": warning, **{k: v for k, v in reply.items() if k != "ok"}}
    return json.dumps(reply, indent=1)


TOOLS = [run_command_in_container]
