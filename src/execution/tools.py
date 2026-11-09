"""
src/execution/tools.py
LangChain agent tool wrapper for container execution.
"""

from __future__ import annotations
import json
import os
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


# The longest a single agent command may run. A smoke test needs seconds; in the EP/BNP run the agent
# asked for 600 s six times while train.py downloaded CIFAR-10 (170 MB at ~100 kB/s), and lost 48 minutes.
MAX_AGENT_TIMEOUT = int(os.environ.get("TML_MAX_COMMAND_SECONDS", "300"))
OUTPUT_LIMIT = 6000   # characters kept per stream: the head and, mostly, the tail

TIMEOUT_HINT = ("The command ran longer than {t}s and was killed. If it was downloading data or training a "
                "model, running it again (even with a longer timeout) will not finish either: a killed "
                "download starts over. Do not repeat it. Test the code another way: import the modules, "
                "build the model, and run one step on a small random input.")


def collapse_progress(text: str) -> str:
    """Keep only the last state of each carriage-return progress bar (tqdm, pip, wget).
    A 10-minute download printed ~200 kB of '\r  3%|...' updates, which buried the real output."""
    if not text or "\r" not in text:
        return text
    lines = []
    for line in text.split("\n"):
        parts = [p for p in line.split("\r") if p.strip()]
        lines.append(parts[-1] if parts else "")
    return "\n".join(lines)


def shorten(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Head and tail of long output, with a marker saying how much was cut."""
    if not text or len(text) <= limit:
        return text
    head, tail = limit // 4, limit - limit // 4
    return f"{text[:head]}\n[... {len(text) - head - tail} characters cut ...]\n{text[-tail:]}"


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
        timeout: Maximum seconds to wait before terminating (default: 300, and at most 300:
            a smoke test should take seconds; do not download datasets or train for real).

    Returns:
        JSON string containing:
        - ok (bool): True if exit_code == 0 and not timed out
        - warning (str, only when it applies): the container runs an older build of its image
        - exit_code (int): 0 for success, non-zero for failure
        - stdout (str): Standard output text
        - stderr (str): Standard error text
        - duration (float): Execution time in seconds
        - timed_out (bool): Whether execution hit the timeout limit
        - hint (str, only after a timeout): what to do instead of running it again
    """
    capped = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_AGENT_TIMEOUT))
    res = execute_in_container(container=container, command=command, cwd=cwd, timeout=capped)
    reply = res.to_dict()
    for stream in ("stdout", "stderr"):
        if isinstance(reply.get(stream), str):
            reply[stream] = shorten(collapse_progress(reply[stream]))
    if reply.get("timed_out"):
        reply["hint"] = TIMEOUT_HINT.format(t=capped)
    if warning := stale_image_warning(container):
        reply = {"ok": reply["ok"], "warning": warning, **{k: v for k, v in reply.items() if k != "ok"}}
    return json.dumps(reply, indent=1)


TOOLS = [run_command_in_container]
