"""
src/execution/tools.py
LangChain agent tool wrapper for container execution.
"""

from __future__ import annotations
import json
from typing import Union, List, Optional
from langchain_core.tools import tool
from .exec import execute_in_container, DEFAULT_TIMEOUT


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
        - exit_code (int): 0 for success, non-zero for failure
        - stdout (str): Standard output text
        - stderr (str): Standard error text
        - duration (float): Execution time in seconds
        - timed_out (bool): Whether execution hit the timeout limit
    """
    res = execute_in_container(container=container, command=command, cwd=cwd, timeout=timeout)
    return json.dumps(res.to_dict(), indent=1)


TOOLS = [run_command_in_container]
