"""
src/execution/exec.py
Executes arbitrary commands inside a running Docker container.
"""

from __future__ import annotations
import time
import subprocess
from typing import List, Union, Optional, Dict
from .models import ExecResult

DEFAULT_TIMEOUT = 300  # 5 minutes


def execute_in_container(
    container: str,
    command: Union[str, List[str]],
    cwd: str = "/workspace",
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
) -> ExecResult:
    """
    Executes a command inside a running Docker container via docker exec.

    Args:
        container: Name or ID of the running container.
        command: Command string or list of argv strings.
        cwd: Working directory inside the container (default: /workspace).
        timeout: Execution timeout in seconds.
        env: Optional dictionary of environment variables to set.

    Returns:
        ExecResult: Structured object with exit_code, stdout, stderr, duration, timed_out.
    """
    if not container or not container.strip():
        return ExecResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="No container specified.",
            duration=0.0,
            error="empty_container_identifier",
        )

    # Build `docker exec` command list
    exec_cmd = ["docker", "exec", "-w", cwd]

    if env:
        for k, v in env.items():
            exec_cmd.extend(["-e", f"{k}={v}"])

    exec_cmd.append(container)

    if isinstance(command, list):
        exec_cmd.extend(command)
    else:
        # Wrap string command in sh -c so pipes, redirects, and chains work
        exec_cmd.extend(["sh", "-c", command])

    start_time = time.time()
    try:
        proc = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start_time
        return ExecResult(
            command=command,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=duration,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return ExecResult(
            command=command,
            exit_code=-1,
            stdout=stdout,
            stderr=stderr + f"\n[TIMEOUT] Command exceeded {timeout}s limit.",
            duration=duration,
            timed_out=True,
            error="timeout_expired",
        )
    except FileNotFoundError:
        duration = time.time() - start_time
        return ExecResult(
            command=command,
            exit_code=127,
            stdout="",
            stderr="Docker binary not found on PATH.",
            duration=duration,
            error="docker_not_found",
        )
    except Exception as e:
        duration = time.time() - start_time
        return ExecResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=str(e),
            duration=duration,
            error="execution_exception",
        )
