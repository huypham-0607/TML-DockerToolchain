"""
src/execution/models.py
Data models and structured outputs for container command execution.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Union, Optional


@dataclass
class ExecResult:
    """Structured result returned by container command execution."""
    command: Union[str, List[str]]
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.exit_code == 0 and not self.timed_out,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": round(self.duration, 3),
            "timed_out": self.timed_out,
            "error": self.error,
        }

    def summary(self) -> dict:
        """Compact summary suitable for LLM prompt context."""
        return {
            "ok": self.exit_code == 0 and not self.timed_out,
            "exit_code": self.exit_code,
            "duration": round(self.duration, 3),
            "stdout_tail": self.stdout[-1000:] if len(self.stdout) > 1000 else self.stdout,
            "stderr_tail": self.stderr[-1000:] if len(self.stderr) > 1000 else self.stderr,
            "timed_out": self.timed_out,
        }
