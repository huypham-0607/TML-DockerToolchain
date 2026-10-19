"""
CLI entry point for manual testing of container execution:
    python -m src.execution <container> <command> [--cwd DIR] [--timeout SECONDS]
"""

from __future__ import annotations
import sys
import json
import argparse
from .exec import execute_in_container, DEFAULT_TIMEOUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.execution",
        description="Execute commands inside a provisioned Docker container."
    )
    parser.add_argument("container", help="Container name or ID")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("--cwd", default="/workspace", help="Working directory (default: /workspace)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")

    args = parser.parse_args(argv)

    res = execute_in_container(
        container=args.container,
        command=args.command,
        cwd=args.cwd,
        timeout=args.timeout,
    )

    if args.compact:
        print(json.dumps(res.to_dict(), separators=(",", ":")))
    else:
        print(json.dumps(res.to_dict(), indent=2))

    return res.exit_code if res.exit_code != -1 else 1


if __name__ == "__main__":
    sys.exit(main())
