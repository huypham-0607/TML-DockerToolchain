"""
    Command line for the container lifecycle, for manual checks and scripts:

      python -m src.lifecycle provision --dockerfile path/to/Dockerfile --name demo
      python -m src.lifecycle status [CONTAINER]
      python -m src.lifecycle logs CONTAINER [--tail N] [--since 10m] [--grep REGEX]
      python -m src.lifecycle diagnose CONTAINER
      python -m src.lifecycle stop CONTAINER [--timeout S]
      python -m src.lifecycle destroy CONTAINER [--force]
      python -m src.lifecycle destroy-all [--force]

    Each command prints the same JSON as the agent tool of the same name.
    Exit code: 0 if "ok" is true, 1 if not, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import tools
from .teardown import destroy_all_managed


def _pairs(items: list[str], what: str) -> dict[str, str]:
    """["KEY=VALUE", ...] -> {"KEY": "VALUE"}"""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"error: {what} must look like KEY=VALUE, not {item!r}")
        out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.lifecycle",
        description="Start, check, diagnose, stop and remove the containers of the TML toolchain.",
    )
    parser.add_argument("--compact", action="store_true", help="print the JSON on one line")
    commands = parser.add_subparsers(dest="action", required=True, metavar="COMMAND")   # not "command": provision has a --command option

    p = commands.add_parser("provision", help="start a container and wait until it is ready")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default="", help="tag or id of an image on this machine")
    source.add_argument("--dockerfile", default="", help="path to a Dockerfile to build and run")
    p.add_argument("--context", default="", help="build context (default: the Dockerfile's directory)")
    p.add_argument("--name", default="", help="container name (default: generated)")
    p.add_argument("--mode", choices=["idle", "native"], default="idle")
    p.add_argument("--command", default="", help="native mode: command instead of the image's CMD")
    p.add_argument("--gpu", choices=["auto", "all", "none"], default="auto")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable (repeatable)")
    p.add_argument("--mount", action="append", default=[], metavar="HOST:CONTAINER[:ro]", help="mount (repeatable)")
    p.add_argument("--port", action="append", default=[], metavar="[HOST:]CONTAINER", help="published port (repeatable)")
    p.add_argument("--memory", default="", help="memory limit, e.g. 8g")
    p.add_argument("--shm-size", default="2g", help="size of /dev/shm, e.g. 8g")
    p.add_argument("--user", default="", help="user to run as, e.g. 1000:1000")
    p.add_argument("--ready-timeout", type=float, default=0, help="seconds to wait for readiness (0: default)")
    p.add_argument("--replace", action="store_true", help="replace a container of this name with other settings")
    p.add_argument("--label", action="append", default=[], metavar="KEY=VALUE", help="extra container label (repeatable)")

    p = commands.add_parser("status", help="state and health of a container; all containers without an argument")
    p.add_argument("container", nargs="?", default="")

    p = commands.add_parser("logs", help="runtime logs of a container")
    p.add_argument("container")
    p.add_argument("--tail", type=int, default=tools.DEFAULT_TAIL, help="lines of each stream")
    p.add_argument("--since", default="", help="a duration such as 10m, or an ISO timestamp")
    p.add_argument("--grep", default="", help="regular expression; only matching lines")

    p = commands.add_parser("diagnose", help="status, log tail, hints, processes, resources, events")
    p.add_argument("container")

    p = commands.add_parser("stop", help="stop a container and keep it")
    p.add_argument("container")
    p.add_argument("--timeout", type=int, default=tools.STOP_TIMEOUT, help="seconds between SIGTERM and SIGKILL")

    p = commands.add_parser("destroy", help="stop if needed, then remove a container")
    p.add_argument("container")
    p.add_argument("--force", action="store_true", help="kill at once")

    p = commands.add_parser("destroy-all", help="destroy every container this toolchain made")
    p.add_argument("--force", action="store_true", help="kill at once")
    p.add_argument("--label", action="append", default=[], metavar="KEY=VALUE", help="only containers with this label (repeatable)")
    return parser


def _destroy_all(force: bool, labels: dict[str, str]) -> str:
    def build() -> dict:
        results = destroy_all_managed(labels=labels or None, force=force)
        return {"ok": True, "results": [tools.teardown_view(r) for r in results]}
    return tools._reply(build)


def run(args: argparse.Namespace) -> str:
    """Runs one parsed command and returns its JSON reply."""
    if args.action == "provision":
        labels, env = _pairs(args.label, "--label"), _pairs(args.env, "--env")
        saved = dict(tools.SESSION_LABELS)
        tools.SESSION_LABELS.update(labels)
        try:
            return tools.provision_container(
                image=args.image, dockerfile=args.dockerfile, context=args.context, name=args.name,
                mode=args.mode, command=args.command, gpu=args.gpu, env=env,
                mounts=args.mount, ports=args.port, memory=args.memory, shm_size=args.shm_size,
                user=args.user, ready_timeout=args.ready_timeout, replace=args.replace,
            )
        finally:
            tools.SESSION_LABELS.clear()
            tools.SESSION_LABELS.update(saved)
    if args.action == "status":
        return tools.container_status(args.container)
    if args.action == "logs":
        return tools.container_logs(args.container, tail=args.tail, since=args.since, grep=args.grep)
    if args.action == "diagnose":
        return tools.diagnose_container(args.container)
    if args.action == "stop":
        return tools.stop_container(args.container, timeout=args.timeout)
    if args.action == "destroy":
        return tools.destroy_container(args.container, force=args.force)
    if args.action == "destroy-all":
        return _destroy_all(args.force, _pairs(args.label, "--label"))
    raise SystemExit(f"error: unknown command {args.action!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reply = json.loads(run(args))
    except SystemExit as e:   # a usage error found after parsing (KEY=VALUE pairs)
        print(e.code, file=sys.stderr)
        return 2
    print(json.dumps(reply, separators=(",", ":")) if args.compact else json.dumps(reply, indent=2))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
