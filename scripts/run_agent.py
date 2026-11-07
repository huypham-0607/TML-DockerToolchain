"""
Runs the agent like `python -m src.main`, but prints every step while it works:
each tool the agent calls (with its arguments) and the start of each tool's reply.

Use it to see what the agent did, also when it crashes or runs out of steps.

    uv run python -m scripts.run_agent "Containerize the repository at /abs/path/to/repo"
    uv run python -m scripts.run_agent --limit 200 "..."       # more steps (default 150)
    uv run python -m scripts.run_agent --log runs/ffb-agent.log "..."

The log file gets the full tool arguments and replies; the terminal shows a short version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SHORT = 300  # characters of each tool reply shown in the terminal


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _one_line(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def run_agent(prompt: str, limit: int = 0, log_path: str = "", agent=None) -> dict:
    """Runs the agent on `prompt`, printing every step. Returns a summary:
    rounds, seconds, tools_used, images_built (tags of successful build_image calls, in order),
    last_ai (the agent's last text), crashed (error text or "")."""
    if agent is None:
        from src.agent import agent  # imported here so --help works without an API key
    from src.main import step_limit

    limit = limit or step_limit()
    log = open(log_path, "w") if log_path else None

    def out(line: str = "", full: str | None = None) -> None:
        print(line, flush=True)
        if log:
            log.write((full if full is not None else line) + "\n")
            log.flush()

    out(f"PROMPT: {prompt}")
    out(f"step limit: {limit}\n")

    start = time.time()
    summary = {"rounds": 0, "seconds": 0.0, "tools_used": {}, "images_built": [], "last_ai": "", "crashed": ""}
    tools_used = summary["tools_used"]
    try:
        for update in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": limit},
            stream_mode="updates",
        ):
            for node, data in update.items():
                messages = (data or {}).get("messages", []) if isinstance(data, dict) else []
                if not isinstance(messages, list):
                    messages = [messages]
                for m in messages:
                    kind = getattr(m, "type", "")
                    stamp = f"[{time.time() - start:6.0f}s]"
                    if kind == "ai":
                        summary["rounds"] += 1
                        rounds = summary["rounds"]
                        text = _text(m.content).strip()
                        if text:
                            summary["last_ai"] = text
                            out(f"{stamp} round {rounds} AGENT SAYS: {_one_line(text, SHORT)}",
                                f"{stamp} round {rounds} AGENT SAYS:\n{text}")
                        for call in getattr(m, "tool_calls", []) or []:
                            name = call.get("name", "?")
                            tools_used[name] = tools_used.get(name, 0) + 1
                            a = json.dumps(call.get("args", {}), ensure_ascii=False)
                            out(f"{stamp} round {rounds} CALLS {name}({_one_line(a, 200)})",
                                f"{stamp} round {rounds} CALLS {name}({a})")
                    elif kind == "tool":
                        text = _text(m.content)
                        name = getattr(m, "name", "?")
                        if name == "build_image":
                            try:
                                reply = json.loads(text)
                                if reply.get("success") and reply.get("image"):
                                    summary["images_built"].append(reply["image"])
                            except (ValueError, AttributeError):
                                pass
                        out(f"{stamp}   <- {name}: {_one_line(text, SHORT)}", f"{stamp}   <- {name}:\n{text}")
    except KeyboardInterrupt:
        summary["crashed"] = "stopped by Ctrl+C"
        out("\nSTOPPED by Ctrl+C")
    except Exception as e:  # noqa: BLE001 - report any crash, then the summary
        summary["crashed"] = f"{type(e).__name__}: {e}"
        out(f"\nCRASHED: {type(e).__name__}: {_one_line(str(e), 500)}", f"\nCRASHED: {type(e).__name__}: {e}")
    finally:
        summary["seconds"] = round(time.time() - start, 1)
        out("\n==== SUMMARY ====")
        out(f"time: {summary['seconds']:.0f}s, agent rounds: {summary['rounds']}")
        out("tool calls: " + (", ".join(f"{k} x{v}" for k, v in sorted(tools_used.items())) or "none"))
        if summary["images_built"]:
            out("images built: " + ", ".join(summary["images_built"]))
        if summary["last_ai"]:
            out("\nLAST THING THE AGENT SAID:\n" + summary["last_ai"])
        if log:
            log.close()
            print(f"\nfull transcript: {Path(log_path).resolve()}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="+", help="the instruction for the agent")
    p.add_argument("--limit", type=int, default=0, help="LangGraph recursion limit, 2 per agent round (default: TML_STEP_LIMIT or 150)")
    p.add_argument("--log", default="", help="also write the full transcript to this file")
    args = p.parse_args()
    summary = run_agent(" ".join(args.prompt), limit=args.limit, log_path=args.log)
    return 1 if summary["crashed"] else 0


if __name__ == "__main__":
    sys.exit(main())
