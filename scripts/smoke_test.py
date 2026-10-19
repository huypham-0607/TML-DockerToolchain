"""
Smoke test for the TML-DockerToolchain agent stack.

Checks four layers, cheapest first, and stops at the first failure so you know
exactly which layer is broken instead of staring at one giant traceback.

    1. config      .env is found and the API key for the chosen provider is set
    2. chat        the LLM endpoint answers a trivial question
    3. tools       the model can actually call a tool (agents are useless without this)
    4. agent       the real agent runs inspect_repo on a repository and reports back

Layer 4 deliberately tells the agent to stop after inspecting. It does not build
images or start containers, so it is safe to run without Docker.

Usage:
    uv run python -m scripts.smoke_test                  # layers 1-3
    uv run python -m scripts.smoke_test --repo .         # layers 1-4 on this repo
    uv run python -m scripts.smoke_test --repo /path/to/some/research-repo
    uv run python -m scripts.smoke_test --provider openai
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ANSI colors, disabled when output is piped to a file
_tty = sys.stdout.isatty()
GREEN = "\033[32m" if _tty else ""
RED = "\033[31m" if _tty else ""
DIM = "\033[2m" if _tty else ""
OFF = "\033[0m" if _tty else ""


def ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}PASS{OFF}  {label}" + (f"{DIM}  {detail}{OFF}" if detail else ""))


def fail(label: str, detail: str, hint: str = "") -> None:
    print(f"  {RED}FAIL{OFF}  {label}")
    print(f"        {detail}")
    if hint:
        print(f"        {DIM}hint: {hint}{OFF}")


def layer(n: int, name: str) -> None:
    print(f"\n[{n}] {name}")


# --------------------------------------------------------------------------
# Layer 1: configuration
# --------------------------------------------------------------------------
def check_config(provider: str | None) -> dict:
    layer(1, "Configuration")

    from dotenv import load_dotenv

    from src.llm import PROVIDERS

    load_dotenv()

    provider = (provider or os.environ.get("LLM_PROVIDER", "rcac")).lower()
    if provider not in PROVIDERS:
        fail("provider", f"{provider!r} is not a known provider",
             f"pick one of {sorted(PROVIDERS)}")
        sys.exit(1)

    cfg = PROVIDERS[provider]
    key_name = cfg["api_key_env"]
    key = os.environ.get(key_name)

    if not key:
        fail(f"{key_name} is set", "variable is empty or missing",
             "create a .env file in the repo root containing "
             f'{key_name}="your-key-here"')
        sys.exit(1)

    # Never print a key. Length and last four characters are enough to tell
    # whether you pasted the right thing.
    ok(f"{key_name} is set", f"{len(key)} chars, ends ...{key[-4:]}")
    ok("provider", provider)
    ok("endpoint", cfg["base_url"] or "api.openai.com (OpenAI default)")
    ok("model", os.environ.get("LLM_MODEL") or cfg["model"])
    return {"provider": provider, "model": os.environ.get("LLM_MODEL") or cfg["model"]}


# --------------------------------------------------------------------------
# Layer 2: a plain chat completion
# --------------------------------------------------------------------------
def check_chat(provider: str):
    layer(2, "Chat completion")

    from src.llm import make_model

    model = make_model(provider=provider)

    start = time.monotonic()
    try:
        reply = model.invoke("Reply with exactly one word: pong")
    except Exception as exc:  # noqa: BLE001 - we want to explain any failure
        fail("endpoint answers", f"{type(exc).__name__}: {exc}",
             "401/403 means a bad key. A connection error usually means you are "
             "off the Purdue network - connect to the VPN and try again.")
        sys.exit(1)

    elapsed = time.monotonic() - start
    text = (reply.content or "").strip()
    ok("endpoint answers", f"{elapsed:.1f}s, said {text[:40]!r}")
    return model


# --------------------------------------------------------------------------
# Layer 3: tool calling
# --------------------------------------------------------------------------
def check_tool_calling(model) -> None:
    layer(3, "Tool calling")

    from langchain_core.tools import tool

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two integers together and returns the sum."""
        return a + b

    bound = model.bind_tools([add_numbers])

    try:
        reply = bound.invoke("What is 17 plus 25? Use the add_numbers tool.")
    except Exception as exc:  # noqa: BLE001
        fail("model accepts tools", f"{type(exc).__name__}: {exc}",
             "this model may not support tool calling - try another model "
             "via LLM_MODEL")
        sys.exit(1)

    calls = getattr(reply, "tool_calls", None)
    if not calls:
        fail("model requests a tool call",
             f"no tool call in the reply; model said {str(reply.content)[:120]!r}",
             "the agent cannot work without tool calling. Try a different model "
             "with LLM_MODEL, or check the provider supports the OpenAI tools API.")
        sys.exit(1)

    call = calls[0]
    ok("model requests a tool call", f"{call['name']}({call['args']})")

    # Run the tool by hand so you can see the whole loop:
    # model asks -> harness executes -> result goes back to the model.
    result = add_numbers.invoke(call["args"])
    ok("tool executes", f"returned {result}")
    if result != 42:
        fail("tool arguments are correct", f"expected 42, got {result}",
             "the model called the tool but with the wrong numbers")
        sys.exit(1)
    ok("tool arguments are correct", "17 + 25 = 42")


# --------------------------------------------------------------------------
# Layer 4: the real agent
# --------------------------------------------------------------------------
def check_agent(repo_path: str) -> None:
    layer(4, "Full agent run")

    from src.agent import TOOLS, agent

    ok("agent assembled", f"{len(TOOLS)} tools: {', '.join(t.name for t in TOOLS)}")

    # The integration system prompt tells the agent to work the whole pipeline
    # and not skip stages. For a connection test we want the first stage only,
    # so the instruction below stops it explicitly. Without this the agent may
    # start building Docker images.
    question = (
        f"Inspect the repository at {repo_path} using the inspect_repo tool. "
        "This is a connection test only: do NOT generate a Dockerfile, do NOT "
        "build an image, and do NOT provision a container. Stop after inspecting. "
        "Then tell me, in under 100 words: the primary language, the dependency "
        "files you found, and whether it already has a Dockerfile."
    )

    start = time.monotonic()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 50},
        )
    except Exception as exc:  # noqa: BLE001
        fail("agent completes", f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = time.monotonic() - start
    messages = result["messages"]

    used_tools = [
        c["name"]
        for m in messages
        for c in (getattr(m, "tool_calls", None) or [])
    ]

    # Did it actually use the tool, or did it just make something up?
    if "inspect_repo" in used_tools:
        ok("agent called inspect_repo", f"{len(messages)} messages, {elapsed:.1f}s")
    else:
        fail("agent called inspect_repo",
             f"tools used: {used_tools or 'none'}",
             "the agent answered without inspecting anything, so the answer below "
             "is guesswork. Try making the request more direct.")

    # Did it overshoot into the rest of the pipeline despite being told not to?
    overshoot = [t for t in used_tools
                 if t in {"generate_dockerfile", "build_image",
                          "provision_container", "run_command_in_container"}]
    if overshoot:
        fail("agent stayed in scope", f"it also called: {', '.join(sorted(set(overshoot)))}",
             "the system prompt is overriding the request. Worth reporting - it means "
             "the agent will not stop when a user asks it to.")
    else:
        ok("agent stayed in scope", "no build or container tools called")

    print(f"\n{DIM}--- agent's answer ---{OFF}")
    print(messages[-1].content)
    print(f"{DIM}----------------------{OFF}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="rcac or openai (default: $LLM_PROVIDER, else rcac)")
    parser.add_argument("--repo", help="run layer 4 against this repository path")
    args = parser.parse_args()

    print("TML-DockerToolchain smoke test")

    cfg = check_config(args.provider)
    model = check_chat(cfg["provider"])
    check_tool_calling(model)

    if args.repo:
        check_agent(args.repo)
    else:
        layer(4, "Full agent run")
        print(f"  {DIM}SKIP  pass --repo PATH to run the agent end to end{OFF}")

    print(f"\n{GREEN}All checks passed.{OFF}\n")


if __name__ == "__main__":
    main()