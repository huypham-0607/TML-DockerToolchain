"""
    Command-line entry point: sends one request to the agent and prints its last reply.

      uv run python -m src.main "Containerize the repository at /abs/path/to/repo"

    TML_STEP_LIMIT sets the LangGraph recursion limit (default 150). Every agent round uses two
    steps (model, then tools), and a full run with 5 build attempts, 3 provision attempts and the
    smoke tests needs far more than LangGraph's 25 or the 50 used before.
    To watch every step, use scripts/run_agent.py instead.
"""

import logging
import os
import sys

DEFAULT_STEP_LIMIT = 150


def step_limit() -> int:
    return int(os.environ.get("TML_STEP_LIMIT") or DEFAULT_STEP_LIMIT)


def main() -> int:
    from langgraph.errors import GraphRecursionError

    from .agent import agent

    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

    question = " ".join(sys.argv[1:]) or "Hello world!"
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": step_limit()},
        )
    except GraphRecursionError:
        print(f"The agent used all {step_limit()} steps without finishing. Raise TML_STEP_LIMIT, or run "
              "scripts/run_agent.py to see what it was doing.", file=sys.stderr)
        return 1
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
