"""
    Temp tools for now
"""

from langchain_core.tools import tool

@tool
def meow() -> str:
    """Returns string Meow
    """

    return f"Meow"

@tool
def woof() -> str:
    """Returns woof
    """
    return f"Woof"

@tool
def inspect_repo(repo_path: str, search: bool = False) -> str:
    """Inspects a local repository and returns a JSON profile: file tree, languages,
    important files, existing Docker config, dependencies, README install commands,
    and the inferred Python / CUDA / framework versions.

    Args:
        repo_path: path to the repository root on this machine.
        search: if True, run a web search when the README has no install commands.
    """
    import json
    from .repo.inspect import inspect_repository

    return json.dumps(inspect_repository(repo_path, search=search).summary(), indent=1)
