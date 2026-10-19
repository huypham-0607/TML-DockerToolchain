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


@tool
def build_image(repo_path: str, dockerfile: str, tag: str = "") -> str:
    """Builds a Docker image from Dockerfile text, using the repository as the build context.
    Returns JSON: success, image tag, exit code, and the tail of the build log.
    If the build fails, read log_tail, fix the Dockerfile, and call this again.

    Args:
        repo_path: path to the repository root on this machine.
        dockerfile: complete Dockerfile text.
        tag: image tag; default tml/<repo-name>:latest.
    """
    import json
    from .docker.build import build_image as _build

    return json.dumps(_build(repo_path, dockerfile, tag or None).to_dict(), indent=1)


@tool
def generate_dockerfile(repo_path: str) -> str:
    """Inspects a local repository and generates a starter Dockerfile for it from a template
    (Python version, framework pins, requirements files). Does not build.
    Returns JSON: dockerfile text, the path it was saved to, and notes (e.g. needs --gpus all).
    Review or edit the text, then pass it to build_image.

    Args:
        repo_path: path to the repository root on this machine.
    """
    import json
    from .docker.generate import generate_dockerfile as _generate

    text, path, notes = _generate(repo_path)
    return json.dumps({"dockerfile": text, "dockerfile_path": str(path), "notes": notes}, indent=1)
