"""
    Sandbox file exploratory tools for agents.

    Every raw paths the agent request will be rendered as virtual path under SANDBOX_ROOT
    
    Three tools:
    - grep
    - list_dir
    - read_file
"""

import subprocess

from langchain_core.tools import tool
from pathlib import Path, PurePosixPath

SANDBOX_ROOT = (Path(__file__).resolve().parent.parent.parent / "notes").resolve()
BLOCKED_NAMES = {
    ".env", ".env.local", ".git", ".ssh", ".aws", ".netrc", "id_rsa", "id_ed25519"
}
BLOCKED_SUFFIXES = {".pem", ".key", ".crt"}
GREP_TIMEOUT_S = 10

def _safe_path(raw: Path | str) -> Path:
    """Resolve `raw` as a virtual path under SANDBOX_ROOT. Raises ValueError on escape."""
    vpath = PurePosixPath("/" + str(raw).replace("\\", "/").lstrip("/"))
    if ".." in vpath.parts or any(part.startswith("~") for part in vpath.parts):
        raise ValueError("access denied: path traversal not allowed")
    try:
        resolved = (SANDBOX_ROOT / vpath.relative_to("/")).resolve()
    except (OSError, RuntimeError) as e:          # ELOOP, symlink recursion
        raise ValueError(f"cannot resolve path: {e}") from None
    if not resolved.is_relative_to(SANDBOX_ROOT):  # symlink pointing out of the tree
        raise ValueError(f"access denied: outside {SANDBOX_ROOT.name}/")
    relative_parts = resolved.relative_to(SANDBOX_ROOT).parts
    if any(part in BLOCKED_NAMES for part in relative_parts):
        raise ValueError("access denied: blocked filename")
    if resolved.suffix in BLOCKED_SUFFIXES:
        raise ValueError(f"access denied: blocked file type ({resolved.suffix})")
    return resolved

def _vpath(path: Path) -> str:
    """Render a sandboxed real path as the virtual path the agent should use."""
    rel = path.relative_to(SANDBOX_ROOT).as_posix()
    return "/" if rel == "." else "/" + rel

@tool
def grep(
    patterns: str,
    file: Path | str,
    regexp: bool = False,
    regexpf: bool = False,
    patterns_file: Path | str | None = None,
    recursive: bool = False,
) -> str:
    """Search for a pattern in a file or directory under the notes/ directory.

    Args:
        patterns: pattern to search for
        file: path to search, relative to notes/
        regexp: treat patterns as an extended regular expression
        regexpf: additionally read patterns from patterns_file
        patterns_file: file of patterns, required when regexpf is true
        recursive: search directories recursively

    Return:
        Matching lines, or a message when nothing matched.
    """
    try:
        file = _safe_path(file)
    except ValueError as e:
        return f"error: {e}"

    command = ["grep"]
    if recursive:
        command.append("-r")  # -r ignores symlinks found during traversal; -R would follow them
    for name in sorted(BLOCKED_NAMES):
        command += [f"--exclude={name}", f"--exclude-dir={name}"]
    command += [f"--exclude=*{suffix}" for suffix in sorted(BLOCKED_SUFFIXES)]
    if regexp:
        command.append("-E")
    if regexpf:
        if patterns_file is None:
            return "Invalid command: -f flag is set to true, but no patterns_file is found."

        try:
            patterns_file = _safe_path(patterns_file)
        except ValueError as e:
            return f"error: {e}"
        command += ["-f", str(patterns_file.relative_to(SANDBOX_ROOT))]

    # -e keeps a pattern starting with "-" from being read as an option, and
    # "--" ends option parsing before the path operand.
    command += ["-e", patterns, "--", str(file.relative_to(SANDBOX_ROOT))]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=SANDBOX_ROOT,  # so matches are reported as notes/-relative paths
            timeout=GREP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"error: grep timed out after {GREP_TIMEOUT_S}s"
    except OSError as e:
        return f"error: could not run grep: {e}"

    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 1:
        return "no matches found"
    return f"error: grep exited {result.returncode}: {result.stderr.strip()}"

@tool
def list_dir(path: Path | str) -> list[str] | str:
    """List the direct children of a directory under the notes/ directory.

    Args:
        path: directory to list, relative to notes/; "/" is the top of the tree

    Return:
        The children as virtual paths, or an error message. Does not recurse,
        and omits files that are blocked from reading.
    """
    try:
        path = _safe_path(path)
    except ValueError as e:
        return f"error: {e}"

    try:
        result = []
        for item in path.iterdir():
            # Hide what _safe_path would refuse to open anyway.
            if item.name in BLOCKED_NAMES or item.suffix in BLOCKED_SUFFIXES:
                continue
            result.append(_vpath(item))

        return result
    except (OSError, ValueError) as e:
        return f"Cannot list {_vpath(path)}: {e}"

@tool
def read_file(
    path: Path | str,
    start: int = 1,
    end: int = 200
) -> str:
    """Read lines [start, end] (1-indexed, inclusive) from a text file under the notes/ directory.
    Returns the lines prefixed with their line numbers.

    Args:
        path: path to file, relative to notes/
        start: starting line, defaults to 1
        end: ending line, defaults to 200

    Return:
        Aggregated string of the file content, truncated at 8000 characters.
    """
    try:
        path = _safe_path(path)
    except ValueError as e:
        return f"error: {e}"
    if not path.is_file():
        return f"error: not a file: {_vpath(path)}"

    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as e:
        return f"error: cannot read {_vpath(path)}: {e}"
    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return f"error: empty range (file has {len(lines)} lines)"

    out = "\n".join(f"{i}\t{lines[i-1]}" for i in range(start, end + 1))
    if len(out) > 8000:
        return out[:8000] + f"\n... truncated at 8000 chars (requested lines {start}-{end})"
    return out
