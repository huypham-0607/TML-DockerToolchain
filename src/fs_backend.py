"""
    File access for the agent's built-in file tools (ls, read_file, glob, grep, write_file, edit_file)

    deepagents gives every agent these tools. By default they work on an in-memory, empty
    filesystem (StateBackend), so the agent could not see the repository it was asked to
    containerize: `read_file` said "not found" for files that inspect_repo had just listed.

    make_backend() builds the backend the agent uses instead:

      - the REAL disk, READ-ONLY: ls / read_file / glob / grep see the actual repository.
        write_file / edit_file / delete are refused, so the agent cannot change the repository
        or anything else on the machine. It changes Dockerfile text and passes it to build_image.
      - secret files are never served: .env files, SSH and cloud keys, credential stores
        (reads are refused, grep skips them, listings still show that they exist).
      - SCRATCH_ROOT (/agent_scratch/) is an in-memory area where writes are allowed: deepagents
        puts large tool results and conversation history there, and the agent may keep notes.
        Nothing there reaches the disk.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GrepResult,
    ReadResult,
    WriteResult,
)

SCRATCH_ROOT = "/agent_scratch"

# File names (or path parts) that are never read. fnmatch patterns, matched case-insensitively
# against the file name and against every directory name in the path.
SECRET_NAMES = (
    ".env", ".env.*", "*.env", ".envrc",
    ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials", "credentials.json", "secrets.*",
)
SECRET_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config/gcloud")

READ_ONLY_MESSAGE = (
    "the real filesystem is read-only for the agent: change Dockerfile text and pass it to "
    f"build_image instead. Notes can be written under {SCRATCH_ROOT}/ (in memory only)."
)


def is_secret(path: str) -> bool:
    """True if `path` names a file the agent must not read."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    name = p.name.lower()
    if any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_NAMES):
        return True
    parts = [part.lower() for part in p.parts]
    joined = "/".join(parts)
    return any(d in parts or f"/{d}/" in f"/{joined}/" for d in SECRET_DIRS)


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """The real filesystem: reads allowed (except secrets), every write refused."""

    def __init__(self) -> None:
        # virtual_mode=False: the agent uses the same absolute paths as the other tools
        super().__init__(root_dir="/", virtual_mode=False)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if is_secret(file_path):
            return ReadResult(error=f"refused: {file_path} may contain secrets and is not readable by the agent")
        return super().read(file_path, offset=offset, limit=limit)

    def grep(self, *args, **kwargs) -> GrepResult:
        result = super().grep(*args, **kwargs)
        if result.matches:
            result.matches = [m for m in result.matches if not is_secret(m.get("path", ""))]
        return result

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        allowed = [p for p in paths if not is_secret(p)]
        served = iter(super().download_files(allowed) if allowed else [])
        # one response per requested path, in order: secrets are refused, the rest come from disk
        return [FileDownloadResponse(path=p, content=None, error="permission_denied") if is_secret(p) else next(served)
                for p in paths]

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=f"refused to write {file_path}: {READ_ONLY_MESSAGE}")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error=f"refused to edit {file_path}: {READ_ONLY_MESSAGE}")

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=f"refused to delete {file_path}: {READ_ONLY_MESSAGE}")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def make_backend() -> CompositeBackend:
    """The agent's file backend: real disk read-only, SCRATCH_ROOT in memory and writable."""
    return CompositeBackend(
        default=ReadOnlyFilesystemBackend(),
        routes={f"{SCRATCH_ROOT}/": StateBackend()},
        artifacts_root=SCRATCH_ROOT,
    )
