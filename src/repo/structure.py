"""
    Section A - Repository Structure

    Inspects a local repository and reports:
      - file tree (pruned of noise like .git, venvs, caches)
      - languages (by file extension)
      - important files (dependency files, docs, entry points, configs, ...)
      - existing Docker / container configuration

    Plain Python helper, not an agent tool. Use inspect_structure(path).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Directories that are never useful for inspection
IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    ".venv", "venv", ".conda",
    "node_modules", ".ipynb_checkpoints",
    "build", "dist", ".eggs", "site-packages",
    ".idea", ".vscode",
}

LANGUAGES = {
    ".py": "Python", ".pyx": "Cython", ".pxd": "Cython", ".ipynb": "Jupyter Notebook",
    ".c": "C", ".h": "C/C++ Header", ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++",
    ".cu": "CUDA", ".cuh": "CUDA",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".r": "R", ".jl": "Julia", ".m": "MATLAB",
    ".js": "JavaScript", ".ts": "TypeScript",
    ".java": "Java", ".scala": "Scala", ".go": "Go", ".rs": "Rust",
    ".lua": "Lua", ".f90": "Fortran", ".f": "Fortran",
}

# Category -> filename glob patterns (matched case-insensitively against the basename)
IMPORTANT_PATTERNS = {
    "dependencies": [
        "requirements*.txt", "requirements*.in", "pyproject.toml", "setup.py", "setup.cfg",
        "environment*.yml", "environment*.yaml", "conda*.yml", "conda*.yaml",
        "pipfile", "pipfile.lock", "poetry.lock", "uv.lock", "package.json",
    ],
    "documentation": ["readme*", "install*", "getting_started*", "usage*"],
    "entry_points": [
        "main.py", "train*.py", "test.py", "eval*.py", "run*.py", "demo*.py",
        "run*.sh", "train*.sh", "eval*.sh", "setup.sh", "install*.sh",
        "makefile", "cmakelists.txt",
    ],
    "configs": ["*.yaml", "*.yml", "*.cfg", "*.ini", "*.toml"],
    "license": ["license*", "copying*"],
    "ci": [],  # filled by path check (.github/workflows, .gitlab-ci.yml, ...)
    "submodules": [".gitmodules"],
}

DOCKER_PATTERNS = {
    "dockerfiles": ["dockerfile", "dockerfile.*", "*.dockerfile", "containerfile"],
    "compose": ["docker-compose*.yml", "docker-compose*.yaml", "compose.yml", "compose.yaml"],
    "dockerignore": [".dockerignore"],
    "devcontainer": ["devcontainer.json"],
    "apptainer": ["singularity", "singularity.*", "*.def", "apptainer*"],
}

# Files bigger than this get flagged (model weights, datasets, ...): matters for the Docker build context
LARGE_FILE_BYTES = 50 * 1024 * 1024


@dataclass
class DockerfileInfo:
    path: str
    base_images: list[str] = field(default_factory=list)
    exposed_ports: list[str] = field(default_factory=list)
    entrypoint: str | None = None
    cmd: str | None = None


@dataclass
class DockerConfig:
    dockerfiles: list[DockerfileInfo] = field(default_factory=list)
    compose: list[str] = field(default_factory=list)
    dockerignore: list[str] = field(default_factory=list)
    devcontainer: list[str] = field(default_factory=list)
    apptainer: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return bool(self.dockerfiles or self.compose or self.devcontainer or self.apptainer)


@dataclass
class RepoStructure:
    root: str
    total_files: int
    total_dirs: int
    tree: str
    languages: dict[str, int]             # language -> file count, sorted desc
    primary_language: str | None
    important_files: dict[str, list[str]]  # category -> relative paths
    docker: DockerConfig
    large_files: list[str]
    truncated: bool                        # True if max_files was hit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["docker"]["present"] = self.docker.present
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def _matches(name: str, patterns: list[str]) -> bool:
    name = name.lower()
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _is_ci(rel: str) -> bool:
    rel = rel.replace(os.sep, "/").lower()
    return (
        rel.startswith(".github/workflows/")
        or rel in {".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", "jenkinsfile"}
        or rel.startswith(".circleci/")
    )


def _walk(root: Path, max_files: int):
    """Yields (relative dir, subdirs, files) in sorted order, skipping ignored dirs."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in IGNORED_DIRS and not d.endswith(".egg-info")
            and not os.path.islink(os.path.join(dirpath, d))
        )
        filenames = sorted(filenames)
        rel = Path(dirpath).relative_to(root)
        if count + len(filenames) > max_files:
            filenames = filenames[: max(0, max_files - count)]
            yield rel, dirnames, filenames, True
            return
        count += len(filenames)
        yield rel, dirnames, filenames, False


def render_tree(root: str | Path, max_depth: int = 3, max_entries_per_dir: int = 25) -> str:
    """Renders an ASCII tree of the repository (ignored dirs pruned)."""
    root = Path(root)
    lines = [root.name + "/"]

    def visit(path: Path, prefix: str, depth: int):
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        entries = [
            e for e in entries
            if not (e.is_dir() and (e.name in IGNORED_DIRS or e.name.endswith(".egg-info")))
        ]
        hidden = len(entries) - max_entries_per_dir
        entries = entries[:max_entries_per_dir]
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1 and hidden <= 0
            branch = "└── " if last else "├── "
            is_dir = entry.is_dir() and not entry.is_symlink()
            lines.append(prefix + branch + entry.name + ("/" if is_dir else ""))
            if is_dir:
                ext = "    " if last else "│   "
                if depth + 1 < max_depth:
                    visit(entry, prefix + ext, depth + 1)
        if hidden > 0:
            lines.append(prefix + f"└── ... ({hidden} more)")

    visit(root, "", 0)
    return "\n".join(lines)


def parse_dockerfile(path: str | Path, rel: str) -> DockerfileInfo:
    """Extracts base images, exposed ports, ENTRYPOINT and CMD from a Dockerfile."""
    info = DockerfileInfo(path=rel)
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return info

    # join line continuations
    text = re.sub(r"\\\r?\n", " ", text)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        instr, _, args = line.partition(" ")
        instr = instr.upper()
        args = args.strip()
        if instr == "FROM":
            # FROM [--platform=...] image [AS name]
            tokens = [t for t in args.split() if not t.startswith("--")]
            if tokens:
                info.base_images.append(tokens[0])
        elif instr == "EXPOSE":
            info.exposed_ports.extend(args.split())
        elif instr == "ENTRYPOINT":
            info.entrypoint = args
        elif instr == "CMD":
            info.cmd = args
    return info


def inspect_structure(
    repo_path: str | Path,
    max_files: int = 20000,
    tree_depth: int = 3,
) -> RepoStructure:
    """Inspects the structure of a local repository.

    Args:
        repo_path: path to the repository root.
        max_files: stop scanning after this many files (huge repos).
        tree_depth: depth of the rendered file tree.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    total_files = 0
    total_dirs = 0
    truncated = False
    lang_counts: dict[str, int] = {}
    important: dict[str, list[str]] = {k: [] for k in IMPORTANT_PATTERNS}
    docker = DockerConfig()
    large_files: list[str] = []

    for rel_dir, dirnames, filenames, hit_limit in _walk(root, max_files):
        total_dirs += len(dirnames)
        truncated = truncated or hit_limit
        for name in filenames:
            total_files += 1
            rel = (rel_dir / name).as_posix()
            full = root / rel_dir / name

            lang = LANGUAGES.get(Path(name).suffix.lower())
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            try:
                if not full.is_symlink() and full.stat().st_size > LARGE_FILE_BYTES:
                    large_files.append(rel)
            except OSError:
                pass

            # Docker config
            if _matches(name, DOCKER_PATTERNS["dockerfiles"]):
                docker.dockerfiles.append(parse_dockerfile(full, rel))
            elif _matches(name, DOCKER_PATTERNS["compose"]):
                docker.compose.append(rel)
            elif _matches(name, DOCKER_PATTERNS["dockerignore"]):
                docker.dockerignore.append(rel)
            elif _matches(name, DOCKER_PATTERNS["devcontainer"]) and ".devcontainer" in rel_dir.parts:
                docker.devcontainer.append(rel)
            elif _matches(name, DOCKER_PATTERNS["apptainer"]) and _looks_like_apptainer(full):
                docker.apptainer.append(rel)

            # Important files (configs only at the top two levels, otherwise too noisy)
            if _is_ci(rel):
                important["ci"].append(rel)
                continue
            for category, patterns in IMPORTANT_PATTERNS.items():
                if category == "configs" and (len(rel_dir.parts) > 1 or _matches(name, IMPORTANT_PATTERNS["dependencies"])
                                              or _matches(name, DOCKER_PATTERNS["compose"])):
                    continue
                if category == "documentation" and len(rel_dir.parts) > 1:
                    continue
                if _matches(name, patterns):
                    important[category].append(rel)
                    break

        if hit_limit:
            break

    languages = dict(sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    # Notebooks/headers alone shouldn't decide the primary language if real code exists
    code_langs = [l for l in languages if l not in {"Jupyter Notebook", "C/C++ Header", "Shell"}]
    primary = code_langs[0] if code_langs else next(iter(languages), None)

    return RepoStructure(
        root=str(root),
        total_files=total_files,
        total_dirs=total_dirs,
        tree=render_tree(root, max_depth=tree_depth),
        languages=languages,
        primary_language=primary,
        important_files={k: v for k, v in important.items() if v},
        docker=docker,
        large_files=large_files,
        truncated=truncated,
    )


def _looks_like_apptainer(path: Path) -> bool:
    """*.def is too generic; require an Apptainer/Singularity 'Bootstrap:' header."""
    try:
        with open(path, errors="replace") as f:
            head = f.read(2048)
    except OSError:
        return False
    return re.search(r"^\s*bootstrap\s*:", head, re.IGNORECASE | re.MULTILINE) is not None


if __name__ == "__main__":
    result = inspect_structure(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(result.to_json(indent=2))
