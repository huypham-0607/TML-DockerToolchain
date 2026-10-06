"""
    Section E - Repository Inspection (combines sections A-D)

    inspect_repository(path) runs:
      A  structure.inspect_structure       file tree, languages, important files, Docker config
      B  dependencies.detect_dependencies  dependency files and package versions
      C  documentation.extract_documentation  README setup sections and install commands
      D  environment.detect_environment    Python / CUDA / framework versions from all of the above

    and returns one RepoProfile. RepoProfile.summary() is a compact view for the LLM;
    RepoProfile.to_dict() is the full result.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from .dependencies import DependencyReport, detect_dependencies
from .documentation import DocumentationReport, extract_documentation
from .environment import EnvironmentReport, detect_environment
from .structure import RepoStructure, inspect_structure


@dataclass
class RepoProfile:
    root: str
    structure: RepoStructure
    dependencies: DependencyReport
    documentation: DocumentationReport
    environment: EnvironmentReport

    def summary(self, max_items: int = 30) -> dict:
        """Compact, LLM-friendly view (no raw evidence lists, truncated long lists)."""
        s, d, doc, env = self.structure, self.dependencies, self.documentation, self.environment
        return {
            "root": self.root,
            "primary_language": s.primary_language,
            "languages": s.languages,
            "total_files": s.total_files,
            "tree": s.tree,
            "important_files": s.important_files,
            "existing_docker": {
                "present": s.docker.present,
                "dockerfiles": [asdict(f) for f in s.docker.dockerfiles],
                "compose": s.docker.compose,
                "apptainer": s.docker.apptainer,
            },
            "large_files": s.large_files[:max_items],
            "dependency_files": d.files,
            "python_requires": d.python_requires,
            "packages": dict(list(d.packages().items())[:max_items * 2]),
            "conda_channels": d.conda_channels,
            "index_urls": d.index_urls,
            "install_commands": [c.command for c in doc.install_commands][:max_items],
            "setup_sections": [f"{x.source} > {x.heading}" for x in doc.setup_sections][:max_items],
            "search_results": [asdict(r) for r in doc.search_results],
            "environment": {
                k: {"value": v.value, "confidence": v.confidence, "constraints": v.constraints, "conflict": v.conflict}
                for k, v in env.resolved.items()
            },
            "frameworks": env.frameworks,
            "gpu": {
                "uses_cuda": env.gpu.uses_cuda,
                "gpu_required": env.gpu.gpu_required,
                "needs_nvcc": env.gpu.needs_nvcc,
                "hardcoded_cuda": env.gpu.hardcoded_cuda[:10],
            },
            "warnings": (
                [f"{e['source']}: {e['error']}" for e in d.errors]
                + doc.notes + env.notes
                + (["file scan truncated (very large repository)"] if s.truncated else [])
            ),
        }

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "structure": self.structure.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "documentation": self.documentation.to_dict(),
            "environment": self.environment.to_dict(),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def inspect_repository(repo_path: str | Path, search: bool = False) -> RepoProfile:
    """Runs sections A-D on a local repository and combines the results.

    Args:
        repo_path: path to the repository root.
        search: allow the web search fallback when the docs have no install commands.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    structure = inspect_structure(root)
    deps = detect_dependencies(root)
    docs = extract_documentation(root, search=search)
    env = detect_environment(root, structure=structure, deps=deps, docs=docs)
    return RepoProfile(root=str(root), structure=structure, dependencies=deps, documentation=docs, environment=env)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    profile = inspect_repository(args[0] if args else ".", search="--search" in sys.argv)
    out = profile.to_dict() if "--full" in sys.argv else profile.summary()
    print(json.dumps(out, indent=2))
