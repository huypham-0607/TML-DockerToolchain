"""
    Dockerization - generate_dockerfile

    Turns the inspection profile (sections A-D) into a starter Dockerfile from one template.
    The LLM can edit the result before / after calling build_image.

    CLI:
      uv run python -m src.docker.generate /tmp/augmix
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..repo.inspect import inspect_repository
from .build import RUNS_DIR

DEFAULT_PYTHON = "3.10"
FRAMEWORKS = ("torch", "torchvision", "torchaudio", "tensorflow", "jax", "jaxlib")


def _major_minor(v: str | None) -> str | None:
    return ".".join(v.split(".")[:2]) if v else None


def generate_dockerfile(repo_path: str | Path) -> tuple[str, Path, list[str]]:
    """Returns (dockerfile_text, saved_path, notes)."""
    profile = inspect_repository(repo_path)
    root = Path(profile.root)
    deps, env = profile.dependencies, profile.environment
    notes: list[str] = []

    py = _major_minor(env.resolved["python"].value) or DEFAULT_PYTHON
    if not env.resolved["python"].value:
        notes.append(f"Python version not found; defaulting to {DEFAULT_PYTHON}")

    # Frameworks installed first in their own layer (largest download, cached across rebuilds)
    pins = [f"{name}=={v}" if v else name for name, v in deps.packages().items() if name in FRAMEWORKS]

    reqs = [f for f in deps.files if Path(f).name == "requirements.txt"]
    installable = (root / "setup.py").is_file() or (root / "pyproject.toml").is_file()

    lines = [
        f"FROM python:{py}-slim",
        "ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "WORKDIR /workspace",
    ]
    if pins:
        lines.append("RUN pip install " + " ".join(f'"{p}"' for p in pins))
    lines.append("COPY . /workspace")
    lines += [f"RUN pip install -r {r}" for r in reqs]
    if installable:
        lines.append("RUN pip install -e .")
    lines.append('CMD ["bash"]')

    if env.gpu.uses_cuda:
        notes.append("repo uses CUDA: run the container with --gpus all")
    if env.gpu.needs_nvcc:
        notes.append("repo compiles CUDA code: base image likely needs to be nvidia/cuda:<ver>-devel")
    if profile.structure.docker.present:
        notes.append(f"repo has its own Docker config: {[d.path for d in profile.structure.docker.dockerfiles]}")

    text = "\n".join(lines) + "\n"
    out = RUNS_DIR / root.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "Dockerfile"
    path.write_text(text)
    return text, path, notes


if __name__ == "__main__":
    text, path, notes = generate_dockerfile(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(text)
    for n in notes:
        print("# note:", n)
    print("# saved to", path)
