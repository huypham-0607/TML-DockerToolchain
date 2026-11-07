"""
    Dockerization - generate_dockerfile

    Turns the inspection profile (sections A-D) into a starter Dockerfile from one template.
    The LLM can edit the result before / after calling build_image.

    PyTorch packages (torch, torchvision, torchaudio) are installed from the PyTorch index, in the
    build that fits this machine:
      - CPU build when Docker cannot give containers an NVIDIA GPU, CUDA build when it can
        (TML_TORCH_BUILD=cpu|cuda|auto overrides; default auto);
      - a pin with a local version (torch==1.13.1+cu116) is installed as torch==1.13.1 from the
        matching index: "+cuNNN" builds exist only there, so a plain `pip install` fails;
      - those lines are filtered out of requirements files before `pip install -r`, and every later
        install runs with the installed versions as pip constraints, so nothing replaces that build.

    CLI:
      uv run python -m src.docker.generate /tmp/augmix
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from ..repo.inspect import inspect_repository
from .build import RUNS_DIR

DEFAULT_PYTHON = "3.10"
FRAMEWORKS = ("torch", "torchvision", "torchaudio", "tensorflow", "jax", "jaxlib")
TORCH_FAMILY = ("torch", "torchvision", "torchaudio")

PYTORCH_INDEX = "https://download.pytorch.org/whl"
PYPI = "https://pypi.org/simple"
TORCH_BUILD_ENV = "TML_TORCH_BUILD"
TORCH_CONSTRAINTS = "/tmp/torch-constraints.txt"   # the torch-family versions installed first

# Import name -> pip name, where they differ (imports not listed here are installed under their own name)
IMPORT_TO_PIP = {
    "PIL": "Pillow", "sklearn": "scikit-learn", "cv2": "opencv-python-headless", "yaml": "PyYAML",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "attr": "attrs", "dateutil": "python-dateutil",
    "Crypto": "pycryptodome", "art": "adversarial-robustness-toolbox", "tensorboardX": "tensorboardX",
    "google": "protobuf", "dotenv": "python-dotenv", "jwt": "PyJWT", "serial": "pyserial", "magic": "python-magic",
    "Levenshtein": "python-Levenshtein", "fitz": "PyMuPDF", "docx": "python-docx", "lightgbm": "lightgbm",
    "pytorch_lightning": "pytorch-lightning", "timm": "timm", "einops": "einops", "keras": "keras",
}
# Imports that are never installed from a guess: torch-family packages come from the PyTorch index,
# and these are not on PyPI under that name or need a manual setup
NOT_GUESSED = set(TORCH_FAMILY) | {"apex", "mpi4py", "horovod", "IPython", "pkg_resources", "setuptools", "distutils"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def guessed_packages(imports: list[str], declared: dict[str, str | None]) -> list[str]:
    """pip names for third-party imports that no dependency file declares."""
    have = {_norm(n) for n in declared}
    out = []
    for module in imports:
        if module in NOT_GUESSED:
            continue
        pip_name = IMPORT_TO_PIP.get(module, module)
        if _norm(pip_name) not in have and _norm(module) not in have and pip_name not in out:
            out.append(pip_name)
    return out


# torch major.minor -> (torchvision, torchaudio) major.minor released with it
TORCH_COMPANIONS = {
    "1.4": ("0.5", "0.4"), "1.5": ("0.6", "0.5"), "1.6": ("0.7", "0.6"), "1.7": ("0.8", "0.7"),
    "1.8": ("0.9", "0.8"), "1.9": ("0.10", "0.9"), "1.10": ("0.11", "0.10"), "1.11": ("0.12", "0.11"),
    "1.12": ("0.13", "0.12"), "1.13": ("0.14", "0.13"), "2.0": ("0.15", "2.0"), "2.1": ("0.16", "2.1"),
    "2.2": ("0.17", "2.2"), "2.3": ("0.18", "2.3"), "2.4": ("0.19", "2.4"), "2.5": ("0.20", "2.5"),
    "2.6": ("0.21", "2.6"), "2.7": ("0.22", "2.7"), "2.8": ("0.23", "2.8"),
}


def fix_companions(torch_pins: list[str]) -> tuple[list[str], list[str]]:
    """Re-pins torchvision / torchaudio that do not match the pinned torch. Returns (pins, notes).

    Each torch release works with one torchvision / torchaudio release; pip cannot install a
    mismatched pair (torch==2.2.0 with torchvision==0.16.0 needs torch 2.1).
    """
    pins = {p.split("==")[0]: (p.split("==")[1] if "==" in p else None) for p in torch_pins}
    mm = _major_minor(pins.get("torch"))
    if mm not in TORCH_COMPANIONS:
        return torch_pins, []
    notes = []
    for name, want in zip(("torchvision", "torchaudio"), TORCH_COMPANIONS[mm]):
        have = pins.get(name)
        if have and _major_minor(have) != want:
            pins[name] = f"{want}.*"
            notes.append(f"{name}=={have} does not work with torch=={pins['torch']}: changed to {name}=={want}.*")
    return [f"{n}=={v}" if v else n for n, v in pins.items()], notes


def _old_torch(torch_pins: list[str]) -> bool:
    """True if torch is pinned below 2.3: those builds break with numpy 2."""
    for pin in torch_pins:
        if pin.startswith("torch==") and (m := re.match(r"(\d+)\.(\d+)", pin.split("==")[1])):
            return (int(m.group(1)), int(m.group(2))) < (2, 3)
    return False


# A requirements line that pins (or names) a torch-family package: "torch==1.13.1+cu116", "torchvision>=0.8"
TORCH_LINE_RE = r"^\s*(torch|torchvision|torchaudio)\s*($|[=<>!~;\[])"


def _major_minor(v: str | None) -> str | None:
    return ".".join(v.split(".")[:2]) if v else None


def _split_local(version: str | None) -> tuple[str | None, str | None]:
    """"1.13.1+cu116" -> ("1.13.1", "cu116"); "2.0.1" -> ("2.0.1", None)."""
    if not version:
        return None, None
    public, _, local = version.partition("+")
    return public, (local or None)


def host_gpu_usable() -> bool:
    """True if Docker on this machine can give a container an NVIDIA GPU.

    Uses the outcome remembered by src.lifecycle.client when there is one. Otherwise, if the daemon
    lists a GPU, tries one real attach with an image that is already on this machine. Any problem
    (no Docker, no image to probe with) counts as no GPU: the CPU build works everywhere.
    """
    try:
        from ..lifecycle import client

        info = client.get_client().info()
        if not client.gpu_claimed(info):
            return False
        if client.gpu_attach_known(info):
            return not client.gpu_attach_failed(info)
        images = [t for img in client.get_client().images.list() for t in (img.tags or [])]
        return bool(images) and client.probe_gpu_attach(images[0], info)
    except Exception:  # noqa: BLE001 - best effort: unknown means CPU
        return False


def choose_torch_build(requested: str | None = None) -> str:
    """"cpu" or "cuda": the argument, else $TML_TORCH_BUILD, else what this machine can run."""
    want = (requested or os.environ.get(TORCH_BUILD_ENV) or "auto").lower()
    if want not in ("cpu", "cuda", "auto"):
        raise ValueError(f"{TORCH_BUILD_ENV} must be cpu, cuda or auto, not {want!r}")
    if want == "auto":
        return "cuda" if host_gpu_usable() else "cpu"
    return want


def torch_index(build: str, local_tags: list[str], cuda_version: str | None) -> str | None:
    """The index URL to install torch-family packages from; None means PyPI as-is.

    cpu: the CPU index. cuda: the index named by a pin's "+cuNNN", else the repository's CUDA
    version (11.6 -> cu116), else PyPI, whose default builds include CUDA.
    """
    if build == "cpu":
        return f"{PYTORCH_INDEX}/cpu"
    tag = next((t for t in local_tags if re.fullmatch(r"cu\d+", t)), None)
    if not tag and cuda_version and (m := re.match(r"(\d+)\.(\d+)", cuda_version)):
        tag = f"cu{m.group(1)}{m.group(2)}"
    return f"{PYTORCH_INDEX}/{tag}" if tag else None


def generate_dockerfile(repo_path: str | Path, torch_build: str | None = None) -> tuple[str, Path, list[str]]:
    """Returns (dockerfile_text, saved_path, notes).

    Args:
        repo_path: the repository root.
        torch_build: "cpu", "cuda" or "auto" (default: $TML_TORCH_BUILD, else auto).
    """
    profile = inspect_repository(repo_path)
    root = Path(profile.root)
    deps, env = profile.dependencies, profile.environment
    notes: list[str] = []

    py = _major_minor(env.resolved["python"].value) or DEFAULT_PYTHON
    if not env.resolved["python"].value:
        notes.append(f"Python version not found; defaulting to {DEFAULT_PYTHON}")

    packages = deps.packages()
    torch_pins, local_tags, other_pins = [], [], []
    for name, version in packages.items():
        if name in TORCH_FAMILY:
            public, local = _split_local(version)
            torch_pins.append(f"{name}=={public}" if public else name)
            if local:
                local_tags.append(local)
        elif name in FRAMEWORKS:
            other_pins.append(f"{name}=={version}" if version else name)
    # imported but not declared anywhere (repositories without a requirements file): install unpinned
    for name in TORCH_FAMILY:
        if name in env.imports and name not in packages:
            torch_pins.append(name)
    torch_pins.sort(key=lambda pin: TORCH_FAMILY.index(pin.split("==")[0]))
    torch_pins, companion_notes = fix_companions(torch_pins)
    notes += companion_notes
    guessed = guessed_packages(env.imports, packages)

    reqs = [f for f in deps.files if Path(f).name == "requirements.txt"]
    installable = (root / "setup.py").is_file() or (root / "pyproject.toml").is_file()

    lines = [
        f"FROM python:{py}-slim",
        "ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "WORKDIR /workspace",
    ]

    # Frameworks first, in their own layers (largest downloads, cached across rebuilds)
    if torch_pins:
        build = choose_torch_build(torch_build)
        index = torch_index(build, local_tags, env.resolved["cuda"].value)
        if _old_torch(torch_pins) and not packages.get("numpy"):
            torch_pins.append("numpy<2")
            notes.append("numpy<2: torch below 2.3 does not work with numpy 2")
        quoted = " ".join(f'"{p}"' for p in torch_pins)
        if index:
            lines.append(f"RUN pip install --index-url {index} --extra-index-url {PYPI} {quoted}")
        else:
            lines.append(f"RUN pip install {quoted}")
        # later installs must keep this torch: without a constraint, a package that asks for another
        # torch version would make pip replace it (for example with the multi-GB CUDA build from PyPI)
        lines.append(f"RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > {TORCH_CONSTRAINTS}")
        if build == "cpu":
            notes.append(f"torch: CPU build ({PYTORCH_INDEX}/cpu). Docker on this machine cannot attach an NVIDIA GPU, "
                         f"or {TORCH_BUILD_ENV}=cpu. On a GPU host, use the CUDA index instead "
                         f"(for example {PYTORCH_INDEX}/cu118)")
        else:
            notes.append(f"torch: CUDA build from {index or 'PyPI (default CUDA build)'}")
        if local_tags:
            notes.append(f"requirements pin local versions ({', '.join(sorted(set(local_tags)))}) that exist only on "
                         "the PyTorch index: installed without the suffix from the index above, and filtered out "
                         "of requirements files")
    if other_pins:
        lines.append("RUN pip install " + " ".join(f'"{p}"' for p in other_pins))

    lines.append("COPY . /workspace")
    for i, r in enumerate(reqs):
        if torch_pins:
            filtered = f"/tmp/requirements-{i}.txt"
            lines.append(f"RUN grep -viE '{TORCH_LINE_RE}' {r} > {filtered}; pip install -r {filtered} -c {TORCH_CONSTRAINTS}")
        else:
            lines.append(f"RUN pip install -r {r}")
    if installable:
        lines.append("RUN pip install -e .")
    if guessed:
        # one at a time and never fatal: a wrong guess must not break the build
        names = " ".join(guessed)
        keep = f" -c {TORCH_CONSTRAINTS}" if torch_pins else ""
        lines.append(f'RUN for p in {names}; do pip install "$p"{keep} || echo "could not install $p (guessed from imports)"; done')
        notes.append(f"installed from imports that no dependency file declares (unpinned, best effort): {names}. "
                     "Pin them, or remove any that are not real dependencies")
    lines.append('CMD ["bash"]')

    gpu = env.gpu
    if gpu.gpu_required:
        where = ", ".join(gpu.hardcoded_cuda[:3]) + (" ..." if len(gpu.hardcoded_cuda) > 3 else "")
        notes.append(f"code calls CUDA without checking for a GPU ({where}): that code needs an NVIDIA GPU "
                     "(run the container with --gpus all on a GPU host); the rest may run on CPU")
    elif gpu.uses_cuda:
        notes.append("code uses a GPU when one is available and falls back to CPU")
    if gpu.needs_nvcc:
        notes.append("repo compiles CUDA code: base image likely needs to be nvidia/cuda:<ver>-devel")
    docker_cfg = profile.structure.docker
    if docker_cfg.present:
        files = [d.path for d in docker_cfg.dockerfiles] + docker_cfg.compose
        notes.append(f"repo ships its own Docker config {files}: read it and try it first; this template ignores it")

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
