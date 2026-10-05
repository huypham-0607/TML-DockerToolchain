"""
    Section D - Environment Detection

    Infers the Python, CUDA, cuDNN, PyTorch, TensorFlow and JAX versions a repository needs.
    Every hint becomes an Evidence item (field, value, source, kind, weight); the evidence is
    then combined per field into one resolved value plus any conflicts.

    Evidence kinds (default weight):
      docker   0.95  existing Dockerfile base images
      config   0.90  .python-version, runtime.txt, python_requires, pinned dependency versions
      docs     0.60  versions mentioned in README / install docs, pip commands in docs
      code     0.50  syntax features, imports, CUDA usage in source
      inferred 0.30  compatibility tables (e.g. torch 2.1 -> CUDA 12.1, Python 3.8-3.11)

    Plain Python helper, not an agent tool. Use detect_environment(path).
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .dependencies import DependencyReport, detect_dependencies, parse_requirement
from .documentation import DocumentationReport, extract_documentation
from .structure import IGNORED_DIRS, RepoStructure, inspect_structure


WEIGHTS = {"docker": 0.95, "config": 0.9, "docs": 0.6, "code": 0.5, "inferred": 0.3}
FIELDS = ("python", "cuda", "cudnn", "torch", "tensorflow", "jax")

# Python versions we would consider picking, newest first
PYTHON_CANDIDATES = ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8", "3.7", "3.6", "2.7"]

# torch major.minor -> (default CUDA of the pip wheel, min python, max python)
TORCH_COMPAT = {
    "1.0": ("9.0", "3.5", "3.7"), "1.1": ("10.0", "3.5", "3.7"), "1.2": ("10.0", "3.5", "3.7"),
    "1.3": ("10.1", "3.5", "3.7"), "1.4": ("10.1", "3.5", "3.8"), "1.5": ("10.2", "3.5", "3.8"), "1.6": ("10.2", "3.6", "3.8"),
    "1.7": ("10.2", "3.6", "3.9"), "1.8": ("10.2", "3.6", "3.9"), "1.9": ("10.2", "3.6", "3.9"),
    "1.10": ("10.2", "3.6", "3.9"), "1.11": ("10.2", "3.7", "3.10"), "1.12": ("10.2", "3.7", "3.10"),
    "1.13": ("11.7", "3.7", "3.10"), "2.0": ("11.7", "3.8", "3.11"), "2.1": ("12.1", "3.8", "3.11"),
    "2.2": ("12.1", "3.8", "3.12"), "2.3": ("12.1", "3.8", "3.12"), "2.4": ("12.1", "3.8", "3.12"),
    "2.5": ("12.4", "3.9", "3.13"), "2.6": ("12.4", "3.9", "3.13"), "2.7": ("12.6", "3.9", "3.13"),
    "2.8": ("12.8", "3.9", "3.13"),
}

# tensorflow major.minor -> (tested CUDA, cuDNN, min python, max python)
TF_COMPAT = {
    "1.14": ("10.0", "7.4", "3.5", "3.7"), "1.15": ("10.0", "7.4", "3.5", "3.7"),
    "2.0": ("10.0", "7.4", "3.5", "3.7"), "2.1": ("10.1", "7.6", "3.5", "3.7"),
    "2.2": ("10.1", "7.6", "3.5", "3.8"), "2.3": ("10.1", "7.6", "3.5", "3.8"),
    "2.4": ("11.0", "8.0", "3.6", "3.8"), "2.5": ("11.2", "8.1", "3.6", "3.9"),
    "2.6": ("11.2", "8.1", "3.6", "3.9"), "2.7": ("11.2", "8.1", "3.7", "3.9"),
    "2.8": ("11.2", "8.1", "3.7", "3.10"), "2.9": ("11.2", "8.1", "3.7", "3.10"),
    "2.10": ("11.2", "8.1", "3.7", "3.10"), "2.11": ("11.2", "8.1", "3.7", "3.10"),
    "2.12": ("11.8", "8.6", "3.8", "3.11"), "2.13": ("11.8", "8.6", "3.8", "3.11"),
    "2.14": ("11.8", "8.7", "3.9", "3.11"), "2.15": ("12.2", "8.9", "3.9", "3.11"),
    "2.16": ("12.3", "8.9", "3.9", "3.12"), "2.17": ("12.3", "8.9", "3.9", "3.12"),
    "2.18": ("12.5", "9.3", "3.9", "3.12"),
}

FRAMEWORK_PACKAGES = {
    "torch": {"torch", "pytorch"},
    "tensorflow": {"tensorflow", "tensorflow-gpu", "tensorflow-cpu", "tf-nightly"},
    "jax": {"jax", "jaxlib"},
}
FRAMEWORK_IMPORTS = {
    "torch": {"torch"}, "tensorflow": {"tensorflow", "keras"}, "jax": {"jax", "flax"},
}

VERSION = r"v?(\d+\.\d+(?:\.\d+)?)"
DOC_PATTERNS = {
    "python": re.compile(r"\bpython\s*(?:version)?\s*(?:>=|==|=|:|\bv)?\s*(?:\(?\s*)" + r"v?([23]\.\d{1,2}(?:\.\d+)?)\b", re.I),
    "cuda": re.compile(r"\bcuda(?:\s*toolkit)?\s*(?:version)?\s*(?:>=|==|=|:)?\s*v?(\d{1,2}\.\d)(?:\.\d+)?\b", re.I),
    "cudnn": re.compile(r"\bcudnn\s*(?:version)?\s*(?:>=|==|=|:)?\s*v?(\d\.\d+)(?:\.\d+)?\b", re.I),
    "torch": re.compile(r"\b(?:pytorch|torch)\s*(?:version)?\s*(?:>=|==|=|:)?\s*" + VERSION + r"\b", re.I),
    "tensorflow": re.compile(r"\b(?:tensorflow|tf)\s*(?:version)?\s*(?:>=|==|=|:)?\s*" + VERSION + r"\b", re.I),
    "jax": re.compile(r"\bjax\s*(?:version)?\s*(?:>=|==|=|:)?\s*" + VERSION + r"\b", re.I),
}

# CUDA usage in code
CUDA_CALL_RE = re.compile(r"""\.cuda\(\)|device\s*=\s*['"]cuda|torch\.device\(\s*['"]cuda|['"]cuda:\d['"]|\.to\(\s*['"]cuda""")
CUDA_GUARD_RE = re.compile(r"cuda\.is_available\(\)|list_physical_devices\(\s*['\"]GPU|device_count\(\)")
CUDA_EXT_RE = re.compile(r"\bCUDAExtension\b|torch\.utils\.cpp_extension|\bnvcc\b|cupy|numba\.cuda|triton")

MAX_CODE_FILES = 3000


@dataclass
class Evidence:
    field: str               # python | cuda | cudnn | torch | tensorflow | jax
    value: str | None        # concrete version, e.g. "3.10", "11.8", "2.1.0"
    constraint: str | None   # version spec, e.g. ">=3.8" (python_requires, torch>=1.10)
    source: str              # file path or description
    kind: str                # docker | config | docs | code | inferred
    detail: str = ""
    weight: float = 0.0


@dataclass
class Resolved:
    value: str | None
    confidence: float        # 0..1
    constraints: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    alternatives: dict[str, float] = field(default_factory=dict)  # other candidate values -> score
    conflict: bool = False


@dataclass
class GPUUsage:
    uses_cuda: bool = False
    hardcoded_cuda: list[str] = field(default_factory=list)  # file:line where CUDA is used without a guard
    guarded: bool = False                                    # any torch.cuda.is_available() style checks
    custom_cuda_code: list[str] = field(default_factory=list)  # .cu files, CUDAExtension, cupy, triton...
    needs_nvcc: bool = False                                 # use a -devel CUDA image

    @property
    def gpu_required(self) -> bool:
        return bool(self.hardcoded_cuda or self.needs_nvcc)


@dataclass
class EnvironmentReport:
    root: str
    frameworks: list[str]
    resolved: dict[str, Resolved]
    gpu: GPUUsage
    evidence: list[Evidence]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, str | None]:
        return {k: v.value for k, v in self.resolved.items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpu"]["gpu_required"] = self.gpu.gpu_required
        d["summary"] = self.summary()
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------- version helpers

def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def mm(v: str | None) -> str | None:
    """major.minor"""
    if not v:
        return None
    t = _vtuple(v)
    return ".".join(map(str, t[:2])) if len(t) >= 2 else (str(t[0]) if t else None)


def satisfies(version: str, spec: str) -> bool:
    """Minimal PEP 440 / conda spec check (>=, <=, >, <, ==, !=, ~=, =, wildcards, commas)."""
    v = _vtuple(version)
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.match(r"^(~=|==|!=|>=|<=|>|<|=)?\s*([\d.*]+)", part)
        if not m:
            continue
        op, target = m.group(1) or "==", m.group(2)
        if target.endswith(".*") or op == "=":  # prefix match
            t = _vtuple(target.rstrip(".*"))
            ok = v[: len(t)] == t
            if (op == "!=" and ok) or (op != "!=" and not ok):
                return False
            continue
        t = _vtuple(target)
        n = max(len(v), len(t))
        vp, tp = v + (0,) * (n - len(v)), t + (0,) * (n - len(t))
        if op == "~=":
            prefix = t[:-1] if len(t) > 1 else t
            if not (vp >= tp and v[: len(prefix)] == prefix):
                return False
        elif op == "==" and v[: len(t)] != t:
            return False
        elif (op == "!=" and v[: len(t)] == t) or (op == ">=" and vp < tp) or (op == "<=" and vp > tp) \
                or (op == ">" and vp <= tp) or (op == "<" and vp >= tp):
            return False
    return True


def _key(fld: str, value: str) -> str:
    """Grouping key: CUDA/Python/cuDNN compare at major.minor, frameworks at full version."""
    return mm(value) if fld in ("python", "cuda", "cudnn") else value


# ---------------------------------------------------------------- evidence collectors

def _from_base_image(image: str, source: str) -> list[Evidence]:
    ev = []
    name, _, tag = image.partition(":")
    first = name.split("/")[0]
    registry = first if "/" in name and "." in first else None
    if registry:
        name = name.split("/", 1)[1]  # strip registry (nvcr.io/nvidia/...)
    ngc = registry == "nvcr.io"  # NGC tags are YY.MM releases, not framework versions
    tag = tag.split("@")[0]
    base = name.split("/")[-1]

    def add(fld, val, detail):
        ev.append(Evidence(fld, val, None, source, "docker", f"base image {image}: {detail}"))

    if re.search(r"(^|/)python$", name) and (m := re.match(r"(\d+\.\d+(?:\.\d+)?)", tag)):
        add("python", m.group(1), "python image")
    if base == "cuda" and (m := re.match(r"(\d+\.\d+(?:\.\d+)?)", tag)):
        add("cuda", m.group(1), "nvidia/cuda image")
    if base == "pytorch" and not ngc and (m := re.match(r"(\d+\.\d+(?:\.\d+)?)", tag)):
        add("torch", m.group(1), "pytorch image")
    if base == "tensorflow" and not ngc and (m := re.match(r"(\d+\.\d+(?:\.\d+)?)", tag)):
        add("tensorflow", m.group(1), "tensorflow image")
    if m := re.search(r"cuda[-_]?(\d+\.\d+(?:\.\d+)?)", tag):
        add("cuda", m.group(1), "cuda in tag")
    if m := re.search(r"cudnn[-_]?(\d+(?:\.\d+)?)", tag):
        add("cudnn", m.group(1), "cudnn in tag")
    if m := re.search(r"(?:^|-)py(?:thon)?(\d)(\d{1,2})(?:-|$)", tag):
        add("python", f"{m.group(1)}.{m.group(2)}", "python in tag")
    return ev


def _from_dependency(dep, kind: str) -> list[Evidence]:
    ev = []
    name, src = dep.name, dep.source
    detail = f"{name}{dep.spec}" if dep.spec else name
    for fw, pkgs in FRAMEWORK_PACKAGES.items():
        if name in pkgs:
            version = dep.version
            if version and "+cu" in version:  # torch==1.13.1+cu117
                version, _, local = version.partition("+")
                if m := re.match(r"cu(\d+)(\d)$", local):
                    ev.append(Evidence("cuda", f"{m.group(1)}.{m.group(2)}", None, src, kind, detail))
            constraint = None if version else (dep.spec or None)
            ev.append(Evidence(fw, version, constraint, src, kind, detail))
    if name in ("cudatoolkit", "pytorch-cuda", "cuda-version", "cuda-toolkit", "cuda-runtime"):
        if dep.version:
            ev.append(Evidence("cuda", dep.version, None, src, kind, detail))
        elif dep.spec:
            ev.append(Evidence("cuda", None, dep.spec, src, kind, detail))
    if name == "cudnn" and dep.version:
        ev.append(Evidence("cudnn", dep.version, None, src, kind, detail))
    if m := re.match(r"nvidia-[\w-]+-cu(\d+)$", name):  # nvidia-cudnn-cu12, nvidia-cuda-runtime-cu11
        ev.append(Evidence("cuda", None, f"=={m.group(1)}.*", src, kind, detail))
        if name.startswith("nvidia-cudnn") and dep.version:
            ev.append(Evidence("cudnn", mm(dep.version), None, src, kind, detail))
    if name == "jax" and (cu := [e for e in dep.extras if re.match(r"cuda\d+", e)]):
        ev.append(Evidence("cuda", None, f"=={cu[0][4:]}.*", src, kind, f"jax[{cu[0]}]"))
    return ev


def _from_index_url(url: str, source: str, kind: str) -> list[Evidence]:
    if m := re.search(r"/whl/(?:nightly/)?cu(\d+)(\d)\b", url):
        return [Evidence("cuda", f"{m.group(1)}.{m.group(2)}", None, source, kind, f"index {url}")]
    return []


def _collect_config(root: Path, deps: DependencyReport) -> list[Evidence]:
    ev = []
    pv = root / ".python-version"
    if pv.is_file():
        val = pv.read_text(errors="replace").strip().splitlines()
        if val and re.match(r"\d+\.\d+", val[0]):
            ev.append(Evidence("python", val[0].strip(), None, ".python-version", "config"))
    rt = root / "runtime.txt"
    if rt.is_file() and (m := re.search(r"python-(\d+\.\d+(?:\.\d+)?)", rt.read_text(errors="replace"))):
        ev.append(Evidence("python", m.group(1), None, "runtime.txt", "config"))
    for pr in deps.python_requires:
        spec = pr["spec"]
        exact = re.fullmatch(r"(?:==?)?\s*(\d+\.\d+(?:\.\d+)?)\*?", spec.strip())
        if exact:
            ev.append(Evidence("python", exact.group(1), None, pr["source"], "config", f"python {spec}"))
        else:
            ev.append(Evidence("python", None, spec, pr["source"], "config", f"python {spec}"))
    for dep in deps.dependencies:
        ev += _from_dependency(dep, "config")
    for url in deps.index_urls:
        ev += _from_index_url(url, "dependency files", "config")
    return ev


def _collect_docs(root: Path, docs: DocumentationReport) -> list[Evidence]:
    ev = []
    for rel in docs.doc_files:
        try:
            text = (root / rel).read_text(errors="replace")
        except OSError:
            continue
        for fld, rx in DOC_PATTERNS.items():
            for m in rx.finditer(text):
                # skip versions that are clearly something else (e.g. "python 3.14159")
                ev.append(Evidence(fld, m.group(1), None, rel, "docs", m.group(0).strip()))
    for cmd in docs.commands:
        if cmd.kind in ("pip", "uv"):
            args = re.sub(r"^.*?\binstall\b", "", cmd.command)
            for url in re.findall(r"(?:--index-url|--extra-index-url|-f|--find-links)[\s=]+(\S+)", args):
                ev += _from_index_url(url, cmd.source, "docs")
            args = re.sub(r"(--index-url|--extra-index-url|-f|--find-links|-r|-c)[\s=]+\S+", "", args)
            for token in re.findall(r"""(?:"[^"]+"|'[^']+'|\S+)""", args):
                token = token.strip("'\"")
                if token.startswith("-"):
                    continue
                if dep := parse_requirement(token, cmd.source):
                    ev += _from_dependency(dep, "docs")
        elif cmd.kind == "conda":
            for m in re.finditer(r"\b(python|cudatoolkit|pytorch-cuda|pytorch|cudnn)=+([\d.]+)", cmd.command):
                fld = {"pytorch": "torch", "cudatoolkit": "cuda", "pytorch-cuda": "cuda"}.get(m.group(1), m.group(1))
                ev.append(Evidence(fld, m.group(2), None, cmd.source, "docs", cmd.command))
        elif cmd.kind == "docker":
            if m := re.search(r"\b(?:docker\s+(?:run|pull)\b.*?\s)([\w./-]+:[\w.-]+)", cmd.command):
                ev += [Evidence(e.field, e.value, None, cmd.source, "docs", e.detail) for e in _from_base_image(m.group(1), cmd.source)]
    return ev


def _iter_code_files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            if name.endswith((".py", ".ipynb", ".cu", ".cuh")):
                yield Path(dirpath) / name
                count += 1
                if count >= MAX_CODE_FILES:
                    return


def _notebook_source(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return ""
    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    out = []
    for c in cells:
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            # drop IPython magics / shell escapes so ast can parse
            out.append("\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in src.splitlines()))
    return "\n".join(out)


def _collect_code(root: Path) -> tuple[list[Evidence], set[str], GPUUsage]:
    ev: list[Evidence] = []
    imported: set[str] = set()
    gpu = GPUUsage()
    min_py: tuple[str, str] | None = None  # (version, file)
    py2_files: list[str] = []
    unguarded: list[str] = []

    for path in _iter_code_files(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix in (".cu", ".cuh"):
            gpu.custom_cuda_code.append(rel)
            gpu.needs_nvcc = True
            continue
        text = _notebook_source(path) if path.suffix == ".ipynb" else path.read_text(errors="replace")

        try:
            tree = ast.parse(text)
        except SyntaxError:
            if re.search(r"^\s*print\s+[\"'\w]", text, re.M) or re.search(r"except\s+\w+\s*,\s*\w+\s*:", text):
                py2_files.append(rel)
            tree = None

        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])
                need = None
                if type(node).__name__ == "Match":
                    need = "3.10"
                elif isinstance(node, ast.NamedExpr):
                    need = "3.8"
                elif isinstance(node, ast.JoinedStr):
                    need = "3.6"
                if need and (min_py is None or _vtuple(need) > _vtuple(min_py[0])):
                    min_py = (need, rel)

        has_guard = bool(CUDA_GUARD_RE.search(text))
        gpu.guarded |= has_guard
        for i, line in enumerate(text.splitlines(), 1):
            if CUDA_CALL_RE.search(line):
                gpu.uses_cuda = True
                if not has_guard and not line.lstrip().startswith("#"):
                    unguarded.append(f"{rel}:{i}")
        if CUDA_EXT_RE.search(text):
            gpu.custom_cuda_code.append(rel)
            if re.search(r"\bCUDAExtension\b|\bnvcc\b|load_inline|cpp_extension\.load", text):
                gpu.needs_nvcc = True

    gpu.hardcoded_cuda = unguarded[:50]
    gpu.uses_cuda |= bool(gpu.custom_cuda_code)
    if py2_files:
        ev.append(Evidence("python", "2.7", None, py2_files[0], "code",
                           f"Python 2 syntax in {len(py2_files)} file(s)"))
    elif min_py:
        ev.append(Evidence("python", None, f">={min_py[0]}", min_py[1], "code",
                           f"syntax needs Python >= {min_py[0]}"))
    return ev, imported, gpu


def _inferred(evidence: list[Evidence]) -> list[Evidence]:
    """Uses compatibility tables to infer CUDA/Python from the framework versions."""
    ev = []
    for e in evidence:
        if e.kind == "inferred" or not e.value:
            continue
        if e.field == "torch" and (row := TORCH_COMPAT.get(mm(e.value))):
            cuda, lo, hi = row
            ev.append(Evidence("cuda", cuda, None, e.source, "inferred", f"default CUDA for torch {e.value}"))
            ev.append(Evidence("python", None, f">={lo},<={hi}", e.source, "inferred", f"torch {e.value} supports Python {lo}-{hi}"))
        if e.field == "tensorflow" and (row := TF_COMPAT.get(mm(e.value))):
            cuda, cudnn, lo, hi = row
            ev.append(Evidence("cuda", cuda, None, e.source, "inferred", f"tested CUDA for tensorflow {e.value}"))
            ev.append(Evidence("cudnn", cudnn, None, e.source, "inferred", f"tested cuDNN for tensorflow {e.value}"))
            ev.append(Evidence("python", None, f">={lo},<={hi}", e.source, "inferred", f"tensorflow {e.value} supports Python {lo}-{hi}"))
    return ev


# ---------------------------------------------------------------- combining

def _resolve(fld: str, evidence: list[Evidence]) -> Resolved:
    items = [e for e in evidence if e.field == fld]
    constraints = list(dict.fromkeys(e.constraint for e in items if e.constraint))
    scores: dict[str, float] = {}
    display: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    for e in items:
        if not e.value:
            continue
        k = _key(fld, e.value)
        # each (kind, source) counts once per value; sum across independent sources, cap at 1
        tag = f"{e.kind}:{e.source}"
        if tag in sources.setdefault(k, []):
            continue
        sources[k].append(tag)
        scores[k] = min(1.0, scores.get(k, 0.0) + e.weight * (1 - scores.get(k, 0.0)))
        if k not in display or len(e.value) > len(display[k]):
            display[k] = e.value  # keep the most specific spelling (11.8.0 over 11.8)

    # drop candidates that violate hard (non-inferred) constraints, if any candidate survives
    hard = [e.constraint for e in items if e.constraint and e.kind != "inferred"]
    ok = {k for k in scores if all(satisfies(display[k], c) for c in hard)}
    pool = ok or set(scores)

    if pool:
        best = max(pool, key=lambda k: (scores[k], _vtuple(display[k])))
        alts = {display[k]: round(scores[k], 2) for k in scores if k != best}
        strong_alts = [k for k in scores if k != best and scores[k] >= 0.5]
        return Resolved(
            value=display[best], confidence=round(scores[best], 2), constraints=constraints,
            sources=sources[best], alternatives=alts, conflict=bool(strong_alts),
        )

    if fld == "python" and constraints:
        # nothing concrete: pick the newest Python allowed by every constraint
        for cand in PYTHON_CANDIDATES:
            if all(satisfies(cand, c) for c in constraints):
                return Resolved(value=cand, confidence=0.4, constraints=constraints,
                                sources=["picked newest version satisfying constraints"])
        hard_only = [c for c in constraints if c in hard]
        for cand in PYTHON_CANDIDATES:  # inferred ranges conflict with config: trust config
            if all(satisfies(cand, c) for c in hard_only):
                return Resolved(value=cand, confidence=0.3, constraints=constraints, conflict=True,
                                sources=["constraints conflict; satisfied hard constraints only"])
    return Resolved(value=None, confidence=0.0, constraints=constraints)


def detect_environment(
    repo_path: str | Path,
    structure: RepoStructure | None = None,
    deps: DependencyReport | None = None,
    docs: DocumentationReport | None = None,
) -> EnvironmentReport:
    """Combines evidence from code, config and docs into Python/CUDA/framework versions.

    Pass in results from sections A-C if already computed, otherwise they are computed here.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    structure = structure or inspect_structure(root)
    deps = deps or detect_dependencies(root)
    docs = docs or extract_documentation(root)

    evidence: list[Evidence] = []
    for df in structure.docker.dockerfiles:
        for image in df.base_images:
            evidence += _from_base_image(image, df.path)
    evidence += _collect_config(root, deps)
    evidence += _collect_docs(root, docs)
    code_ev, imported, gpu = _collect_code(root)
    evidence += code_ev
    evidence += _inferred(evidence)
    for e in evidence:
        e.weight = WEIGHTS[e.kind]

    frameworks = [
        fw for fw in FRAMEWORK_PACKAGES
        if any(e.field == fw for e in evidence) or imported & FRAMEWORK_IMPORTS[fw]
    ]

    resolved = {fld: _resolve(fld, evidence) for fld in FIELDS}
    # only report framework versions for frameworks that are actually used
    resolved = {k: v for k, v in resolved.items() if k in ("python", "cuda", "cudnn") or k in frameworks}

    notes = []
    if not gpu.uses_cuda and not any(e.field in ("cuda", "cudnn") and e.kind != "inferred" for e in evidence):
        guess = resolved["cuda"]
        resolved["cuda"] = Resolved(value=None, confidence=0.0,
                                    alternatives={guess.value: guess.confidence} if guess.value else {})
        resolved["cudnn"] = Resolved(value=None, confidence=0.0)
        notes.append("no CUDA usage found; a CPU image may be enough")
    if gpu.hardcoded_cuda:
        notes.append(
            f"CUDA is used without an availability check in {len(gpu.hardcoded_cuda)} place(s) "
            "(e.g. .cuda() / device='cuda'); the code will likely fail on a machine without a GPU"
        )
    if gpu.needs_nvcc:
        notes.append("custom CUDA code is compiled; use a CUDA -devel image (needs nvcc)")
    for fld, r in resolved.items():
        if r.conflict:
            notes.append(f"conflicting {fld} versions: chose {r.value}, also seen {list(r.alternatives)}")

    return EnvironmentReport(
        root=str(root), frameworks=frameworks, resolved=resolved, gpu=gpu, evidence=evidence, notes=notes,
    )


if __name__ == "__main__":
    result = detect_environment(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(result.to_json(indent=2))
