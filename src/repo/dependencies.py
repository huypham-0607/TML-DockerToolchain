"""
    Section B - Dependency Detection

    Finds and parses dependency files in a local repository:
      - requirements*.txt / *.in  (follows -r / -c includes)
      - pyproject.toml            (PEP 621, Poetry, build-system)
      - setup.py                  (static AST read, never executed)
      - setup.cfg
      - environment.yml           (conda + nested pip list)
      - Pipfile

    Plain Python helper, not an agent tool. Use detect_dependencies(path).
"""

from __future__ import annotations

import ast
import configparser
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".conda", "node_modules",
    ".tox", ".nox", "build", "dist", "site-packages", ".ipynb_checkpoints",
}

# How deep to look for dependency files (root = 0)
MAX_DEPTH = 2

# name[extras] spec ; marker
_REQ_RE = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
    \s*(?:\[(?P<extras>[^\]]*)\])?
    \s*(?P<spec>[^;]*?)
    \s*(?:;\s*(?P<marker>.*))?$""",
    re.VERBOSE,
)

# conda: name[=version[=build]] or name>=version
_CONDA_RE = re.compile(r"^\s*(?:(?P<channel>[\w.-]+)::)?(?P<name>[A-Za-z0-9_.-]+)\s*(?P<spec>.*)$")


@dataclass
class Dependency:
    name: str                 # normalized (lowercase, - instead of _)
    spec: str = ""            # e.g. ">=1.0,<2" or "==2.1.0"
    version: str | None = None  # exact version if pinned with == / =
    source: str = ""          # relative path of the file it came from
    manager: str = "pip"      # pip | conda
    group: str = "main"       # main | dev | <extra name> | build
    extras: list[str] = field(default_factory=list)
    marker: str | None = None
    url: str | None = None    # VCS / direct URL / local path installs
    channel: str | None = None  # conda channel prefix (e.g. conda-forge::x)


@dataclass
class DependencyReport:
    root: str
    files: list[str] = field(default_factory=list)        # parsed dependency files
    dependencies: list[Dependency] = field(default_factory=list)
    python_requires: list[dict] = field(default_factory=list)  # [{"spec", "source"}]
    conda_channels: list[str] = field(default_factory=list)
    index_urls: list[str] = field(default_factory=list)   # --index-url / --extra-index-url / -f
    errors: list[dict] = field(default_factory=list)      # [{"source", "error"}]

    def packages(self) -> dict[str, str | None]:
        """name -> pinned version (None if not pinned). First occurrence wins."""
        out: dict[str, str | None] = {}
        for d in self.dependencies:
            if d.name not in out or (out[d.name] is None and d.version):
                out[d.name] = d.version
        return out

    def get(self, name: str) -> list[Dependency]:
        name = normalize(name)
        return [d for d in self.dependencies if d.name == name]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["packages"] = self.packages()
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pinned(spec: str) -> str | None:
    m = re.fullmatch(r"\s*===?\s*([^,\s]+)\s*", spec)
    if m and "*" not in m.group(1):
        return m.group(1)
    return None


def parse_requirement(line: str, source: str = "", group: str = "main") -> Dependency | None:
    """Parses one PEP 508 / pip requirement string. Returns None for options/blank lines."""
    line = re.sub(r"(^|\s)#.*$", "", line).strip()
    if not line or line.startswith("-") and not line.startswith(("-e", "--editable")):
        return None

    # editable / direct URL / local path installs
    if line.startswith(("-e ", "--editable ")):
        target = line.split(None, 1)[1].strip()
        egg = re.search(r"#egg=([\w.-]+)", target)
        name = egg.group(1) if egg else Path(target.rstrip("/")).name or target
        return Dependency(name=normalize(name), source=source, group=group, url=target)
    if " @ " in line:
        name, url = line.split(" @ ", 1)
        name, _, extras = name.partition("[")
        return Dependency(
            name=normalize(name.strip()), source=source, group=group, url=url.split(";")[0].strip(),
            extras=[e.strip() for e in extras.rstrip("]").split(",") if e.strip()],
        )
    if re.match(r"^(git\+|hg\+|https?://|file:|\.{0,2}/)", line):
        egg = re.search(r"#egg=([\w.-]+)", line)
        name = egg.group(1) if egg else Path(line.split("#")[0].rstrip("/")).stem
        return Dependency(name=normalize(name), source=source, group=group, url=line)

    m = _REQ_RE.match(line)
    if not m:
        return None
    spec = m.group("spec").strip().strip("()").replace(" ", "")
    return Dependency(
        name=normalize(m.group("name")),
        spec=spec,
        version=_pinned(spec),
        source=source,
        group=group,
        extras=[e.strip() for e in (m.group("extras") or "").split(",") if e.strip()],
        marker=(m.group("marker") or "").strip() or None,
    )


def parse_conda_spec(item: str, source: str = "") -> Dependency | None:
    m = _CONDA_RE.match(item.strip())
    if not m:
        return None
    spec = m.group("spec").replace(" ", "")
    version = None
    # conda "=1.2" means 1.2.*, "==1.2" means exact; "1.2=build" carries a build string
    if spec.startswith("=") and not spec.startswith("=="):
        version = spec[1:].split("=")[0].rstrip(".*") or None
    elif spec.startswith("=="):
        version = spec[2:].split("=")[0]
    elif re.fullmatch(r"[\d.]+", spec):  # "python 3.8" style
        version = spec
        spec = "=" + spec
    return Dependency(
        name=normalize(m.group("name")), spec=spec, version=version,
        source=source, manager="conda", channel=m.group("channel"),
    )


# ---------------------------------------------------------------- file parsers

def _parse_requirements(path: Path, rel: str, report: DependencyReport, root: Path, seen: set[Path]):
    if path in seen:
        return
    seen.add(path)
    group = "dev" if re.search(r"dev|test|lint|doc", path.name, re.I) else "main"
    text = path.read_text(errors="replace")
    text = re.sub(r"\\\r?\n", " ", text)
    for raw in text.splitlines():
        line = raw.strip()
        opt = re.match(r"^(-r|--requirement|-c|--constraint)\s*(\S+)", line)
        if opt:
            inc = (path.parent / opt.group(2)).resolve()
            if inc.is_file() and inc.is_relative_to(root):
                report.files.append(inc.relative_to(root).as_posix())
                _parse_requirements(inc, inc.relative_to(root).as_posix(), report, root, seen)
            continue
        idx = re.match(r"^(--index-url|-i|--extra-index-url|--find-links|-f)[\s=]+(\S+)", line)
        if idx:
            report.index_urls.append(idx.group(2))
            continue
        dep = parse_requirement(line, rel, group)
        if dep:
            report.dependencies.append(dep)


def _parse_pyproject(path: Path, rel: str, report: DependencyReport):
    data = tomllib.loads(path.read_text(errors="replace"))
    project = data.get("project", {})
    if "requires-python" in project:
        report.python_requires.append({"spec": project["requires-python"], "source": rel})
    for req in project.get("dependencies", []):
        if dep := parse_requirement(req, rel):
            report.dependencies.append(dep)
    for extra, reqs in project.get("optional-dependencies", {}).items():
        for req in reqs:
            if dep := parse_requirement(req, rel, extra):
                report.dependencies.append(dep)
    for grp, reqs in data.get("dependency-groups", {}).items():  # PEP 735
        for req in reqs:
            if isinstance(req, str) and (dep := parse_requirement(req, rel, grp)):
                report.dependencies.append(dep)
    for req in data.get("build-system", {}).get("requires", []):
        if dep := parse_requirement(req, rel, "build"):
            report.dependencies.append(dep)

    poetry = data.get("tool", {}).get("poetry", {})
    sections = [("main", poetry.get("dependencies", {})), ("dev", poetry.get("dev-dependencies", {}))]
    sections += [(g, v.get("dependencies", {})) for g, v in poetry.get("group", {}).items()]
    for group, deps in sections:
        for name, val in deps.items():
            spec = val if isinstance(val, str) else (val.get("version", "") if isinstance(val, dict) else "")
            if name.lower() == "python":
                report.python_requires.append({"spec": _poetry_spec(spec), "source": rel})
                continue
            url = val.get("git") or val.get("url") or val.get("path") if isinstance(val, dict) else None
            spec = _poetry_spec(spec)
            report.dependencies.append(Dependency(
                name=normalize(name), spec=spec, version=_pinned(spec) or _bare_version(spec),
                source=rel, group=group, url=url,
            ))


def _bare_version(spec: str) -> str | None:
    return spec if re.fullmatch(r"\d+(\.\d+)*", spec) else None


def _poetry_spec(spec: str) -> str:
    """Converts Poetry ^ / ~ constraints to PEP 440-ish ranges; leaves others alone."""
    spec = spec.strip()
    if spec in ("", "*"):
        return ""
    m = re.fullmatch(r"\^(\d+)(?:\.(\d+))?(?:\.(\d+))?", spec)
    if m:
        parts = [int(p) if p else 0 for p in m.groups()]
        bump = next((i for i, p in enumerate(parts) if p != 0), len(parts) - 1)
        upper = parts[:bump] + [parts[bump] + 1]
        return f">={spec[1:]},<{'.'.join(map(str, upper))}"
    m = re.fullmatch(r"~(\d+(?:\.\d+)*)", spec)
    if m:
        return f"~={m.group(1)}" if m.group(1).count(".") >= 1 else f">={m.group(1)}"
    if re.fullmatch(r"\d+(\.\d+)*", spec):
        return "==" + spec
    return spec.replace(" ", "")


def _literal(node: ast.AST, names: dict[str, ast.AST]):
    """Evaluates a literal AST node, resolving simple module-level variable names."""
    if isinstance(node, ast.Name) and node.id in names:
        node = names[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _parse_setup_py(path: Path, rel: str, report: DependencyReport):
    tree = ast.parse(path.read_text(errors="replace"))
    names: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names[node.targets[0].id] = node.value

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "setup"):
            continue
        for kw in node.keywords:
            val = _literal(kw.value, names)
            if kw.arg == "python_requires" and isinstance(val, str):
                report.python_requires.append({"spec": val, "source": rel})
            elif kw.arg in ("install_requires", "setup_requires", "tests_require"):
                group = {"install_requires": "main", "setup_requires": "build", "tests_require": "dev"}[kw.arg]
                if val is None:
                    report.errors.append({"source": rel, "error": f"{kw.arg} is not a static literal"})
                    continue
                for req in ([val] if isinstance(val, str) else val):
                    if dep := parse_requirement(req, rel, group):
                        report.dependencies.append(dep)
            elif kw.arg == "extras_require" and isinstance(val, dict):
                for extra, reqs in val.items():
                    for req in ([reqs] if isinstance(reqs, str) else reqs):
                        if dep := parse_requirement(req, rel, extra):
                            report.dependencies.append(dep)


def _parse_setup_cfg(path: Path, rel: str, report: DependencyReport):
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(path.read_text(errors="replace"))
    if cfg.has_option("options", "python_requires"):
        report.python_requires.append({"spec": cfg.get("options", "python_requires").strip(), "source": rel})
    if cfg.has_option("options", "install_requires"):
        for req in cfg.get("options", "install_requires").splitlines():
            if dep := parse_requirement(req, rel):
                report.dependencies.append(dep)
    if cfg.has_section("options.extras_require"):
        for extra, reqs in cfg.items("options.extras_require"):
            for req in reqs.splitlines():
                if dep := parse_requirement(req, rel, extra):
                    report.dependencies.append(dep)


def _load_env_yaml(text: str) -> dict:
    """Uses PyYAML if installed, otherwise a small parser for the usual environment.yml shape."""
    if yaml is not None:
        return yaml.safe_load(text) or {}
    data: dict = {}
    key = None
    in_pip = False
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0 and ":" in s and not s.startswith("-"):
            key, _, val = s.partition(":")
            key = key.strip()
            val = val.strip()
            data[key] = val.strip("'\"") if val else []
            in_pip = False
        elif s.startswith("-") and key is not None and isinstance(data.get(key), list):
            item = s[1:].strip().strip("'\"")
            if item.rstrip(":") == "pip" and item.endswith(":"):
                data[key].append({"pip": []})
                in_pip = True
            elif in_pip and indent > 2 and data[key] and isinstance(data[key][-1], dict):
                data[key][-1]["pip"].append(item)
            else:
                in_pip = False
                data[key].append(item)
    return data


def _parse_environment_yml(path: Path, rel: str, report: DependencyReport):
    data = _load_env_yaml(path.read_text(errors="replace"))
    for ch in data.get("channels") or []:
        if ch not in report.conda_channels:
            report.conda_channels.append(ch)
    for item in data.get("dependencies") or []:
        if isinstance(item, dict):
            for req in item.get("pip", []) or []:
                opt = re.match(r"^(--index-url|-i|--extra-index-url|--find-links|-f)[\s=]+(\S+)", req)
                if opt:
                    report.index_urls.append(opt.group(2))
                elif dep := parse_requirement(req, rel):
                    report.dependencies.append(dep)
            continue
        dep = parse_conda_spec(str(item), rel)
        if not dep:
            continue
        if dep.name == "python":
            report.python_requires.append({"spec": dep.spec, "source": rel})
        else:
            report.dependencies.append(dep)


def _parse_pipfile(path: Path, rel: str, report: DependencyReport):
    data = tomllib.loads(path.read_text(errors="replace"))
    pyver = data.get("requires", {}).get("python_version") or data.get("requires", {}).get("python_full_version")
    if pyver:
        report.python_requires.append({"spec": "==" + pyver, "source": rel})
    for section, group in (("packages", "main"), ("dev-packages", "dev")):
        for name, val in data.get(section, {}).items():
            spec = val if isinstance(val, str) else (val.get("version", "") if isinstance(val, dict) else "")
            spec = "" if spec == "*" else spec.replace(" ", "")
            report.dependencies.append(Dependency(
                name=normalize(name), spec=spec, version=_pinned(spec), source=rel, group=group,
                url=(val.get("git") or val.get("path")) if isinstance(val, dict) else None,
            ))


PARSERS = [
    (re.compile(r"^requirements.*\.(txt|in)$|.*requirements\.txt$", re.I), "requirements"),
    (re.compile(r"^pyproject\.toml$"), "pyproject"),
    (re.compile(r"^setup\.py$"), "setup_py"),
    (re.compile(r"^setup\.cfg$"), "setup_cfg"),
    (re.compile(r"^(environment|conda|env)[\w.-]*\.ya?ml$", re.I), "environment"),
    (re.compile(r"^Pipfile$"), "pipfile"),
]


def find_dependency_files(root: Path) -> list[tuple[Path, str]]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        for name in sorted(filenames):
            for pattern, kind in PARSERS:
                if pattern.match(name):
                    found.append((Path(dirpath) / name, kind))
                    break
    # root-level files first
    found.sort(key=lambda t: (len(t[0].relative_to(root).parts), t[0].as_posix()))
    return found


def detect_dependencies(repo_path: str | Path) -> DependencyReport:
    """Finds and parses every dependency file in the repository (top MAX_DEPTH levels)."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    report = DependencyReport(root=str(root))
    seen_reqs: set[Path] = set()
    for path, kind in find_dependency_files(root):
        rel = path.relative_to(root).as_posix()
        if path.resolve() in seen_reqs:  # already pulled in via -r
            continue
        report.files.append(rel)
        try:
            if kind == "requirements":
                _parse_requirements(path.resolve(), rel, report, root, seen_reqs)
            elif kind == "pyproject":
                _parse_pyproject(path, rel, report)
            elif kind == "setup_py":
                _parse_setup_py(path, rel, report)
            elif kind == "setup_cfg":
                _parse_setup_cfg(path, rel, report)
            elif kind == "environment":
                _parse_environment_yml(path, rel, report)
            elif kind == "pipfile":
                _parse_pipfile(path, rel, report)
        except Exception as e:  # one broken file must not kill the whole inspection
            report.errors.append({"source": rel, "error": f"{type(e).__name__}: {e}"})

    report.files = list(dict.fromkeys(report.files))
    report.index_urls = list(dict.fromkeys(report.index_urls))
    return report


if __name__ == "__main__":
    result = detect_dependencies(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(result.to_json(indent=2))
