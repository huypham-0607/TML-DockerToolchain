"""
    Section C - Documentation / Search

    Extracts setup information from a repository's documentation:
      - finds README / INSTALL / docs files
      - pulls out setup-related sections (Installation, Requirements, Getting Started, ...)
      - extracts shell commands from code blocks and inline lines, classified by kind
      - optional web search fallback (Google Programmable Search) when docs give nothing

    Plain Python helper, not an agent tool. Use extract_documentation(path).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path


DOC_NAME_RE = re.compile(r"^(readme|install|installation|setup|getting[_-]?started|usage|quickstart)([._-].*)?$", re.I)
DOC_EXTS = {"", ".md", ".markdown", ".rst", ".txt"}
DOC_DIRS = {"docs", "doc", "documentation"}

SETUP_HEADING_RE = re.compile(
    r"install|setup|set up|requirement|dependenc|environment|getting started|quick ?start|"
    r"prerequisite|usage|how to run|running|reproduc|train|docker|build",
    re.I,
)

# kind -> regex matched against the start of a command
COMMAND_KINDS = [
    ("pip", re.compile(r"^(python[\d.]*\s+-m\s+)?pip[\d.]*\s+install\b")),
    ("uv", re.compile(r"^uv\s+(pip|sync|add|venv|run)\b")),
    ("poetry", re.compile(r"^poetry\s+(install|add)\b")),
    ("conda", re.compile(r"^(conda|mamba|micromamba)\s+(install|create|env|activate)\b")),
    ("apt", re.compile(r"^(sudo\s+)?apt(-get)?\s+(install|update)\b")),
    ("git", re.compile(r"^git\s+(clone|submodule|lfs)\b")),
    ("docker", re.compile(r"^(sudo\s+)?docker(-compose)?\s+|^docker\s+compose\b")),
    ("setup", re.compile(r"^python[\d.]*\s+setup\.py\s+(install|develop|build)")),
    ("build", re.compile(r"^(make|cmake|bash\s+\S*(install|setup|build)\S*\.sh|sh\s+\S*(install|setup|build)\S*\.sh)\b")),
    ("download", re.compile(r"^(wget|curl|gdown|huggingface-cli\s+download)\b")),
    ("env", re.compile(r"^(export\s+\w+=|source\s+|\.\s+\S+activate|cd\s+)")),
    ("run", re.compile(r"^(python[\d.]*|bash|sh|torchrun|accelerate\s+launch|deepspeed)\s+\S+")),
]

SHELL_FENCE_LANGS = {"", "bash", "sh", "shell", "console", "zsh", "terminal", "cmd", "powershell", "text"}

# Commands that tend to matter for building an environment
SETUP_KINDS = {"pip", "uv", "poetry", "conda", "apt", "git", "setup", "build", "docker", "download"}


@dataclass
class Command:
    command: str
    kind: str           # see COMMAND_KINDS
    source: str         # relative file path
    line: int           # 1-indexed line in the source file
    section: str | None  # heading the command appeared under


@dataclass
class DocSection:
    source: str
    heading: str
    level: int
    text: str


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class DocumentationReport:
    root: str
    doc_files: list[str] = field(default_factory=list)
    setup_sections: list[DocSection] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    search_used: bool = False
    search_query: str | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def install_commands(self) -> list[Command]:
        return [c for c in self.commands if c.kind in SETUP_KINDS]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["install_commands"] = [asdict(c) for c in self.install_commands]
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def find_doc_files(root: Path) -> list[Path]:
    """README/INSTALL-like files at the root and in docs/ (one level)."""
    out = []
    candidates = [p for p in sorted(root.iterdir()) if p.is_file()]
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.lower() in DOC_DIRS:
            candidates += [p for p in sorted(d.iterdir()) if p.is_file()]
    for p in candidates:
        if DOC_NAME_RE.match(p.stem) and p.suffix.lower() in DOC_EXTS:
            out.append(p)
    # READMEs first
    out.sort(key=lambda p: (not p.stem.lower().startswith("readme"), len(p.relative_to(root).parts), p.name))
    return out


def classify_command(cmd: str) -> str | None:
    cmd = cmd.strip()
    for kind, rx in COMMAND_KINDS:
        if rx.search(cmd):
            return kind
    return None


def _split_headings(text: str, is_rst: bool) -> list[tuple[int, str, int]]:
    """Returns [(line_index, heading, level)]."""
    lines = text.splitlines()
    heads = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            heads.append((i, m.group(2), len(m.group(1))))
            continue
        # setext (Markdown) or RST underline
        if i + 1 < len(lines) and line.strip() and re.fullmatch(r"(=+|-+|~+|\^+|\*+)", lines[i + 1].strip() or "x"):
            if len(lines[i + 1].strip()) >= max(3, len(line.strip()) // 2) and not line.lstrip().startswith(("-", "*", "|")):
                char = lines[i + 1].strip()[0]
                level = {"=": 1, "-": 2, "~": 3, "^": 4, "*": 5}[char] if is_rst or char in "=-" else 3
                heads.append((i, line.strip(), level))
        # HTML headings are common in fancy READMEs
        m = re.match(r"^\s*<h([1-6])[^>]*>(.*?)</h\1>", line, re.I)
        if m:
            heads.append((i, re.sub(r"<[^>]+>", "", m.group(2)).strip(), int(m.group(1))))
    return heads


def _strip_prompt(line: str) -> str:
    line = re.sub(r"^\s*(\$|>>>|>|%|PS>|\(\S+\)\s*\$?)\s+", "", line)
    return line.strip()


def _extract_commands(text: str, rel: str, heads: list[tuple[int, str, int]]) -> list[Command]:
    lines = text.splitlines()

    def section_at(i: int) -> str | None:
        current = None
        for idx, heading, _ in heads:
            if idx > i:
                break
            current = heading
        return current

    cmds: list[Command] = []
    in_fence = False
    fence_lang = ""
    rst_block_indent = None
    pending = ""
    pending_start = 0
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        fence = re.match(r"^\s*(```|~~~)\s*([\w+-]*)", raw)
        if fence:
            in_fence = not in_fence
            fence_lang = fence.group(2).lower() if in_fence else ""
            continue

        # RST: ".. code-block:: bash" / "::" literal blocks
        if not in_fence and (
            re.match(r"^\s*\.\.\s+(code-block|code|sourcecode)::\s*(bash|sh|shell|console)?\s*$", raw)
            or (stripped.endswith("::") and not stripped.startswith(".."))
        ):
            rst_block_indent = -1
            continue
        if rst_block_indent is not None:
            if not stripped:
                continue
            indent = len(raw) - len(raw.lstrip())
            if rst_block_indent == -1 and indent > 0:
                rst_block_indent = indent
            elif indent < max(rst_block_indent, 1):
                rst_block_indent = None

        in_code = (in_fence and fence_lang in SHELL_FENCE_LANGS) or rst_block_indent is not None
        if in_code:
            candidate = _strip_prompt(raw)
        else:
            # inline `pip install foo` or a bare indented command line
            inline = re.findall(r"`([^`]+)`", raw)
            candidate = next((_strip_prompt(c) for c in inline if classify_command(_strip_prompt(c)) in SETUP_KINDS), "")
            if not candidate and raw.startswith(("    ", "\t")) and not in_fence:
                candidate = _strip_prompt(raw)
        if not candidate or candidate.startswith("#"):
            continue

        # line continuations
        if candidate.endswith("\\"):
            if not pending:
                pending_start = i
            pending += candidate[:-1].strip() + " "
            continue
        if pending:
            candidate = pending + candidate
            i = pending_start
            pending = ""

        for part in re.split(r"\s*&&\s*|\s*;\s+", candidate):
            kind = classify_command(part)
            if kind:
                cmds.append(Command(command=part.strip(), kind=kind, source=rel, line=i + 1, section=section_at(i)))
    return cmds


def _setup_sections(text: str, rel: str, heads: list[tuple[int, str, int]], max_chars: int) -> list[DocSection]:
    lines = text.splitlines()
    out = []
    for n, (idx, heading, level) in enumerate(heads):
        if not SETUP_HEADING_RE.search(heading):
            continue
        end = len(lines)
        children = []
        for idx2, heading2, level2 in heads[n + 1:]:
            if level2 <= level:
                end = idx2
                break
            children.append(heading2)
        # a parent (e.g. the README title "Robust Training") whose subsections match on their own
        # would just duplicate them; keep the specific subsections instead
        if any(SETUP_HEADING_RE.search(c) for c in children):
            continue
        body = "\n".join(lines[idx + 1:end]).strip()
        if body:
            out.append(DocSection(source=rel, heading=heading, level=level, text=body[:max_chars]))
    return out


def google_search(query: str, num: int = 5, timeout: float = 10.0) -> list[SearchResult]:
    """Google Programmable Search (Custom Search JSON API).

    Needs GOOGLE_API_KEY and GOOGLE_CSE_ID in the environment; returns [] otherwise.
    """
    key, cx = os.environ.get("GOOGLE_API_KEY"), os.environ.get("GOOGLE_CSE_ID")
    if not key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(
        {"key": key, "cx": cx, "q": query, "num": num}
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)
    return [
        SearchResult(title=it.get("title", ""), url=it.get("link", ""), snippet=it.get("snippet", ""))
        for it in data.get("items", [])
    ]


def _repo_name(root: Path) -> str:
    """Prefers the GitHub owner/name from .git/config, falls back to the folder name."""
    cfg = root / ".git" / "config"
    if cfg.is_file():
        m = re.search(r"url\s*=\s*\S*github\.com[:/]([\w.-]+/[\w.-]+?)(\.git)?\s*$", cfg.read_text(errors="replace"), re.M)
        if m:
            return m.group(1)
    return root.name


def extract_documentation(
    repo_path: str | Path,
    search: bool = False,
    search_fn=google_search,
    max_section_chars: int = 4000,
) -> DocumentationReport:
    """Extracts setup sections and commands from the repository's docs.

    Args:
        repo_path: path to the repository root.
        search: if True and the docs contain no install commands, run a web search fallback.
        search_fn: search backend (query -> list[SearchResult]); default is Google Programmable Search.
        max_section_chars: truncate each extracted section to this length.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    report = DocumentationReport(root=str(root))
    for path in find_doc_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            report.notes.append(f"could not read {rel}: {e}")
            continue
        report.doc_files.append(rel)
        heads = _split_headings(text, is_rst=path.suffix.lower() == ".rst")
        report.setup_sections += _setup_sections(text, rel, heads, max_section_chars)
        report.commands += _extract_commands(text, rel, heads)

    # de-duplicate commands (same command text), keep first occurrence
    seen = set()
    report.commands = [c for c in report.commands if not (c.command in seen or seen.add(c.command))]

    if not report.doc_files:
        report.notes.append("no README or installation docs found")

    if search and not report.install_commands:
        report.search_used = True
        report.search_query = f"{_repo_name(root)} github installation requirements"
        try:
            report.search_results = search_fn(report.search_query)
        except Exception as e:
            report.notes.append(f"search failed: {type(e).__name__}: {e}")
        if not report.search_results and search_fn is google_search:
            report.notes.append("search returned nothing (are GOOGLE_API_KEY and GOOGLE_CSE_ID set?)")

    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--search"]
    result = extract_documentation(args[0] if args else ".", search="--search" in sys.argv)
    print(result.to_json(indent=2))
