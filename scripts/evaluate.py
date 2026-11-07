"""
Scores the pipeline against the hand-made reference Dockerfiles (references/).

For each repository in references/repos.txt:
  1. clone it at the pinned commit into references/.repos/<name>;
  2. let the pipeline make an image:
       template mode (default): generate_dockerfile, then one build_image. No LLM, no API key.
       agent mode (--agent):    the full agent (needs the LLM API key; 10-40 minutes per repository);
  3. score the image:
       builds   did the pipeline produce an image?
       python   Python major.minor, same as the reference image?
       torch    torch / torchvision versions (without +cpu / +cuNNN), same as the reference?
       smoke    does references/<name>/smoke.py pass in the pipeline's image (no network)?
  4. add the rows to runs/eval/results.json (one row per repository and mode: a new run replaces
     that row and keeps the others) and write runs/eval/results.md from all rows.

The reference images (tml-ref/<name>:latest) come from references/run.sh. Without them, the
python / torch columns show the pipeline's versions only.

Usage:
    uv run python -m scripts.evaluate                    # template mode, all four
    uv run python -m scripts.evaluate ffb ep_bnp         # some of them
    uv run python -m scripts.evaluate --agent ffb        # full agent on FFB
    uv run python -m scripts.evaluate --skip-build ffb   # re-score the image built last time
    uv run python -m scripts.evaluate --clean            # remove the images it made at the end
    uv run python -m scripts.evaluate --fresh            # forget earlier results first
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
CLONES = REFERENCES / ".repos"
OUT = ROOT / "runs" / "eval"
SMOKE_TIMEOUT = 1800

AGENT_PROMPT = ("Containerize the repository at {repo} so that its code can run. "
                "At the end, show the final Dockerfile you used.")

# Prints the versions in an image as one JSON line; a missing package is null
VERSION_PROBE = r"""
import json, sys
v = {"python": "%d.%d" % sys.version_info[:2]}
for name in ("torch", "torchvision"):
    try:
        v[name] = __import__(name).__version__
    except Exception:
        v[name] = None
print("VERSIONS " + json.dumps(v))
"""


@dataclass
class Result:
    name: str
    mode: str
    image: str = ""
    built: bool = False
    build_note: str = ""
    versions: dict = field(default_factory=dict)
    reference_versions: dict = field(default_factory=dict)
    smoke_passed: bool | None = None
    smoke_tail: str = ""
    agent: dict = field(default_factory=dict)
    seconds: float = 0.0


# ---------------------------------------------------------------- helpers

def read_repos(path: Path = REFERENCES / "repos.txt") -> dict[str, tuple[str, str]]:
    """name -> (url, commit), in file order."""
    repos = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            repos[parts[0]] = (parts[1], parts[2])
    return repos


def clone(name: str, url: str, commit: str) -> Path:
    """The repository at `commit` in references/.repos/<name> (cloned on first use)."""
    path = CLONES / name
    if not (path / ".git").is_dir():
        CLONES.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(path)], check=True)
    if subprocess.run(["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--quiet", commit], check=True)
    return path


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def probe_versions(image: str) -> dict:
    """Python / torch / torchvision versions inside `image`; {} if the probe cannot run."""
    p = subprocess.run(["docker", "run", "--rm", "--network", "none", image, "python", "-c", VERSION_PROBE],
                       capture_output=True, text=True, timeout=300)
    for line in p.stdout.splitlines():
        if line.startswith("VERSIONS "):
            return json.loads(line[len("VERSIONS "):])
    return {}


def run_smoke(image: str, smoke: Path) -> tuple[bool, str]:
    """Pipes the reference smoke test into `image`, without network. (passed, last lines)."""
    try:
        with smoke.open() as script:
            p = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", image, "python", "-"],
                               stdin=script, capture_output=True, text=True, timeout=SMOKE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {SMOKE_TIMEOUT}s"
    passed = p.returncode == 0 and "SMOKE OK" in p.stdout
    # a pass: the end of the normal output; a failure: the end of the errors (the traceback)
    text = p.stdout if passed or not p.stderr.strip() else p.stderr
    return passed, "\n".join(text.strip().splitlines()[-12:])


def public(version: str | None) -> str | None:
    """'1.13.1+cpu' -> '1.13.1'."""
    return version.split("+")[0] if version else version


def compare(field_name: str, got: dict, want: dict) -> str:
    """One table cell: the pipeline's version, and whether it matches the reference."""
    g = got.get(field_name)
    if not got:
        return "-"
    if field_name != "python":
        g = public(g)
    if not want:
        return f"{g}" if g else "missing"
    w = want.get(field_name) if field_name == "python" else public(want.get(field_name))
    if g == w:
        return f"{g} ✅"
    return f"{g or 'missing'} ❌ (ref {w})"


# ---------------------------------------------------------------- one repository

def pipeline_image(name: str, repo: Path, mode: str, skip_build: bool, result: Result) -> None:
    """Fills result.image / built / build_note / agent."""
    if mode == "template":
        tag = f"tml-eval/{name}:template"
        result.image = tag
        if skip_build:
            result.built = image_exists(tag)
            result.build_note = "reused" if result.built else "no image from an earlier run"
            return
        from src.docker.build import build_image
        from src.docker.generate import generate_dockerfile

        text, path, notes = generate_dockerfile(repo)
        print(f"--- {name}: generated {path}")
        for n in notes:
            print(f"    note: {n}")
        build = build_image(repo, text, tag=tag)
        result.built = build.success
        result.build_note = "built" if build.success else "build failed: " + (build.log_tail.strip().splitlines() or ["?"])[-1][:200]
        if not build.success:
            print(f"--- {name}: build failed, log: {build.log_path}")
        return

    # agent mode
    if skip_build:
        result.image = f"tml/{repo.name.lower()}:latest"
        result.built = image_exists(result.image)
        result.build_note = "reused" if result.built else "no image from an earlier run"
        return
    from scripts.run_agent import run_agent

    OUT.mkdir(parents=True, exist_ok=True)
    summary = run_agent(AGENT_PROMPT.format(repo=repo), log_path=str(OUT / f"{name}-agent.log"))
    result.agent = {k: summary[k] for k in ("rounds", "seconds", "tools_used", "images_built", "crashed")}
    result.agent["builds"] = summary["tools_used"].get("build_image", 0)
    if summary["images_built"]:
        result.image = summary["images_built"][-1]
        result.built = image_exists(result.image)
        result.build_note = f"built in {result.agent['builds']} build attempt(s)"
    else:
        result.build_note = "the agent built no image" + (f" ({summary['crashed'][:120]})" if summary["crashed"] else "")


def evaluate_one(name: str, url: str, commit: str, mode: str, skip_build: bool) -> Result:
    start = time.time()
    result = Result(name=name, mode=mode)
    print(f"\n=== {name}: {url} @ {commit[:7]} ({mode})")
    repo = clone(name, url, commit)
    try:
        pipeline_image(name, repo, mode, skip_build, result)
    except Exception as e:  # noqa: BLE001 - one repository's failure must not stop the others
        result.build_note = f"pipeline error: {type(e).__name__}: {e}"[:300]
    ref = f"tml-ref/{name}:latest"
    if image_exists(ref):
        result.reference_versions = probe_versions(ref)
    if result.built:
        result.versions = probe_versions(result.image)
        print(f"--- {name}: smoke test in {result.image}")
        result.smoke_passed, result.smoke_tail = run_smoke(result.image, REFERENCES / name / "smoke.py")
        print(result.smoke_tail)
    result.seconds = round(time.time() - start, 1)
    return result


# ---------------------------------------------------------------- report

def report(results: list[Result]) -> str:
    rows = ["| Repo | Mode | Builds | Python | torch | torchvision | Smoke test | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        smoke = "-" if r.smoke_passed is None else ("✅ pass" if r.smoke_passed else "❌ fail")
        note = r.build_note
        if r.smoke_passed is False and r.smoke_tail:
            note += "; smoke: " + r.smoke_tail.strip().splitlines()[-1][:120]
        rows.append(f"| {r.name} | {r.mode} | {'✅' if r.built else '❌'} | "
                    f"{compare('python', r.versions, r.reference_versions)} | "
                    f"{compare('torch', r.versions, r.reference_versions)} | "
                    f"{compare('torchvision', r.versions, r.reference_versions)} | {smoke} | "
                    f"{note.replace('|', '/')} |")
    passed = sum(1 for r in results if r.smoke_passed)
    return "\n".join(rows) + f"\n\n**{passed} of {len(results)}** pipeline images pass the reference smoke test.\n"


def merge(new: list[Result], path: Path) -> list[Result]:
    """Earlier rows from `path`, with each (name, mode) row of `new` replacing its old row."""
    old: list[Result] = []
    try:
        for row in json.loads(path.read_text()):
            old.append(Result(**{k: v for k, v in row.items() if k in Result.__dataclass_fields__}))
    except (OSError, ValueError, TypeError):
        pass
    replaced = {(r.name, r.mode) for r in new}
    kept = [r for r in old if (r.name, r.mode) not in replaced]
    order = {n: i for i, n in enumerate(read_repos())}
    return sorted(kept + new, key=lambda r: (r.mode, order.get(r.name, 99)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="repositories from references/repos.txt (default: all)")
    p.add_argument("--agent", action="store_true", help="run the full agent instead of the template step")
    p.add_argument("--skip-build", action="store_true", help="score the image an earlier run built")
    p.add_argument("--clean", action="store_true", help="remove the pipeline images at the end")
    p.add_argument("--fresh", action="store_true", help="forget the results of earlier runs")
    args = p.parse_args(argv)

    repos = read_repos()
    names = args.names or list(repos)
    unknown = [n for n in names if n not in repos]
    if unknown:
        p.error(f"unknown repositories {unknown}; choose from {list(repos)}")
    mode = "agent" if args.agent else "template"
    # without a daemon every image check says "missing": stop with the real reason instead
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("Docker is not reachable: start Docker Desktop (or the Docker daemon), wait until it runs, "
              "and try again. Check with: docker version", file=sys.stderr)
        return 2

    results = [evaluate_one(n, *repos[n], mode=mode, skip_build=args.skip_build) for n in names]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = merge(results, Path("/nonexistent") if args.fresh else OUT / "results.json")
    table = report(rows)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    (OUT / "results.md").write_text(f"# Pipeline vs reference (updated {stamp})\n\n{table}")
    (OUT / "results.json").write_text(json.dumps([asdict(r) for r in rows], indent=1))
    print("\nthis run:\n" + report(results))
    if len(rows) > len(results):
        print("all results so far:\n" + table)
    print(f"saved: {OUT / 'results.md'}, {OUT / 'results.json'}")

    if args.clean:
        for r in results:
            if r.image and mode == "template":
                subprocess.run(["docker", "rmi", "-f", r.image], capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
