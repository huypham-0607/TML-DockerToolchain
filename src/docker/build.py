"""
    Dockerization - build_image

    Thin wrapper around `docker build`. The LLM writes / repairs the Dockerfile;
    this code only runs the build and returns a structured result (README: "Key Split").

    The Dockerfile is written to runs/<repo>/ (not into the repository) and built
    with the repository as the build context.

    CLI:
      uv run python -m src.docker.build /tmp/augmix path/to/Dockerfile [tag]
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
LOG_TAIL_CHARS = 4000


@dataclass
class BuildResult:
    success: bool
    image: str | None       # tag of the built image, None if the build failed
    exit_code: int | None   # None if docker could not run or the build timed out
    log_tail: str           # last part of the build log, for the LLM to diagnose
    log_path: str | None = None
    dockerfile_path: str | None = None
    timed_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def build_image(repo_path: str | Path, dockerfile: str, tag: str | None = None,
                timeout: int = 3600) -> BuildResult:
    """Builds `dockerfile` (Dockerfile text) with the repository as the build context."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        return BuildResult(False, None, None, f"not a directory: {repo}")
    tag = tag or f"tml/{repo.name.lower()}:latest"

    out = RUNS_DIR / repo.name
    out.mkdir(parents=True, exist_ok=True)
    df = out / "Dockerfile"
    df.write_text(dockerfile)
    log_path = out / "build.log"

    cmd = ["docker", "build", "--progress=plain", "-f", str(df), "-t", tag, str(repo)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return BuildResult(False, None, None, "docker is not installed or not on PATH", dockerfile_path=str(df))
    except subprocess.TimeoutExpired:
        return BuildResult(False, None, None, f"build timed out after {timeout}s",
                           dockerfile_path=str(df), timed_out=True)

    log = p.stdout + p.stderr
    log_path.write_text(log)
    ok = p.returncode == 0
    return BuildResult(ok, tag if ok else None, p.returncode, log[-LOG_TAIL_CHARS:],
                       str(log_path), str(df))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m src.docker.build <repo_path> <Dockerfile> [tag]")
    result = build_image(sys.argv[1], Path(sys.argv[2]).read_text(), sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps(result.to_dict(), indent=2))
