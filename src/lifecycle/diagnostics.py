"""
    Section L3 - Diagnostics

    Reads what a managed container left behind and labels the evidence:
      - get_logs(container)   runtime logs, stdout and stderr apart, cut to a character budget
      - diagnose(container)   status + log tail + hints + processes + resource use + events
      - classify_failure()    deterministic hints from the status and the logs (no Docker calls)

    A Hint names a known failure pattern, the line that shows it, and an owner: the toolchain
    that can act on it (lifecycle: start it differently; dockerization: change the image;
    execution: change the command; host: a person has to look). The helper only labels
    evidence. The LLM makes the repair decision.

    These are runtime logs (`docker logs`). Build logs belong to the Dockerization toolchain.

    Plain Python helper, not an agent tool. Use diagnose(container).
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

from . import client
from .models import ContainerLogs, ContainerStatus, Diagnosis, Hint, LifecycleError
from .status import get_status, require_managed


# Lines of each stream that get_logs returns by default, and the most it accepts
DEFAULT_TAIL = 200
MAX_TAIL = 2000

# Log entries read from Docker for one request. Docker counts `tail` over stdout and stderr
# together, so the per-stream tail and the grep are applied here, on these entries.
SCAN_LINES = 5000

# Character budget of one get_logs result (stdout + stderr): keeps a log from flooding the LLM
MAX_LOG_CHARS = 8000

# A diagnosis carries a shorter log tail
DIAGNOSIS_TAIL = 60
DIAGNOSIS_LOG_CHARS = 4000

MAX_EVIDENCE_CHARS = 300
MAX_PROCESSES = 30
MAX_EVENTS = 30
MAX_PATTERN_CHARS = 200

# Container events worth showing (exec_* and top are noise from probes and from diagnose itself)
_EVENT_ACTIONS = ("create", "start", "die", "kill", "stop", "oom", "restart", "pause", "unpause", "destroy")

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_DURATION_RE = re.compile(r"^(\d+)\s*([smhd])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


# ---------------------------------------------------------------- failure patterns

# Checked against the Docker start error, stderr and stdout. Order = priority:
# a line that already explains one hint is not used again for a later one.
# (code, owner, pattern, suggestion)
LOG_RULES: list[tuple[str, str, re.Pattern, str]] = [
    (
        "gpu_unavailable", "lifecycle",
        re.compile(r"could not select device driver|nvidia-container-cli|failed to discover GPU vendor", re.I),
        "this daemon cannot give the container a GPU: provision again with gpu='none'",
    ),
    (
        "cuda_out_of_memory", "execution",
        re.compile(r"CUDA out of memory|CUDA error: out of memory", re.I),
        "the GPU ran out of memory: run with a smaller batch size or model; the environment is fine",
    ),
    (
        "cuda_runtime", "",   # the owner depends on whether the container has a GPU, see _cuda_hint
        re.compile(
            r"Found no NVIDIA driver|Torch not compiled with CUDA enabled|CUDA (is )?(not available|unavailable)"
            r"|no CUDA-capable device|libcuda[\w.]*: cannot open|libcudnn[\w.]*: cannot open|libnvidia[\w.-]*: cannot open"
            r"|CUDA driver version is insufficient|Could not load dynamic library 'libcud",
            re.I,
        ),
        "",
    ),
    (
        "missing_shared_library", "dockerization",
        re.compile(r"error while loading shared libraries|cannot open shared object file", re.I),
        "a system library is not in the image: install the package that provides it in the Dockerfile, then rebuild",
    ),
    (
        "missing_python_module", "dockerization",
        re.compile(r"ModuleNotFoundError|No module named|ImportError: cannot import name"),
        "a Python package is missing or has the wrong version: add or pin it in the Dockerfile, then rebuild",
    ),
    (
        "arch_mismatch", "dockerization",
        re.compile(r"exec format error", re.I),
        "the image or a binary in it is for another CPU architecture: build for this platform (--platform)",
    ),
    (
        "missing_executable", "dockerization",
        re.compile(r"executable file not found|command not found|: not found\s*$|exec \S+ failed: No such file", re.I),
        "the program is not in the image or not on PATH: install it in the Dockerfile, or correct the command",
    ),
    (
        "shm_too_small", "lifecycle",
        re.compile(r"Bus error|insufficient shared memory|shared memory \(shm\)|unable to (open|write to|allocate) shared memory|/dev/shm", re.I),
        "shared memory is too small (typical for PyTorch DataLoader workers): provision again with a larger shm_size",
    ),
    (
        "host_disk", "host",
        re.compile(r"No space left on device|ENOSPC|Disk quota exceeded", re.I),
        "a disk is full: stop and report it; free space on the host or in the mounted directory",
    ),
    (
        "out_of_memory", "lifecycle",
        re.compile(r"\bMemoryError\b|Cannot allocate memory|std::bad_alloc|^Killed\s*$|\bout of memory\b", re.I),
        "the process ran out of memory: provision again with a larger memory limit",
    ),
    (
        "port_conflict", "lifecycle",
        re.compile(r"port is already allocated|Address already in use|EADDRINUSE", re.I),
        "the port is taken: use another port",
    ),
    (
        "permission_denied", "dockerization",
        re.compile(r"Permission denied|PermissionError|EACCES|Operation not permitted", re.I),
        "a file or directory is not accessible to the user of the container: fix ownership in the Dockerfile; "
        "for a mounted directory, start the container with a matching user",
    ),
    (
        "missing_file", "dockerization",
        re.compile(r"FileNotFoundError|No such file or directory|can't open file", re.I),
        "a file the code needs is not in the container: COPY it in the Dockerfile, correct the WORKDIR, "
        "or mount the data directory",
    ),
    (
        "network", "host",
        re.compile(
            r"Temporary failure in name resolution|Could not resolve host|Name or service not known"
            r"|Network is unreachable|Connection refused|Max retries exceeded|CERTIFICATE_VERIFY_FAILED",
            re.I,
        ),
        "a download or a connection failed: check the network of the host; a container started with "
        "network='none' has no network at all",
    ),
    (
        "segfault", "dockerization",
        re.compile(r"Segmentation fault|core dumped|SIGSEGV", re.I),
        "a native library crashed: usually a version mismatch between compiled packages (framework, CUDA, numpy)",
    ),
    (
        "python_version", "dockerization",
        re.compile(r"\bSyntaxError\b"),
        "a SyntaxError in code that worked for its authors often means a different Python version: "
        "compare the image with the version the repository needs",
    ),
]


def _evidence(line: str) -> str:
    return line.strip()[:MAX_EVIDENCE_CHARS]


def _cuda_hint(line: str, has_gpu: bool) -> Hint:
    if re.search(r"not compiled with CUDA", line, re.I):
        owner, text = "dockerization", "the framework in the image is a CPU-only build: install the CUDA build, then rebuild"
    elif has_gpu:
        owner, text = "dockerization", (
            "the container has a GPU but the CUDA libraries of the image do not work with it: "
            "use a base image that matches the framework's CUDA version"
        )
    else:
        owner, text = "lifecycle", (
            "the code asked for CUDA but the container has no GPU: provision with gpu='all' on a GPU host. "
            "If the code must also run without a GPU, it hardcodes CUDA and needs a CPU fallback"
        )
    return Hint(code="cuda_runtime", evidence=_evidence(line), suggestion=text, owner=owner)


def _scan(status: ContainerStatus, logs: ContainerLogs | None) -> dict[str, Hint]:
    """Runs LOG_RULES over the start error and the logs. Returns code -> Hint, in rule order."""
    # the last matching line counts: in a traceback the final line names the error
    lines = [status.error] if status.error else []
    if logs is not None:
        lines += logs.stdout.splitlines() + logs.stderr.splitlines()
    has_gpu = bool(status.handle and status.handle.gpu)

    found: dict[str, Hint] = {}
    used: set[int] = set()
    for code, owner, pattern, suggestion in LOG_RULES:
        for i in range(len(lines) - 1, -1, -1):
            if i in used or not pattern.search(lines[i]):
                continue
            used.add(i)
            if code == "cuda_runtime":
                found[code] = _cuda_hint(lines[i], has_gpu)
            else:
                found[code] = Hint(code=code, evidence=_evidence(lines[i]), suggestion=suggestion, owner=owner)
            break
    return found


def classify_failure(status: ContainerStatus, logs: ContainerLogs | None = None) -> list[Hint]:
    """Returns the known failure patterns that the status and the logs show. No Docker calls.

    Hints from the container state (exit code, OOM, health) come first, then the ones
    found only in the logs. An empty list means: no known pattern, read the logs.

    Args:
        status: status of the container.
        logs: its logs; None looks at the status only.
    """
    found = _scan(status, logs)
    hints: list[Hint] = []
    mode = status.handle.mode if status.handle else ""
    code = status.exit_code

    def add(hint_code: str, owner: str, evidence: str, suggestion: str) -> None:
        # a log line for the same pattern is better evidence than the bare state
        seen = found.pop(hint_code, None)
        if seen is not None:
            evidence, owner = f"{evidence}; {seen.evidence}"[:MAX_EVIDENCE_CHARS], owner or seen.owner
        hints.append(Hint(code=hint_code, evidence=evidence, suggestion=suggestion, owner=owner))

    if status.oom_killed:
        add("out_of_memory", "lifecycle", f"OOMKilled (exit code {code})",
            "the kernel killed the container at its memory limit: provision again with a larger memory limit")
    elif code == 137:
        add("sigkill", "lifecycle", "exit code 137 (SIGKILL)",
            "it was killed: by a stop that ran into its timeout, by a forced removal, or because the host "
            "ran out of memory. If nothing stopped it, provision again with a larger memory limit")
    if code == 139:
        add("segfault", "dockerization", "exit code 139 (SIGSEGV)",
            "a native library crashed: usually a version mismatch between compiled packages (framework, CUDA, numpy)")
    if code == 127:
        add("missing_executable", "dockerization", "exit code 127",
            "idle mode runs `sleep infinity` and the image has no `sleep`: add basic tools to the image, "
            "or use mode='native'" if mode == "idle" else
            "the command or entrypoint is not in the image or not on PATH: install it in the Dockerfile, or correct the command")
    if code == 126:
        add("not_executable", "dockerization", "exit code 126",
            "the command exists but cannot be run: make it executable (chmod +x) or call it through its interpreter")
    if status.state == "restarting" or status.restart_count > 0:
        add("crash_loop", "lifecycle", f"restart count {status.restart_count}, state {status.state}",
            "the container keeps restarting: read the logs for the first error, then act on that cause")
    if status.health == "unhealthy" and status.state == "running":
        if status.healthcheck_output or "HEALTHCHECK" in status.reason:
            add("healthcheck_failing", "dockerization", f"HEALTHCHECK: {status.healthcheck_output}"[:MAX_EVIDENCE_CHARS],
                "the image's own HEALTHCHECK fails: the service in the container is not ready or not working")
        elif "readiness probe" in status.reason:
            add("no_exec", "dockerization", status.reason[:MAX_EVIDENCE_CHARS],
                "no command can be started in the container: the image has no basic tools (`true`), or the container is frozen")
    if status.health == "starting":
        add("slow_start", "lifecycle", status.reason[:MAX_EVIDENCE_CHARS],
            "the image HEALTHCHECK has not passed yet: wait longer (ready_timeout), then check its command")

    return hints + list(found.values())


# ---------------------------------------------------------------- log helpers (pure)

def clean_log(text: str) -> str:
    """Removes terminal colour codes and collapses progress bars that redraw one line with \\r."""
    text = _ANSI_RE.sub("", text)
    return "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))


def filter_lines(text: str, pattern: re.Pattern | None, tail: int) -> str:
    """Keeps the lines that match `pattern` (all lines if None), then the last `tail` of them."""
    lines = text.splitlines()
    if pattern is not None:
        lines = [line for line in lines if pattern.search(line)]
    return "\n".join(lines[-tail:])


def truncate_tail(text: str, max_chars: int) -> tuple[str, bool]:
    """Cuts `text` to its last `max_chars` characters, at a line start. Returns (text, was_cut)."""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:
        return "", True
    cut = text[-max_chars:]
    newline = cut.find("\n")
    # drop the partial first line, unless the cut is one very long line
    return (cut[newline + 1:] if 0 <= newline < len(cut) - 1 else cut), True


def split_budget(stdout: str, stderr: str, max_chars: int) -> tuple[int, int]:
    """Shares the character budget between the streams: a short stream leaves its rest to the other."""
    half = max_chars // 2
    if len(stdout) + len(stderr) <= max_chars:
        return len(stdout), len(stderr)
    if len(stderr) <= half:
        return max_chars - len(stderr), len(stderr)
    if len(stdout) <= half:
        return len(stdout), max_chars - len(stdout)
    return half, max_chars - half


def parse_time(value: str) -> datetime:
    """Parses a Docker / ISO 8601 timestamp (nanoseconds and a trailing Z are accepted)."""
    text = value.strip().replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)   # datetime keeps microseconds only
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_since(since: str, now: datetime) -> datetime | None:
    """Turns `since` into a point in time.

    Args:
        since: "" (no limit), a duration back from now ("30s", "10m", "2h", "1d"),
            or an ISO 8601 timestamp.
        now: the present, by the clock of the Docker daemon.

    Raises LifecycleError("invalid_argument") for anything else.
    """
    since = (since or "").strip()
    if not since:
        return None
    if m := _DURATION_RE.match(since):
        return now - timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})
    try:
        return parse_time(since)
    except ValueError:
        raise LifecycleError(
            "invalid_argument", f"invalid since {since!r}",
            "use a duration such as 30s, 10m, 2h, or an ISO timestamp such as 2026-10-09T14:00:00Z",
        ) from None


# ---------------------------------------------------------------- Docker access

def _daemon_now() -> datetime:
    """The present by the daemon's clock. It can differ from this machine's clock (Docker Desktop runs in a VM)."""
    try:
        with client.translated("read the daemon time"):
            return parse_time(client.get_client().info().get("SystemTime") or "")
    except (ValueError, LifecycleError):
        return datetime.now(timezone.utc)


def get_logs(
    container: str,
    tail: int = DEFAULT_TAIL,
    since: str = "",
    grep: str = "",
    max_chars: int = MAX_LOG_CHARS,
) -> ContainerLogs:
    """Returns the runtime logs of a managed container, stdout and stderr apart.

    The newest output is kept: at most `tail` lines of each stream, then at most
    `max_chars` characters in total (ContainerLogs.truncated says if older output was cut).
    Colour codes are removed and redrawn progress bars are collapsed to their last state.
    Only the last 5000 log entries of the container are read.

    Args:
        container: container name or id; it may be stopped.
        tail: lines of each stream to return.
        since: only output after this point: a duration back from now ("10m") or an ISO timestamp.
        grep: regular expression; only matching lines are returned.
        max_chars: character budget for stdout and stderr together.

    Raises:
        LifecycleError: not_found, not_managed, invalid_argument.
    """
    if isinstance(tail, bool) or not isinstance(tail, int) or not 1 <= tail <= MAX_TAIL:
        raise LifecycleError("invalid_argument", f"invalid tail {tail!r}", f"use a whole number from 1 to {MAX_TAIL}")
    pattern = None
    if grep:
        if len(grep) > MAX_PATTERN_CHARS:
            raise LifecycleError("invalid_argument", f"grep pattern is longer than {MAX_PATTERN_CHARS} characters")
        try:
            pattern = re.compile(grep)
        except re.error as e:
            raise LifecycleError("invalid_argument", f"invalid grep pattern {grep!r}: {e}", "use a regular expression") from e

    since = (since or "").strip()
    parse_since(since, datetime.now(timezone.utc))   # only to refuse a bad value before any Docker call

    obj = require_managed(container)
    options: dict = {"tail": SCAN_LINES}
    if since:
        # a duration counts back from the daemon's clock, which stamps the log lines
        options["since"] = max(1, math.floor(parse_since(since, _daemon_now()).timestamp()))

    with client.translated(f"read the logs of container {container!r}"):
        raw_out = obj.logs(stdout=True, stderr=False, **options)
        raw_err = obj.logs(stdout=False, stderr=True, **options)

    stdout = filter_lines(clean_log(raw_out.decode("utf-8", errors="replace")), pattern, tail)
    stderr = filter_lines(clean_log(raw_err.decode("utf-8", errors="replace")), pattern, tail)
    out_budget, err_budget = split_budget(stdout, stderr, max_chars)
    stdout, cut_out = truncate_tail(stdout, out_budget)
    stderr, cut_err = truncate_tail(stderr, err_budget)
    return ContainerLogs(
        container=container,
        stdout=stdout,
        stderr=stderr,
        lines=len(stdout.splitlines()) + len(stderr.splitlines()),
        truncated=cut_out or cut_err,
        since=since,
    )


def _processes(obj) -> list[dict]:
    top = obj.top()
    titles = [t.lower() for t in top.get("Titles") or []]
    return [dict(zip(titles, row)) for row in (top.get("Processes") or [])[:MAX_PROCESSES]]


def summarize_stats(stats: dict) -> dict:
    """CPU, memory and process figures from one `docker stats` sample. No Docker calls."""
    cpu, pre = stats.get("cpu_stats") or {}, stats.get("precpu_stats") or {}
    memory, pids = stats.get("memory_stats") or {}, stats.get("pids_stats") or {}
    out: dict = {}

    cpu_delta = (cpu.get("cpu_usage") or {}).get("total_usage", 0) - (pre.get("cpu_usage") or {}).get("total_usage", 0)
    system_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    if system_delta > 0 and cpu_delta >= 0:
        # 100 % = one full CPU, as `docker stats` shows it
        out["cpu_percent"] = round(cpu_delta / system_delta * (cpu.get("online_cpus") or 1) * 100, 1)

    if memory.get("usage") is not None:
        # like `docker stats`: the page cache that can be dropped does not count
        used = memory["usage"] - (memory.get("stats") or {}).get("inactive_file", 0)
        out["memory_bytes"] = used
        if limit := memory.get("limit"):
            out["memory_limit_bytes"] = limit
            out["memory_percent"] = round(used / limit * 100, 1)
    if pids.get("current") is not None:
        out["processes"] = pids["current"]
        if pids.get("limit"):
            out["processes_limit"] = pids["limit"]
    return out


def summarize_events(events: list[dict]) -> list[dict]:
    """Keeps the container events that explain a failure (start, die, kill, oom, health changes)."""
    out = []
    for event in events:
        action = str(event.get("Action") or event.get("status") or "")
        if action not in _EVENT_ACTIONS and not action.startswith("health_status"):
            continue
        attributes = (event.get("Actor") or {}).get("Attributes") or {}
        item = {"time": datetime.fromtimestamp(event.get("time", 0), timezone.utc).isoformat(), "action": action}
        if "exitCode" in attributes:
            item["exit_code"] = int(attributes["exitCode"])
        if "signal" in attributes:
            item["signal"] = int(attributes["signal"])
        out.append(item)
    return out[-MAX_EVENTS:]


def _events(obj, status: ContainerStatus) -> list[dict]:
    # both ends by the daemon's clock: since the container was created, until now
    created = parse_time(status.handle.created_at) if status.handle and status.handle.created_at else None
    now = _daemon_now()
    since = math.floor((created or now - timedelta(hours=1)).timestamp())
    stream = client.get_client().events(
        since=since, until=math.ceil(now.timestamp()), filters={"container": obj.id}, decode=True,
    )
    return summarize_events(list(stream))


def diagnose(container: str, tail: int = DIAGNOSIS_TAIL) -> Diagnosis:
    """Collects everything that explains the state of a managed container.

    Status and logs are required. Processes, resource use (running containers only; the
    sample takes about one second) and events are best effort: a part that cannot be
    read is left empty and named in Diagnosis.notes.
    A container that does not exist gives a Diagnosis with health "missing".

    Args:
        container: container name or id.
        tail: log lines of each stream to include.

    Raises:
        LifecycleError: not_managed, invalid_argument, docker_unavailable.
    """
    status = get_status(container)
    if status.state == "missing":
        return Diagnosis(status=status)
    try:
        obj = require_managed(container)
        logs = get_logs(container, tail=tail, max_chars=DIAGNOSIS_LOG_CHARS)
    except LifecycleError as e:
        if e.code != "not_found":
            raise
        # removed between the two looks
        return Diagnosis(status=get_status(container))

    diagnosis = Diagnosis(status=status, logs=logs, hints=classify_failure(status, logs))

    def best_effort(part: str, read):
        try:
            with client.translated(f"read the {part} of container {container!r}"):
                return read()
        except (LifecycleError, ValueError, KeyError, TypeError) as e:
            diagnosis.notes.append(f"{part} not available: {getattr(e, 'message', e)}")
            return None

    if status.running:
        diagnosis.processes = best_effort("processes", lambda: _processes(obj)) or []
        diagnosis.resources = best_effort("resource use", lambda: summarize_stats(obj.stats(stream=False))) or {}
    diagnosis.events = best_effort("events", lambda: _events(obj, status)) or []
    return diagnosis
