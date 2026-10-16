import json
import unittest
from datetime import datetime, timezone

import requests

from src.lifecycle import diagnostics
from src.lifecycle.diagnostics import (
    LOG_RULES, MAX_EVENTS, MAX_EVIDENCE_CHARS, MAX_PROCESSES,
    classify_failure, diagnose, summarize_events, summarize_stats,
)
from src.lifecycle.models import OWNERS, ContainerHandle, ContainerLogs, ContainerStatus, Diagnosis, LifecycleError
from src.lifecycle.teardown import stop
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, api_error, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker


def tearDownModule():
    samples.cleanup()


def status(
    state: str = "exited", health: str = "failed", exit_code: int | None = 1,
    mode: str = "native", gpu: bool = False, **fields,
) -> ContainerStatus:
    handle = ContainerHandle(id="d7d9d9bb3b41", name="tml-demo", image="demo:1", image_id="51dafde81dbd", mode=mode, gpu=gpu)
    return ContainerStatus(container="tml-demo", state=state, health=health, exit_code=exit_code, handle=handle, **fields)


def logs(stderr: str = "", stdout: str = "") -> ContainerLogs:
    return ContainerLogs(container="tml-demo", stdout=stdout, stderr=stderr)


def codes(hints) -> list[str]:
    return [h.code for h in hints]


# ---------------------------------------------------------------- pure suites (no daemon)

class TestClassifyFailure(unittest.TestCase):
    """One log line per known pattern: the hint, and the toolchain that can act on it."""

    # (log line, hint code, owner)
    LOG_CASES = [
        ("docker: Error response from daemon: could not select device driver \"\" with capabilities: [[gpu]].", "gpu_unavailable", "lifecycle"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 10.76 GiB total capacity)", "cuda_out_of_memory", "execution"),
        ("ImportError: libGL.so.1: cannot open shared object file: No such file or directory", "missing_shared_library", "dockerization"),
        ("python: error while loading shared libraries: libpython3.8.so.1.0: cannot open shared object file", "missing_shared_library", "dockerization"),
        ("ModuleNotFoundError: No module named 'torch'", "missing_python_module", "dockerization"),
        ("ImportError: cannot import name 'container_abcs' from 'torch._six'", "missing_python_module", "dockerization"),
        ("exec /usr/local/bin/python: exec format error", "arch_mismatch", "dockerization"),
        ("sh: 1: nvidia-smi: not found", "missing_executable", "dockerization"),
        ("bash: line 1: conda: command not found", "missing_executable", "dockerization"),
        ("[FATAL tini (7)] exec sleep failed: No such file or directory", "missing_executable", "dockerization"),
        ("RuntimeError: DataLoader worker (pid 123) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory.", "shm_too_small", "lifecycle"),
        ("OSError: [Errno 28] No space left on device", "host_disk", "host"),
        ("MemoryError", "out_of_memory", "lifecycle"),
        ("Killed", "out_of_memory", "lifecycle"),
        ("OSError: [Errno 98] Address already in use", "port_conflict", "lifecycle"),
        ("PermissionError: [Errno 13] Permission denied: '/workspace/output/log.txt'", "permission_denied", "dockerization"),
        ("python: can't open file '/workspace/train.py': [Errno 2] No such file or directory", "missing_file", "dockerization"),
        ("FileNotFoundError: [Errno 2] No such file or directory: 'data/cifar10/train.bin'", "missing_file", "dockerization"),
        ("urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>", "network", "host"),
        ("Segmentation fault (core dumped)", "segfault", "dockerization"),
        ("SyntaxError: invalid syntax", "python_version", "dockerization"),
    ]

    def test_each_known_log_line(self):
        for line, code, owner in self.LOG_CASES:
            hints = classify_failure(status(), logs(stderr=f"Traceback (most recent call last):\n  File \"train.py\", line 3\n{line}\n"))
            self.assertEqual([(h.code, h.owner) for h in hints], [(code, owner)], line)
            self.assertEqual(hints[0].evidence, line[:MAX_EVIDENCE_CHARS], line)
            self.assertTrue(hints[0].suggestion, line)

    def test_every_rule_has_a_case(self):
        covered = {code for _, code, _ in self.LOG_CASES} | {"cuda_runtime"}        # cuda_runtime: tests below
        self.assertEqual({rule[0] for rule in LOG_RULES}, covered)

    def test_every_owner_is_a_known_toolchain(self):
        for code, owner, _, suggestion in LOG_RULES:
            if code != "cuda_runtime":
                self.assertIn(owner, OWNERS, code)
                self.assertTrue(suggestion, code)

    def test_memory_error_class_name_is_recognised(self):
        hints = classify_failure(status(), logs(stderr="numpy.core._exceptions.MemoryError: Unable to allocate 64.0 GiB"))
        self.assertEqual(codes(hints), ["out_of_memory"])

    def test_lines_in_stdout_count_too(self):
        hints = classify_failure(status(), logs(stdout="ModuleNotFoundError: No module named 'yaml'"))
        self.assertEqual(codes(hints), ["missing_python_module"])

    def test_nothing_known(self):
        self.assertEqual(classify_failure(status(exit_code=3), logs(stderr="boom")), [])

    def test_healthy_container_has_no_hints(self):
        healthy = status(state="running", health="healthy", exit_code=None)
        self.assertEqual(classify_failure(healthy, logs(stdout="epoch 1\nepoch 2")), [])

    def test_without_logs_only_the_state_counts(self):
        self.assertEqual(classify_failure(status(exit_code=3)), [])
        self.assertEqual(codes(classify_failure(status(exit_code=139))), ["segfault"])

    # --- CUDA: the owner depends on the container

    def test_cuda_without_a_gpu_in_the_container(self):
        (hint,) = classify_failure(status(gpu=False), logs(stderr="RuntimeError: Found no NVIDIA driver on your system."))
        self.assertEqual((hint.code, hint.owner), ("cuda_runtime", "lifecycle"))
        self.assertIn("gpu='all'", hint.suggestion)
        self.assertIn("hardcodes CUDA", hint.suggestion)          # the IPmix case from the README

    def test_cuda_with_a_gpu_in_the_container(self):
        (hint,) = classify_failure(status(gpu=True), logs(stderr="RuntimeError: Found no NVIDIA driver on your system."))
        self.assertEqual((hint.code, hint.owner), ("cuda_runtime", "dockerization"))
        self.assertIn("base image", hint.suggestion)

    def test_cpu_only_framework_build_is_an_image_problem(self):
        for gpu in (False, True):
            (hint,) = classify_failure(status(gpu=gpu), logs(stderr="AssertionError: Torch not compiled with CUDA enabled"))
            self.assertEqual((hint.code, hint.owner), ("cuda_runtime", "dockerization"))
            self.assertIn("CPU-only", hint.suggestion)

    def test_other_cuda_wordings(self):
        for line in (
            "RuntimeError: CUDA is not available",
            "RuntimeError: cuda unavailable, invalid device ordinal",
            "OSError: libcudnn.so.8: cannot open shared object file: No such file or directory",
            "Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file",
            "RuntimeError: The NVIDIA driver on your system is too old (found version 10010). CUDA driver version is insufficient",
        ):
            self.assertEqual(codes(classify_failure(status(), logs(stderr=line))), ["cuda_runtime"], line)

    # --- one line explains one hint

    def test_cuda_out_of_memory_is_not_also_host_memory(self):
        hints = classify_failure(status(gpu=True), logs(stderr="RuntimeError: CUDA out of memory."))
        self.assertEqual(codes(hints), ["cuda_out_of_memory"])

    def test_missing_library_is_not_also_a_missing_file(self):
        hints = classify_failure(status(), logs(stderr="ImportError: libGL.so.1: cannot open shared object file: No such file or directory"))
        self.assertEqual(codes(hints), ["missing_shared_library"])

    def test_missing_cuda_library_is_a_cuda_problem_first(self):
        hints = classify_failure(status(), logs(stderr="OSError: libcudnn.so.8: cannot open shared object file: No such file or directory"))
        self.assertEqual(codes(hints), ["cuda_runtime"])

    def test_two_different_problems_give_two_hints(self):
        text = "ModuleNotFoundError: No module named 'yaml'\nOSError: [Errno 28] No space left on device"
        self.assertEqual(codes(classify_failure(status(), logs(stderr=text))), ["missing_python_module", "host_disk"])

    def test_evidence_is_the_last_matching_line(self):
        text = "ModuleNotFoundError: No module named 'first'\nretrying\nModuleNotFoundError: No module named 'second'"
        (hint,) = classify_failure(status(), logs(stderr=text))
        self.assertIn("second", hint.evidence)

    def test_stderr_is_preferred_over_stdout(self):
        hints = classify_failure(status(), logs(stdout="No module named 'from_stdout'", stderr="No module named 'from_stderr'"))
        self.assertIn("from_stderr", hints[0].evidence)

    def test_evidence_is_cut(self):
        (hint,) = classify_failure(status(), logs(stderr="ModuleNotFoundError: " + "x" * 2000))
        self.assertEqual(len(hint.evidence), MAX_EVIDENCE_CHARS)

    def test_docker_start_error_is_read(self):
        failed = status(state="created", exit_code=None, error='could not select device driver "" with capabilities: [[gpu]]')
        (hint,) = classify_failure(failed)
        self.assertEqual((hint.code, hint.owner), ("gpu_unavailable", "lifecycle"))

    # --- hints from the container state

    def test_out_of_memory_kill(self):
        (hint,) = classify_failure(status(exit_code=137, oom_killed=True), logs())
        self.assertEqual((hint.code, hint.owner), ("out_of_memory", "lifecycle"))
        self.assertIn("OOMKilled", hint.evidence)
        self.assertIn("memory limit", hint.suggestion)

    def test_out_of_memory_kill_takes_the_log_line_as_evidence(self):
        (hint,) = classify_failure(status(exit_code=137, oom_killed=True), logs(stderr="Killed"))
        self.assertEqual(hint.code, "out_of_memory")               # one hint, not two
        self.assertEqual(hint.evidence, "OOMKilled (exit code 137); Killed")

    def test_sigkill_without_out_of_memory(self):
        (hint,) = classify_failure(status(health="stopped", exit_code=137), logs())
        self.assertEqual((hint.code, hint.owner), ("sigkill", "lifecycle"))
        self.assertIn("timeout", hint.suggestion)

    def test_graceful_stop_has_no_hint(self):
        self.assertEqual(classify_failure(status(health="stopped", exit_code=143), logs()), [])

    def test_segfault_exit_code(self):
        (hint,) = classify_failure(status(exit_code=139), logs(stderr="Segmentation fault (core dumped)"))
        self.assertEqual((hint.code, hint.owner), ("segfault", "dockerization"))
        self.assertEqual(hint.evidence, "exit code 139 (SIGSEGV); Segmentation fault (core dumped)")

    def test_exit_127_in_idle_mode_is_about_sleep(self):
        (hint,) = classify_failure(status(exit_code=127, mode="idle"), logs(stderr="[FATAL tini (7)] exec sleep failed: No such file or directory"))
        self.assertEqual((hint.code, hint.owner), ("missing_executable", "dockerization"))
        self.assertIn("sleep", hint.suggestion)
        self.assertIn("mode='native'", hint.suggestion)
        self.assertIn("exec sleep failed", hint.evidence)

    def test_exit_127_in_native_mode_is_about_the_command(self):
        (hint,) = classify_failure(status(exit_code=127, mode="native"), logs())
        self.assertEqual(hint.code, "missing_executable")
        self.assertNotIn("sleep", hint.suggestion)
        self.assertIn("PATH", hint.suggestion)

    def test_exit_126(self):
        (hint,) = classify_failure(status(exit_code=126), logs())
        self.assertEqual((hint.code, hint.owner), ("not_executable", "dockerization"))

    def test_crash_loop(self):
        looping = status(state="restarting", health="unhealthy", exit_code=1, restart_count=4)
        hints = classify_failure(looping, logs())
        self.assertEqual(codes(hints), ["crash_loop"])
        self.assertIn("restart count 4", hints[0].evidence)

    def test_failing_healthcheck(self):
        sick = status(state="running", health="unhealthy", exit_code=None, healthcheck_output="db down", reason="running; the image HEALTHCHECK fails: db down")
        (hint,) = classify_failure(sick, logs())
        self.assertEqual((hint.code, hint.owner), ("healthcheck_failing", "dockerization"))
        self.assertEqual(hint.evidence, "HEALTHCHECK: db down")

    def test_failed_readiness_probe(self):
        frozen = status(state="running", health="unhealthy", exit_code=None, reason="running, but the readiness probe failed (cannot run `true` in the container)")
        (hint,) = classify_failure(frozen, logs())
        self.assertEqual((hint.code, hint.owner), ("no_exec", "dockerization"))

    def test_still_starting(self):
        slow = status(state="running", health="starting", exit_code=None, reason="running; the image HEALTHCHECK has not passed yet (still starting after 30 s)")
        (hint,) = classify_failure(slow, logs())
        self.assertEqual((hint.code, hint.owner), ("slow_start", "lifecycle"))
        self.assertIn("ready_timeout", hint.suggestion)

    def test_state_hints_come_before_log_hints(self):
        hints = classify_failure(status(exit_code=137, oom_killed=True), logs(stderr="OSError: [Errno 28] No space left on device"))
        self.assertEqual(codes(hints), ["out_of_memory", "host_disk"])

    def test_no_code_appears_twice(self):
        text = "\n".join(line for line, _, _ in self.LOG_CASES)
        hints = classify_failure(status(exit_code=127, oom_killed=True), logs(stderr=text))
        self.assertEqual(len(codes(hints)), len(set(codes(hints))))
        for hint in hints:
            self.assertIn(hint.owner, OWNERS)

    def test_missing_container(self):
        gone = ContainerStatus(container="gone", state="missing", health="missing")
        self.assertEqual(classify_failure(gone), [])


class TestSummarizeStats(unittest.TestCase):
    SAMPLE = {
        "cpu_stats": {"cpu_usage": {"total_usage": 242_237_000}, "system_cpu_usage": 5_291_420_000_000, "online_cpus": 20},
        "precpu_stats": {"cpu_usage": {"total_usage": 197_403_000}, "system_cpu_usage": 5_271_390_000_000},
        "memory_stats": {"usage": 6_733_824, "limit": 9_400_438_784, "stats": {"inactive_file": 528_384}},
        "pids_stats": {"current": 3, "limit": 8192},
    }

    def test_real_sample(self):
        out = summarize_stats(self.SAMPLE)
        self.assertEqual(out["cpu_percent"], 4.5)                  # 100 % = one full CPU
        self.assertEqual(out["memory_bytes"], 6_733_824 - 528_384)  # droppable page cache does not count
        self.assertEqual(out["memory_limit_bytes"], 9_400_438_784)
        self.assertEqual(out["memory_percent"], 0.1)
        self.assertEqual((out["processes"], out["processes_limit"]), (3, 8192))
        json.dumps(out)

    def test_one_busy_cpu(self):
        stats = {
            "cpu_stats": {"cpu_usage": {"total_usage": 2_000}, "system_cpu_usage": 8_000, "online_cpus": 4},
            "precpu_stats": {"cpu_usage": {"total_usage": 1_000}, "system_cpu_usage": 4_000},
        }
        self.assertEqual(summarize_stats(stats), {"cpu_percent": 100.0})

    def test_first_sample_has_no_cpu_figure(self):
        stats = dict(self.SAMPLE, precpu_stats={"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 5_291_420_000_000})
        self.assertNotIn("cpu_percent", summarize_stats(stats))

    def test_stopped_container_has_no_figures(self):
        self.assertEqual(summarize_stats({"memory_stats": {}, "pids_stats": {}, "cpu_stats": {}, "precpu_stats": {}}), {})
        self.assertEqual(summarize_stats({}), {})

    def test_memory_without_a_limit(self):
        out = summarize_stats({"memory_stats": {"usage": 1000}})
        self.assertEqual(out, {"memory_bytes": 1000})


class TestSummarizeEvents(unittest.TestCase):
    def event(self, action: str, time: int = 1791581150, **attributes) -> dict:
        return {"Type": "container", "Action": action, "time": time, "Actor": {"ID": "d7d9", "Attributes": attributes}}

    def test_keeps_what_explains_a_failure(self):
        events = [
            self.event("create"), self.event("start"), self.event("oom"),
            self.event("kill", signal="15"), self.event("kill", signal="9"), self.event("stop"),
            self.event("die", exitCode="137"), self.event("destroy"),
        ]
        out = summarize_events(events)
        self.assertEqual([e["action"] for e in out], ["create", "start", "oom", "kill", "kill", "stop", "die", "destroy"])
        self.assertEqual([e.get("signal") for e in out if e["action"] == "kill"], [15, 9])
        self.assertEqual(out[6]["exit_code"], 137)
        json.dumps(out)

    def test_probe_noise_is_left_out(self):
        events = [
            self.event("start"),
            self.event("exec_create: true", execID="495c"), self.event("exec_start: true", execID="495c"),
            self.event("exec_die", exitCode="0", execID="495c"), self.event("top"),
            self.event("die", exitCode="3"),
        ]
        self.assertEqual([e["action"] for e in summarize_events(events)], ["start", "die"])

    def test_health_changes_are_kept(self):
        out = summarize_events([self.event("health_status: healthy"), self.event("health_status: unhealthy")])
        self.assertEqual([e["action"] for e in out], ["health_status: healthy", "health_status: unhealthy"])

    def test_time_is_readable(self):
        (item,) = summarize_events([self.event("start", time=1791581150)])
        self.assertEqual(datetime.fromisoformat(item["time"]), datetime.fromtimestamp(1791581150, timezone.utc))

    def test_only_the_newest_events(self):
        events = [self.event("start", time=1791581150 + i) for i in range(MAX_EVENTS + 20)]
        out = summarize_events(events)
        self.assertEqual(len(out), MAX_EVENTS)
        self.assertEqual(out[-1]["time"], datetime.fromtimestamp(1791581150 + MAX_EVENTS + 19, timezone.utc).isoformat())

    def test_no_events(self):
        self.assertEqual(summarize_events([]), [])


class TestDiagnoseLogic(unittest.TestCase):
    """diagnose() against a fake client."""

    def one(self, stderr: str = "", **attrs) -> tuple[FakeContainer, FakeClient]:
        container = FakeContainer(make_attrs(**attrs))
        container.stderr_log = stderr.encode()
        return container, use_fake(self, FakeClient(container, info={"SystemTime": "2026-10-09T10:30:00Z"}))

    def test_failed_container(self):
        container, _ = self.one("Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'\n", status="exited", exit_code=1)
        d = diagnose("tml-demo")
        self.assertIsInstance(d, Diagnosis)
        self.assertEqual((d.status.state, d.status.health, d.status.exit_code), ("exited", "failed", 1))
        self.assertTrue(d.logs.stderr.endswith("No module named 'torch'"))
        self.assertEqual([(h.code, h.owner) for h in d.hints], [("missing_python_module", "dockerization")])
        self.assertEqual(d.notes, [])

    def test_stopped_container_is_not_asked_for_processes_or_resources(self):
        container, _ = self.one(status="exited", exit_code=3)
        d = diagnose("tml-demo")
        self.assertEqual(container.read_calls, [])
        self.assertEqual((d.processes, d.resources), ([], {}))

    def test_running_container_reports_processes_and_resources(self):
        container, _ = self.one(status="running")
        container.top_result = {"Titles": ["UID", "PID", "CMD"], "Processes": [["root", "1", "sleep infinity"], ["root", "7", "python train.py"]]}
        container.stats_result = TestSummarizeStats.SAMPLE
        d = diagnose("tml-demo")
        self.assertEqual(container.read_calls, ["top", "stats"])
        self.assertEqual(d.processes, [{"uid": "root", "pid": "1", "cmd": "sleep infinity"}, {"uid": "root", "pid": "7", "cmd": "python train.py"}])
        self.assertEqual(d.resources["processes"], 3)
        self.assertEqual(d.hints, [])

    def test_process_list_is_capped(self):
        container, _ = self.one(status="running")
        container.top_result = {"Titles": ["PID"], "Processes": [[str(i)] for i in range(MAX_PROCESSES + 50)]}
        self.assertEqual(len(diagnose("tml-demo").processes), MAX_PROCESSES)

    def test_part_that_cannot_be_read_becomes_a_note(self):
        container, _ = self.one(status="running")
        container.top_result = api_error(409, "container 746d is not running")
        container.stats_result = requests.exceptions.ReadTimeout("Read timed out.")
        d = diagnose("tml-demo")
        self.assertEqual((d.processes, d.resources), ([], {}))
        self.assertEqual(len(d.notes), 2)
        self.assertTrue(d.notes[0].startswith("processes not available"))
        self.assertTrue(d.notes[1].startswith("resource use not available"))
        self.assertEqual(d.status.state, "running")                 # the rest of the diagnosis is still there
        self.assertIsNotNone(d.logs)

    def test_events(self):
        container, fake = self.one(status="exited", exit_code=3)
        fake.events_result = [
            {"Action": "start", "time": 1791541201, "Actor": {"Attributes": {}}},
            {"Action": "exec_die", "time": 1791541202, "Actor": {"Attributes": {"exitCode": "0"}}},
            {"Action": "die", "time": 1791541205, "Actor": {"Attributes": {"exitCode": "3"}}},
        ]
        d = diagnose("tml-demo")
        self.assertEqual([(e["action"], e.get("exit_code")) for e in d.events], [("start", None), ("die", 3)])

    def test_events_are_asked_by_the_daemon_clock(self):
        container, fake = self.one(status="exited", exit_code=3, created="2026-10-09T10:00:00.123456789Z")
        diagnose("tml-demo")
        (call,) = fake.event_calls
        self.assertEqual(call["since"], int(datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc).timestamp()))   # created
        self.assertEqual(call["until"], int(datetime(2026, 10, 9, 10, 30, 0, tzinfo=timezone.utc).timestamp()))  # daemon's now
        self.assertEqual(call["filters"], {"container": container.id})
        self.assertIs(call["decode"], True)

    def test_events_that_cannot_be_read_become_a_note(self):
        _, fake = self.one(status="exited", exit_code=3)
        fake.events_result = api_error(500, "events are not available")
        d = diagnose("tml-demo")
        self.assertEqual(d.events, [])
        self.assertTrue(d.notes[0].startswith("events not available"))

    def test_log_tail_is_shorter_than_a_logs_request(self):
        container, _ = self.one("".join(f"line {i}\n" for i in range(500)), status="exited", exit_code=1)
        d = diagnose("tml-demo")
        self.assertEqual(len(d.logs.stderr.splitlines()), diagnostics.DIAGNOSIS_TAIL)
        self.assertEqual(len(diagnose("tml-demo", tail=5).logs.stderr.splitlines()), 5)

    def test_missing_container(self):
        use_fake(self, FakeClient())
        d = diagnose("nope")
        self.assertEqual((d.status.state, d.status.health), ("missing", "missing"))
        self.assertIsNone(d.logs)
        self.assertEqual((d.hints, d.processes, d.resources, d.events, d.notes), ([], [], {}, [], []))

    def test_unmanaged_container_is_not_read(self):
        container = FakeContainer(make_attrs(name="caddy", managed=False))
        use_fake(self, FakeClient(container))
        with self.assertRaises(LifecycleError) as ctx:
            diagnose("caddy")
        self.assertEqual(ctx.exception.code, "not_managed")
        self.assertEqual((container.log_calls, container.read_calls), ([], []))

    def test_logs_that_cannot_be_read_are_an_error(self):
        container, _ = self.one(status="exited", exit_code=1)
        container.stdout_log = requests.exceptions.ConnectionError("Connection refused")
        with self.assertRaises(LifecycleError) as ctx:
            diagnose("tml-demo")
        self.assertEqual(ctx.exception.code, "docker_unavailable")

    def test_result_is_json_serializable(self):
        container, fake = self.one("ModuleNotFoundError: No module named 'torch'\n", status="exited", exit_code=1)
        fake.events_result = [{"Action": "die", "time": 1791541205, "Actor": {"Attributes": {"exitCode": "1"}}}]
        data = json.loads(diagnose("tml-demo").to_json())
        self.assertEqual(set(data), {"status", "logs", "hints", "processes", "resources", "events", "notes"})
        self.assertEqual(data["hints"][0]["owner"], "dockerization")
        self.assertIn("usable", data["status"])


# ---------------------------------------------------------------- live suites (real daemon)

class LiveCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends

    def actions(self, diagnosis: Diagnosis) -> list[str]:
        return [e["action"] for e in diagnosis.events]


@requires_docker
class TestLiveDiagnosis(LiveCase):
    """Failed containers from the sample Dockerfiles."""

    def test_missing_python_module(self):
        name = samples.start("missing-module", "diag-module", mode="native").handle.name
        d = diagnose(name)
        self.assertEqual((d.status.health, d.status.exit_code), ("failed", 1))
        (hint,) = d.hints
        self.assertEqual((hint.code, hint.owner), ("missing_python_module", "dockerization"))
        self.assertEqual(hint.evidence, "ModuleNotFoundError: No module named 'torch'")
        self.assertIn("Traceback", d.logs.stderr)

    def test_out_of_memory(self):
        name = samples.start("oom", "diag-oom", mode="native", memory="64m").handle.name
        d = diagnose(name)
        self.assertTrue(d.status.oom_killed)
        self.assertEqual([(h.code, h.owner) for h in d.hints], [("out_of_memory", "lifecycle")])
        self.assertIn("oom", self.actions(d))

    def test_image_that_cannot_idle(self):
        name = samples.start("empty", "diag-empty").handle.name           # FROM scratch: no `sleep`
        d = diagnose(name)
        (hint,) = d.hints
        self.assertEqual((hint.code, hint.owner), ("missing_executable", "dockerization"))
        self.assertIn("exit code 127", hint.evidence)
        self.assertIn("mode='native'", hint.suggestion)

    def test_crash_without_a_known_pattern(self):
        name = samples.start("crash", "diag-crash", mode="native").handle.name
        d = diagnose(name)
        self.assertEqual((d.status.health, d.status.exit_code), ("failed", 3))
        self.assertEqual(d.logs.stderr, "boom")
        self.assertEqual(d.hints, [])                                      # nothing known: the LLM reads the log
        self.assertEqual((d.processes, d.resources, d.notes), ([], {}, []))
        die = [e for e in d.events if e["action"] == "die"]
        self.assertEqual([e["exit_code"] for e in die], [3])

    def test_failing_healthcheck(self):
        name = samples.start("unhealthy", "diag-unhealthy", mode="native").handle.name
        d = diagnose(name)
        self.assertEqual((d.status.state, d.status.health), ("running", "unhealthy"))
        self.assertEqual([(h.code, h.evidence) for h in d.hints], [("healthcheck_failing", "HEALTHCHECK: db down")])
        self.assertIn("health_status: unhealthy", self.actions(d))
        self.assertGreaterEqual(len(d.processes), 1)                       # it runs, so its processes are listed

    def test_stop_that_needed_sigkill(self):
        name = samples.start("stubborn", "diag-stubborn", mode="native").handle.name
        stop(name, timeout=1)
        d = diagnose(name)
        self.assertEqual(d.status.exit_code, 137)
        self.assertEqual([h.code for h in d.hints], ["sigkill"])
        self.assertEqual([e.get("signal") for e in d.events if e["action"] == "kill"], [15, 9])   # asked, then forced
        self.assertEqual(d.logs.stdout, "ignoring SIGTERM")

    def test_graceful_stop_is_not_a_failure(self):
        name = samples.start("idle", "diag-stopped").handle.name
        stop(name)
        d = diagnose(name)
        self.assertEqual((d.status.health, d.status.exit_code), ("stopped", 143))
        self.assertEqual(d.hints, [])

    def test_missing_file(self):
        name = samples.start("idle", "diag-file", mode="native", command="python /workspace/train.py").handle.name
        (hint,) = diagnose(name).hints
        self.assertEqual((hint.code, hint.owner), ("missing_file", "dockerization"))
        self.assertIn("/workspace/train.py", hint.evidence)

    def test_no_network(self):
        fetch = "python -c \"import urllib.request; urllib.request.urlopen('http://example.com')\""
        name = samples.start("idle", "diag-network", mode="native", network="none", command=fetch).handle.name
        hints = diagnose(name).hints
        self.assertEqual([(h.code, h.owner) for h in hints], [("network", "host")])


@requires_docker
class TestRunningDiagnosis(LiveCase):
    @classmethod
    def setUpClass(cls):
        cls.name = samples.start("service", "diag-service", mode="native").handle.name
        cls.diagnosis = diagnose(cls.name)

    def test_healthy_container_has_no_hints(self):
        d = self.diagnosis
        self.assertEqual((d.status.state, d.status.health), ("running", "healthy"))
        self.assertEqual((d.hints, d.notes), ([], []))

    def test_processes(self):
        processes = self.diagnosis.processes
        self.assertGreaterEqual(len(processes), 2)                         # init and the service
        self.assertTrue(all("pid" in p and "cmd" in p for p in processes))
        self.assertTrue(any("tick" in p["cmd"] for p in processes))

    def test_resources(self):
        resources = self.diagnosis.resources
        self.assertGreater(resources["memory_bytes"], 0)
        self.assertGreater(resources["memory_limit_bytes"], resources["memory_bytes"])
        self.assertGreaterEqual(resources["processes"], 2)
        self.assertEqual(resources["processes_limit"], 8192)
        self.assertGreaterEqual(resources["cpu_percent"], 0)

    def test_events(self):
        actions = self.actions(self.diagnosis)
        self.assertEqual(actions[:2], ["create", "start"])
        self.assertIn("health_status: healthy", actions)
        self.assertFalse([a for a in actions if a.startswith("exec") or a == "top"])

    def test_logs(self):
        self.assertIn("tick", self.diagnosis.logs.stdout)

    def test_idle_container(self):
        d = diagnose(samples.start("idle", "diag-idle").handle.name)
        self.assertEqual((d.status.health, d.hints, d.notes), ("healthy", [], []))
        self.assertEqual(d.logs.lines, 0)
        self.assertTrue(any("sleep" in p["cmd"] for p in d.processes))

    def test_result_is_json_serializable(self):
        data = json.loads(self.diagnosis.to_json())
        self.assertEqual(data["status"]["handle"]["name"], self.name)
        self.assertGreater(data["resources"]["memory_bytes"], 0)


@requires_docker
class TestDiagnoseRefused(LiveCase):
    def test_missing_container(self):
        d = diagnose(samples.container_name("diag-never-made"))
        self.assertEqual(d.status.health, "missing")
        self.assertIsNone(d.logs)

    def test_unmanaged_container(self):
        other = samples.start_unmanaged("diag-unmanaged")
        with self.assertRaises(LifecycleError) as ctx:
            diagnose(other.name)
        self.assertEqual(ctx.exception.code, "not_managed")


if __name__ == "__main__":
    unittest.main()
