import re
import time
import unittest
from datetime import datetime, timedelta, timezone

import requests

from src.lifecycle import diagnostics
from src.lifecycle.diagnostics import (
    DEFAULT_TAIL, MAX_LOG_CHARS, MAX_TAIL, SCAN_LINES,
    clean_log, filter_lines, get_logs, parse_since, parse_time, split_budget, truncate_tail,
)
from src.lifecycle.models import ContainerLogs, LifecycleError
from src.lifecycle.teardown import stop
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker

NOW = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)


def tearDownModule():
    samples.cleanup()


# ---------------------------------------------------------------- pure suites (no daemon)

class TestCleanLog(unittest.TestCase):
    def test_plain_text_is_unchanged(self):
        text = "epoch 1\nloss 0.42\n\ndone"
        self.assertEqual(clean_log(text), text)

    def test_colour_codes_are_removed(self):
        self.assertEqual(clean_log("\x1b[31mERROR\x1b[0m: bad \x1b[1;32mvalue\x1b[0m"), "ERROR: bad value")

    def test_cursor_codes_are_removed(self):
        self.assertEqual(clean_log("\x1b[2K\x1b[1Atraining"), "training")

    def test_redrawn_progress_bar_keeps_its_last_state(self):
        bar = "  0%|          | 0/100\r 50%|#####     | 50/100\r100%|##########| 100/100"
        self.assertEqual(clean_log(f"start\n{bar}\nend"), "start\n100%|##########| 100/100\nend")

    def test_windows_line_ends(self):
        self.assertEqual(clean_log("one\r\ntwo\r\n"), "one\ntwo\n")

    def test_empty(self):
        self.assertEqual(clean_log(""), "")


class TestFilterLines(unittest.TestCase):
    TEXT = "\n".join(f"line {i}" for i in range(10))

    def test_tail(self):
        self.assertEqual(filter_lines(self.TEXT, None, 3), "line 7\nline 8\nline 9")

    def test_tail_longer_than_the_text(self):
        self.assertEqual(filter_lines(self.TEXT, None, 500), self.TEXT)

    def test_pattern(self):
        self.assertEqual(filter_lines("ok\nERROR a\nok\nERROR b", re.compile("ERROR"), 10), "ERROR a\nERROR b")

    def test_tail_counts_matching_lines_only(self):
        text = "ERROR 1\n" + "ok\n" * 50 + "ERROR 2\nok\nERROR 3\nok"
        self.assertEqual(filter_lines(text, re.compile("^ERROR"), 2), "ERROR 2\nERROR 3")

    def test_no_match(self):
        self.assertEqual(filter_lines(self.TEXT, re.compile("nothing"), 10), "")

    def test_empty(self):
        self.assertEqual(filter_lines("", None, 10), "")


class TestTruncation(unittest.TestCase):
    def test_short_text_is_not_cut(self):
        self.assertEqual(truncate_tail("a\nb", 100), ("a\nb", False))

    def test_text_that_just_fits(self):
        self.assertEqual(truncate_tail("x" * 50, 50), ("x" * 50, False))

    def test_keeps_the_end(self):
        text = "\n".join(f"line {i:03}" for i in range(100))
        cut, was_cut = truncate_tail(text, 40)
        self.assertTrue(was_cut)
        self.assertTrue(cut.endswith("line 099"))
        self.assertLessEqual(len(cut), 40)

    def test_cuts_at_a_line_start(self):
        text = "\n".join(f"line {i:03}" for i in range(100))
        cut, _ = truncate_tail(text, 40)
        for line in cut.splitlines():
            self.assertRegex(line, r"^line \d{3}$")       # no half line at the top

    def test_one_very_long_line_keeps_its_end(self):
        cut, was_cut = truncate_tail("a" * 500 + "END", 10)
        self.assertEqual((cut, was_cut), ("aaaaaaaEND", True))

    def test_no_budget(self):
        self.assertEqual(truncate_tail("text", 0), ("", True))
        self.assertEqual(truncate_tail("", 0), ("", False))

    def test_budget_when_everything_fits(self):
        self.assertEqual(split_budget("a" * 10, "b" * 20, 100), (10, 20))

    def test_short_stderr_leaves_its_rest_to_stdout(self):
        self.assertEqual(split_budget("a" * 1000, "b" * 10, 100), (90, 10))

    def test_short_stdout_leaves_its_rest_to_stderr(self):
        self.assertEqual(split_budget("a" * 10, "b" * 1000, 100), (10, 90))

    def test_two_long_streams_share_the_budget(self):
        self.assertEqual(split_budget("a" * 1000, "b" * 1000, 100), (50, 50))
        self.assertEqual(sum(split_budget("a" * 1000, "b" * 1000, 101)), 101)


class TestParseSince(unittest.TestCase):
    def test_empty_means_no_limit(self):
        self.assertIsNone(parse_since("", NOW))
        self.assertIsNone(parse_since("   ", NOW))
        self.assertIsNone(parse_since(None, NOW))

    def test_durations_count_back_from_now(self):
        for text, delta in (("30s", timedelta(seconds=30)), ("10m", timedelta(minutes=10)), ("2h", timedelta(hours=2)), ("1d", timedelta(days=1))):
            self.assertEqual(parse_since(text, NOW), NOW - delta, text)

    def test_duration_with_spaces(self):
        self.assertEqual(parse_since(" 10 m ", NOW), NOW - timedelta(minutes=10))

    def test_iso_timestamp(self):
        self.assertEqual(parse_since("2026-10-09T13:30:00Z", NOW), datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc))

    def test_iso_timestamp_with_offset(self):
        self.assertEqual(parse_since("2026-10-09T09:30:00-04:00", NOW), datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc))

    def test_timestamp_without_zone_is_utc(self):
        self.assertEqual(parse_since("2026-10-09T13:30:00", NOW).tzinfo, timezone.utc)

    def test_docker_timestamp_with_nanoseconds(self):
        parsed = parse_time("2026-10-09T14:50:03.046614918Z")
        self.assertEqual(parsed, datetime(2026, 10, 9, 14, 50, 3, 46614, tzinfo=timezone.utc))

    def test_invalid_values(self):
        for bad in ("yesterday", "10", "10x", "-5m", "5 minutes", "2026-13-45"):
            with self.assertRaises(LifecycleError) as ctx:
                parse_since(bad, NOW)
            self.assertEqual(ctx.exception.code, "invalid_argument", bad)
            self.assertIn("10m", ctx.exception.hint)


class TestGetLogsLogic(unittest.TestCase):
    """get_logs() against a fake client."""

    def container(self, stdout: str = "", stderr: str = "", **attrs) -> FakeContainer:
        container = FakeContainer(make_attrs(**attrs))
        container.stdout_log, container.stderr_log = stdout.encode(), stderr.encode()
        self.fake = use_fake(self, FakeClient(container))
        return container

    def assertInvalid(self, container: FakeContainer, **kwargs):
        with self.assertRaises(LifecycleError) as ctx:
            get_logs("tml-demo", **kwargs)
        self.assertEqual(ctx.exception.code, "invalid_argument", kwargs)
        self.assertEqual(container.log_calls, [], kwargs)      # refused before any Docker call

    def test_streams_are_kept_apart(self):
        self.container(stdout="epoch 1\nepoch 2\n", stderr="Traceback\nValueError: bad\n")
        logs = get_logs("tml-demo")
        self.assertIsInstance(logs, ContainerLogs)
        self.assertEqual((logs.stdout, logs.stderr), ("epoch 1\nepoch 2", "Traceback\nValueError: bad"))
        self.assertEqual((logs.container, logs.lines, logs.truncated, logs.since), ("tml-demo", 4, False, ""))

    def test_one_request_for_each_stream(self):
        container = self.container(stdout="a\n", stderr="b\n")
        get_logs("tml-demo")
        self.assertEqual(
            [(c["stdout"], c["stderr"]) for c in container.log_calls],
            [(True, False), (False, True)],
        )

    def test_tail_is_applied_to_each_stream_here_not_by_docker(self):
        out = "".join(f"out {i}\n" for i in range(50))
        err = "".join(f"err {i}\n" for i in range(50))
        container = self.container(stdout=out, stderr=err)
        logs = get_logs("tml-demo", tail=3)
        self.assertEqual(logs.stdout, "out 47\nout 48\nout 49")
        self.assertEqual(logs.stderr, "err 47\nerr 48\nerr 49")
        self.assertEqual(logs.lines, 6)
        # Docker counts its own tail over both streams together, so it is asked for many entries
        self.assertEqual({c["tail"] for c in container.log_calls}, {SCAN_LINES})

    def test_default_tail(self):
        self.container(stdout="".join(f"out {i}\n" for i in range(DEFAULT_TAIL + 100)))
        self.assertEqual(len(get_logs("tml-demo").stdout.splitlines()), DEFAULT_TAIL)

    def test_grep(self):
        self.container(stdout="epoch 1 loss 0.9\nsaving\nepoch 2 loss 0.5\n", stderr="warning: slow\nError: bad value\n")
        logs = get_logs("tml-demo", grep=r"loss \d|^Error")
        self.assertEqual((logs.stdout, logs.stderr), ("epoch 1 loss 0.9\nepoch 2 loss 0.5", "Error: bad value"))
        self.assertEqual(logs.lines, 3)

    def test_grep_is_case_sensitive_like_a_regular_expression(self):
        self.container(stderr="Error one\nerror two\n")
        self.assertEqual(get_logs("tml-demo", grep="Error").stderr, "Error one")
        self.assertEqual(get_logs("tml-demo", grep="(?i)error").stderr, "Error one\nerror two")

    def test_grep_with_tail_returns_the_last_matches(self):
        self.container(stdout="".join(f"step {i}\nnoise\n" for i in range(100)))
        self.assertEqual(get_logs("tml-demo", grep="^step", tail=2).stdout, "step 98\nstep 99")

    def test_grep_without_a_match(self):
        self.container(stdout="a\nb\n")
        logs = get_logs("tml-demo", grep="nothing")
        self.assertEqual((logs.stdout, logs.stderr, logs.lines), ("", "", 0))

    def test_long_output_is_cut_to_the_budget(self):
        self.container(stdout="".join(f"line {i:05} {'x' * 60}\n" for i in range(2000)), stderr="err\n")
        logs = get_logs("tml-demo", tail=2000)
        self.assertTrue(logs.truncated)
        self.assertLessEqual(len(logs.stdout) + len(logs.stderr), MAX_LOG_CHARS)
        self.assertTrue(logs.stdout.endswith(f"line 01999 {'x' * 60}"))     # the newest output is kept
        self.assertEqual(logs.stderr, "err")                                 # a short stream is never cut
        self.assertEqual(logs.lines, len(logs.stdout.splitlines()) + 1)

    def test_budget_can_be_set(self):
        self.container(stdout="".join(f"line {i}\n" for i in range(100)))
        logs = get_logs("tml-demo", max_chars=30)
        self.assertTrue(logs.truncated)
        self.assertLessEqual(len(logs.stdout), 30)
        self.assertTrue(logs.stdout.endswith("line 99"))

    def test_output_is_cleaned(self):
        self.container(stdout="\x1b[32mok\x1b[0m\n 10%\r 60%\r100%\n")
        self.assertEqual(get_logs("tml-demo").stdout, "ok\n100%")

    def test_bytes_that_are_not_text(self):
        container = self.container()
        container.stdout_log = b"good\n\xff\xfe broken\n"
        logs = get_logs("tml-demo")
        self.assertEqual(logs.stdout.splitlines()[0], "good")
        self.assertIn("broken", logs.stdout)

    def test_no_output(self):
        self.container()
        logs = get_logs("tml-demo")
        self.assertEqual((logs.stdout, logs.stderr, logs.lines, logs.truncated), ("", "", 0, False))

    def test_stopped_container_can_still_be_read(self):
        self.container(stderr="boom\n", status="exited", exit_code=3)
        self.assertEqual(get_logs("tml-demo").stderr, "boom")

    def test_without_since_docker_gets_no_time_limit(self):
        container = self.container(stdout="a\n")
        get_logs("tml-demo")
        for call in container.log_calls:
            self.assertNotIn("since", call)

    def test_duration_counts_back_from_the_daemon_clock(self):
        # the daemon is 5 hours ahead of this machine: the duration must follow the daemon
        container = FakeContainer(make_attrs())
        use_fake(self, FakeClient(container, info={"SystemTime": "2031-01-01T00:10:00.123456789Z"}))
        logs = get_logs("tml-demo", since="10m")
        expected = int(datetime(2031, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        self.assertEqual({c["since"] for c in container.log_calls}, {expected})
        self.assertEqual(logs.since, "10m")

    def test_duration_falls_back_to_this_machine_clock(self):
        container = self.container()                                         # this fake daemon reports no SystemTime
        before = time.time()
        get_logs("tml-demo", since="1h")
        (since,) = {c["since"] for c in container.log_calls}
        self.assertAlmostEqual(since, before - 3600, delta=5)

    def test_timestamp_is_passed_as_seconds(self):
        container = self.container()
        get_logs("tml-demo", since="2026-10-09T13:30:00Z")
        expected = int(datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc).timestamp())
        self.assertEqual({c["since"] for c in container.log_calls}, {expected})

    def test_invalid_tail(self):
        container = self.container()
        for bad in (0, -1, MAX_TAIL + 1, 2.5, "10", True, None):
            self.assertInvalid(container, tail=bad)

    def test_tail_limits(self):
        self.container(stdout="a\nb\n")
        self.assertEqual(get_logs("tml-demo", tail=1).stdout, "b")
        self.assertEqual(get_logs("tml-demo", tail=MAX_TAIL).stdout, "a\nb")

    def test_invalid_grep(self):
        container = self.container()
        self.assertInvalid(container, grep="(unclosed")
        self.assertInvalid(container, grep="x" * (diagnostics.MAX_PATTERN_CHARS + 1))

    def test_invalid_since(self):
        self.assertInvalid(self.container(), since="yesterday")

    def test_missing_container(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            get_logs("nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_unmanaged_container_is_not_read(self):
        container = FakeContainer(make_attrs(name="caddy", managed=False))
        container.stdout_log = b"private access log\n"
        use_fake(self, FakeClient(container))
        with self.assertRaises(LifecycleError) as ctx:
            get_logs("caddy")
        self.assertEqual(ctx.exception.code, "not_managed")
        self.assertEqual(container.log_calls, [])

    def test_daemon_error_is_translated(self):
        container = self.container()
        container.stderr_log = requests.exceptions.ConnectionError("Connection refused")
        with self.assertRaises(LifecycleError) as ctx:
            get_logs("tml-demo")
        self.assertEqual(ctx.exception.code, "docker_unavailable")
        self.assertIn("read the logs", ctx.exception.message)


# ---------------------------------------------------------------- live suites (real daemon)

# 3000 long stdout lines, then 50 short stderr lines. Python buffers stdout when it is not a
# terminal, so Docker receives the stderr lines FIRST and the stdout lines when the process ends.
NOISY = (
    "python -c \"import sys\n"
    "for i in range(3000): print('out line', i, 'x' * 40)\n"
    "for i in range(50): print('ERR line', i, file=sys.stderr)\""
)

DECORATED = "python -c \"print('\\x1b[31mred\\x1b[0m text'); print(' 10%\\r 60%\\r100%')\""

BOTH_STREAMS = "python -c \"import sys; print('to stdout'); print('to stderr', file=sys.stderr)\""


class LiveCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends


@requires_docker
class TestStreams(LiveCase):
    def test_stderr_of_a_crash(self):
        name = samples.start("crash", "logs-crash", mode="native").handle.name
        logs = get_logs(name)
        self.assertEqual((logs.stdout, logs.stderr, logs.lines), ("", "boom", 1))

    def test_stdout(self):
        name = samples.start("idle", "logs-stdout", mode="native").handle.name      # CMD prints "done"
        logs = get_logs(name)
        self.assertEqual((logs.stdout, logs.stderr), ("done", ""))

    def test_both_streams(self):
        name = samples.start("idle", "logs-both", mode="native", command=BOTH_STREAMS).handle.name
        logs = get_logs(name)
        self.assertEqual((logs.stdout, logs.stderr, logs.lines), ("to stdout", "to stderr", 2))

    def test_traceback(self):
        name = samples.start("missing-module", "logs-traceback", mode="native").handle.name
        logs = get_logs(name)
        self.assertTrue(logs.stderr.startswith("Traceback (most recent call last):"))
        self.assertTrue(logs.stderr.endswith("ModuleNotFoundError: No module named 'torch'"))

    def test_idle_container_has_no_output(self):
        name = samples.start("idle", "logs-idle").handle.name
        logs = get_logs(name)
        self.assertEqual((logs.stdout, logs.stderr, logs.lines), ("", "", 0))

    def test_by_id(self):
        handle = samples.start("crash", "logs-by-id", mode="native").handle
        self.assertEqual(get_logs(handle.id).stderr, "boom")

    def test_colour_codes_and_progress_bars_are_cleaned(self):
        name = samples.start("idle", "logs-decorated", mode="native", command=DECORATED).handle.name
        self.assertEqual(get_logs(name).stdout, "red text\n100%")


@requires_docker
class TestFilters(LiveCase):
    @classmethod
    def setUpClass(cls):
        cls.service = samples.start("service", "logs-service", mode="native").handle.name   # prints "tick" every second
        cls.noisy = samples.start("idle", "logs-noisy", mode="native", command=NOISY).handle.name
        time.sleep(3)

    def test_tail(self):
        self.assertEqual(get_logs(self.service, tail=2).stdout, "tick\ntick")
        self.assertGreater(len(get_logs(self.service).stdout.splitlines()), 2)

    def test_since_a_duration(self):
        recent = len(get_logs(self.service, since="2s").stdout.splitlines())
        everything = len(get_logs(self.service, since="1h").stdout.splitlines())
        self.assertGreaterEqual(recent, 1)
        self.assertLess(recent, everything)

    def test_since_a_time_in_the_future(self):
        self.assertEqual(get_logs(self.service, since="2100-01-01T00:00:00Z").lines, 0)

    def test_grep(self):
        logs = get_logs(self.noisy, grep=r"out line 2999 |ERR line 4[89]$")
        self.assertEqual(logs.stdout, f"out line 2999 {'x' * 40}")
        self.assertEqual(logs.stderr, "ERR line 48\nERR line 49")

    def test_grep_with_tail(self):
        self.assertEqual(get_logs(self.noisy, grep="^ERR", tail=1).stderr, "ERR line 49")

    def test_tail_holds_for_each_stream(self):
        # the stderr lines are older than the 3000 stdout lines in Docker's log; they must still be there
        logs = get_logs(self.noisy, tail=5)
        self.assertEqual(logs.stderr.splitlines(), [f"ERR line {i}" for i in range(45, 50)])
        self.assertEqual(len(logs.stdout.splitlines()), 5)
        self.assertTrue(logs.stdout.endswith(f"out line 2999 {'x' * 40}"))
        self.assertEqual(logs.lines, 10)

    def test_long_output_is_cut_to_the_budget(self):
        logs = get_logs(self.noisy, tail=2000)
        self.assertTrue(logs.truncated)
        self.assertLessEqual(len(logs.stdout) + len(logs.stderr), MAX_LOG_CHARS)
        self.assertEqual(len(logs.stderr.splitlines()), 50)                  # the short stream is whole
        self.assertTrue(logs.stdout.endswith(f"out line 2999 {'x' * 40}"))   # the newest stdout is kept
        self.assertLess(len(logs.stdout.splitlines()), 2000)

    def test_short_request_is_not_truncated(self):
        self.assertFalse(get_logs(self.noisy, tail=10).truncated)


@requires_docker
class TestStoppedAndRefused(LiveCase):
    def test_logs_survive_a_stop(self):
        name = samples.start("service", "logs-stopped", mode="native").handle.name
        time.sleep(1.5)
        stop(name)
        self.assertIn("tick", get_logs(name).stdout)

    def test_missing_container(self):
        with self.assertRaises(LifecycleError) as ctx:
            get_logs(samples.container_name("logs-never-made"))
        self.assertEqual(ctx.exception.code, "not_found")

    def test_unmanaged_container(self):
        other = samples.start_unmanaged("logs-unmanaged")
        with self.assertRaises(LifecycleError) as ctx:
            get_logs(other.name)
        self.assertEqual(ctx.exception.code, "not_managed")

    def test_invalid_arguments(self):
        name = samples.start("idle", "logs-invalid").handle.name
        for kwargs in ({"tail": 0}, {"grep": "(unclosed"}, {"since": "yesterday"}):
            with self.assertRaises(LifecycleError) as ctx:
                get_logs(name, **kwargs)
            self.assertEqual(ctx.exception.code, "invalid_argument", kwargs)


if __name__ == "__main__":
    unittest.main()
