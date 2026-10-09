import time
import unittest

import requests

from src.lifecycle.models import LifecycleError, TeardownResult
from src.lifecycle.status import get_status
from src.lifecycle.teardown import MAX_STOP_TIMEOUT, STOP_TIMEOUT, stop
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker

# A process that ends itself cleanly when it is asked to stop
EXITS_ON_SIGTERM = (
    "python -c \"import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(600)\""
)


def tearDownModule():
    samples.cleanup()


# ---------------------------------------------------------------- pure suites (no daemon)

class TestStopLogic(unittest.TestCase):
    """stop() against a fake client."""

    def container(self, **kwargs) -> FakeContainer:
        fake_kwargs = {k: kwargs.pop(k) for k in ("stop_exit", "stop_oom") if k in kwargs}
        container = FakeContainer(make_attrs(**kwargs), **fake_kwargs)
        use_fake(self, FakeClient(container))
        return container

    def test_running_container_is_stopped(self):
        container = self.container(status="running")
        r = stop("tml-demo")
        self.assertIsInstance(r, TeardownResult)
        self.assertEqual(container.calls, [("stop", STOP_TIMEOUT)])
        self.assertEqual((r.container, r.stopped, r.killed, r.removed, r.exit_code), ("tml-demo", True, False, False, 143))
        self.assertEqual((r.status.state, r.status.health), ("exited", "stopped"))

    def test_timeout_is_passed_on(self):
        container = self.container(status="running")
        stop("tml-demo", timeout=45)
        self.assertEqual(container.calls, [("stop", 45)])

    def test_fractional_timeout_becomes_whole_seconds(self):
        container = self.container(status="running")
        stop("tml-demo", timeout=2.9)
        self.assertEqual(container.calls, [("stop", 2)])

    def test_exit_137_means_the_stop_needed_sigkill(self):
        self.container(status="running", stop_exit=137)
        r = stop("tml-demo", timeout=5)
        self.assertEqual((r.stopped, r.killed, r.exit_code), (True, True, 137))
        self.assertIn("did not end within 5 s", r.status.reason)

    def test_out_of_memory_kill_is_not_reported_as_a_stop_timeout(self):
        self.container(status="running", stop_exit=137, stop_oom=True)
        r = stop("tml-demo")
        self.assertFalse(r.killed)
        self.assertEqual(r.status.health, "failed")

    def test_process_that_ends_itself_cleanly(self):
        self.container(status="running", stop_exit=0)
        r = stop("tml-demo")
        self.assertEqual((r.stopped, r.killed, r.exit_code), (True, False, 0))
        self.assertEqual(r.status.health, "completed")

    def test_paused_and_restarting_containers_are_stopped_too(self):
        for state in ("paused", "restarting"):
            container = self.container(status=state)
            self.assertTrue(stop("tml-demo").stopped, state)
            self.assertEqual(container.calls, [("stop", STOP_TIMEOUT)], state)

    def test_container_that_is_not_running_is_left_alone(self):
        for state, code, health in (("exited", 3, "failed"), ("exited", 0, "completed"), ("created", 0, "stopped"), ("dead", 1, "failed")):
            container = self.container(status=state, exit_code=code)
            r = stop("tml-demo")
            self.assertEqual(container.calls, [], state)           # no stop request at all
            self.assertEqual((r.stopped, r.killed, r.removed), (False, False, False), state)
            self.assertEqual((r.status.state, r.status.health), (state, health))

    def test_exit_code_of_an_already_stopped_container_is_kept(self):
        self.container(status="exited", exit_code=3)
        self.assertEqual(stop("tml-demo").exit_code, 3)
        self.container(status="created")
        self.assertIsNone(stop("tml-demo").exit_code)              # never ran

    def test_missing_container(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            stop("nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_unmanaged_container_is_not_stopped(self):
        container = FakeContainer(make_attrs(name="caddy", status="running", managed=False))
        use_fake(self, FakeClient(container))
        with self.assertRaises(LifecycleError) as ctx:
            stop("caddy")
        self.assertEqual(ctx.exception.code, "not_managed")
        self.assertEqual(container.calls, [])

    def test_empty_name(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            stop("")
        self.assertEqual(ctx.exception.code, "invalid_argument")

    def test_invalid_timeouts_are_refused_before_anything_happens(self):
        container = self.container(status="running")
        for bad in (-1, MAX_STOP_TIMEOUT + 1, True, "10", None, float("nan")):
            with self.assertRaises(LifecycleError) as ctx:
                stop("tml-demo", timeout=bad)
            self.assertEqual(ctx.exception.code, "invalid_argument", repr(bad))
        self.assertEqual(container.calls, [])

    def test_timeout_limits(self):
        container = self.container(status="running")
        stop("tml-demo", timeout=0)                                # kill at once
        self.assertEqual(container.calls, [("stop", 0)])
        container = self.container(status="running")
        stop("tml-demo", timeout=MAX_STOP_TIMEOUT)
        self.assertEqual(container.calls, [("stop", MAX_STOP_TIMEOUT)])

    def test_daemon_errors_are_translated(self):
        for exc, code in (
            (requests.exceptions.ConnectionError("Connection refused"), "docker_unavailable"),
            (requests.exceptions.ReadTimeout("Read timed out."), "timeout"),
        ):
            self.container(status="running", stop_exit=exc)
            with self.assertRaises(LifecycleError) as ctx:
                stop("tml-demo")
            self.assertEqual(ctx.exception.code, code)
            self.assertIn("stop container 'tml-demo'", ctx.exception.message)

    def test_result_is_json_serializable(self):
        self.container(status="running")
        data = stop("tml-demo").to_dict()
        self.assertEqual(data["stopped"], True)
        self.assertEqual(data["status"]["health"], "stopped")
        self.assertIn("usable", data["status"])                    # the nested status keeps its derived fields


# ---------------------------------------------------------------- live suites (real daemon)

class LiveCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends


@requires_docker
class TestGraceful(LiveCase):
    def test_idle_container_stops_quickly(self):
        name = samples.start("idle", "stop-idle").handle.name
        started = time.monotonic()
        r = stop(name)
        self.assertLess(time.monotonic() - started, 5)             # init forwards SIGTERM; no wait for the timeout
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (True, False, False, 143))
        self.assertEqual((r.status.state, r.status.health), ("exited", "stopped"))

    def test_container_is_kept(self):
        name = samples.start("service", "stop-kept", mode="native").handle.name
        handle = stop(name).status.handle
        self.assertEqual(handle.name, name)
        obj = samples.sdk_container(name)                          # still there, with its logs
        self.assertEqual(obj.status, "exited")
        self.assertIn(b"tick", obj.logs())
        self.assertEqual(get_status(name).health, "stopped")

    def test_process_that_handles_sigterm_exits_with_its_own_code(self):
        name = samples.start("idle", "stop-clean", mode="native", command=EXITS_ON_SIGTERM).handle.name
        r = stop(name)
        self.assertEqual((r.stopped, r.killed, r.exit_code), (True, False, 0))
        self.assertEqual(r.status.health, "completed")

    def test_stop_by_id(self):
        handle = samples.start("idle", "stop-by-id").handle
        r = stop(handle.id)
        self.assertTrue(r.stopped)
        self.assertEqual((r.container, r.status.handle.name), (handle.id, handle.name))

    def test_provision_starts_a_stopped_container_again(self):
        first = samples.start("idle", "stop-restart")
        stop(first.handle.name)
        again = samples.start("idle", "stop-restart")
        self.assertEqual(again.action, "restarted")
        self.assertEqual(again.handle.id, first.handle.id)
        self.assertEqual(again.status.health, "healthy")


@requires_docker
class TestEscalation(LiveCase):
    def test_stubborn_process_is_killed_after_the_timeout(self):
        name = samples.start("stubborn", "stop-stubborn", mode="native").handle.name
        started = time.monotonic()
        r = stop(name, timeout=2)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 2)                        # it waited for the whole timeout first
        self.assertLess(elapsed, 10)
        self.assertEqual((r.stopped, r.killed, r.exit_code), (True, True, 137))
        self.assertEqual(r.status.state, "exited")
        self.assertIn("did not end within 2 s", r.status.reason)

    def test_timeout_zero_kills_at_once(self):
        name = samples.start("stubborn", "stop-stubborn-now", mode="native").handle.name
        started = time.monotonic()
        r = stop(name, timeout=0)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual((r.stopped, r.killed, r.exit_code), (True, True, 137))

    def test_long_timeout_is_not_used_up_by_a_cooperative_process(self):
        name = samples.start("idle", "stop-long-timeout").handle.name
        started = time.monotonic()
        r = stop(name, timeout=60)
        self.assertLess(time.monotonic() - started, 5)
        self.assertFalse(r.killed)


@requires_docker
class TestIdempotent(LiveCase):
    def test_second_stop_changes_nothing(self):
        name = samples.start("idle", "stop-twice").handle.name
        first = stop(name)
        finished_at = first.status.finished_at
        second = stop(name)
        self.assertEqual((first.stopped, second.stopped), (True, False))
        self.assertEqual((second.status.state, second.exit_code), ("exited", 143))
        self.assertEqual(second.status.finished_at, finished_at)   # really untouched

    def test_stop_of_a_finished_container(self):
        name = samples.start("idle", "stop-finished", mode="native").handle.name      # CMD exits with 0
        r = stop(name)
        self.assertEqual((r.stopped, r.killed, r.exit_code), (False, False, 0))
        self.assertEqual(r.status.health, "completed")

    def test_stop_of_a_crashed_container_keeps_its_exit_code(self):
        name = samples.start("crash", "stop-crashed", mode="native").handle.name
        r = stop(name)
        self.assertEqual((r.stopped, r.exit_code, r.status.health), (False, 3, "failed"))


@requires_docker
class TestNotManaged(LiveCase):
    def test_unmanaged_container_keeps_running(self):
        other = samples.start_unmanaged("stop-unmanaged")
        with self.assertRaises(LifecycleError) as ctx:
            stop(other.name)
        self.assertEqual(ctx.exception.code, "not_managed")
        other.reload()
        self.assertEqual(other.status, "running")

    def test_missing_container(self):
        with self.assertRaises(LifecycleError) as ctx:
            stop(samples.container_name("stop-never-made"))
        self.assertEqual(ctx.exception.code, "not_found")

    def test_invalid_timeout_does_not_stop_the_container(self):
        name = samples.start("idle", "stop-bad-timeout").handle.name
        with self.assertRaises(LifecycleError) as ctx:
            stop(name, timeout=-5)
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertEqual(samples.sdk_container(name).status, "running")


if __name__ == "__main__":
    unittest.main()
