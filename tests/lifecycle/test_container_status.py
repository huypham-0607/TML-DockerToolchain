import unittest
from unittest import mock

import requests

from src.lifecycle import status
from src.lifecycle.models import LABEL_TEST, ContainerStatus, LifecycleError
from src.lifecycle.status import (
    derive_health, get_status, handle_from_attrs, list_managed,
    require_managed, require_running, status_from_attrs, wait_ready,
)
from tests.lifecycle import live
from tests.lifecycle.fakes import FakeClient, FakeContainer, api_error, make_attrs, use_fake
from tests.lifecycle.live import requires_docker


def tearDownModule():
    live.remove_test_containers()


# ---------------------------------------------------------------- pure suites (no daemon)

class TestDeriveHealth(unittest.TestCase):
    def verdict(self, probe_ok=None, **attrs) -> str:
        health, reason = derive_health(make_attrs(**attrs), probe_ok)
        self.assertTrue(reason, "every verdict needs a reason")
        return health

    def test_running_and_probe_passes(self):
        self.assertEqual(self.verdict(True, status="running"), "healthy")

    def test_running_and_probe_fails(self):
        health, reason = derive_health(make_attrs(status="running"), probe_ok=False)
        self.assertEqual(health, "unhealthy")
        self.assertIn("readiness probe failed", reason)

    def test_running_and_not_probed(self):
        health, reason = derive_health(make_attrs(status="running"))
        self.assertEqual(health, "healthy")
        self.assertIn("not probed", reason)

    def test_healthcheck_starting(self):
        self.assertEqual(self.verdict(True, status="running", health="starting"), "starting")

    def test_healthcheck_healthy(self):
        self.assertEqual(self.verdict(status="running", health="healthy"), "healthy")

    def test_healthcheck_unhealthy_reports_its_output(self):
        health, reason = derive_health(make_attrs(status="running", health="unhealthy", health_output="db down\n"))
        self.assertEqual(health, "unhealthy")
        self.assertIn("db down", reason)

    def test_healthcheck_wins_over_the_probe(self):
        self.assertEqual(self.verdict(True, status="running", health="unhealthy"), "unhealthy")
        self.assertEqual(self.verdict(False, status="running", health="healthy"), "healthy")

    def test_restarting_is_a_crash_loop(self):
        health, reason = derive_health(make_attrs(status="restarting", exit_code=1, restart_count=4))
        self.assertEqual(health, "unhealthy")
        self.assertIn("restart count 4", reason)

    def test_paused(self):
        self.assertEqual(self.verdict(status="paused"), "unhealthy")

    def test_created_but_never_started(self):
        self.assertEqual(self.verdict(status="created"), "stopped")

    def test_created_with_start_error(self):
        health, reason = derive_health(make_attrs(status="created", exit_code=127, error="exec: \"sleep\": executable file not found"))
        self.assertEqual(health, "failed")
        self.assertIn("executable file not found", reason)

    def test_exit_zero_is_completed(self):
        self.assertEqual(self.verdict(status="exited", exit_code=0), "completed")

    def test_nonzero_exit_is_failed(self):
        health, reason = derive_health(make_attrs(status="exited", exit_code=3))
        self.assertEqual(health, "failed")
        self.assertIn("code 3", reason)

    def test_sigterm_and_sigkill_are_stopped(self):
        self.assertEqual(self.verdict(status="exited", exit_code=143), "stopped")
        self.assertEqual(self.verdict(status="exited", exit_code=137), "stopped")

    def test_oom_kill_is_failed_not_stopped(self):
        health, reason = derive_health(make_attrs(status="exited", exit_code=137, oom=True))
        self.assertEqual(health, "failed")
        self.assertIn("out of memory", reason)

    def test_dead(self):
        self.assertEqual(self.verdict(status="dead", exit_code=1), "failed")

    def test_removing(self):
        self.assertEqual(self.verdict(status="removing"), "missing")


class TestStatusFromAttrs(unittest.TestCase):
    def test_handle_fields(self):
        attrs = make_attrs(
            name="tml-demo", image="demo:latest", mode="idle", labels={"maintainer": "someone", LABEL_TEST: "abc"},
            HostConfig={
                "DeviceRequests": [{"Driver": "", "Count": -1, "Capabilities": [["gpu"]]}],
                "PortBindings": {"8888/tcp": [{"HostIp": "", "HostPort": "8888"}]},
            },
            Mounts=[
                {"Type": "bind", "Source": "/data", "Destination": "/data", "RW": False},
                {"Type": "bind", "Source": "/runs/a", "Destination": "/workspace/out", "RW": True},
            ],
            Config={"Image": "sha256:51dafde81dbd", "WorkingDir": "/workspace", "Labels": None},
        )
        # Config was overridden above, so put the labels back the way make_attrs builds them
        attrs["Config"]["Labels"] = {"tml.managed": "true", "tml.mode": "idle", "tml.image": "demo:latest", "maintainer": "someone", LABEL_TEST: "abc"}
        handle = handle_from_attrs(attrs)
        self.assertEqual(len(handle.id), 12)
        self.assertEqual(handle.name, "tml-demo")              # no leading slash
        self.assertEqual(handle.image, "demo:latest")          # from the label, not the resolved id
        self.assertEqual(handle.image_id, "51dafde81dbd")
        self.assertEqual(handle.mode, "idle")
        self.assertTrue(handle.gpu)
        self.assertEqual(handle.workdir, "/workspace")
        self.assertEqual(handle.ports, {"8888/tcp": "8888"})
        self.assertEqual(handle.mounts, ["/data:/data:ro", "/runs/a:/workspace/out"])
        self.assertEqual(set(handle.labels), {"tml.managed", "tml.mode", "tml.image", LABEL_TEST})  # image labels dropped

    def test_live_port_map_wins_over_configured_bindings(self):
        attrs = make_attrs(
            HostConfig={"PortBindings": {"8888/tcp": [{"HostIp": "", "HostPort": ""}]}},
            NetworkSettings={"Ports": {"8888/tcp": [{"HostIp": "0.0.0.0", "HostPort": "32768"}], "6006/tcp": None}},
        )
        self.assertEqual(handle_from_attrs(attrs).ports, {"8888/tcp": "32768"})

    def test_running_container_has_no_exit_code(self):
        # restarted after an earlier run: Docker still reports the old ExitCode and FinishedAt
        s = status_from_attrs(make_attrs(status="running", exit_code=143, finished="2026-10-09T09:00:00Z"), probe_ok=True)
        self.assertEqual(s.state, "running")
        self.assertIsNone(s.exit_code)
        self.assertIsNone(s.finished_at)
        self.assertIsNotNone(s.started_at)
        self.assertTrue(s.usable)

    def test_exited_container(self):
        s = status_from_attrs(make_attrs(status="exited", exit_code=3))
        self.assertEqual((s.state, s.health, s.exit_code), ("exited", "failed", 3))
        self.assertIsNotNone(s.finished_at)
        self.assertFalse(s.running)

    def test_never_started_container(self):
        s = status_from_attrs(make_attrs(status="created"))
        self.assertIsNone(s.exit_code)
        self.assertIsNone(s.started_at)      # Docker's zero time becomes None
        self.assertIsNone(s.finished_at)

    def test_container_field_is_what_was_asked(self):
        self.assertEqual(status_from_attrs(make_attrs(name="tml-demo"), asked="746d6c2d").container, "746d6c2d")
        self.assertEqual(status_from_attrs(make_attrs(name="tml-demo")).container, "tml-demo")

    def test_healthcheck_output_is_truncated(self):
        s = status_from_attrs(make_attrs(health="unhealthy", health_output="x" * 5000 + "TAIL"))
        self.assertEqual(len(s.healthcheck_output), status.MAX_HEALTHCHECK_CHARS)
        self.assertTrue(s.healthcheck_output.endswith("TAIL"))

    def test_oom_and_restart_count(self):
        s = status_from_attrs(make_attrs(status="exited", exit_code=137, oom=True, restart_count=2))
        self.assertTrue(s.oom_killed)
        self.assertEqual(s.restart_count, 2)


class TestGetStatus(unittest.TestCase):
    """get_status / require_managed / require_running against a fake client."""

    def test_missing_container_is_a_verdict_not_an_error(self):
        use_fake(self, FakeClient())
        s = get_status("nope")
        self.assertEqual((s.container, s.state, s.health), ("nope", "missing", "missing"))
        self.assertIsNone(s.handle)

    def test_unmanaged_container_is_refused(self):
        use_fake(self, FakeClient(FakeContainer(make_attrs(name="caddy", managed=False))))
        with self.assertRaises(LifecycleError) as ctx:
            get_status("caddy")
        self.assertEqual(ctx.exception.code, "not_managed")
        self.assertIn("tml.managed=true", ctx.exception.hint)

    def test_empty_name(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            get_status("")
        self.assertEqual(ctx.exception.code, "invalid_argument")

    def test_lookup_by_id_prefix(self):
        container = FakeContainer(make_attrs(name="tml-demo"))
        use_fake(self, FakeClient(container))
        s = get_status(container.id[:12])
        self.assertEqual(s.handle.name, "tml-demo")
        self.assertEqual(s.container, container.id[:12])

    def test_probe_runs_for_a_running_container_without_healthcheck(self):
        container = FakeContainer(make_attrs(status="running"))
        use_fake(self, FakeClient(container))
        self.assertEqual(get_status("tml-demo").health, "healthy")
        self.assertEqual(container.exec_calls, [status.PROBE_COMMAND])

    def test_failed_probe_is_unhealthy(self):
        use_fake(self, FakeClient(FakeContainer(make_attrs(status="running"), probe=127)))
        self.assertEqual(get_status("tml-demo").health, "unhealthy")

    def test_probe_that_raises_is_unhealthy(self):
        # e.g. the container stopped between the inspect and the probe
        gone = api_error(409, "container 746d is not running")
        use_fake(self, FakeClient(FakeContainer(make_attrs(status="running"), probe=gone)))
        self.assertEqual(get_status("tml-demo").health, "unhealthy")

    def test_no_probe_when_the_image_has_a_healthcheck(self):
        container = FakeContainer(make_attrs(status="running", health="healthy"), probe=1)
        use_fake(self, FakeClient(container))
        self.assertEqual(get_status("tml-demo").health, "healthy")
        self.assertEqual(container.exec_calls, [])

    def test_no_probe_when_not_running(self):
        container = FakeContainer(make_attrs(status="exited", exit_code=3))
        use_fake(self, FakeClient(container))
        self.assertEqual(get_status("tml-demo").health, "failed")
        self.assertEqual(container.exec_calls, [])

    def test_probe_can_be_switched_off(self):
        container = FakeContainer(make_attrs(status="running"), probe=1)
        use_fake(self, FakeClient(container))
        s = get_status("tml-demo", probe=False)
        self.assertEqual(s.health, "healthy")
        self.assertIn("not probed", s.reason)
        self.assertEqual(container.exec_calls, [])

    def test_daemon_down(self):
        use_fake(self, FakeClient(error=requests.exceptions.ConnectionError("Connection refused")))
        with self.assertRaises(LifecycleError) as ctx:
            get_status("tml-demo")
        self.assertEqual(ctx.exception.code, "docker_unavailable")

    def test_require_managed_returns_the_container(self):
        container = FakeContainer(make_attrs())
        use_fake(self, FakeClient(container))
        self.assertIs(require_managed("tml-demo"), container)

    def test_require_managed_missing(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            require_managed("nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_require_running_returns_the_handle(self):
        container = FakeContainer(make_attrs(status="running", mode="idle"))
        use_fake(self, FakeClient(container))
        handle = require_running("tml-demo")
        self.assertEqual((handle.name, handle.mode), ("tml-demo", "idle"))
        self.assertEqual(container.exec_calls, [])     # cheap: no probe before every command

    def test_require_running_refuses_other_states(self):
        for state, code in (("exited", 3), ("created", 0), ("paused", 0)):
            use_fake(self, FakeClient(FakeContainer(make_attrs(status=state, exit_code=code))))
            with self.assertRaises(LifecycleError) as ctx:
                require_running("tml-demo")
            self.assertEqual(ctx.exception.code, "not_running", state)
            self.assertIn(state, ctx.exception.message)

    def test_require_running_missing_and_unmanaged(self):
        use_fake(self, FakeClient(FakeContainer(make_attrs(name="caddy", managed=False))))
        for name, code in (("nope", "not_found"), ("caddy", "not_managed")):
            with self.assertRaises(LifecycleError) as ctx:
                require_running(name)
            self.assertEqual(ctx.exception.code, code)


class TestListing(unittest.TestCase):
    def setUp(self):
        self.fake = use_fake(self, FakeClient(
            FakeContainer(make_attrs(name="tml-b", status="running", created="2026-10-09T10:05:00Z", labels={LABEL_TEST: "run1"})),
            FakeContainer(make_attrs(name="tml-a", status="exited", exit_code=3, created="2026-10-09T10:00:00Z", labels={LABEL_TEST: "run1"})),
            FakeContainer(make_attrs(name="tml-c", status="running", created="2026-10-09T10:10:00Z", labels={LABEL_TEST: "run2"})),
            FakeContainer(make_attrs(name="caddy", status="running", managed=False)),
        ))

    def test_only_managed_containers_oldest_first(self):
        self.assertEqual([s.container for s in list_managed()], ["tml-a", "tml-b", "tml-c"])

    def test_running_only(self):
        self.assertEqual([s.container for s in list_managed(all=False)], ["tml-b", "tml-c"])

    def test_extra_label_filter(self):
        self.assertEqual([s.container for s in list_managed(labels={LABEL_TEST: "run1"})], ["tml-a", "tml-b"])
        self.assertEqual(list_managed(labels={LABEL_TEST: "other"}), [])

    def test_each_entry_is_a_full_status(self):
        by_name = {s.container: s for s in list_managed()}
        self.assertEqual(by_name["tml-a"].health, "failed")
        self.assertEqual(by_name["tml-b"].health, "healthy")
        self.assertIsInstance(by_name["tml-b"], ContainerStatus)

    def test_empty(self):
        use_fake(self, FakeClient())
        self.assertEqual(list_managed(), [])

    def test_daemon_down(self):
        use_fake(self, FakeClient(error=requests.exceptions.ConnectionError("Connection refused")))
        with self.assertRaises(LifecycleError) as ctx:
            list_managed()
        self.assertEqual(ctx.exception.code, "docker_unavailable")


class TestWaitReady(unittest.TestCase):
    """wait_ready against a scripted sequence of statuses (no sleeping, no daemon)."""

    def script(self, *healths: str) -> mock.Mock:
        statuses = [ContainerStatus(container="c", state="running", health=h, reason=h) for h in healths]
        patcher = mock.patch.object(status, "get_status", side_effect=statuses)
        get = patcher.start()
        self.addCleanup(patcher.stop)
        sleep = mock.patch.object(status.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        return get

    def test_waits_through_starting(self):
        get = self.script("starting", "starting", "healthy", "healthy")
        self.assertEqual(wait_ready("c").health, "healthy")
        self.assertEqual(get.call_count, 4)     # 2 x starting, healthy, then the settle check

    def test_crash_just_after_start_is_not_reported_as_ready(self):
        self.script("healthy", "failed")
        self.assertEqual(wait_ready("c", settle=1.0).health, "failed")
        self.sleep.assert_called_with(1.0)

    def test_no_settle_check(self):
        get = self.script("healthy")
        self.assertEqual(wait_ready("c", settle=0).health, "healthy")
        self.assertEqual(get.call_count, 1)

    def test_terminal_verdict_returns_at_once(self):
        for health in ("failed", "completed", "stopped", "unhealthy", "missing"):
            get = self.script(health)
            self.assertEqual(wait_ready("c").health, health)
            self.assertEqual(get.call_count, 1)

    def test_timeout_returns_the_last_status(self):
        self.script("starting")
        s = wait_ready("c", timeout=0)
        self.assertEqual(s.health, "starting")
        self.assertIn("still starting after 0 s", s.reason)


# ---------------------------------------------------------------- live suites (real daemon)

@requires_docker
class TestLiveStatus(unittest.TestCase):
    """Containers made directly through the SDK; lifecycle.provision is not written yet."""

    @classmethod
    def tearDownClass(cls):
        live.remove_test_containers()

    def exited(self, suffix: str, command, **kwargs) -> ContainerStatus:
        obj = live.start_raw(suffix, command, **kwargs)
        obj.wait(timeout=30)
        return get_status(obj.name)

    def test_idle_container_is_healthy(self):
        obj = live.start_idle("status-idle")
        s = get_status(obj.name)
        self.assertEqual((s.state, s.health), ("running", "healthy"))
        self.assertIn("readiness probe passes", s.reason)
        self.assertTrue(s.usable)
        self.assertIsNone(s.exit_code)
        self.assertIsNotNone(s.started_at)
        self.assertEqual(s.handle.name, obj.name)
        self.assertEqual(s.handle.mode, "idle")
        self.assertEqual(s.handle.image, live.BASE_IMAGE)
        self.assertFalse(s.handle.gpu)
        self.assertEqual(s.handle.labels[LABEL_TEST], live.RUN_ID)

    def test_lookup_by_id(self):
        obj = live.start_idle("status-by-id")
        self.assertEqual(get_status(obj.id[:12]).handle.name, obj.name)

    def test_passing_healthcheck(self):
        obj = live.start_raw(
            "status-healthy", ["sh", "-c", "sleep 1; touch /tmp/ready; sleep 120"],
            healthcheck=live.healthcheck("test -f /tmp/ready", retries=5),
        )
        self.assertEqual(get_status(obj.name).health, "starting")
        s = wait_ready(obj.name, timeout=20)
        self.assertEqual(s.health, "healthy")
        self.assertIn("HEALTHCHECK passes", s.reason)

    def test_failing_healthcheck(self):
        obj = live.start_raw("status-unhealthy", ["sleep", "120"], healthcheck=live.healthcheck("echo db down; exit 1"))
        s = wait_ready(obj.name, timeout=20)
        self.assertEqual((s.state, s.health), ("running", "unhealthy"))
        self.assertEqual(s.healthcheck_output, "db down")
        self.assertIn("db down", s.reason)
        self.assertFalse(s.usable)

    def test_crash(self):
        s = self.exited("status-crash", ["python", "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"])
        self.assertEqual((s.state, s.health, s.exit_code), ("exited", "failed", 3))
        self.assertIsNotNone(s.finished_at)

    def test_clean_exit(self):
        s = self.exited("status-done", ["python", "-c", "print('done')"])
        self.assertEqual((s.state, s.health, s.exit_code), ("exited", "completed", 0))

    def test_missing_executable(self):
        s = self.exited("status-noexec", ["x"], entrypoint=["/no/such/binary"])
        self.assertEqual((s.health, s.exit_code), ("failed", 127))

    def test_out_of_memory(self):
        # touch memory in 8 MB steps until the 32 MB limit kills the process
        hog = "chunks = []\nwhile True:\n    chunks.append(bytearray(b'x' * (8 * 1024 * 1024)))"
        s = self.exited("status-oom", ["python", "-c", hog], mem_limit="32m", memswap_limit="32m")
        self.assertEqual(s.health, "failed")
        self.assertTrue(s.oom_killed)
        self.assertIn("out of memory", s.reason)

    def test_stopped_from_outside(self):
        obj = live.start_idle("status-stopped")
        obj.stop(timeout=5)
        s = get_status(obj.name)
        self.assertEqual((s.state, s.health, s.exit_code), ("exited", "stopped", 143))

    def test_created_but_never_started(self):
        obj = live.start_raw("status-created", ["sleep", "120"], start=False)
        s = get_status(obj.name)
        self.assertEqual((s.state, s.health), ("created", "stopped"))
        self.assertIsNone(s.exit_code)
        self.assertIsNone(s.started_at)

    def test_wait_ready_catches_a_crash_just_after_start(self):
        obj = live.start_raw("status-late-crash", ["sh", "-c", "sleep 0.5; exit 4"])
        s = wait_ready(obj.name, timeout=20, settle=1.5)
        self.assertEqual((s.health, s.exit_code), ("failed", 4))

    def test_missing(self):
        s = get_status(live.container_name("status-never-made"))
        self.assertEqual((s.state, s.health), ("missing", "missing"))


@requires_docker
class TestLiveListing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.running = live.start_idle("list-running")
        cls.exited = live.start_raw("list-exited", ["python", "-c", "raise SystemExit(3)"])
        cls.exited.wait(timeout=30)
        cls.unmanaged = live.start_idle("list-unmanaged", managed=False)

    @classmethod
    def tearDownClass(cls):
        live.remove_test_containers()

    def test_lists_managed_containers_of_this_run(self):
        by_name = {s.container: s for s in list_managed(labels=live.TEST_LABEL)}
        self.assertEqual(set(by_name), {self.running.name, self.exited.name})
        self.assertEqual(by_name[self.running.name].health, "healthy")
        self.assertEqual(by_name[self.exited.name].health, "failed")

    def test_running_only(self):
        names = [s.container for s in list_managed(all=False, labels=live.TEST_LABEL)]
        self.assertEqual(names, [self.running.name])

    def test_unmanaged_container_never_appears(self):
        self.assertNotIn(self.unmanaged.name, [s.container for s in list_managed()])

    def test_unmanaged_container_is_refused(self):
        for call in (get_status, require_managed, require_running):
            with self.assertRaises(LifecycleError) as ctx:
                call(self.unmanaged.name)
            self.assertEqual(ctx.exception.code, "not_managed", call.__name__)


@requires_docker
class TestLiveRequireRunning(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        live.remove_test_containers()

    def test_running_container_gives_its_handle(self):
        obj = live.start_idle("require-running")
        handle = require_running(obj.name)
        self.assertEqual(handle.name, obj.name)
        self.assertEqual(handle.id, obj.id[:12])

    def test_stopped_container_is_refused(self):
        obj = live.start_idle("require-stopped")
        obj.stop(timeout=5)
        with self.assertRaises(LifecycleError) as ctx:
            require_running(obj.name)
        self.assertEqual(ctx.exception.code, "not_running")
        self.assertIn("exited", ctx.exception.message)

    def test_missing_container(self):
        with self.assertRaises(LifecycleError) as ctx:
            require_running(live.container_name("require-never-made"))
        self.assertEqual(ctx.exception.code, "not_found")


if __name__ == "__main__":
    unittest.main()
