import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.lifecycle import client
from src.lifecycle.models import (
    LABEL_IMAGE, LABEL_MANAGED, LABEL_MODE, LABEL_SPEC, LABEL_TEST,
    ContainerSpec, LifecycleError,
)
from src.lifecycle.provision import (
    DEFAULT_MOUNT_ROOT, MOUNT_ROOTS_ENV, PIDS_LIMIT,
    allowed_mount_roots, build_create_kwargs, default_memory, generate_name,
    parse_mounts, parse_ports, provision, spec_fingerprint,
)
from src.lifecycle.status import get_status
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, FakeImage, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker

GB = 1024 ** 3


def tearDownModule():
    samples.cleanup()


class InvalidMixin:
    def assertInvalid(self, fragment: str, call, *args, **kwargs) -> LifecycleError:
        with self.assertRaises(LifecycleError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertIn(fragment, ctx.exception.message)
        return ctx.exception


# ---------------------------------------------------------------- pure suites (no daemon)

class TestCreateKwargs(InvalidMixin, unittest.TestCase):
    def kwargs(self, use_gpu=False, memory_default=None, **fields) -> dict:
        return build_create_kwargs(ContainerSpec(image="demo:1", **fields), "demo:1", "tml-demo", use_gpu, memory_default)

    def test_always_set(self):
        k = self.kwargs()
        self.assertEqual((k["image"], k["name"]), ("demo:1", "tml-demo"))
        self.assertIs(k["init"], True)
        self.assertIs(k["privileged"], False)
        self.assertEqual(k["pids_limit"], PIDS_LIMIT)
        self.assertEqual(k["shm_size"], 2 * GB)
        self.assertEqual(k["network_mode"], "bridge")

    def test_toolchain_labels(self):
        self.assertEqual(self.kwargs()["labels"], {LABEL_MANAGED: "true", LABEL_MODE: "idle", LABEL_IMAGE: "demo:1"})

    def test_spec_labels_are_added_but_cannot_replace_toolchain_labels(self):
        labels = self.kwargs(labels={LABEL_TEST: "run1", LABEL_MANAGED: "false", LABEL_MODE: "x"})["labels"]
        self.assertEqual(labels[LABEL_TEST], "run1")
        self.assertEqual((labels[LABEL_MANAGED], labels[LABEL_MODE]), ("true", "idle"))

    def test_idle_mode_only_keeps_the_container_alive(self):
        k = self.kwargs(mode="idle")
        self.assertEqual((k["entrypoint"], k["command"]), (["sleep"], ["infinity"]))
        self.assertEqual(k["healthcheck"], {"test": ["NONE"]})     # the image's service is not running

    def test_native_mode_leaves_the_image_alone(self):
        k = self.kwargs(mode="native")
        for key in ("entrypoint", "command", "healthcheck"):
            self.assertNotIn(key, k)
        self.assertEqual(k["labels"][LABEL_MODE], "native")

    def test_native_command_is_split_like_a_shell_would(self):
        k = self.kwargs(mode="native", command="python train.py --name 'run one' --lr 0.1")
        self.assertEqual(k["command"], ["python", "train.py", "--name", "run one", "--lr", "0.1"])
        self.assertNotIn("entrypoint", k)

    def test_command_with_broken_quoting(self):
        self.assertInvalid("cannot parse command", self.kwargs, mode="native", command="echo 'unclosed")

    def test_no_memory_limit_when_the_host_size_is_unknown(self):
        k = self.kwargs()
        self.assertNotIn("mem_limit", k)
        self.assertNotIn("memswap_limit", k)

    def test_default_memory_limit_has_no_swap(self):
        k = self.kwargs(memory_default=12 * GB)
        self.assertEqual((k["mem_limit"], k["memswap_limit"]), (12 * GB, 12 * GB))

    def test_explicit_memory_wins_over_the_default(self):
        k = self.kwargs(memory="512m", memory_default=12 * GB)
        self.assertEqual((k["mem_limit"], k["memswap_limit"]), (512 * 1024 ** 2, 512 * 1024 ** 2))

    def test_invalid_sizes(self):
        self.assertInvalid("invalid memory", self.kwargs, memory="lots")
        self.assertInvalid("invalid memory", self.kwargs, memory="0")
        self.assertInvalid("invalid shm_size", self.kwargs, shm_size="big")

    def test_cpus(self):
        self.assertNotIn("nano_cpus", self.kwargs())               # no limit by default
        self.assertEqual(self.kwargs(cpus=1.5)["nano_cpus"], 1_500_000_000)

    def test_gpu(self):
        self.assertNotIn("device_requests", self.kwargs(use_gpu=False))
        (request,) = self.kwargs(use_gpu=True)["device_requests"]
        self.assertEqual((request["Count"], request["Capabilities"]), (-1, [["gpu"]]))

    def test_env_values_become_strings(self):
        self.assertEqual(self.kwargs(env={"SEED": 1, "DEBUG": True})["environment"], {"SEED": "1", "DEBUG": "True"})

    def test_optional_settings_are_left_out_when_empty(self):
        k = self.kwargs()
        for key in ("environment", "volumes", "ports", "working_dir", "user"):
            self.assertNotIn(key, k)

    def test_workdir_user_network(self):
        k = self.kwargs(workdir="/workspace", user="1000:1000", network="none")
        self.assertEqual((k["working_dir"], k["user"], k["network_mode"]), ("/workspace", "1000:1000", "none"))

    def test_ports(self):
        self.assertEqual(self.kwargs(ports=["8888:8888"])["ports"], {"8888/tcp": ("127.0.0.1", 8888)})


class TestParsePorts(InvalidMixin, unittest.TestCase):
    def test_host_and_container_port(self):
        self.assertEqual(parse_ports(["18080:8080"]), {"8080/tcp": ("127.0.0.1", 18080)})

    def test_container_port_gets_a_free_host_port(self):
        self.assertEqual(parse_ports(["8888"]), {"8888/tcp": ("127.0.0.1", None)})

    def test_protocol(self):
        self.assertEqual(parse_ports(["5353:53/udp", "6006/tcp"]), {"53/udp": ("127.0.0.1", 5353), "6006/tcp": ("127.0.0.1", None)})

    def test_every_port_is_published_on_loopback_only(self):
        for address, _ in parse_ports(["80", "443:443", "53/udp"]).values():
            self.assertEqual(address, "127.0.0.1")

    def test_an_address_cannot_be_chosen(self):
        self.assertInvalid("invalid port", parse_ports, ["0.0.0.0:8888:8888"])

    def test_invalid_ports(self):
        for bad in ("http", "8888:", ":8888", "80/sctp", "1-10", ""):
            self.assertInvalid("invalid port", parse_ports, [bad])
        for bad in ("0", "70000", "70000:80"):
            self.assertInvalid("out of range", parse_ports, [bad])

    def test_no_ports(self):
        self.assertEqual(parse_ports([]), {})


class TestParseMounts(InvalidMixin, unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name).resolve()
        self.root, self.outside = tmp / "allowed", tmp / "outside"
        (self.root / "data").mkdir(parents=True)
        self.outside.mkdir()
        env = mock.patch.dict(os.environ, {MOUNT_ROOTS_ENV: str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_read_write_by_default(self):
        self.assertEqual(parse_mounts([f"{self.root}/data:/data"]), {str(self.root / "data"): {"bind": "/data", "mode": "rw"}})

    def test_read_only(self):
        self.assertEqual(parse_mounts([f"{self.root}/data:/data:ro"])[str(self.root / "data")]["mode"], "ro")

    def test_the_root_itself_may_be_mounted(self):
        self.assertIn(str(self.root), parse_mounts([f"{self.root}:/all"]))

    def test_outside_the_allowed_roots(self):
        e = self.assertInvalid("outside the allowed roots", parse_mounts, [f"{self.outside}:/data"])
        self.assertIn(MOUNT_ROOTS_ENV, e.hint)
        self.assertInvalid("outside the allowed roots", parse_mounts, ["/etc:/host-etc:ro"])

    def test_dot_dot_cannot_leave_a_root(self):
        self.assertInvalid("outside the allowed roots", parse_mounts, [f"{self.root}/data/../../outside:/data"])

    def test_symlink_cannot_leave_a_root(self):
        (self.root / "link").symlink_to(self.outside)
        self.assertInvalid("outside the allowed roots", parse_mounts, [f"{self.root}/link:/data"])

    def test_docker_socket_is_always_refused(self):
        self.assertInvalid("Docker socket", parse_mounts, ["/var/run/docker.sock:/var/run/docker.sock"])
        (self.root / "docker.sock").touch()                  # even one inside an allowed root
        self.assertInvalid("Docker socket", parse_mounts, [f"{self.root}/docker.sock:/var/run/docker.sock"])

    def test_source_must_exist(self):
        self.assertInvalid("does not exist", parse_mounts, [f"{self.root}/not-made:/data"])

    def test_source_must_be_an_absolute_path(self):
        self.assertInvalid("absolute host path", parse_mounts, ["data:/data"])           # also rules out named volumes
        self.assertInvalid("absolute host path", parse_mounts, ["./data:/data"])

    def test_target_must_be_an_absolute_path(self):
        self.assertInvalid("absolute container path", parse_mounts, [f"{self.root}/data:data"])

    def test_invalid_format(self):
        for bad in (f"{self.root}/data", f"{self.root}/data:/data:rw:extra", f"{self.root}/data:/data:exec"):
            self.assertInvalid("invalid mount", parse_mounts, [bad])

    def test_allowed_roots(self):
        self.assertEqual(allowed_mount_roots(), [DEFAULT_MOUNT_ROOT.resolve(), self.root])
        with mock.patch.dict(os.environ, {MOUNT_ROOTS_ENV: f"{self.root}{os.pathsep}{self.outside}"}):
            self.assertEqual(allowed_mount_roots()[1:], [self.root, self.outside])
        with mock.patch.dict(os.environ, {MOUNT_ROOTS_ENV: ""}):
            self.assertEqual(allowed_mount_roots(), [DEFAULT_MOUNT_ROOT.resolve()])

    def test_default_root_is_runs_in_the_repository(self):
        repo = Path(__file__).resolve().parents[2]
        self.assertEqual(DEFAULT_MOUNT_ROOT, repo / "runs")


class TestFingerprintAndNames(unittest.TestCase):
    def kwargs(self, name="tml-a", **fields) -> dict:
        return build_create_kwargs(ContainerSpec(image="demo:1", **fields), "demo:1", name)

    def test_same_image_and_settings(self):
        self.assertEqual(spec_fingerprint(self.kwargs(), "abc"), spec_fingerprint(self.kwargs(), "abc"))
        self.assertRegex(spec_fingerprint(self.kwargs(), "abc"), r"^[0-9a-f]{12}$")

    def test_name_and_labels_do_not_count(self):
        a = spec_fingerprint(self.kwargs(name="tml-a"), "abc")
        b = spec_fingerprint(self.kwargs(name="tml-b", labels={LABEL_TEST: "run2"}), "abc")
        self.assertEqual(a, b)

    def test_image_and_every_setting_count(self):
        base = spec_fingerprint(self.kwargs(), "abc")
        self.assertNotEqual(base, spec_fingerprint(self.kwargs(), "def"))        # rebuilt image
        for changed in ({"memory": "1g"}, {"mode": "native"}, {"env": {"A": "1"}}, {"ports": ["80"]}, {"cpus": 2}, {"network": "none"}):
            self.assertNotEqual(base, spec_fingerprint(self.kwargs(**changed), "abc"), changed)

    def test_gpu_counts(self):
        spec = ContainerSpec(image="demo:1")
        without = build_create_kwargs(spec, "demo:1", "n", use_gpu=False)
        with_gpu = build_create_kwargs(spec, "demo:1", "n", use_gpu=True)
        self.assertNotEqual(spec_fingerprint(without, "abc"), spec_fingerprint(with_gpu, "abc"))

    def test_generated_names(self):
        self.assertRegex(generate_name("tml-local/robust-training:3d51101d19ac"), r"^tml-robust-training-[0-9a-f]{6}$")
        self.assertRegex(generate_name("python:3.14-slim"), r"^tml-python-[0-9a-f]{6}$")
        self.assertRegex(generate_name("registry.example.com:5000/team/img@sha256:abcd"), r"^tml-img-[0-9a-f]{6}$")
        self.assertNotEqual(generate_name("demo"), generate_name("demo"))
        ContainerSpec(image="demo", name=generate_name("demo")).validate()         # a generated name is a valid name

    def test_default_memory(self):
        self.assertEqual(default_memory({"MemTotal": 16 * GB}), 12 * GB)
        self.assertIsNone(default_memory({}))
        self.assertIsNone(default_memory({"MemTotal": 0}))


class TestGuardrails(InvalidMixin, unittest.TestCase):
    """provision() refuses a bad request before it creates anything. Fake client: no image exists,
    so a request that got past its argument checks would fail with image_not_found instead."""

    def setUp(self):
        self.fake = use_fake(self, FakeClient())
        self.fake.containers.create = mock.Mock(side_effect=AssertionError("a container was created"))

    def test_mount_outside_the_allowed_roots(self):
        self.assertInvalid("outside the allowed roots", provision, ContainerSpec(image="demo:1", mounts=["/etc:/host-etc"]))

    def test_docker_socket(self):
        self.assertInvalid("Docker socket", provision, ContainerSpec(image="demo:1", mounts=["/var/run/docker.sock:/var/run/docker.sock"]))

    def test_port_on_all_interfaces(self):
        self.assertInvalid("invalid port", provision, ContainerSpec(image="demo:1", ports=["0.0.0.0:8888:8888"]))

    def test_host_network(self):
        self.assertInvalid("network", provision, ContainerSpec(image="demo:1", network="host"))

    def test_needs_exactly_one_of_image_and_dockerfile(self):
        self.assertInvalid("exactly one", provision, ContainerSpec())
        self.assertInvalid("exactly one", provision, ContainerSpec(image="demo:1", dockerfile="Dockerfile"))

    def test_there_is_no_way_to_ask_for_privileged(self):
        with self.assertRaises(TypeError):
            ContainerSpec(image="demo:1", privileged=True)

    def test_arguments_are_checked_before_the_image(self):
        self.assertInvalid("outside the allowed roots", provision, ContainerSpec(image="demo:1", mounts=["/etc:/x"]))
        self.assertEqual(self.fake.images.get_calls, [])

    def test_missing_image_is_not_pulled(self):
        with self.assertRaises(LifecycleError) as ctx:
            provision(ContainerSpec(image="pytorch/pytorch:latest"))
        self.assertEqual(ctx.exception.code, "image_not_found")
        self.fake.containers.create.assert_not_called()

    def test_name_of_an_unmanaged_container_is_refused_even_with_replace(self):
        other = FakeContainer(make_attrs(name="caddy", managed=False))
        other.remove = mock.Mock(side_effect=AssertionError("an unmanaged container was removed"))
        fake = use_fake(self, FakeClient(other, images=[FakeImage("demo:1")]))
        fake.containers.create = mock.Mock(side_effect=AssertionError("a container was created"))
        with self.assertRaises(LifecycleError) as ctx:
            provision(ContainerSpec(image="demo:1", name="caddy"), replace=True)
        self.assertEqual(ctx.exception.code, "name_conflict")
        self.assertIn("did not create", ctx.exception.message)


# ---------------------------------------------------------------- live suites (real daemon)

class LiveCase(unittest.TestCase):
    """Base of the live suites: containers are removed after each class."""

    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends

    def attrs(self, result) -> dict:
        return samples.sdk_container(result.handle.name).attrs


@requires_docker
class TestIdleMode(LiveCase):
    def test_keeps_a_one_shot_image_alive(self):
        r = samples.start("idle", "idle-basic")           # the image's CMD prints one line and exits
        self.assertTrue(r.ok)
        self.assertEqual(r.action, "created")
        self.assertEqual((r.status.state, r.status.health), ("running", "healthy"))
        self.assertEqual(r.handle.name, samples.container_name("idle-basic"))
        self.assertEqual(r.handle.mode, "idle")
        self.assertEqual(r.handle.workdir, "/workspace")   # WORKDIR of the image is kept
        self.assertEqual(r.handle.image, r.image.tag)
        self.assertRegex(r.image.tag, r"^tml-local/idle:[0-9a-f]{12}$")

    def test_labels(self):
        labels = samples.start("idle", "idle-labels").handle.labels
        self.assertEqual((labels[LABEL_MANAGED], labels[LABEL_MODE]), ("true", "idle"))
        self.assertEqual(labels[LABEL_TEST], samples.RUN_ID)
        self.assertRegex(labels[LABEL_SPEC], r"^[0-9a-f]{12}$")

    def test_settings_docker_really_has(self):
        attrs = self.attrs(samples.start("idle", "idle-settings"))
        host, config = attrs["HostConfig"], attrs["Config"]
        self.assertIs(host["Init"], True)
        self.assertIs(host["Privileged"], False)
        self.assertEqual(host["NetworkMode"], "bridge")
        self.assertEqual(host["PidsLimit"], PIDS_LIMIT)
        self.assertEqual(host["ShmSize"], 2 * GB)
        self.assertGreater(host["Memory"], 0)              # default limit: a share of the host memory
        self.assertEqual(host["MemorySwap"], host["Memory"])
        self.assertFalse(host["DeviceRequests"])
        self.assertEqual((config["Entrypoint"], config["Cmd"]), (["sleep"], ["infinity"]))

    def test_overrides_the_entrypoint_of_the_image(self):
        r = samples.start("entrypoint", "idle-entrypoint")
        self.assertEqual((r.status.state, r.status.health), ("running", "healthy"))

    def test_image_healthcheck_is_switched_off(self):
        # in idle mode the service of the image does not run, so its HEALTHCHECK could never pass
        r = samples.start("service", "idle-service")
        self.assertEqual(r.status.health, "healthy")
        self.assertIn("readiness probe passes", r.status.reason)
        self.assertEqual(self.attrs(r)["Config"]["Healthcheck"]["Test"], ["NONE"])

    def test_image_without_sleep_cannot_idle(self):
        r = samples.start("empty", "idle-empty")           # FROM scratch: there is no `sleep`
        self.assertFalse(r.ok)
        self.assertEqual((r.status.state, r.status.health, r.status.exit_code), ("exited", "failed", 127))
        # the failed container is kept, so it can be diagnosed
        self.assertEqual(get_status(r.handle.name).health, "failed")

    def test_from_an_existing_image(self):
        built = samples.build("idle")
        r = provision(ContainerSpec(image=built.tag, name=samples.container_name("idle-from-image"), labels=samples.TEST_LABEL))
        self.assertTrue(r.ok)
        self.assertEqual((r.image.tag, r.image.id, r.image.built), (built.tag, built.id, False))
        self.assertEqual(r.handle.image_id, built.id)

    def test_generated_name(self):
        built = samples.build("idle")
        a = provision(ContainerSpec(image=built.tag, labels=samples.TEST_LABEL))
        b = provision(ContainerSpec(image=built.tag, labels=samples.TEST_LABEL))
        self.assertRegex(a.handle.name, r"^tml-idle-[0-9a-f]{6}$")
        self.assertNotEqual(a.handle.name, b.handle.name)
        self.assertEqual((a.action, b.action), ("created", "created"))


@requires_docker
class TestNativeMode(LiveCase):
    def test_command_that_ends_cleanly_is_ok(self):
        r = samples.start("idle", "native-done", mode="native")
        self.assertTrue(r.ok)                               # "completed" is a success in native mode
        self.assertEqual((r.status.state, r.status.health, r.status.exit_code), ("exited", "completed", 0))
        self.assertEqual(r.handle.mode, "native")

    def test_crash(self):
        r = samples.start("crash", "native-crash", mode="native")
        self.assertFalse(r.ok)
        self.assertEqual((r.status.state, r.status.health, r.status.exit_code), ("exited", "failed", 3))
        self.assertIsNone(r.diagnosis)                      # filled by the diagnostics section later

    def test_missing_python_module(self):
        r = samples.start("missing-module", "native-module", mode="native")
        self.assertFalse(r.ok)
        self.assertEqual((r.status.health, r.status.exit_code), ("failed", 1))

    def test_crash_just_after_start_is_not_reported_as_ready(self):
        r = samples.start("idle", "native-late-crash", mode="native", command="sh -c 'sleep 0.5; exit 4'")
        self.assertFalse(r.ok)
        self.assertEqual((r.status.health, r.status.exit_code), ("failed", 4))

    def test_command_replaces_the_cmd_of_the_image(self):
        r = samples.start("idle", "native-command", mode="native", command="python -c 'import time; time.sleep(300)'")
        self.assertTrue(r.ok)
        self.assertEqual((r.status.state, r.status.health), ("running", "healthy"))
        config = self.attrs(r)["Config"]
        self.assertEqual(config["Cmd"], ["python", "-c", "import time; time.sleep(300)"])
        self.assertIsNone(config["Entrypoint"])

    def test_waits_for_the_image_healthcheck(self):
        r = samples.start("service", "native-service", mode="native")
        self.assertTrue(r.ok)
        self.assertEqual(r.status.health, "healthy")
        self.assertIn("HEALTHCHECK passes", r.status.reason)

    def test_short_timeout_returns_while_still_starting(self):
        r = samples.start("service", "native-service-slow", mode="native", ready_timeout=0.01)
        self.assertFalse(r.ok)
        self.assertEqual((r.status.state, r.status.health), ("running", "starting"))
        self.assertIn("still starting", r.status.reason)

    def test_failing_healthcheck(self):
        r = samples.start("unhealthy", "native-unhealthy", mode="native")
        self.assertFalse(r.ok)
        self.assertEqual((r.status.state, r.status.health), ("running", "unhealthy"))
        self.assertEqual(r.status.healthcheck_output, "db down")

    def test_out_of_memory(self):
        r = samples.start("oom", "native-oom", mode="native", memory="64m")
        self.assertFalse(r.ok)
        self.assertEqual(r.status.health, "failed")
        self.assertTrue(r.status.oom_killed)
        self.assertIn("out of memory", r.status.reason)


@requires_docker
class TestConflicts(LiveCase):
    def test_same_name_and_settings_is_reused(self):
        first = samples.start("idle", "conflict-reuse")
        again = samples.start("idle", "conflict-reuse")
        self.assertEqual((first.action, again.action), ("created", "reused"))
        self.assertEqual(again.handle.id, first.handle.id)          # the very same container
        self.assertTrue(again.ok)

    def test_stopped_container_is_started_again(self):
        first = samples.start("idle", "conflict-restart")
        samples.sdk_container(first.handle.name).stop(timeout=5)
        self.assertEqual(get_status(first.handle.name).health, "stopped")
        again = samples.start("idle", "conflict-restart")
        self.assertEqual(again.action, "restarted")
        self.assertEqual(again.handle.id, first.handle.id)
        self.assertEqual((again.status.state, again.status.health), ("running", "healthy"))
        self.assertIsNone(again.status.exit_code)

    def test_different_settings_is_a_conflict(self):
        first = samples.start("idle", "conflict-settings")
        with self.assertRaises(LifecycleError) as ctx:
            samples.start("idle", "conflict-settings", memory="256m")
        self.assertEqual(ctx.exception.code, "name_conflict")
        self.assertIn("replace=True", ctx.exception.hint)
        self.assertEqual(samples.sdk_container(first.handle.name).id[:12], first.handle.id)   # untouched

    def test_different_image_is_a_conflict(self):
        samples.start("idle", "conflict-image")
        with self.assertRaises(LifecycleError) as ctx:
            samples.start("entrypoint", "conflict-image")
        self.assertEqual(ctx.exception.code, "name_conflict")

    def test_different_mode_is_a_conflict(self):
        samples.start("idle", "conflict-mode")
        with self.assertRaises(LifecycleError) as ctx:
            samples.start("idle", "conflict-mode", mode="native")
        self.assertEqual(ctx.exception.code, "name_conflict")

    def test_replace(self):
        first = samples.start("idle", "conflict-replace")
        second = samples.start("idle", "conflict-replace", memory="256m", replace=True)
        self.assertEqual(second.action, "replaced")
        self.assertNotEqual(second.handle.id, first.handle.id)
        self.assertEqual(self.attrs(second)["HostConfig"]["Memory"], 256 * 1024 ** 2)

    def test_replace_with_same_settings_still_reuses(self):
        first = samples.start("idle", "conflict-replace-same")
        again = samples.start("idle", "conflict-replace-same", replace=True)
        self.assertEqual(again.action, "reused")
        self.assertEqual(again.handle.id, first.handle.id)

    def test_unmanaged_container_keeps_its_name(self):
        other = samples.start_unmanaged("conflict-unmanaged")
        for replace in (False, True):
            with self.assertRaises(LifecycleError) as ctx:
                provision(ContainerSpec(image=samples.build("idle").tag, name=other.name, labels=samples.TEST_LABEL), replace=replace)
            self.assertEqual(ctx.exception.code, "name_conflict")
        other.reload()
        self.assertEqual(other.status, "running")                  # not stopped, not removed


@requires_docker
class TestOptions(LiveCase):
    def run_in(self, result, command: str) -> str:
        """Runs a shell command with the SDK (the tests may; lifecycle itself never does)."""
        return samples.sdk_container(result.handle.name).exec_run(["sh", "-c", command]).output.decode().strip()

    def test_env_and_workdir(self):
        r = samples.start("idle", "opt-env", env={"SEED": "7", "MODE": "eval"}, workdir="/tmp")
        self.assertEqual(self.run_in(r, "echo $SEED-$MODE; pwd"), "7-eval\n/tmp")
        self.assertEqual(r.handle.workdir, "/tmp")

    def test_mount_from_the_default_root(self):
        data = samples.mount_dir()
        (data / "x.txt").write_text("hello")
        r = samples.start("idle", "opt-mount", mounts=[f"{data}:/data"])
        self.assertTrue(r.ok)
        self.assertEqual(r.handle.mounts, [f"{data}:/data"])
        self.assertEqual(self.run_in(r, "cat /data/x.txt; echo written > /data/y.txt"), "hello")
        self.assertEqual((data / "y.txt").read_text().strip(), "written")

    def test_read_only_mount(self):
        data = samples.mount_dir()
        r = samples.start("idle", "opt-mount-ro", mounts=[f"{data}:/data:ro"])
        self.assertEqual(r.handle.mounts, [f"{data}:/data:ro"])
        self.assertIn("Read-only file system", self.run_in(r, "touch /data/y.txt 2>&1"))
        self.assertEqual(list(data.iterdir()), [])

    def test_ports_are_published_on_loopback(self):
        r = samples.start("idle", "opt-ports", ports=["8888", "18473:8080"])
        self.assertEqual(r.handle.ports["8080/tcp"], "18473")
        self.assertTrue(r.handle.ports["8888/tcp"].isdigit())       # a free port chosen by Docker
        for binds in self.attrs(r)["NetworkSettings"]["Ports"].values():
            self.assertEqual({b["HostIp"] for b in binds}, {"127.0.0.1"})

    def test_port_conflict_leaves_no_container(self):
        samples.start("idle", "opt-port-a", ports=["18474:8080"])
        with self.assertRaises(LifecycleError) as ctx:
            samples.start("idle", "opt-port-b", ports=["18474:8080"])
        self.assertEqual(ctx.exception.code, "port_conflict")
        self.assertEqual(get_status(samples.container_name("opt-port-b")).health, "missing")

    def test_memory_and_cpus(self):
        host = self.attrs(samples.start("idle", "opt-limits", memory="256m", cpus=1.5))["HostConfig"]
        self.assertEqual((host["Memory"], host["MemorySwap"], host["NanoCpus"]), (256 * 1024 ** 2, 256 * 1024 ** 2, 1_500_000_000))

    def test_no_network(self):
        r = samples.start("idle", "opt-no-network", network="none")
        self.assertTrue(r.ok)
        self.assertEqual(self.attrs(r)["HostConfig"]["NetworkMode"], "none")

    def test_user(self):
        r = samples.start("idle", "opt-user", user="1234:1234")
        self.assertEqual(self.run_in(r, "id -u; id -g"), "1234\n1234")

    def test_gpu_none(self):
        self.assertFalse(samples.start("idle", "opt-gpu-none", gpu="none").handle.gpu)

    def test_gpu_auto_follows_the_daemon(self):
        r = samples.start("idle", "opt-gpu-auto", gpu="auto")
        self.assertTrue(r.ok)
        self.assertEqual(r.handle.gpu, client.has_nvidia_runtime())

    def test_gpu_all(self):
        # with a GPU: the container gets it. Without one: a clear error and no leftover container.
        try:
            r = samples.start("idle", "opt-gpu-all", gpu="all")
        except LifecycleError as e:
            self.assertEqual(e.code, "gpu_unavailable")
            self.assertIn("gpu='none'", e.hint)
            self.assertEqual(get_status(samples.container_name("opt-gpu-all")).health, "missing")
        else:
            self.assertTrue(r.handle.gpu)


@requires_docker
class TestDockerfileInput(LiveCase):
    def test_builds_once_and_then_reuses_the_image(self):
        samples.build("entrypoint", rebuild=True)
        a = samples.start("entrypoint", "df-a")
        b = samples.start("entrypoint", "df-b")
        self.assertFalse(a.image.built)
        self.assertEqual((a.image.tag, a.image.id), (b.image.tag, b.image.id))
        self.assertNotEqual(a.handle.id, b.handle.id)

    def test_first_provision_builds_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Dockerfile"
            path.write_text(samples.DOCKERFILES["idle"] + f"ENV BUILD_MARK={samples.RUN_ID}\n")
            spec = ContainerSpec(dockerfile=str(path), name=samples.container_name("df-fresh"), labels=samples.TEST_LABEL)
            r = provision(spec)
        self.assertTrue(r.image.built)
        self.assertTrue(r.ok)

    def test_failed_build_creates_no_container(self):
        with self.assertRaises(LifecycleError) as ctx:
            samples.start("broken-build", "df-broken")
        self.assertEqual(ctx.exception.code, "build_failed")
        self.assertEqual(get_status(samples.container_name("df-broken")).health, "missing")

    def test_missing_dockerfile(self):
        spec = ContainerSpec(dockerfile="/nonexistent/Dockerfile", name=samples.container_name("df-missing"))
        with self.assertRaises(LifecycleError) as ctx:
            provision(spec)
        self.assertEqual(ctx.exception.code, "invalid_argument")


@requires_docker
class TestMissingImage(LiveCase):
    IMAGE = "hello-world:linux"      # real, tiny, and not something this project uses

    def test_image_is_not_pulled(self):
        sdk = client.get_client()
        if self.present(sdk):
            self.skipTest(f"{self.IMAGE} is already on this machine")
        with self.assertRaises(LifecycleError) as ctx:
            provision(ContainerSpec(image=self.IMAGE, name=samples.container_name("missing-image"), labels=samples.TEST_LABEL))
        self.assertEqual(ctx.exception.code, "image_not_found")
        self.assertIn("does not pull", ctx.exception.hint)
        self.assertFalse(self.present(sdk))                         # still not there: nothing was pulled
        self.assertEqual(get_status(samples.container_name("missing-image")).health, "missing")

    def present(self, sdk) -> bool:
        try:
            sdk.images.get(self.IMAGE)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    unittest.main()
