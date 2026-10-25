import gc
import json
import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import requests
from docker import errors as docker_errors

from src.lifecycle import client
from src.lifecycle.models import ERROR_CODES, LifecycleError
from tests.lifecycle.fakes import FakeClient, api_error, use_fake
from tests.lifecycle.samples import requires_docker


class TestTranslate(unittest.TestCase):
    def code(self, exc: Exception) -> str:
        err = client.translate(exc, "do the thing")
        self.assertIsInstance(err, LifecycleError)
        self.assertIn(err.code, ERROR_CODES)
        self.assertTrue(err.message.startswith("could not do the thing: "))
        return err.code

    def test_image_not_found(self):
        exc = docker_errors.ImageNotFound("404", explanation="No such image: no/such-image:zzz")
        err = client.translate(exc)
        self.assertEqual(err.code, "image_not_found")
        self.assertIn("No such image: no/such-image:zzz", err.message)
        self.assertIn("does not pull", err.hint)

    def test_container_not_found(self):
        self.assertEqual(self.code(docker_errors.NotFound("404", explanation="No such container: x")), "not_found")

    def test_name_conflict(self):
        exc = api_error(409, 'Conflict. The container name "/demo" is already in use by container "d7d9". You have to remove (or rename) that container.')
        self.assertEqual(self.code(exc), "name_conflict")

    def test_not_running(self):
        self.assertEqual(self.code(api_error(409, "container d7d9d9bb3b41 is not running")), "not_running")

    def test_other_conflict_is_a_plain_docker_error(self):
        exc = api_error(409, "You cannot remove a running container d7d9. Stop the container before attempting removal or force remove")
        self.assertEqual(self.code(exc), "docker_error")

    def test_gpu_unavailable(self):
        exc = api_error(500, 'could not select device driver "" with capabilities: [[gpu]]')
        err = client.translate(exc)
        self.assertEqual(err.code, "gpu_unavailable")
        self.assertIn("gpu='none'", err.hint)

    def test_gpu_unavailable_cdi_wording(self):
        # Engine 28+ finds GPUs through CDI and words the failure differently
        exc = api_error(500, "failed to discover GPU vendor from CDI: no known GPU vendor found")
        self.assertEqual(self.code(exc), "gpu_unavailable")

    def test_mount_denied_by_docker_desktop(self):
        exc = api_error(500, "mounts denied: \nThe path /tmp/data is not shared from the host and is not known to Docker.")
        err = client.translate(exc)
        self.assertEqual(err.code, "invalid_argument")
        self.assertIn("File Sharing", err.hint)

    def test_gpu_unavailable_other_wordings(self):
        for text in (
            "failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: "
            "unable to start container process: error during container init: error running prestart hook #0: exit status 1, "
            "stdout: , stderr: Auto-detected mode as 'legacy'\nnvidia-container-cli: initialization error: nvml error: driver not loaded",
            "could not start container: unresolvable CDI devices nvidia.com/gpu=all",
            "failed to inject CDI devices: unresolvable CDI devices nvidia.com/gpu=0",
        ):
            err = client.translate(api_error(500, text))
            self.assertEqual(err.code, "gpu_unavailable", text)
            self.assertIn("gpu='none'", err.hint)

    def test_docker_desktop_without_an_nvidia_gpu(self):
        # the tester's exact message: Docker Desktop (WSL) lists the NVIDIA runtime but has no GPU adapter
        err = client.translate(api_error(500, "failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'\nnvidia-container-cli: initialization error: WSL environment detected but no adapters were found: unknown"), "start container 'tml-x'")
        self.assertEqual(err.code, "gpu_unavailable")
        self.assertIn("gpu='none'", err.hint)

    def test_cuda_error_inside_a_container_is_not_a_daemon_gpu_error(self):
        # these come from the code in the container (its logs), not from the daemon
        for text in ("RuntimeError: Found no NVIDIA driver on your system.", "CUDA error: no kernel image is available"):
            self.assertIsNone(client.GPU_UNAVAILABLE_RE.search(text), text)

    def test_port_conflict(self):
        exc = api_error(500, "driver failed programming external connectivity: Bind for 0.0.0.0:8888 failed: port is already allocated")
        self.assertEqual(self.code(exc), "port_conflict")

    def test_server_error(self):
        self.assertEqual(self.code(api_error(500, "something broke in the daemon")), "docker_error")

    def test_daemon_unreachable(self):
        self.assertEqual(self.code(requests.exceptions.ConnectionError("Connection refused")), "docker_unavailable")
        exc = docker_errors.DockerException("Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))")
        self.assertEqual(self.code(exc), "docker_unavailable")

    def test_socket_permission(self):
        exc = docker_errors.DockerException("Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))")
        err = client.translate(exc)
        self.assertEqual(err.code, "docker_unavailable")
        self.assertIn("docker group", err.hint)
        exc = requests.exceptions.ConnectionError("('Connection aborted.', PermissionError(13, 'Permission denied'))")
        self.assertIn("docker group", client.translate(exc).hint)

    def test_permission_error_from_the_daemon_is_not_about_the_socket(self):
        # an APIError comes from a daemon that answered: the socket itself was fine
        err = client.translate(api_error(500, "error while creating mount source path '/data': permission denied"))
        self.assertEqual(err.code, "docker_error")

    def test_timeout(self):
        self.assertEqual(self.code(requests.exceptions.ReadTimeout("Read timed out. (read timeout=30)")), "timeout")

    def test_lifecycle_error_passes_through(self):
        original = LifecycleError("not_managed", "not ours")
        self.assertIs(client.translate(original, "anything"), original)

    def test_message_without_operation(self):
        err = client.translate(docker_errors.NotFound("404", explanation="No such container: x"))
        self.assertEqual(err.message, "No such container: x")


class TestTranslated(unittest.TestCase):
    def test_sdk_error_becomes_lifecycle_error(self):
        with self.assertRaises(LifecycleError) as ctx:
            with client.translated("inspect container 'x'"):
                raise docker_errors.NotFound("404", explanation="No such container: x")
        self.assertEqual(ctx.exception.code, "not_found")
        self.assertIsInstance(ctx.exception.__cause__, docker_errors.NotFound)

    def test_other_exceptions_are_not_hidden(self):
        # a bug in our own code must stay a normal traceback, not a "docker_error"
        with self.assertRaises(KeyError):
            with client.translated("anything"):
                raise KeyError("bug")

    def test_no_error(self):
        with client.translated("anything"):
            value = 1
        self.assertEqual(value, 1)


class TestAvailability(unittest.TestCase):
    def setUp(self):
        # the shared client must be rebuilt after each test here, whatever happened to it
        self.addCleanup(client.reset_client)

    def unreachable(self):
        client.reset_client()
        env = mock.patch.dict(os.environ, {"DOCKER_HOST": "unix:///nonexistent/tml-test-docker.sock"})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.collect_failed_sockets)

    @staticmethod
    def collect_failed_sockets():
        # the SDK leaves the socket of a failed connect to the garbage collector;
        # collect it here so its ResourceWarning does not show up in a later test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()

    def test_unreachable_daemon_is_not_available(self):
        self.unreachable()
        self.assertFalse(client.docker_available())

    def test_unreachable_daemon_raises_docker_unavailable(self):
        self.unreachable()
        with self.assertRaises(LifecycleError) as ctx:
            client.get_client()
        self.assertEqual(ctx.exception.code, "docker_unavailable")
        self.assertIn("daemon", ctx.exception.hint)

    def test_failed_ping_is_not_available(self):
        fake = FakeClient()
        fake.ping = mock.Mock(side_effect=requests.exceptions.ConnectionError("gone"))
        use_fake(self, fake)
        self.assertFalse(client.docker_available())

    @requires_docker
    def test_reachable_daemon_is_available(self):
        client.reset_client()
        self.assertTrue(client.docker_available())

    @requires_docker
    def test_client_is_shared(self):
        client.reset_client()
        self.assertIs(client.get_client(), client.get_client())


class TestGpuMemory(unittest.TestCase):
    """The remembered outcome of a real GPU attach, per daemon (a listed GPU does not have to work)."""

    DESKTOP = {"ID": "daemon-1", "ServerVersion": "29.0.1", "Runtimes": {"runc": {}, "nvidia": {}}}   # lists a GPU

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = mock.patch.dict(os.environ, {client.CACHE_DIR_ENV: self._tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def test_listed_gpu_counts_until_an_attach_fails(self):
        self.assertTrue(client.gpu_claimed(self.DESKTOP))
        self.assertTrue(client.gpu_available(self.DESKTOP))
        client.remember_gpu_attach(self.DESKTOP, worked=False, detail="WSL environment detected but no adapters were found")
        self.assertTrue(client.gpu_claimed(self.DESKTOP))           # still listed
        self.assertFalse(client.gpu_available(self.DESKTOP))        # but known not to work
        saved = json.loads(client.gpu_state_path().read_text())
        self.assertIn("no adapters", next(iter(saved.values()))["detail"])

    def test_a_later_success_wins(self):
        client.remember_gpu_attach(self.DESKTOP, worked=False)
        client.remember_gpu_attach(self.DESKTOP, worked=True)
        self.assertTrue(client.gpu_available(self.DESKTOP))
        self.assertTrue(client.gpu_attach_known(self.DESKTOP))

    def test_a_failure_is_forgotten_after_the_time_limit(self):
        client.remember_gpu_attach(self.DESKTOP, worked=False)
        later = time.time() + client.GPU_STATE_TTL + 1
        with mock.patch.object(client.time, "time", return_value=later):
            self.assertTrue(client.gpu_available(self.DESKTOP))
            self.assertFalse(client.gpu_attach_known(self.DESKTOP))

    def test_a_new_gpu_setup_is_tried_again(self):
        client.remember_gpu_attach(self.DESKTOP, worked=False)
        fixed = dict(self.DESKTOP, DiscoveredDevices=[{"Source": "cdi", "ID": "nvidia.com/gpu=0"}])
        self.assertTrue(client.gpu_available(fixed))                # e.g. a driver installed later: a new key
        other_daemon = dict(self.DESKTOP, ID="daemon-2")
        self.assertTrue(client.gpu_available(other_daemon))

    def test_nothing_known_at_first(self):
        self.assertFalse(client.gpu_attach_known(self.DESKTOP))
        self.assertFalse(client.gpu_state_path().exists())

    def test_forget(self):
        client.remember_gpu_attach(self.DESKTOP, worked=False)
        client.forget_gpu_attach()
        self.assertTrue(client.gpu_available(self.DESKTOP))
        client.forget_gpu_attach()                                  # nothing to forget: no error

    def test_damaged_cache_file_counts_as_empty(self):
        client.gpu_state_path().write_text("{not json")
        self.assertTrue(client.gpu_available(self.DESKTOP))
        client.gpu_state_path().write_text('["a list"]')
        self.assertTrue(client.gpu_available(self.DESKTOP))
        client.remember_gpu_attach(self.DESKTOP, worked=False)      # and is replaced on the next write
        self.assertFalse(client.gpu_available(self.DESKTOP))

    def test_cache_that_cannot_be_written(self):
        blocker = Path(self._tmp.name) / "a-file"
        blocker.write_text("")
        with mock.patch.dict(os.environ, {client.CACHE_DIR_ENV: str(blocker / "below")}):
            client.remember_gpu_attach(self.DESKTOP, worked=False)  # no error: it only costs one more try
            self.assertTrue(client.gpu_available(self.DESKTOP))

    def test_cache_location(self):
        with mock.patch.dict(os.environ, {client.CACHE_DIR_ENV: "/somewhere/tml"}):
            self.assertEqual(client.gpu_state_path(), Path("/somewhere/tml") / client.GPU_STATE_FILE)
        env = {k: v for k, v in os.environ.items() if k != client.CACHE_DIR_ENV}
        with mock.patch.dict(os.environ, {**env, "XDG_CACHE_HOME": "/xdg"}, clear=True):
            self.assertEqual(client.cache_dir(), Path("/xdg/tml-dockertoolchain"))
        env.pop("XDG_CACHE_HOME", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(client.cache_dir(), Path.home() / ".cache" / "tml-dockertoolchain")

    # --- the one-off probe

    def probe(self, start_error=None) -> tuple[bool, mock.Mock]:
        container = mock.Mock()
        container.start.side_effect = start_error
        sdk = mock.Mock()
        sdk.containers.create.return_value = container
        with mock.patch.object(client, "get_client", return_value=sdk):
            worked = client.probe_gpu_attach("tml-local/repo:abc", self.DESKTOP)
        kwargs = sdk.containers.create.call_args.kwargs
        self.assertEqual(sdk.containers.create.call_args.args, ("tml-local/repo:abc",))
        self.assertTrue(kwargs["device_requests"])
        container.remove.assert_called_once_with(force=True, v=True)   # never left behind
        return worked, container

    def test_probe_on_docker_desktop_without_a_gpu(self):
        worked, _ = self.probe(api_error(500, "failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'\nnvidia-container-cli: initialization error: WSL environment detected but no adapters were found: unknown"))
        self.assertFalse(worked)
        self.assertFalse(client.gpu_available(self.DESKTOP))
        self.assertTrue(client.gpu_attach_known(self.DESKTOP))

    def test_probe_with_a_working_gpu(self):
        worked, _ = self.probe(None)
        self.assertTrue(worked)
        self.assertTrue(client.gpu_available(self.DESKTOP))
        self.assertTrue(client.gpu_attach_known(self.DESKTOP))

    def test_probe_problem_that_is_not_about_the_gpu(self):
        with self.assertRaises(LifecycleError):
            self.probe(api_error(500, "something else broke"))
        self.assertFalse(client.gpu_attach_known(self.DESKTOP))     # nothing learned


class TestRuntimeDetection(unittest.TestCase):
    def test_nvidia_runtime_present(self):
        info = {"Runtimes": {"runc": {"path": "runc"}, "nvidia": {"path": "nvidia-container-runtime"}}}
        self.assertTrue(client.has_nvidia_runtime(info))

    def test_only_runc(self):
        self.assertFalse(client.has_nvidia_runtime({"Runtimes": {"runc": {}, "io.containerd.runc.v2": {}}}))

    def test_no_runtime_section(self):
        self.assertFalse(client.has_nvidia_runtime({}))
        self.assertFalse(client.has_nvidia_runtime({"Runtimes": None}))

    def test_reads_daemon_info_when_not_given(self):
        use_fake(self, FakeClient(info={"Runtimes": {"nvidia": {}}}))
        self.assertTrue(client.has_nvidia_runtime())
        use_fake(self, FakeClient(info={"Runtimes": {"runc": {}}}))
        self.assertFalse(client.has_nvidia_runtime())

    @requires_docker
    def test_live_daemon_answers(self):
        self.assertIsInstance(client.has_nvidia_runtime(), bool)
        self.assertIsInstance(client.gpu_available(), bool)

    def test_gpu_through_the_nvidia_runtime(self):
        self.assertTrue(client.gpu_available({"Runtimes": {"nvidia": {}}}))

    def test_gpu_as_a_cdi_device(self):
        info = {"Runtimes": {"runc": {}}, "DiscoveredDevices": [{"Source": "cdi", "ID": "nvidia.com/gpu=0"}]}
        self.assertTrue(client.gpu_available(info))
        self.assertFalse(client.has_nvidia_runtime(info))             # the runtime check alone would miss it

    def test_no_gpu(self):
        for info in (
            {"Runtimes": {"runc": {}}},
            {"Runtimes": {"runc": {}}, "DiscoveredDevices": None},
            {"Runtimes": {"runc": {}}, "DiscoveredDevices": [{"Source": "cdi", "ID": "vendor.example/fpga=0"}]},
            {},
        ):
            self.assertFalse(client.gpu_available(info), info)

    def test_gpu_reads_daemon_info_when_not_given(self):
        use_fake(self, FakeClient(info={"DiscoveredDevices": [{"Source": "cdi", "ID": "nvidia.com/gpu=all"}]}))
        self.assertTrue(client.gpu_available())


if __name__ == "__main__":
    unittest.main()
