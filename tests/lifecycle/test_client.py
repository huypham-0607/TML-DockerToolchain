import gc
import os
import unittest
import warnings
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

    def test_port_conflict(self):
        exc = api_error(500, "driver failed programming external connectivity: Bind for 0.0.0.0:8888 failed: port is already allocated")
        self.assertEqual(self.code(exc), "port_conflict")

    def test_server_error(self):
        self.assertEqual(self.code(api_error(500, "something broke in the daemon")), "docker_error")

    def test_daemon_unreachable(self):
        self.assertEqual(self.code(requests.exceptions.ConnectionError("Connection refused")), "docker_unavailable")
        exc = docker_errors.DockerException("Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))")
        self.assertEqual(self.code(exc), "docker_unavailable")

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


if __name__ == "__main__":
    unittest.main()
