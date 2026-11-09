"""
Unit tests for src/execution module.
"""

import unittest
from unittest import mock
import subprocess
from src.execution.exec import execute_in_container
from src.execution.models import ExecResult


class TestContainerExecution(unittest.TestCase):

    def test_empty_container(self):
        res = execute_in_container("", "ls")
        self.assertEqual(res.exit_code, 1)
        self.assertFalse(res.to_dict()["ok"])
        self.assertIn("No container", res.stderr)

    def test_successful_execution(self):
        fake_proc = subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="Python 3.10.12\n",
            stderr=""
        )
        with mock.patch("subprocess.run", return_value=fake_proc) as mock_run:
            res = execute_in_container("my-container", "python --version")
            self.assertEqual(res.exit_code, 0)
            self.assertEqual(res.stdout, "Python 3.10.12\n")
            self.assertFalse(res.timed_out)
            mock_run.assert_called_once()
            # Verify docker exec and container name were in argv
            call_args = mock_run.call_args[0][0]
            self.assertIn("docker", call_args)
            self.assertIn("my-container", call_args)

    def test_failed_command(self):
        fake_proc = subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=2,
            stdout="",
            stderr="command not found: pytest\n"
        )
        with mock.patch("subprocess.run", return_value=fake_proc):
            res = execute_in_container("my-container", "pytest")
            self.assertEqual(res.exit_code, 2)
            self.assertFalse(res.to_dict()["ok"])
            self.assertIn("command not found", res.stderr)

    def test_timeout_handling(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5)
        with mock.patch("subprocess.run", side_effect=timeout_exc):
            res = execute_in_container("my-container", "sleep 10", timeout=5)
            self.assertEqual(res.exit_code, -1)
            self.assertTrue(res.timed_out)
            self.assertIn("[TIMEOUT]", res.stderr)


if __name__ == "__main__":
    unittest.main()


class TestStaleImageWarning(unittest.TestCase):
    """A container created before the latest build_image of its tag tests the old image.
    Seen in the FFB and UAP agent runs: the agent rebuilt, then kept testing the old container."""

    def sdk(self, container_image_id, tag_image_id, tag="tml/ffb:latest"):
        import types
        sdk = mock.Mock()
        sdk.containers.get.return_value = types.SimpleNamespace(attrs={"Image": container_image_id, "Config": {"Image": tag}})
        sdk.images.get.return_value = types.SimpleNamespace(id=tag_image_id)
        return sdk

    def warning(self, sdk):
        from src.execution import tools
        from src.lifecycle import client
        with mock.patch.object(client, "get_client", return_value=sdk):
            return tools.stale_image_warning("c")

    def test_older_build_warns(self):
        w = self.warning(self.sdk("sha256:old", "sha256:new"))
        self.assertIn("OLDER build of tml/ffb:latest", w)
        self.assertIn("provision a new one", w)

    def test_current_build_is_quiet(self):
        self.assertIsNone(self.warning(self.sdk("sha256:same", "sha256:same")))

    def test_container_made_from_an_id_is_quiet(self):
        self.assertIsNone(self.warning(self.sdk("sha256:a", "sha256:b", tag="sha256:a")))

    def test_docker_problems_are_quiet(self):
        sdk = mock.Mock()
        sdk.containers.get.side_effect = RuntimeError("no such container")
        self.assertIsNone(self.warning(sdk))

    def test_tool_reply_carries_the_warning(self):
        import json
        from src.execution import tools
        ok = ExecResult(command="true", exit_code=0, stdout="", stderr="", duration=0.1)
        with mock.patch.object(tools, "execute_in_container", return_value=ok), \
                mock.patch.object(tools, "stale_image_warning", return_value="OLDER build"):
            reply = json.loads(tools.run_command_in_container.invoke({"container": "c", "command": "true"}))
        self.assertEqual(list(reply)[:2], ["ok", "warning"])
        with mock.patch.object(tools, "execute_in_container", return_value=ok), \
                mock.patch.object(tools, "stale_image_warning", return_value=None):
            reply = json.loads(tools.run_command_in_container.invoke({"container": "c", "command": "true"}))
        self.assertNotIn("warning", reply)


class TestAgentCommandLimits(unittest.TestCase):
    """EP/BNP run: the agent asked for 600 s six times while train.py downloaded CIFAR-10;
    each try was killed, started the download over, and returned ~40 kB of tqdm updates."""

    def run_tool(self, result, **args):
        import json
        from src.execution import tools
        with mock.patch.object(tools, "execute_in_container", return_value=result) as ex, \
                mock.patch.object(tools, "stale_image_warning", return_value=None):
            reply = json.loads(tools.run_command_in_container.invoke({"container": "c", "command": "x", **args}))
        return reply, ex.call_args.kwargs

    def test_timeout_is_capped(self):
        from src.execution import tools
        ok = ExecResult(command="x", exit_code=0, stdout="", stderr="", duration=0.1)
        _, kwargs = self.run_tool(ok, timeout=600)
        self.assertEqual(kwargs["timeout"], tools.MAX_AGENT_TIMEOUT)
        _, kwargs = self.run_tool(ok, timeout=30)
        self.assertEqual(kwargs["timeout"], 30)

    def test_progress_bars_collapse_to_their_last_state(self):
        from src.execution.tools import collapse_progress
        bar = "".join(f"\r  {i}%|#   | {i}M/170M" for i in range(100))
        out = collapse_progress("start\n" + bar + "\nTraceback: boom")
        self.assertEqual(out, "start\n  99%|#   | 99M/170M\nTraceback: boom")
        self.assertEqual(collapse_progress("plain\ntext"), "plain\ntext")

    def test_long_output_keeps_head_and_tail(self):
        from src.execution.tools import shorten
        text = "HEAD" + "x" * 50000 + "TAIL"
        out = shorten(text, 1000)
        self.assertTrue(out.startswith("HEAD") and out.endswith("TAIL"))
        self.assertIn("characters cut", out)
        self.assertLess(len(out), 1100)

    def test_timeout_reply_has_a_hint_and_a_short_stderr(self):
        bar = "".join(f"\r  0%|   | {i}k/170M [00:0{i % 10}<25:00, 110kB/s]" for i in range(3000))
        killed = ExecResult(command="x", exit_code=-1, stdout="", stderr=bar, duration=300.0,
                            timed_out=True, error="timeout_expired")
        reply, _ = self.run_tool(killed, timeout=600)
        self.assertIn("Do not repeat it", reply["hint"])
        self.assertLess(len(reply["stderr"]), 200)
