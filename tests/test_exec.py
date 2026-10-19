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
