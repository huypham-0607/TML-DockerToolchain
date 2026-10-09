import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.docker import build


class TestBuildImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "myrepo"
        self.repo.mkdir()
        self.runs = mock.patch.object(build, "RUNS_DIR", Path(self._tmp.name) / "runs")
        self.runs.start()

    def tearDown(self):
        self.runs.stop()
        self._tmp.cleanup()

    def fake_run(self, returncode=0, stdout="", stderr=""):
        return mock.patch.object(build.subprocess, "run",
                                 return_value=subprocess.CompletedProcess([], returncode, stdout, stderr))

    def test_success(self):
        with self.fake_run(0, "built ok") as run:
            r = build.build_image(self.repo, "FROM python:3.10-slim\n")
        self.assertTrue(r.success)
        self.assertEqual(r.image, "tml/myrepo:latest")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["docker", "build"])
        self.assertEqual(cmd[-1], str(self.repo.resolve()))      # repo is the build context
        self.assertEqual(Path(r.dockerfile_path).read_text(), "FROM python:3.10-slim\n")
        self.assertFalse((self.repo / "Dockerfile").exists())    # repo not modified

    def test_failure_returns_log_tail(self):
        log = "x" * 10000 + "ERROR: No matching distribution found for torch==1.2.0"
        with self.fake_run(1, "", log):
            r = build.build_image(self.repo, "FROM python:3.12-slim\n", tag="t:1")
        self.assertFalse(r.success)
        self.assertIsNone(r.image)
        self.assertEqual(r.exit_code, 1)
        self.assertTrue(r.log_tail.endswith("torch==1.2.0"))
        self.assertLessEqual(len(r.log_tail), build.LOG_TAIL_CHARS)

    def test_timeout(self):
        with mock.patch.object(build.subprocess, "run", side_effect=subprocess.TimeoutExpired("docker", 1)):
            r = build.build_image(self.repo, "FROM x\n", timeout=1)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.success)

    def test_docker_missing(self):
        with mock.patch.object(build.subprocess, "run", side_effect=FileNotFoundError):
            r = build.build_image(self.repo, "FROM x\n")
        self.assertIn("docker is not installed", r.log_tail)

    def test_bad_repo_path(self):
        r = build.build_image(Path(self._tmp.name) / "nope", "FROM x\n")
        self.assertFalse(r.success)


if __name__ == "__main__":
    unittest.main()
