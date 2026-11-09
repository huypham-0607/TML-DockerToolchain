"""
    scripts/evaluate.py: scoring the pipeline against the reference Dockerfiles.

    The orchestration is tested with the pipeline steps replaced (no LLM, no real builds).
    The live tests build two tiny images with a fake `torch` package and check the version
    probe and the smoke-test runner against them.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate as ev
from tests.lifecycle import samples
from tests.lifecycle.samples import requires_docker


class TestHelpers(unittest.TestCase):
    def test_repos_file_lists_the_four_references(self):
        repos = ev.read_repos()
        self.assertEqual(list(repos), ["ipmix", "ffb", "ep_bnp", "uap"])
        for name, (url, commit) in repos.items():
            self.assertTrue(url.startswith("https://github.com/"), name)
            self.assertRegex(commit, r"^[0-9a-f]{40}$")
            self.assertTrue((ev.REFERENCES / name / "smoke.py").is_file(), name)

    def test_self_cloning_dockerfiles_match_the_repos_file(self):
        for name, (url, commit) in ev.read_repos().items():
            text = (ev.REFERENCES / name / "Dockerfile").read_text()
            if "git clone" in text:
                self.assertIn(url, text, name)
                self.assertIn(commit, text, name)

    def test_compare(self):
        ref = {"python": "3.8", "torch": "1.13.1+cpu", "torchvision": "0.14.1+cpu"}
        self.assertEqual(ev.compare("python", {"python": "3.8"}, ref), "3.8 ✅")
        self.assertEqual(ev.compare("torch", {"torch": "1.13.1+cu117"}, ref), "1.13.1 ✅")     # build suffix ignored
        self.assertEqual(ev.compare("torch", {"torch": "2.0.1"}, ref), "2.0.1 ❌ (ref 1.13.1)")
        self.assertEqual(ev.compare("torchvision", {"torchvision": None}, ref), "missing ❌ (ref 0.14.1)")
        self.assertEqual(ev.compare("torch", {"torch": "2.0.1"}, {}), "2.0.1")                   # no reference image
        self.assertEqual(ev.compare("torch", {}, ref), "-")                                       # nothing built

    def test_report(self):
        rs = [ev.Result("ffb", "template", built=True, smoke_passed=True, versions={"python": "3.8"}),
              ev.Result("uap", "template", built=True, smoke_passed=False, smoke_tail="ModuleNotFoundError: uap"),
              ev.Result("ipmix", "template", built=False, build_note="build failed: x | y")]
        text = ev.report(rs)
        self.assertIn("| ffb | template | ✅ | 3.8 |", text)
        self.assertIn("❌ fail | ; smoke: ModuleNotFoundError: uap |", text)
        self.assertIn("build failed: x / y", text)      # a | in a note would break the table
        self.assertIn("**1 of 3**", text)


class TestEvaluateOne(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "ffb"
        self.repo.mkdir()
        patches = [
            mock.patch.object(ev, "clone", return_value=self.repo),
            mock.patch.object(ev, "image_exists", side_effect=lambda tag: True),
            mock.patch.object(ev, "probe_versions", side_effect=lambda tag: {"python": "3.8", "torch": "1.13.1", "tag": tag}),
            mock.patch.object(ev, "run_smoke", return_value=(True, "SMOKE OK")),
            mock.patch.object(ev, "OUT", Path(self._tmp.name) / "out"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_template_mode_builds_and_scores(self):
        build = mock.Mock(success=True, log_tail="", log_path="x")
        with mock.patch("src.docker.generate.generate_dockerfile", return_value=("FROM x\n", self.repo / "Dockerfile", ["n"])), \
                mock.patch("src.docker.build.build_image", return_value=build) as b:
            r = ev.evaluate_one("ffb", "https://github.com/x/ffb", "0" * 40, mode="template", skip_build=False)
        self.assertEqual(b.call_args.kwargs["tag"], "tml-eval/ffb:template")
        self.assertTrue(r.built and r.smoke_passed)
        self.assertEqual(r.reference_versions["tag"], "tml-ref/ffb:latest")
        self.assertEqual(r.versions["tag"], "tml-eval/ffb:template")

    def test_template_build_failure_skips_the_smoke_test(self):
        build = mock.Mock(success=False, log_tail="...\nERROR: No matching distribution found for torch==1.13.1+cu116\n", log_path="x")
        with mock.patch("src.docker.generate.generate_dockerfile", return_value=("FROM x\n", self.repo / "Dockerfile", [])), \
                mock.patch("src.docker.build.build_image", return_value=build), \
                mock.patch.object(ev, "run_smoke") as smoke:
            r = ev.evaluate_one("ffb", "u", "0" * 40, mode="template", skip_build=False)
        self.assertFalse(r.built)
        self.assertIn("No matching distribution", r.build_note)
        self.assertIsNone(r.smoke_passed)
        smoke.assert_not_called()

    def test_agent_mode_scores_the_last_image_the_agent_built(self):
        summary = {"rounds": 9, "seconds": 300.0, "tools_used": {"build_image": 2}, "crashed": "",
                   "images_built": ["tml/ffb:latest"], "last_ai": "done"}
        with mock.patch("scripts.run_agent.run_agent", return_value=summary) as run:
            r = ev.evaluate_one("ffb", "u", "0" * 40, mode="agent", skip_build=False)
        self.assertIn(str(self.repo), run.call_args.args[0])
        self.assertEqual(r.image, "tml/ffb:latest")
        self.assertEqual(r.agent["builds"], 2)
        self.assertTrue(r.smoke_passed)

    def test_agent_that_built_nothing(self):
        summary = {"rounds": 13, "seconds": 50.0, "tools_used": {"build_image": 1}, "crashed": "",
                   "images_built": [], "last_ai": "gave up"}
        with mock.patch("scripts.run_agent.run_agent", return_value=summary):
            r = ev.evaluate_one("ffb", "u", "0" * 40, mode="agent", skip_build=False)
        self.assertFalse(r.built)
        self.assertIn("built no image", r.build_note)

    def test_a_crash_in_the_pipeline_does_not_stop_the_run(self):
        with mock.patch("src.docker.generate.generate_dockerfile", side_effect=RuntimeError("boom")):
            r = ev.evaluate_one("ffb", "u", "0" * 40, mode="template", skip_build=False)
        self.assertIn("pipeline error: RuntimeError: boom", r.build_note)

    def test_docker_down_is_a_clear_error_not_missing_images(self):
        import io
        from contextlib import redirect_stderr
        down = subprocess.CompletedProcess(["docker", "info"], 1, "", "Cannot connect to the Docker daemon")
        err = io.StringIO()
        with mock.patch.object(ev.subprocess, "run", return_value=down), redirect_stderr(err), \
                mock.patch.object(ev, "evaluate_one") as one:
            code = ev.main(["--agent", "--skip-build"])
        self.assertEqual(code, 2)
        self.assertIn("Docker is not reachable", err.getvalue())
        one.assert_not_called()

    def test_unknown_name_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            ev.main(["nope"])


@requires_docker
class TestLiveProbes(unittest.TestCase):
    """Real containers: a fake `torch` module stands in for PyTorch."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        ctx = Path(cls._tmp.name)
        (ctx / "torch").mkdir()
        (ctx / "torch" / "__init__.py").write_text("__version__ = '1.13.1+cpu'\n")
        (ctx / "Dockerfile").write_text(f"FROM {samples.BASE_IMAGE}\nCOPY torch /opt/fake/torch\nENV PYTHONPATH=/opt/fake\n"
                                        f"LABEL tml.test={samples.RUN_ID}\n")
        cls.tag = f"tml-test/eval-{samples.RUN_ID}:x"
        subprocess.run(["docker", "build", "-q", "-t", cls.tag, str(ctx)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["docker", "rmi", "-f", cls.tag], capture_output=True)
        cls._tmp.cleanup()

    def test_probe_versions(self):
        v = ev.probe_versions(self.tag)
        self.assertEqual(v["torch"], "1.13.1+cpu")
        self.assertIsNone(v["torchvision"])
        self.assertRegex(v["python"], r"^3\.\d+$")

    def test_smoke_pass_and_fail(self):
        good = Path(self._tmp.name) / "good.py"
        good.write_text("import torch\nprint('SMOKE OK')\n")
        bad = Path(self._tmp.name) / "bad.py"
        bad.write_text("import torchvision\nprint('SMOKE OK')\n")
        self.assertEqual(ev.run_smoke(self.tag, good), (True, "SMOKE OK"))
        passed, tail = ev.run_smoke(self.tag, bad)
        self.assertFalse(passed)
        self.assertIn("ModuleNotFoundError", tail)

    def test_smoke_has_no_network(self):
        script = Path(self._tmp.name) / "net.py"
        script.write_text("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\nprint('SMOKE OK')\n")
        passed, tail = ev.run_smoke(self.tag, script)
        self.assertFalse(passed)
        self.assertIn("OSError", tail.replace("Network is unreachable", "OSError"))


if __name__ == "__main__":
    unittest.main()


class TestMerge(unittest.TestCase):
    """Running one repository at a time must not erase the others (Sat 7 Nov: only UAP survived)."""

    def test_new_rows_replace_their_old_row_and_keep_the_rest(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "results.json"
            first = [ev.Result("ffb", "agent", built=True, smoke_passed=False), ev.Result("ep_bnp", "agent", built=True, smoke_passed=True)]
            path.write_text(json.dumps([vars(r) for r in first]))
            rows = ev.merge([ev.Result("ffb", "agent", built=True, smoke_passed=True), ev.Result("uap", "template")], path)
        names = [(r.name, r.mode, r.smoke_passed) for r in rows]
        self.assertEqual(names, [("ffb", "agent", True), ("ep_bnp", "agent", True), ("uap", "template", None)])

    def test_missing_or_broken_file_starts_empty(self):
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / "results.json"
            bad.write_text("{not json")
            self.assertEqual([r.name for r in ev.merge([ev.Result("ffb", "agent")], bad)], ["ffb"])
            self.assertEqual([r.name for r in ev.merge([ev.Result("ffb", "agent")], Path(t) / "none.json")], ["ffb"])


class TestOtherModels(unittest.TestCase):
    """Testing "with other agents": the same tools driven by another LLM keep their own rows."""

    def test_rows_of_two_models_are_kept_apart(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "results.json"
            path.write_text(json.dumps([
                {"name": "ffb", "mode": "agent", "built": True, "smoke_passed": True},  # before the model column
                {"name": "ffb", "mode": "template", "built": True},
            ]))
            rows = ev.merge([ev.Result("ffb", "agent", built=True, smoke_passed=False, model="qwen3.6:27b")], path)
        got = {(r.mode, r.model): r.smoke_passed for r in rows}
        self.assertEqual(got[("agent", ev.LEGACY_AGENT_MODEL)], True)
        self.assertEqual(got[("agent", "qwen3.6:27b")], False)
        self.assertIn(("template", ""), got)
        table = ev.report(rows)
        self.assertIn("| ffb | agent (qwen3.6:27b) |", table)
        self.assertIn("| ffb | agent (gpt-oss:120b) |", table)

    def test_current_model_follows_the_environment(self):
        with mock.patch.dict("os.environ", {"LLM_PROVIDER": "rcac", "LLM_MODEL": "gemma4:26b-a4b"}):
            self.assertEqual(ev.current_model(), "gemma4:26b-a4b")
        with mock.patch.dict("os.environ", {"LLM_PROVIDER": "openai", "LLM_MODEL": ""}):
            self.assertEqual(ev.current_model(), "gpt-5-mini")

    def test_model_flag_needs_agent_mode(self):
        import io
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ev.main(["--model", "qwen3.6:27b", "ffb"])

    def test_model_flag_sets_the_environment(self):
        with mock.patch.dict("os.environ", {}, clear=False), \
                mock.patch.object(ev.subprocess, "run", return_value=mock.Mock(returncode=1)):
            import io
            from contextlib import redirect_stderr
            with redirect_stderr(io.StringIO()):
                ev.main(["--agent", "--model", "qwen3.6:27b", "ffb"])
            import os
            self.assertEqual(os.environ["LLM_MODEL"], "qwen3.6:27b")
