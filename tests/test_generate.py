import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.docker import generate
from tests.test_structure import make_repo

CPU_INDEX = "--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"


class GenerateCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.runs = mock.patch.object(generate, "RUNS_DIR", self.tmp / "runs")
        self.runs.start()
        self.env = mock.patch.dict(os.environ, {generate.TORCH_BUILD_ENV: ""})   # tests choose the build
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.runs.stop()
        self._tmp.cleanup()

    def gen(self, name, files, build="cpu"):
        make_repo(self.tmp / name, files)
        return generate.generate_dockerfile(self.tmp / name, torch_build=build)


class TestGenerateDockerfile(GenerateCase):
    def test_old_torch_repo(self):
        text, path, notes = self.gen("augmix", {
            "requirements.txt": "numpy\ntorch==1.2.0\ntorchvision==0.2.2\n",
            "train.py": "import torch\nmodel.cuda()\n",
        })
        self.assertIn("FROM python:3.7-slim", text)
        self.assertIn(f'RUN pip install {CPU_INDEX} "torch==1.2.0" "torchvision==0.2.2"', text)
        self.assertIn("pip install -r /tmp/requirements-0.txt", text)
        # frameworks are installed before the repo is copied (layer cache)
        self.assertLess(text.index("torch==1.2.0"), text.index("COPY . /workspace"))
        self.assertTrue(any("needs an NVIDIA GPU" in n and "train.py" in n for n in notes))
        self.assertEqual(path.read_text(), text)

    def test_defaults_and_installable_project(self):
        text, _, notes = self.gen("plain", {"setup.py": "from setuptools import setup\nsetup(name='x')\n"})
        self.assertIn(f"FROM python:{generate.DEFAULT_PYTHON}-slim", text)
        self.assertIn("RUN pip install -e .", text)
        self.assertNotIn("requirements", text)
        self.assertNotIn("download.pytorch.org", text)
        self.assertTrue(any("defaulting" in n for n in notes))

    def test_requirements_without_torch_are_installed_as_is(self):
        text, _, _ = self.gen("np", {"requirements.txt": "numpy==1.24.2\n", "a.py": "import numpy\n"})
        self.assertIn("RUN pip install -r requirements.txt", text)
        self.assertNotIn("grep", text)


class TestTorchPins(GenerateCase):
    """The FFB case: requirements.txt pins torch==1.13.1+cu116, which plain pip cannot find."""

    # out of order on purpose: the Dockerfile lists torch, torchvision, torchaudio
    FFB = {"requirements.txt": "numpy==1.24.2\ntorchaudio==0.13.1+cu116\ntorch==1.13.1+cu116\ntorchvision==0.14.1+cu116\n",
           "src/train.py": "import torch\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'\n"}

    def test_local_version_cpu_build(self):
        text, _, notes = self.gen("ffb", self.FFB, build="cpu")
        self.assertNotIn("+cu116", text.split("COPY")[0])     # the suffix never reaches pip
        self.assertIn(f'{CPU_INDEX} "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"', text)
        self.assertTrue(any("cu116" in n and "filtered out" in n for n in notes))
        self.assertTrue(any(n.startswith("torch: CPU build") for n in notes))

    def test_local_version_cuda_build_uses_its_index(self):
        text, _, notes = self.gen("ffb", self.FFB, build="cuda")
        self.assertIn("--index-url https://download.pytorch.org/whl/cu116 --extra-index-url https://pypi.org/simple", text)
        self.assertTrue(any("CUDA build from https://download.pytorch.org/whl/cu116" in n for n in notes))

    def test_cuda_build_from_the_repos_cuda_version(self):
        text, _, _ = self.gen("cu113", {"requirements.txt": "torch==1.12.1\n", "README.md": "Tested with CUDA 11.3\n"},
                              build="cuda")
        self.assertIn("https://download.pytorch.org/whl/cu113", text)

    def test_cuda_build_without_a_cuda_version_uses_pypi(self):
        text, _, _ = self.gen("nocuda", {"requirements.txt": "torch==2.2.0\n"}, build="cuda")
        self.assertIn('RUN pip install "torch==2.2.0"', text)
        self.assertNotIn("download.pytorch.org", text)

    def test_requirements_cannot_reinstall_torch(self):
        text, _, _ = self.gen("ffb", self.FFB)
        grep_line = next(line for line in text.splitlines() if "grep" in line)
        self.assertIn("requirements.txt > /tmp/requirements-0.txt", grep_line)

    def test_other_frameworks_stay_on_pypi(self):
        text, _, _ = self.gen("tf", {"requirements.txt": "tensorflow==2.12.0\ntorch==2.0.1\n"})
        self.assertIn('RUN pip install "tensorflow==2.12.0"', text)
        self.assertIn(f'{CPU_INDEX} "torch==2.0.1"', text)

    def test_guarded_cuda_is_not_called_gpu_only(self):
        _, _, notes = self.gen("ffb", self.FFB)
        self.assertTrue(any("falls back to CPU" in n for n in notes))
        self.assertFalse(any("needs an NVIDIA GPU" in n for n in notes))


class TestTorchFilter(unittest.TestCase):
    """The grep pattern written into Dockerfiles, checked with Python's re (same ERE subset)."""

    def test_lines(self):
        import re
        pattern = re.compile(generate.TORCH_LINE_RE.replace(r"\s", r"[ \t]"), re.I)
        dropped = ["torch==1.13.1+cu116", "torchvision>=0.8", "  torchaudio", "Torch==2.0", "torch[opt]", 'torch ; python_version>"3"']
        kept = ["torchmetrics==1.0", "torch-geometric==2.3", "torch_scatter", "numpy==1.24", "# torch"]
        for line in dropped:
            self.assertTrue(pattern.search(line), line)
        for line in kept:
            self.assertFalse(pattern.search(line), line)


class TestChooseBuild(unittest.TestCase):
    def test_explicit(self):
        self.assertEqual(generate.choose_torch_build("cpu"), "cpu")
        self.assertEqual(generate.choose_torch_build("CUDA"), "cuda")

    def test_env(self):
        with mock.patch.dict(os.environ, {generate.TORCH_BUILD_ENV: "cuda"}):
            self.assertEqual(generate.choose_torch_build(), "cuda")

    def test_auto_follows_the_host(self):
        with mock.patch.object(generate, "host_gpu_usable", return_value=True):
            self.assertEqual(generate.choose_torch_build("auto"), "cuda")
        with mock.patch.object(generate, "host_gpu_usable", return_value=False):
            self.assertEqual(generate.choose_torch_build("auto"), "cpu")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            generate.choose_torch_build("tpu")

    def test_no_docker_means_cpu(self):
        from src.lifecycle import client
        with mock.patch.object(client, "get_client", side_effect=RuntimeError("no daemon")):
            self.assertFalse(generate.host_gpu_usable())

    def test_claimed_but_failed_gpu_means_cpu(self):
        # Docker Desktop on Windows lists an "nvidia" runtime without an NVIDIA GPU
        from src.lifecycle import client
        sdk = mock.Mock()
        sdk.info.return_value = {"Runtimes": {"nvidia": {}}}
        with mock.patch.object(client, "get_client", return_value=sdk), \
                mock.patch.object(client, "gpu_attach_known", return_value=True), \
                mock.patch.object(client, "gpu_attach_failed", return_value=True):
            self.assertFalse(generate.host_gpu_usable())

    def test_claimed_and_unknown_probes_once(self):
        from src.lifecycle import client
        sdk = mock.Mock()
        sdk.info.return_value = {"Runtimes": {"nvidia": {}}}
        sdk.images.list.return_value = [mock.Mock(tags=["python:3.10-slim"])]
        with mock.patch.object(client, "get_client", return_value=sdk), \
                mock.patch.object(client, "gpu_attach_known", return_value=False), \
                mock.patch.object(client, "probe_gpu_attach", return_value=True) as probe:
            self.assertTrue(generate.host_gpu_usable())
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[0], "python:3.10-slim")


if __name__ == "__main__":
    unittest.main()
