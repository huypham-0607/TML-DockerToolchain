import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.docker import build, generate
from tests.test_structure import make_repo


class TestGenerateDockerfile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.runs = mock.patch.object(generate, "RUNS_DIR", self.tmp / "runs")
        self.runs.start()

    def tearDown(self):
        self.runs.stop()
        self._tmp.cleanup()

    def test_old_torch_repo(self):
        make_repo(self.tmp / "augmix", {
            "requirements.txt": "numpy\ntorch==1.2.0\ntorchvision==0.2.2\n",
            "train.py": "import torch\nmodel.cuda()\n",
        })
        text, path, notes = generate.generate_dockerfile(self.tmp / "augmix")
        self.assertIn("FROM python:3.7-slim", text)
        self.assertIn('RUN pip install "torch==1.2.0" "torchvision==0.2.2"', text)
        self.assertIn("RUN pip install -r requirements.txt", text)
        # frameworks are installed before the repo is copied (layer cache)
        self.assertLess(text.index("torch==1.2.0"), text.index("COPY . /workspace"))
        self.assertTrue(any("--gpus all" in n for n in notes))
        self.assertEqual(path.read_text(), text)

    def test_defaults_and_installable_project(self):
        make_repo(self.tmp / "plain", {"setup.py": "from setuptools import setup\nsetup(name='x')\n"})
        text, _, notes = generate.generate_dockerfile(self.tmp / "plain")
        self.assertIn(f"FROM python:{generate.DEFAULT_PYTHON}-slim", text)
        self.assertIn("RUN pip install -e .", text)
        self.assertNotIn("requirements.txt", text)
        self.assertTrue(any("defaulting" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
