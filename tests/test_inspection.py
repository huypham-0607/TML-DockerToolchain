import json
import tempfile
import unittest
from pathlib import Path

from src.repo.dependencies import detect_dependencies, parse_conda_spec, parse_requirement
from src.repo.documentation import SearchResult, classify_command, extract_documentation
from src.repo.environment import detect_environment, satisfies
from src.repo.inspect import inspect_repository
from tests.test_structure import make_repo


README = """\
# Robust Training

Code for our paper.

## Requirements

- Python 3.8
- CUDA 11.3
- PyTorch 1.12.1

## Installation

```bash
git clone https://github.com/someone/robust-training.git
cd robust-training
conda create -n robust python=3.8
conda activate robust
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt && python setup.py develop
```

You can also run `pip install -e .` for development.

## Training

```bash
$ python train.py --config configs/cifar.yaml
```

```python
import torch  # not a shell block
```
"""

SETUP_PY = """\
from setuptools import setup
REQS = ["numpy>=1.20", "tqdm"]
setup(
    name="robust",
    python_requires=">=3.7",
    install_requires=REQS,
    extras_require={"dev": ["pytest"]},
)
"""

PYPROJECT = """\
[project]
name = "robust"
requires-python = ">=3.8,<3.11"
dependencies = ["scipy==1.10.1", "pillow[extra]>=9; python_version>='3.8'"]

[project.optional-dependencies]
viz = ["matplotlib"]

[build-system]
requires = ["setuptools>=61"]
"""

ENV_YML = """\
name: robust
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.8
  - pytorch=1.12.1
  - cudatoolkit=11.3
  - pip
  - pip:
    - wandb==0.15.0
    - --extra-index-url https://download.pytorch.org/whl/cu113
"""

TRAIN_PY = """\
import torch
from models.net import Net

model = Net().cuda()
x = torch.randn(1, 3).to("cuda")
"""

GUARDED_PY = """\
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
"""


class TestDependencies(unittest.TestCase):
    def test_parse_requirement(self):
        d = parse_requirement("Torch_Vision[extra]==0.13.1 ; python_version >= '3.8'  # comment")
        self.assertEqual((d.name, d.version, d.extras, d.marker), ("torch-vision", "0.13.1", ["extra"], "python_version >= '3.8'"))
        self.assertIsNone(parse_requirement("--extra-index-url https://x"))
        self.assertIsNone(parse_requirement("# just a comment"))
        self.assertEqual(parse_requirement("numpy>=1.20,<2").spec, ">=1.20,<2")
        self.assertIsNone(parse_requirement("numpy>=1.20").version)
        self.assertEqual(parse_requirement("-e git+https://github.com/a/b.git#egg=bee").name, "bee")
        self.assertEqual(parse_requirement("mylib @ git+https://github.com/a/mylib").url, "git+https://github.com/a/mylib")

    def test_parse_conda(self):
        self.assertEqual(parse_conda_spec("pytorch=1.12.1").version, "1.12.1")
        self.assertEqual(parse_conda_spec("cudatoolkit=11.3=h123").version, "11.3")
        d = parse_conda_spec("conda-forge::numpy>=1.2")
        self.assertEqual((d.channel, d.spec, d.version), ("conda-forge", ">=1.2", None))

    def test_detect(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {
                "requirements.txt": "-r requirements/base.txt\n--find-links https://data.pyg.org/whl\ntorchvision==0.13.1\n",
                "requirements/base.txt": "numpy==1.23.5\n",
                "requirements-dev.txt": "pytest\n",
                "setup.py": SETUP_PY,
                "pyproject.toml": PYPROJECT,
                "environment.yml": ENV_YML,
                "Pipfile": '[packages]\nrequests = "*"\n[requires]\npython_version = "3.8"\n',
                "broken/setup.cfg": "[[[not ini",
            })
            r = detect_dependencies(t)
            pkgs = r.packages()
            self.assertEqual(pkgs["numpy"], "1.23.5")
            self.assertEqual(pkgs["torchvision"], "0.13.1")
            self.assertEqual(pkgs["pytorch"], "1.12.1")
            self.assertEqual(pkgs["cudatoolkit"], "11.3")
            self.assertEqual(pkgs["wandb"], "0.15.0")
            self.assertEqual(pkgs["scipy"], "1.10.1")
            self.assertIn("tqdm", pkgs)
            self.assertIn("requests", pkgs)
            self.assertEqual(r.get("pytest")[0].group, "dev")
            self.assertEqual(r.get("matplotlib")[0].group, "viz")
            self.assertEqual(r.get("setuptools")[0].group, "build")
            self.assertEqual(r.get("pytorch")[0].manager, "conda")
            self.assertEqual(r.conda_channels, ["pytorch", "conda-forge"])
            self.assertIn("https://download.pytorch.org/whl/cu113", r.index_urls)
            self.assertIn("https://data.pyg.org/whl", r.index_urls)
            specs = {p["spec"] for p in r.python_requires}
            self.assertTrue({">=3.7", ">=3.8,<3.11", "=3.8", "==3.8"} <= specs, specs)
            self.assertIn("requirements/base.txt", r.files)
            self.assertEqual(len(r.errors), 1)  # broken setup.cfg reported, not raised
            json.loads(r.to_json())

    def test_poetry(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"pyproject.toml": (
                '[tool.poetry.dependencies]\npython = "^3.9"\ntorch = "2.0.1"\nnumpy = "^1.24"\n'
                '[tool.poetry.group.dev.dependencies]\nblack = "*"\n'
            )})
            r = detect_dependencies(t)
            self.assertEqual(r.python_requires[0]["spec"], ">=3.9,<4")
            self.assertEqual(r.packages()["torch"], "2.0.1")
            self.assertEqual(r.get("numpy")[0].spec, ">=1.24,<2")
            self.assertEqual(r.get("black")[0].group, "dev")


class TestDocumentation(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_command("python -m pip install foo"), "pip")
        self.assertEqual(classify_command("sudo apt-get install -y libgl1"), "apt")
        self.assertEqual(classify_command("conda env create -f environment.yml"), "conda")
        self.assertEqual(classify_command("python train.py"), "run")
        self.assertIsNone(classify_command("hello world"))

    def test_extract(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"README.md": README, "docs/INSTALL.rst": (
                "Install\n=======\n\nRun this::\n\n    sudo apt-get install -y libgl1\n\nDone.\n"
            )})
            r = extract_documentation(t)
            self.assertEqual(r.doc_files, ["README.md", "docs/INSTALL.rst"])
            cmds = [c.command for c in r.commands]
            self.assertIn("conda create -n robust python=3.8", cmds)
            self.assertIn("pip install -r requirements.txt", cmds)
            self.assertIn("python setup.py develop", cmds)  # split on &&
            self.assertIn("pip install -e .", cmds)  # inline code
            self.assertIn("python train.py --config configs/cifar.yaml", cmds)  # prompt stripped
            self.assertIn("sudo apt-get install -y libgl1", cmds)  # rst literal block
            self.assertNotIn("import torch  # not a shell block", cmds)
            pip = next(c for c in r.commands if c.command.startswith("pip install torch"))
            self.assertEqual(pip.section, "Installation")
            self.assertEqual(pip.kind, "pip")
            headings = [s.heading for s in r.setup_sections]
            self.assertEqual(headings, ["Requirements", "Installation", "Training", "Install"])
            self.assertFalse(r.search_used)

    def test_search_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"README.md": "# Nothing useful here\n"})
            calls = []
            fake = lambda q: calls.append(q) or [SearchResult("t", "https://x", "pip install foo")]
            r = extract_documentation(t, search=True, search_fn=fake)
            self.assertTrue(r.search_used)
            self.assertEqual(len(calls), 1)
            self.assertEqual(r.search_results[0].url, "https://x")
            # no fallback when the README already has install commands
            make_repo(Path(t), {"README.md": "```\npip install foo\n```\n"})
            r = extract_documentation(t, search=True, search_fn=fake)
            self.assertFalse(r.search_used)


class TestEnvironment(unittest.TestCase):
    def test_satisfies(self):
        self.assertTrue(satisfies("3.8", ">=3.7"))
        self.assertTrue(satisfies("3.10", ">=3.8,<3.11"))
        self.assertFalse(satisfies("3.11", ">=3.8,<3.11"))
        self.assertTrue(satisfies("3.8.10", "=3.8"))
        self.assertTrue(satisfies("3.9", "~=3.8"))
        self.assertFalse(satisfies("4.0", "~=3.8"))
        self.assertTrue(satisfies("12.1", "==12.*"))
        self.assertFalse(satisfies("3.8", "!=3.8"))

    def test_full_repo(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {
                "README.md": README, "environment.yml": ENV_YML, "setup.py": SETUP_PY,
                "train.py": TRAIN_PY, "models/net.py": "import torch.nn as nn\nclass Net(nn.Module): pass\n",
                "eval.py": GUARDED_PY,
            })
            r = detect_environment(t)
            s = r.summary()
            self.assertEqual(s["python"], "3.8")
            self.assertEqual(s["cuda"], "11.3")
            self.assertEqual(s["torch"], "1.12.1")
            self.assertNotIn("tensorflow", s)
            self.assertEqual(r.frameworks, ["torch"])
            self.assertGreater(r.resolved["python"].confidence, 0.9)
            self.assertTrue(r.gpu.uses_cuda)
            self.assertTrue(r.gpu.guarded)
            self.assertEqual(r.gpu.hardcoded_cuda, ["train.py:4", "train.py:5"])
            self.assertTrue(r.gpu.gpu_required)
            self.assertTrue(any("without an availability check" in n for n in r.notes))

    def test_dockerfile_wins_and_conflict(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {
                "Dockerfile": "FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime\n",
                "requirements.txt": "torch==2.0.1\n",
                "a.py": GUARDED_PY,
            })
            r = detect_environment(t)
            self.assertEqual(r.summary()["cuda"], "11.8")
            self.assertEqual(r.summary()["cudnn"], "8")
            # docker (0.95) beats requirements (0.9) and both are strong -> conflict
            self.assertEqual(r.summary()["torch"], "2.1.0")
            self.assertTrue(r.resolved["torch"].conflict)
            # python not stated anywhere: newest version allowed by torch 2.0/2.1 ranges
            self.assertEqual(r.summary()["python"], "3.11")

    def test_tensorflow_inference_and_cpu_only(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"requirements.txt": "tensorflow==2.13.0\n", "m.py": "import tensorflow as tf\n"})
            r = detect_environment(t)
            self.assertEqual(r.frameworks, ["tensorflow"])
            self.assertEqual(r.summary()["tensorflow"], "2.13.0")
            self.assertIsNone(r.summary()["cuda"])  # no CUDA usage -> CPU is fine
            self.assertEqual(r.resolved["cuda"].alternatives, {"11.8": 0.3})
            self.assertEqual(r.summary()["python"], "3.11")

    def test_old_torch_inference(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"requirements.txt": "torch==1.2.0\n", "a.py": "import torch\nx = torch.zeros(1).cuda()\n"})
            s = detect_environment(t).summary()
            self.assertEqual((s["torch"], s["cuda"], s["python"]), ("1.2.0", "10.0", "3.7"))

    def test_python2_and_custom_cuda(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {
                "old.py": 'print "hello"\n',
                "setup.py": "from torch.utils.cpp_extension import CUDAExtension\n",
                "kernels/k.cu": "",
            })
            r = detect_environment(t)
            self.assertEqual(r.summary()["python"], "2.7")
            self.assertTrue(r.gpu.needs_nvcc)
            self.assertIn("kernels/k.cu", r.gpu.custom_cuda_code)

    def test_ngc_image_is_not_a_torch_version(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"Dockerfile": "FROM nvcr.io/nvidia/pytorch:23.10-py3\n"})
            r = detect_environment(t)
            self.assertNotIn("torch", r.summary())


class TestInspect(unittest.TestCase):
    def test_profile(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {
                "README.md": README, "environment.yml": ENV_YML, "train.py": TRAIN_PY,
                "Dockerfile": "FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04\n",
            })
            p = inspect_repository(t)
            s = p.summary()
            self.assertEqual(s["primary_language"], "Python")
            self.assertTrue(s["existing_docker"]["present"])
            self.assertEqual(s["environment"]["cuda"]["value"], "11.3.1")
            self.assertEqual(s["environment"]["cudnn"]["value"], "8")
            self.assertIn("pip install -r requirements.txt", s["install_commands"])
            self.assertEqual(s["packages"]["pytorch"], "1.12.1")
            json.dumps(s)
            full = json.loads(p.to_json())
            self.assertEqual(set(full), {"root", "structure", "dependencies", "documentation", "environment"})


if __name__ == "__main__":
    unittest.main()
