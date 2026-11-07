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
        self.assertIn(f"pip install -r /tmp/requirements-0.txt -c {generate.TORCH_CONSTRAINTS}", text)
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
        self.assertIn("COPY requirements.txt /tmp/requirements-src-0.txt\nRUN pip install -r /tmp/requirements-src-0.txt", text)
        self.assertNotIn("grep", text)

    def test_requirements_are_installed_before_the_repo_is_copied(self):
        # a rebuild after a change to a later line then reuses the cached install layers
        text, _, _ = self.gen("order", {"requirements.txt": "numpy==1.24.2\n", "a.py": "import numpy, tqdm\n"})
        lines = text.splitlines()
        self.assertLess(lines.index("RUN pip install -r /tmp/requirements-src-0.txt"), lines.index("COPY . /workspace"))
        guess = next(i for i, x in enumerate(lines) if x.startswith("RUN for p in"))
        self.assertLess(guess, lines.index("COPY . /workspace"))

    def test_requirements_that_need_the_repo_are_installed_after_copy(self):
        text, _, _ = self.gen("local", {"requirements.txt": "-r base.txt\n-e .\n", "base.txt": "numpy\n",
                                        "setup.py": "from setuptools import setup\nsetup(name='x')\n"})
        lines = text.splitlines()
        self.assertNotIn("COPY requirements.txt /tmp/requirements-src-0.txt", lines)
        self.assertGreater(lines.index("RUN pip install -r requirements.txt"), lines.index("COPY . /workspace"))


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
        text, _, _ = self.gen("ffb", {**self.FFB, "a.py": "import folktables\n"})
        lines = text.splitlines()
        freeze = next(i for i, x in enumerate(lines) if "pip freeze" in x)
        req = next(i for i, x in enumerate(lines) if "> /tmp/requirements-0.txt" in x)
        guess = next(i for i, x in enumerate(lines) if x.startswith("RUN for p in"))
        self.assertLess(freeze, req)
        self.assertIn(f"-c {generate.TORCH_CONSTRAINTS}", lines[req])     # torch lines filtered AND pinned
        self.assertIn(f"-c {generate.TORCH_CONSTRAINTS}", lines[guess])

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


class TestNoRequirementsFile(GenerateCase):
    """IPMix / EP-BNP case: no dependency file, so the template used to install nothing at all."""

    REPO = {
        "train.py": "import torch, torchvision\nimport numpy as np\nfrom PIL import Image\nimport tqdm\n"
                    "import utils\nfrom models.resnet import net\nimport os, json\n",
        "utils.py": "import torchattacks\n",
        "models/resnet.py": "import torch.nn as nn\n",
    }

    def test_torch_imported_but_not_pinned_is_installed(self):
        text, _, _ = self.gen("noreq", self.REPO)
        self.assertIn(f'{CPU_INDEX} "torch" "torchvision"', text)

    def test_other_imports_are_installed_best_effort(self):
        text, _, notes = self.gen("noreq", self.REPO)
        line = next(x for x in text.splitlines() if x.startswith("RUN for p in"))
        self.assertEqual(line.split(" in ")[1].split(";")[0].split(), ["Pillow", "numpy", "torchattacks", "tqdm"])
        self.assertIn('|| echo "could not install', line)          # a wrong guess does not fail the build
        self.assertTrue(any("guessed" in n or "from imports" in n for n in notes))
        for local_or_stdlib in ("utils", "models", "os", "json"):
            self.assertNotIn(f" {local_or_stdlib} ", f" {line} ")

    def test_declared_packages_are_not_guessed_again(self):
        text, _, _ = self.gen("partial", {"requirements.txt": "Pillow==9.4.0\nnumpy==1.24.2\n",
                                          "a.py": "from PIL import Image\nimport numpy\nimport folktables\n"})
        line = next(x for x in text.splitlines() if x.startswith("RUN for p in"))
        self.assertEqual(line.split(" in ")[1].split(";")[0].split(), ["folktables"])   # FFB's missing package

    def test_python_falls_back_to_310_not_newest(self):
        text, _, _ = self.gen("fstring", {"a.py": "x = 1\nprint(f'{x}')\n"})      # only "needs >= 3.6"
        self.assertIn("FROM python:3.10-slim", text)


class TestTorchCompanions(GenerateCase):
    def test_mismatched_torchvision_is_repinned(self):
        # UAP: torch==2.2.0 with torchvision==0.16.0, which needs torch 2.1
        text, _, notes = self.gen("uap", {"requirements.txt": "torch==2.2.0\ntorchvision==0.16.0\n"})
        self.assertIn('"torch==2.2.0" "torchvision==0.17.*"', text)
        self.assertTrue(any("torchvision==0.16.0 does not work with torch==2.2.0" in n for n in notes))

    def test_matching_pair_is_kept(self):
        text, _, notes = self.gen("ok", {"requirements.txt": "torch==1.13.1+cu116\ntorchvision==0.14.1+cu116\ntorchaudio==0.13.1\n"})
        self.assertIn('"torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"', text)
        self.assertFalse(any("does not work with torch==" in n for n in notes))

    def test_unknown_torch_version_is_left_alone(self):
        pins, notes = generate.fix_companions(["torch==9.9.0", "torchvision==0.1.0"])
        self.assertEqual((pins, notes), (["torch==9.9.0", "torchvision==0.1.0"], []))

    def test_old_torch_gets_numpy_below_2(self):
        text, _, notes = self.gen("old", {"requirements.txt": "torch==1.12.1\n"})
        self.assertIn('"numpy<2"', text)
        self.assertTrue(any("numpy<2" in n for n in notes))

    def test_pinned_numpy_is_not_overridden(self):
        text, _, _ = self.gen("oldnp", {"requirements.txt": "torch==1.12.1\nnumpy==1.21.0\n"})
        self.assertNotIn("numpy<2", text)

    def test_new_torch_keeps_numpy_free(self):
        text, _, _ = self.gen("new", {"requirements.txt": "torch==2.4.0\n"})
        self.assertNotIn("numpy<2", text)


class TestFixesFromTheAgentRuns(GenerateCase):
    """What the agent had to find by hand in the FFB and UAP runs (Sat 7 Nov), now in the template."""

    def test_tkinter_import_adds_libtk(self):
        # FFB: src/dataset.py starts with `from tkinter import S`
        text, _, notes = self.gen("tk", {"dataset.py": "from tkinter import S\nimport numpy\n"})
        self.assertIn("--no-install-recommends build-essential git libtk8.6", text)
        self.assertTrue(any("tkinter" in n for n in notes))

    def test_no_tkinter_no_libtk(self):
        text, _, _ = self.gen("notk", {"a.py": "import numpy\n"})
        self.assertNotIn("libtk", text)

    def test_wandb_runs_disabled(self):
        text, _, notes = self.gen("wb", {"train.py": "import wandb\nwandb.init()\n"})
        self.assertIn("WANDB_MODE=disabled", text.splitlines()[1])
        self.assertTrue(any("wandb" in n for n in notes))

    def test_duplicate_pins_and_module_clash_are_cleaned(self):
        # UAP's requirements.txt, as published
        uap = ("adversarial_robustness_toolbox==1.17.1\nart==6.1\nmatplotlib==3.7.5\nnumpy==1.24.3\n"
               "Pillow==10.1.0\nPillow==10.2.0\nscikit_learn==1.3.2\nscikit_learn==1.3.0\n"
               "torch==2.2.0\ntorchvision==0.16.0\ntqdm==4.66.1\n")
        text, _, notes = self.gen("uapreq", {"requirements.txt": uap, "a.py": "import art, torch\n"})
        line = next(x for x in text.splitlines() if x.startswith("RUN printf"))
        self.assertIn("'Pillow==10.2.0'", line)
        self.assertNotIn("Pillow==10.1.0", line)
        self.assertIn("'scikit_learn==1.3.2'", line)
        self.assertNotIn("scikit_learn==1.3.0", line)
        self.assertNotIn("'art==6.1'", line)
        self.assertIn("'adversarial_robustness_toolbox==1.17.1'", line)
        self.assertTrue(any("pillow is listed 2 times" in n for n in notes))
        self.assertTrue(any("both install the module `art`" in n for n in notes))

    def test_clean_requirements_leaves_clean_files_alone(self):
        self.assertEqual(generate.clean_requirements("numpy==1.24.2\npandas\n"), (None, []))

    def test_quoting_survives_shell_characters(self):
        cleaned, _ = generate.clean_requirements("a==1\na==2\nb; python_version < '3.9'\n")
        quoted = " ".join(generate._sh_quote(x) for x in cleaned)
        import shlex
        self.assertEqual(shlex.split(quoted), ["a==2", "b; python_version < '3.9'"])

    def test_repo_that_imports_itself_gets_a_package_layout(self):
        # UAP: main.py has `from uap.train.model_structure import get_model`
        text, _, notes = self.gen("uap", {"main.py": "from uap.train.model_structure import get_model\n",
                                          "train/model_structure.py": "import models\n", "models/__init__.py": ""})
        lines = text.splitlines()
        self.assertIn("COPY . /workspace/uap", lines)
        self.assertIn("ENV PYTHONPATH=/workspace:/workspace/uap", lines)
        self.assertIn("WORKDIR /workspace/uap", lines)
        self.assertNotIn("COPY . /workspace", lines)
        self.assertTrue(any("imports the repository itself" in n for n in notes))

    def test_other_repos_keep_workspace(self):
        text, _, _ = self.gen("plainrepo", {"main.py": "import os\n"})
        self.assertIn("COPY . /workspace", text.splitlines())
        self.assertNotIn("PYTHONPATH", text)
