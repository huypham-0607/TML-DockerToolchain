import tempfile
import unittest
from pathlib import Path

from src.repo.structure import inspect_structure, parse_dockerfile


def make_repo(root: Path, files: dict[str, str]):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


class TestStructure(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "myrepo"
        make_repo(self.root, {
            "README.md": "# hi",
            "LICENSE": "MIT",
            "requirements.txt": "torch==2.1.0",
            "environment.yml": "name: x",
            "train.py": "print(1)",
            "models/resnet.py": "",
            "models/layers.py": "",
            "kernels/op.cu": "",
            "configs/default.yaml": "lr: 0.1",
            "scripts/run_all.sh": "",
            "notebooks/demo.ipynb": "{}",
            ".github/workflows/ci.yml": "on: push",
            ".gitmodules": "",
            "docker/Dockerfile": (
                "FROM --platform=linux/amd64 nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base\n"
                "RUN apt-get update && \\\n    apt-get install -y python3\n"
                "FROM base\nEXPOSE 8888 6006\n"
                'ENTRYPOINT ["python"]\nCMD ["train.py"]\n'
            ),
            "docker-compose.yml": "services: {}",
            ".dockerignore": "data/",
            "env.def": "Bootstrap: docker\nFrom: ubuntu",
            "other.def": "not apptainer",
            # noise that must be ignored
            ".git/config": "",
            ".venv/lib/foo.py": "",
            "__pycache__/x.pyc": "",
            "node_modules/a/index.js": "",
        })

    def tearDown(self):
        self._tmp.cleanup()

    def test_languages(self):
        r = inspect_structure(self.root)
        self.assertEqual(r.primary_language, "Python")
        self.assertEqual(r.languages["Python"], 3)
        self.assertEqual(r.languages["CUDA"], 1)
        self.assertNotIn("JavaScript", r.languages)  # node_modules ignored

    def test_ignored_dirs(self):
        r = inspect_structure(self.root)
        self.assertNotIn(".venv", r.tree)
        self.assertNotIn("node_modules", r.tree)
        self.assertNotIn("__pycache__", r.tree)
        self.assertIn("models/", r.tree)

    def test_important_files(self):
        imp = inspect_structure(self.root).important_files
        self.assertCountEqual(imp["dependencies"], ["requirements.txt", "environment.yml"])
        self.assertEqual(imp["documentation"], ["README.md"])
        self.assertIn("train.py", imp["entry_points"])
        self.assertIn("scripts/run_all.sh", imp["entry_points"])
        self.assertEqual(imp["configs"], ["configs/default.yaml"])
        self.assertEqual(imp["license"], ["LICENSE"])
        self.assertEqual(imp["ci"], [".github/workflows/ci.yml"])
        self.assertEqual(imp["submodules"], [".gitmodules"])

    def test_docker_config(self):
        d = inspect_structure(self.root).docker
        self.assertTrue(d.present)
        self.assertEqual(len(d.dockerfiles), 1)
        df = d.dockerfiles[0]
        self.assertEqual(df.path, "docker/Dockerfile")
        self.assertEqual(df.base_images, ["nvidia/cuda:11.8.0-devel-ubuntu22.04", "base"])
        self.assertEqual(df.exposed_ports, ["8888", "6006"])
        self.assertEqual(df.entrypoint, '["python"]')
        self.assertEqual(df.cmd, '["train.py"]')
        self.assertEqual(d.compose, ["docker-compose.yml"])
        self.assertEqual(d.dockerignore, [".dockerignore"])
        self.assertEqual(d.apptainer, ["env.def"])

    def test_no_docker(self):
        with tempfile.TemporaryDirectory() as t:
            make_repo(Path(t), {"a.py": ""})
            r = inspect_structure(t)
            self.assertFalse(r.docker.present)
            self.assertFalse(r.to_dict()["docker"]["present"])

    def test_truncation(self):
        r = inspect_structure(self.root, max_files=3)
        self.assertTrue(r.truncated)
        self.assertEqual(r.total_files, 3)

    def test_to_json(self):
        s = inspect_structure(self.root).to_json()
        self.assertIn('"primary_language": "Python"', s)

    def test_not_a_dir(self):
        with self.assertRaises(NotADirectoryError):
            inspect_structure(self.root / "nope")

    def test_parse_missing_dockerfile(self):
        info = parse_dockerfile(self.root / "missing", "missing")
        self.assertEqual(info.base_images, [])


if __name__ == "__main__":
    unittest.main()
