"""
    Repositories that already ship their own Dockerfile, and images that already exist.

    No hand-made Dockerfiles exist yet for the reproduction repositories, so these tests use
    small toy repositories (TOY_REPOS below). Each one copies a situation that real research
    repositories have: a Dockerfile at the root, one under docker/ that needs the repository
    root as the build context, a CPU and a GPU variant next to a compose file, a .dockerignore,
    a Dockerfile that does not build.

    Two groups:
      - inspection / generation tests: no Docker needed, always run
      - live tests (@requires_docker): build the toy Dockerfiles for real, start containers,
        run commands in them. Skipped when no Docker daemon is reachable.

    The live toy Dockerfiles build FROM samples.BASE_IMAGE (python:3.14-slim by default;
    set TML_TEST_BASE_IMAGE to use another local image, e.g. when Docker Hub is unreachable).

    Live containers are started with gpu="none": these tests are about Dockerfiles and images,
    not GPUs, and gpu="auto" currently picks a GPU on Docker Desktop for Windows even without an
    NVIDIA card (client.gpu_available trusts the "nvidia" runtime entry). The GPU paths have
    their own tests in tests/lifecycle/.

    Run only this file:
      uv run python -m unittest tests.test_existing_dockerfiles -v
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.docker import build, generate
from src.execution.exec import execute_in_container
from src.lifecycle.image import ensure_image, resolve_image
from src.lifecycle.models import ContainerSpec, LifecycleError
from src.lifecycle.provision import provision
from src.lifecycle.status import get_status
from src.lifecycle.teardown import destroy
from src.repo.inspect import inspect_repository
from src.repo.structure import inspect_structure
from tests.lifecycle import samples
from tests.lifecycle.samples import requires_docker
from tests.test_structure import make_repo

BASE = samples.BASE_IMAGE

# name -> {relative path: content}. Add new toy repositories here.
TOY_REPOS = {
    # The common case: a Dockerfile at the root that copies the repository and has a CMD.
    # The .dockerignore keeps a secret and a checkpoint out of the image.
    "own-dockerfile": {
        "Dockerfile": (
            f"FROM {BASE}\n"
            "WORKDIR /workspace\n"
            "COPY . /workspace\n"
            "ENV TOY_MARK=own-dockerfile\n"
            'CMD ["python", "train.py"]\n'
        ),
        ".dockerignore": "secrets.txt\n*.ckpt\n",
        "train.py": "print('toy training ok')\n",
        "secrets.txt": "do not ship me\n",
        "weights.ckpt": "fake checkpoint\n",
        "README.md": "# toy repo with its own Dockerfile\n",
    },
    # Dockerfile under docker/ whose COPY lines are relative to the repository root:
    # it only builds when the context is the repository, not the Dockerfile's directory.
    "nested-dockerfile": {
        "docker/Dockerfile": (
            f"FROM {BASE}\n"
            "WORKDIR /workspace\n"
            "COPY src/ /workspace/src/\n"
            "COPY run.py /workspace/run.py\n"
            'CMD ["python", "run.py"]\n'
        ),
        "src/toy/__init__.py": "VALUE = 42\n",
        "run.py": "import sys; sys.path.insert(0, 'src'); from toy import VALUE; print('value', VALUE)\n",
    },
    # A CPU Dockerfile, a GPU variant on a CUDA base image, and a compose file.
    # Only the CPU one is built; the CUDA image is never pulled.
    "two-dockerfiles": {
        "Dockerfile": f"FROM {BASE}\nWORKDIR /workspace\nCOPY . /workspace\n",
        "Dockerfile.gpu": (
            "FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04\n"
            "RUN apt-get update && apt-get install -y python3\n"
            "EXPOSE 8888\n"
        ),
        "docker-compose.yml": "services:\n  train:\n    build: .\n",
        "main.py": "print('cpu variant')\n",
    },
    # A Dockerfile whose build fails with a recognisable message.
    "broken-dockerfile": {
        "Dockerfile": f"FROM {BASE}\nRUN echo toy build failure here && exit 3\n",
        "main.py": "print('never runs')\n",
    },
    # Inspection only: a pinned python base image is evidence for the Python version.
    "python-base": {
        "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\n",
        "main.py": "print(1)\n",
    },
}


class ToyRepos:
    """Writes the toy repositories into one temporary directory per test class."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="tml-toys-")
        self.root = Path(self._tmp.name).resolve()

    def make(self, name: str, copy: str = "") -> Path:
        """The toy repository `name`. With `copy`, a separate copy in its own directory: a new
        build context path, so lifecycle.image gives it a new tag and nothing is cached yet."""
        path = self.root / (f"{name}-{copy}" if copy else name)
        if not path.exists():
            make_repo(path, TOY_REPOS[name])
        return path

    def cleanup(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------- inspection, no Docker

class TestInspectionFindsExistingDockerConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toys = ToyRepos()

    @classmethod
    def tearDownClass(cls):
        cls.toys.cleanup()

    def test_root_dockerfile_and_dockerignore(self):
        docker = inspect_structure(self.toys.make("own-dockerfile")).docker
        self.assertTrue(docker.present)
        self.assertEqual([d.path for d in docker.dockerfiles], ["Dockerfile"])
        df = docker.dockerfiles[0]
        self.assertEqual(df.base_images, [BASE])
        self.assertIn("train.py", df.cmd)
        self.assertEqual(docker.dockerignore, [".dockerignore"])

    def test_nested_dockerfile_path(self):
        docker = inspect_structure(self.toys.make("nested-dockerfile")).docker
        self.assertEqual([d.path for d in docker.dockerfiles], ["docker/Dockerfile"])

    def test_cpu_and_gpu_variants_and_compose(self):
        docker = inspect_structure(self.toys.make("two-dockerfiles")).docker
        by_path = {d.path: d for d in docker.dockerfiles}
        self.assertEqual(set(by_path), {"Dockerfile", "Dockerfile.gpu"})
        self.assertEqual(by_path["Dockerfile.gpu"].base_images, ["nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"])
        self.assertIn("8888", by_path["Dockerfile.gpu"].exposed_ports)
        self.assertEqual(docker.compose, ["docker-compose.yml"])

    def test_cuda_base_image_is_evidence_for_the_cuda_version(self):
        env = inspect_repository(self.toys.make("two-dockerfiles")).environment
        self.assertEqual(env.resolved["cuda"].value, "11.8.0")   # full version, as in test_inspection

    def test_python_base_image_is_evidence_for_the_python_version(self):
        env = inspect_repository(self.toys.make("python-base")).environment
        self.assertEqual(env.resolved["python"].value, "3.11")


class TestGenerateWithExistingDockerfile(unittest.TestCase):
    def setUp(self):
        self.toys = ToyRepos()
        self.runs = mock.patch.object(generate, "RUNS_DIR", self.toys.root / "runs")
        self.runs.start()

    def tearDown(self):
        self.runs.stop()
        self.toys.cleanup()

    def test_notes_name_the_existing_dockerfiles(self):
        _, _, notes = generate.generate_dockerfile(self.toys.make("two-dockerfiles"))
        note = next((n for n in notes if "own Docker config" in n), "")
        self.assertIn("Dockerfile", note)
        self.assertIn("Dockerfile.gpu", note)

    def test_repo_dockerfile_is_not_modified(self):
        repo = self.toys.make("own-dockerfile")
        before = (repo / "Dockerfile").read_text()
        _, path, _ = generate.generate_dockerfile(repo)
        self.assertEqual((repo / "Dockerfile").read_text(), before)
        self.assertNotEqual(path.resolve(), (repo / "Dockerfile").resolve())   # written to runs/, not the repo

    def test_current_behavior_generates_a_template_anyway(self):
        """CURRENT behavior, not necessarily the wanted one: the repository's own Dockerfile is
        only mentioned in a note, and the starter Dockerfile is still the python:X-slim template.
        If the team decides the toolchain should start from the repository's Dockerfile, change
        this test to assert that instead."""
        text, _, _ = generate.generate_dockerfile(self.toys.make("own-dockerfile"))
        self.assertTrue(text.startswith("FROM python:"))
        self.assertNotIn("TOY_MARK", text)


# ---------------------------------------------------------------- live: building

@requires_docker
class TestBuildRepoDockerfile(unittest.TestCase):
    """src.docker.build.build_image (the agent's build_image tool) with a repository's own Dockerfile."""

    @classmethod
    def setUpClass(cls):
        cls.toys = ToyRepos()
        cls.runs = mock.patch.object(build, "RUNS_DIR", cls.toys.root / "runs")
        cls.runs.start()
        cls.tags: list[str] = []

    @classmethod
    def tearDownClass(cls):
        cls.runs.stop()
        # build_image does not label its images, so remove them by tag
        for tag in cls.tags:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
        cls.toys.cleanup()

    def tag(self, name: str) -> str:
        t = f"tml-test/{name}-{samples.RUN_ID}:latest"
        self.tags.append(t)
        return t

    def test_builds_from_the_repo_dockerfile(self):
        repo = self.toys.make("own-dockerfile")
        before = (repo / "Dockerfile").read_text()
        r = build.build_image(repo, before, tag=self.tag("own"))
        self.assertTrue(r.success, r.log_tail)
        self.assertEqual(r.exit_code, 0)
        inspect = subprocess.run(["docker", "image", "inspect", r.image], capture_output=True)
        self.assertEqual(inspect.returncode, 0, "the image should exist after a successful build")
        self.assertEqual((repo / "Dockerfile").read_text(), before)

    def test_failed_build_reports_the_failing_step(self):
        repo = self.toys.make("broken-dockerfile")
        r = build.build_image(repo, (repo / "Dockerfile").read_text(), tag=self.tag("broken"))
        self.assertFalse(r.success)
        self.assertIsNone(r.image)
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("toy build failure here", r.log_tail)


# ---------------------------------------------------------------- live: images that already exist

@requires_docker
class TestExistingImages(unittest.TestCase):
    """lifecycle.image: an image that is already on the machine is reused, never pulled."""

    @classmethod
    def setUpClass(cls):
        cls.toys = ToyRepos()

    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()
        cls.toys.cleanup()

    def test_second_ensure_reuses_the_image(self):
        df = self.toys.make("own-dockerfile", copy="reuse") / "Dockerfile"
        first = ensure_image(df, labels=samples.TEST_LABEL)
        second = ensure_image(df, labels=samples.TEST_LABEL)
        self.assertTrue(first.built)
        self.assertFalse(second.built)
        self.assertEqual((first.tag, first.id), (second.tag, second.id))

    def test_rebuild_builds_again(self):
        df = self.toys.make("own-dockerfile") / "Dockerfile"
        ensure_image(df, labels=samples.TEST_LABEL)
        self.assertTrue(ensure_image(df, labels=samples.TEST_LABEL, rebuild=True).built)

    def test_provision_from_an_existing_image_tag(self):
        image = ensure_image(self.toys.make("own-dockerfile") / "Dockerfile", labels=samples.TEST_LABEL)
        spec = ContainerSpec(image=image.tag, name=samples.container_name("existing-image"),
                             gpu="none", labels=samples.TEST_LABEL)
        r = provision(spec)
        self.assertTrue(r.ok, r.status.reason)
        self.assertFalse(r.image.built)
        self.assertEqual(r.handle.image_id, image.id)

    def test_missing_image_is_not_pulled(self):
        missing = f"tml-test/never-built-{samples.RUN_ID}:x"
        with self.assertRaises(LifecycleError) as ctx:
            resolve_image(missing)
        self.assertEqual(ctx.exception.code, "image_not_found")

    def test_provision_with_a_missing_image_creates_no_container(self):
        name = samples.container_name("missing-image")
        spec = ContainerSpec(image=f"tml-test/never-built-{samples.RUN_ID}:x", name=name,
                             gpu="none", labels=samples.TEST_LABEL)
        with self.assertRaises(LifecycleError) as ctx:
            provision(spec)
        self.assertEqual(ctx.exception.code, "image_not_found")
        self.assertEqual(get_status(name).health, "missing")


@requires_docker
class TestBuildContext(unittest.TestCase):
    """docker/Dockerfile with COPY lines relative to the repository root."""

    @classmethod
    def setUpClass(cls):
        cls.toys = ToyRepos()
        cls.repo = cls.toys.make("nested-dockerfile")

    @classmethod
    def tearDownClass(cls):
        cls.toys.cleanup()

    def test_default_context_fails(self):
        # default context = the Dockerfile's directory (docker/), which has no src/ or run.py
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.repo / "docker" / "Dockerfile", labels=samples.TEST_LABEL)
        self.assertEqual(ctx.exception.code, "build_failed")

    def test_repo_root_context_builds(self):
        ref = ensure_image(self.repo / "docker" / "Dockerfile", context=self.repo, labels=samples.TEST_LABEL)
        self.assertTrue(ref.id)


# ---------------------------------------------------------------- live: repository -> container -> command

@requires_docker
class TestRepoDockerfileEndToEnd(unittest.TestCase):
    """The first milestone with a repository's own Dockerfile:
    Dockerfile -> image -> running container -> command runs -> container destroyed."""

    @classmethod
    def setUpClass(cls):
        cls.toys = ToyRepos()

    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()
        cls.toys.cleanup()

    def start(self, toy: str, suffix: str, dockerfile: str = "Dockerfile", context: str = ""):
        repo = self.toys.make(toy)
        spec = ContainerSpec(dockerfile=str(repo / dockerfile), context=context,
                             name=samples.container_name(suffix), gpu="none", labels=samples.TEST_LABEL)
        r = provision(spec)
        self.assertTrue(r.ok, r.status.reason)
        return r.handle.name

    def test_repo_code_runs_in_the_container(self):
        name = self.start("own-dockerfile", "e2e-own")
        out = execute_in_container(name, "python train.py")
        self.assertEqual(out.exit_code, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "toy training ok")

    def test_dockerfile_env_reaches_the_container(self):
        name = self.start("own-dockerfile", "e2e-env")
        out = execute_in_container(name, "echo $TOY_MARK")
        self.assertEqual(out.stdout.strip(), "own-dockerfile")

    def test_dockerignore_is_respected(self):
        name = self.start("own-dockerfile", "e2e-ignore")
        out = execute_in_container(name, "ls -a /workspace")
        files = set(out.stdout.split())
        self.assertIn("train.py", files)
        self.assertNotIn("secrets.txt", files)
        self.assertNotIn("weights.ckpt", files)

    def test_nested_dockerfile_with_repo_context(self):
        repo = self.toys.make("nested-dockerfile")
        name = self.start("nested-dockerfile", "e2e-nested", dockerfile="docker/Dockerfile", context=str(repo))
        out = execute_in_container(name, "python run.py")
        self.assertEqual(out.stdout.strip(), "value 42")

    def test_cpu_variant_of_a_repo_that_also_has_a_gpu_dockerfile(self):
        name = self.start("two-dockerfiles", "e2e-cpu")
        out = execute_in_container(name, ["python", "main.py"])
        self.assertEqual(out.stdout.strip(), "cpu variant")

    def test_failing_command_reports_its_exit_code(self):
        name = self.start("own-dockerfile", "e2e-fail")
        out = execute_in_container(name, "python -c 'import sys; sys.exit(5)'")
        self.assertEqual(out.exit_code, 5)

    def test_destroy_removes_the_container(self):
        name = self.start("own-dockerfile", "e2e-destroy")
        destroy(name, force=True)
        self.assertEqual(get_status(name).health, "missing")


if __name__ == "__main__":
    unittest.main()


@requires_docker
class TestStaleContainerLive(unittest.TestCase):
    """Real daemon: rebuild a tag while a container from the old build is running."""

    def test_old_container_warns_new_one_does_not(self):
        from src.execution.tools import stale_image_warning

        tag = f"tml-test/stale-{samples.RUN_ID}:latest"
        names = [samples.container_name("stale-old"), samples.container_name("stale-new")]
        with tempfile.TemporaryDirectory() as t:
            ctx = Path(t)
            try:
                for name, version in zip(names, ("1", "2")):
                    (ctx / "Dockerfile").write_text(f"FROM {BASE}\nENV BUILD={version}\nLABEL tml.test={samples.RUN_ID}\n")
                    subprocess.run(["docker", "build", "-q", "-t", tag, str(ctx)], check=True, capture_output=True)
                    subprocess.run(["docker", "run", "-d", "--name", name, "--label", f"tml.test={samples.RUN_ID}",
                                    tag, "sleep", "120"], check=True, capture_output=True)
                self.assertIn("OLDER build", stale_image_warning(names[0]))
                self.assertIsNone(stale_image_warning(names[1]))
            finally:
                subprocess.run(["docker", "rm", "-f", *names], capture_output=True)
                subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
