import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from docker.credentials.errors import StoreError

from src.lifecycle import client, image
from src.lifecycle.image import (
    BUILDER_ENV, build_command, buildkit_available, choose_builder, ensure_image, image_tag,
    needs_buildkit, resolve_image,
)
from src.lifecycle.models import LABEL_BUILT_BY, LABEL_TEST, LifecycleError
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeImage, use_fake
from tests.lifecycle.samples import requires_docker

BUILDX_MISSING = (
    "ERROR: BuildKit is enabled but the buildx component is missing or broken.\n"
    "       Install the buildx component to build images with BuildKit:\n"
    "       https://docs.docker.com/go/buildx/\n"
)


def builder_env(test: unittest.TestCase, value: str | None) -> None:
    """Sets TML_BUILDER for one test (None removes it)."""
    env = {k: v for k, v in os.environ.items() if k != BUILDER_ENV}
    if value is not None:
        env[BUILDER_ENV] = value
    patcher = mock.patch.dict(os.environ, env, clear=True)
    patcher.start()
    test.addCleanup(patcher.stop)


def buildx_answer(test: unittest.TestCase, works: bool | None = None, **outcome) -> mock.Mock:
    """Fakes `docker buildx version` (and forgets an earlier answer)."""
    patcher = mock.patch.object(image, "_buildkit_cache", None)
    patcher.start()
    test.addCleanup(patcher.stop)
    if works is not None:
        outcome = {"return_value": subprocess.CompletedProcess(args=[], returncode=0 if works else 1)}
    run = mock.patch.object(image.subprocess, "run", **outcome)
    test.addCleanup(run.stop)
    return run.start()


def tearDownModule():
    samples.cleanup()


# ---------------------------------------------------------------- pure suites (no daemon)

class TestTagging(unittest.TestCase):
    TEXT = "FROM python:3.14-slim\nCMD [\"python\"]\n"

    def test_format(self):
        tag = image_tag(self.TEXT, "/repos/robust-training")
        self.assertRegex(tag, r"^tml-local/robust-training:[0-9a-f]{12}$")

    def test_same_input_gives_the_same_tag(self):
        self.assertEqual(image_tag(self.TEXT, "/repos/a"), image_tag(self.TEXT, "/repos/a"))

    def test_changes_with_the_dockerfile_text(self):
        self.assertNotEqual(image_tag(self.TEXT, "/repos/a"), image_tag(self.TEXT + "RUN true\n", "/repos/a"))

    def test_changes_with_the_context_path(self):
        # two checkouts with the same directory name and the same Dockerfile must not share an image
        a, b = image_tag(self.TEXT, "/one/repo"), image_tag(self.TEXT, "/two/repo")
        self.assertEqual(a.split(":")[0], b.split(":")[0])
        self.assertNotEqual(a, b)

    def test_name_overrides_the_directory_name(self):
        self.assertTrue(image_tag(self.TEXT, "/repos/a", name="ipmix").startswith("tml-local/ipmix:"))

    def test_name_becomes_a_valid_repository(self):
        for raw, slug in (("My Repo_v2", "my-repo-v2"), ("--Odd..Name--", "odd-name"), ("ÜBER", "ber"), ("___", "image")):
            self.assertEqual(image_tag(self.TEXT, "/x", name=raw).split(":")[0], f"tml-local/{slug}")

    def test_relative_context_is_resolved(self):
        self.assertEqual(image_tag(self.TEXT, "."), image_tag(self.TEXT, Path.cwd()))


class TestBuildCommand(unittest.TestCase):
    def command(self, labels=None) -> list[str]:
        return build_command(Path("/repo/docker/Dockerfile"), Path("/repo"), "tml-local/repo:abc", labels)

    def test_one_plain_build(self):
        cmd = self.command()
        self.assertEqual(cmd[:2], ["docker", "build"])
        self.assertEqual(cmd[cmd.index("--file") + 1], "/repo/docker/Dockerfile")
        self.assertEqual(cmd[cmd.index("--tag") + 1], "tml-local/repo:abc")
        self.assertEqual(cmd[-1], "/repo")                 # the context is the last argument
        self.assertNotIn("--pull", cmd)
        self.assertNotIn("--no-cache", cmd)

    def test_always_labelled_as_built_by_lifecycle(self):
        self.assertIn(f"{LABEL_BUILT_BY}=lifecycle", self.command())

    def test_extra_labels(self):
        cmd = self.command({LABEL_TEST: "run1"})
        labels = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--label"]
        self.assertEqual(set(labels), {f"{LABEL_TEST}=run1", f"{LABEL_BUILT_BY}=lifecycle"})

    def test_extra_label_cannot_replace_the_built_by_label(self):
        cmd = self.command({LABEL_BUILT_BY: "someone-else"})
        labels = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--label"]
        self.assertEqual(labels, [f"{LABEL_BUILT_BY}=lifecycle"])


class TestPathChecks(unittest.TestCase):
    """ensure_image refuses bad paths before it talks to Docker."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.dockerfile = self.root / "Dockerfile"
        self.dockerfile.write_text("FROM scratch\n")
        # any Docker call in these tests is a bug
        patcher = mock.patch.object(client, "get_client", side_effect=AssertionError("Docker was called"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertInvalid(self, fragment: str, *args, **kwargs):
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(*args, **kwargs)
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertIn(fragment, ctx.exception.message)

    def test_missing_dockerfile(self):
        self.assertInvalid("Dockerfile not found", self.root / "nope" / "Dockerfile")

    def test_dockerfile_is_a_directory(self):
        self.assertInvalid("Dockerfile not found", self.root)

    def test_context_is_not_a_directory(self):
        self.assertInvalid("not a directory", self.dockerfile, context=self.root / "nope")

    def test_filesystem_root_as_context(self):
        self.assertInvalid("too broad", self.dockerfile, context="/")

    def test_home_directory_as_context(self):
        self.assertInvalid("too broad", self.dockerfile, context=Path.home())


class TestResolveImage(unittest.TestCase):
    def test_existing_image(self):
        use_fake(self, FakeClient(images=[FakeImage("demo:latest", "sha256:" + "ab" * 32)]))
        ref = resolve_image("demo:latest")
        self.assertEqual((ref.id, ref.tag, ref.built), ("abababababab", "demo:latest", False))

    def test_missing_image_is_never_pulled(self):
        fake = use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            resolve_image("pytorch/pytorch:latest")
        self.assertEqual(ctx.exception.code, "image_not_found")
        self.assertIn("does not pull", ctx.exception.hint)
        self.assertFalse(hasattr(fake.images, "pull"))     # the fake has no pull: a call would have failed


class TestBuildSyntax(unittest.TestCase):
    def test_buildkit_only_syntax(self):
        for text in (
            "# syntax=docker/dockerfile:1\nFROM python:3.12",
            "FROM python:3.12\nRUN --mount=type=cache,target=/root/.cache/pip pip install torch",
            "FROM python:3.12\nRUN <<EOF\npip install torch\nEOF",
            "FROM python:3.12\nCOPY --link . /app",
            "FROM python:3.12\nCOPY --chmod=755 run.sh /run.sh",
            "FROM python:3.12\nRUN --network=none make",
        ):
            self.assertTrue(needs_buildkit(text), text)

    def test_classic_syntax(self):
        text = "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"train.py\"]"
        self.assertFalse(needs_buildkit(text))


class TestBuilderChoice(unittest.TestCase):
    def test_auto_prefers_buildkit(self):
        builder_env(self, None)
        buildx_answer(self, works=True)
        self.assertEqual(choose_builder(), "buildkit")

    def test_auto_without_buildx(self):
        builder_env(self, "auto")
        buildx_answer(self, works=False)
        self.assertEqual(choose_builder(), "legacy")

    def test_auto_without_the_docker_command(self):
        builder_env(self, "")
        buildx_answer(self, side_effect=FileNotFoundError("docker"))
        self.assertEqual(choose_builder(), "legacy")

    def test_detection_runs_once(self):
        run = buildx_answer(self, works=True)
        self.assertTrue(buildkit_available())
        self.assertTrue(buildkit_available())
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["docker", "buildx", "version"])

    def test_detection_that_hangs(self):
        buildx_answer(self, side_effect=subprocess.TimeoutExpired(cmd="docker buildx version", timeout=30))
        self.assertFalse(buildkit_available())

    def test_forced_legacy(self):
        builder_env(self, "LEGACY")
        run = buildx_answer(self, works=True)
        self.assertEqual(choose_builder(), "legacy")
        run.assert_not_called()                                       # no need to look

    def test_forced_buildkit(self):
        builder_env(self, "buildkit")
        buildx_answer(self, works=True)
        self.assertEqual(choose_builder(), "buildkit")

    def test_forced_buildkit_without_buildx(self):
        builder_env(self, "buildkit")
        buildx_answer(self, works=False)
        with self.assertRaises(LifecycleError) as ctx:
            choose_builder()
        self.assertEqual(ctx.exception.code, "docker_unavailable")
        self.assertIn("docker-buildx", ctx.exception.hint)

    def test_unknown_builder(self):
        builder_env(self, "kaniko")
        with self.assertRaises(LifecycleError) as ctx:
            choose_builder()
        self.assertEqual(ctx.exception.code, "invalid_argument")


class TestBuildWithoutDaemon(unittest.TestCase):
    """The BuildKit build step with `docker build` replaced by a mock."""

    def setUp(self):
        for name, value in (("choose_builder", mock.Mock(return_value="buildkit")), ("_buildkit_cache", True)):
            patcher = mock.patch.object(image, name, value)     # a test that finds buildx broken must not change later tests
            patcher.start()
            self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dockerfile = Path(self._tmp.name) / "repo" / "Dockerfile"
        self.dockerfile.parent.mkdir()
        self.dockerfile.write_text("FROM scratch\n")
        self.tag = image_tag("FROM scratch\n", self.dockerfile.parent)

    def run_build(self, images=(), **outcome) -> mock.Mock:
        self.fake = use_fake(self, FakeClient(images=images))
        patcher = mock.patch.object(image.subprocess, "run", **outcome)
        run = patcher.start()
        self.addCleanup(patcher.stop)
        return run

    def completed(self, returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)

    def test_existing_tag_skips_the_build(self):
        run = self.run_build(images=[FakeImage(self.tag)])
        ref = ensure_image(self.dockerfile)
        self.assertEqual((ref.tag, ref.built), (self.tag, False))
        run.assert_not_called()

    def test_rebuild_builds_even_if_the_tag_exists(self):
        run = self.run_build(images=[FakeImage(self.tag)], return_value=self.completed(0))
        ref = ensure_image(self.dockerfile, rebuild=True)
        self.assertEqual((ref.built, ref.builder), (True, "buildkit"))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], ["docker", "build"])
        self.assertEqual(self.fake.api.build_calls, [])                 # the classic builder was not used

    def test_failed_build(self):
        log = "#5 [2/2] RUN pip install nope\n#5 ERROR: No matching distribution found for nope\n"
        self.run_build(return_value=self.completed(1, log))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        e = ctx.exception
        self.assertEqual(e.code, "build_failed")
        self.assertIn("exit code 1", e.message)
        self.assertIn("No matching distribution found for nope", e.message)
        self.assertIn("Dockerization", e.hint)

    def test_failed_build_keeps_only_the_end_of_the_log(self):
        log = "\n".join(f"#1 line {i}" for i in range(1000)) + "\nTHE LAST LINE\n"
        self.run_build(return_value=self.completed(1, log))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        message = ctx.exception.message
        self.assertIn("THE LAST LINE", message)
        self.assertNotIn("#1 line 0\n", message)
        self.assertLess(len(message), image.MAX_LOG_CHARS + 300)

    def test_daemon_down_during_build(self):
        self.run_build(return_value=self.completed(1, "ERROR: Cannot connect to the Docker daemon at unix:///var/run/docker.sock."))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        self.assertEqual(ctx.exception.code, "docker_unavailable")

    def test_docker_command_not_installed_falls_back_to_the_classic_builder(self):
        self.run_build(side_effect=FileNotFoundError("docker"))
        ref = ensure_image(self.dockerfile)
        self.assertEqual((ref.built, ref.builder), (True, "legacy"))
        self.assertEqual(len(self.fake.api.build_calls), 1)

    def test_broken_buildx_falls_back_to_the_classic_builder(self):
        self.run_build(return_value=self.completed(1, BUILDX_MISSING))
        ref = ensure_image(self.dockerfile)
        self.assertEqual(ref.builder, "legacy")
        self.assertFalse(buildkit_available())                          # remembered for the next build

    def test_socket_permission(self):
        message = "ERROR: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock"
        self.run_build(return_value=self.completed(1, message))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        self.assertEqual(ctx.exception.code, "docker_unavailable")
        self.assertIn("docker group", ctx.exception.hint)

    def test_daemon_down_in_newer_wording(self):
        message = "failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and if the daemon is running"
        self.run_build(return_value=self.completed(1, message))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        self.assertEqual(ctx.exception.code, "docker_unavailable")

    def test_build_timeout(self):
        self.run_build(side_effect=subprocess.TimeoutExpired(cmd="docker build", timeout=5))
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile, timeout=5)
        self.assertEqual(ctx.exception.code, "timeout")
        self.assertIn("5 s", ctx.exception.message)

    def test_success_without_an_image_is_an_error(self):
        self.run_build(return_value=self.completed(0))      # "succeeded", but the fake has no such image
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(self.dockerfile)
        self.assertEqual(ctx.exception.code, "docker_error")


class TestClassicBuildWithoutDaemon(unittest.TestCase):
    """The classic-builder step against a fake low-level API."""

    def setUp(self):
        builder_env(self, "legacy")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        self.dockerfile = self.repo / "Dockerfile"
        self.dockerfile.write_text("FROM scratch\nCOPY . /app\n")
        self.fake = use_fake(self, FakeClient())

    def build(self, outcomes=None, **kwargs):
        if outcomes is not None:
            self.fake.api.outcomes = outcomes
        return ensure_image(self.dockerfile, **kwargs)

    def assertFails(self, code: str, **kwargs) -> LifecycleError:
        with self.assertRaises(LifecycleError) as ctx:
            self.build(**kwargs)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_success(self):
        ref = self.build(labels={LABEL_TEST: "run1"}, timeout=99)
        self.assertEqual((ref.built, ref.builder), (True, "legacy"))
        (call,) = self.fake.api.build_calls
        self.assertEqual((call["path"], call["dockerfile"], call["tag"]), (str(self.repo.resolve()), "Dockerfile", ref.tag))
        self.assertEqual(call["labels"], {LABEL_TEST: "run1", LABEL_BUILT_BY: "lifecycle"})
        self.assertTrue(call["rm"] and call["forcerm"] and call["decode"])
        self.assertEqual(call["timeout"], 99)
        self.assertEqual(self.fake.api.closed, 1)

    def test_dockerfile_in_a_subdirectory_of_the_context(self):
        (self.repo / "docker").mkdir()
        inner = self.repo / "docker" / "Dockerfile"
        inner.write_text("FROM scratch\n")
        ensure_image(inner, context=self.repo)
        self.assertEqual(self.fake.api.build_calls[0]["dockerfile"], "docker/Dockerfile")

    def test_dockerfile_outside_the_context(self):
        outside = Path(self._tmp.name) / "Dockerfile.outside"
        outside.write_text("FROM scratch\n")
        ensure_image(outside, context=self.repo)
        self.assertEqual(self.fake.api.build_calls[0]["dockerfile"], str(outside.resolve()))   # the SDK sends it along

    def test_failed_step(self):
        e = self.assertFails("build_failed", outcomes=[[
            {"stream": "Step 1/2 : FROM python:3.12\n"}, {"stream": "Step 2/2 : RUN pip install nope\n"},
            {"stream": "ERROR: No matching distribution found for nope\n"},
            {"error": "The command '/bin/sh -c pip install nope' returned a non-zero code: 1"},
        ]])
        self.assertIn("No matching distribution found for nope", e.message)
        self.assertIn("returned a non-zero code: 1", e.message)
        self.assertIn("Dockerization", e.hint)

    def test_buildkit_only_dockerfile(self):
        self.dockerfile.write_text("FROM scratch\nRUN --mount=type=cache,target=/x true\n")
        e = self.assertFails("build_failed", outcomes=[[
            {"error": "the --mount option requires BuildKit. Refer to https://docs.docker.com/go/buildkit/"},
        ]])
        self.assertIn("needs BuildKit", e.message)
        self.assertIn("docker-buildx", e.hint)
        self.assertNotIn("Dockerization", e.hint)

    def test_buildkit_needed_although_the_syntax_check_missed_it(self):
        e = self.assertFails("build_failed", outcomes=[[{"error": "the --some-new-flag option requires BuildKit"}]])
        self.assertIn("docker-buildx", e.hint)

    def test_stale_credential_helper(self):
        # ~/.docker/config.json names a helper that is gone: the build is done again without credentials
        ref = self.build(outcomes=[StoreError("docker-credential-desktop not installed or not available in PATH"), [{"stream": "ok\n"}]])
        self.assertTrue(ref.built)
        self.assertEqual(self.fake.api.auth_shadowed, [False, True])
        self.assertNotIn("_set_auth_headers", self.fake.api.__dict__)   # the SDK method is back afterwards

    def test_timeout(self):
        with mock.patch.object(image.time, "monotonic", side_effect=[0.0, 5.0, 5.0, 5.0]):
            e = self.assertFails("timeout", outcomes=[[{"stream": "Step 1/9\n"}, {"stream": "Step 2/9\n"}]], timeout=1)
        self.assertIn("did not finish in 1 s", e.message)
        self.assertEqual(self.fake.api.closed, 1)                       # the connection is closed: the daemon stops

    def test_daemon_down(self):
        self.assertFails("docker_unavailable", outcomes=[requests.exceptions.ConnectionError("Connection refused")])


# ---------------------------------------------------------------- live suites (real daemon)

@requires_docker
class TestBuild(unittest.TestCase):
    """Real builds of the sample Dockerfiles."""

    @classmethod
    def setUpClass(cls):
        cls.first = samples.build("idle", rebuild=True)

    def sdk_image(self, tag: str):
        return client.get_client().images.get(tag)

    def test_builds_and_tags(self):
        self.assertTrue(self.first.built)
        self.assertEqual(self.first.builder, choose_builder())          # buildkit if buildx works, legacy otherwise
        self.assertRegex(self.first.tag, r"^tml-local/idle:[0-9a-f]{12}$")
        self.assertRegex(self.first.id, r"^[0-9a-f]{12}$")
        self.assertIn(self.first.tag, self.sdk_image(self.first.tag).tags)

    def test_tag_is_the_documented_hash(self):
        path = Path(samples.dockerfile("idle"))
        self.assertEqual(self.first.tag, image_tag(path.read_text(), path.parent))

    def test_labels(self):
        labels = self.sdk_image(self.first.tag).labels
        self.assertEqual(labels[LABEL_BUILT_BY], "lifecycle")
        self.assertEqual(labels[LABEL_TEST], samples.RUN_ID)

    def test_second_call_reuses_the_image(self):
        again = samples.build("idle")
        self.assertFalse(again.built)
        self.assertEqual((again.id, again.tag), (self.first.id, self.first.tag))

    def test_rebuild(self):
        # its own sample: a rebuild can give the tag a new image id, which would disturb the other tests
        one = samples.build("entrypoint", rebuild=True)
        two = samples.build("entrypoint", rebuild=True)
        self.assertTrue(one.built and two.built)
        self.assertEqual(two.tag, one.tag)
        self.assertEqual(resolve_image(two.tag).id, two.id)     # the tag points at the newest build

    def test_name(self):
        ref = samples.build("idle", name="Renamed Sample")
        self.assertTrue(ref.tag.startswith("tml-local/renamed-sample:"))

    def test_resolve_a_built_image(self):
        ref = resolve_image(self.first.tag)
        self.assertEqual((ref.id, ref.built), (self.first.id, False))

    def test_resolve_by_id(self):
        self.assertEqual(resolve_image(self.first.id).id, self.first.id)

    def test_image_without_base_image(self):
        ref = samples.build("empty")                        # FROM scratch
        self.assertTrue(ref.tag.startswith("tml-local/empty:"))

    def test_changed_dockerfile_gives_a_new_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Dockerfile"
            path.write_text(samples.DOCKERFILES["idle"])
            one = ensure_image(path, labels=samples.TEST_LABEL)
            path.write_text(samples.DOCKERFILES["idle"] + "ENV CHANGED=1\n")
            two = ensure_image(path, labels=samples.TEST_LABEL)
        self.assertTrue(one.built and two.built)
        self.assertNotEqual(one.tag, two.tag)
        self.assertNotEqual(one.id, two.id)


@requires_docker
class TestBuildFailure(unittest.TestCase):
    def test_broken_build(self):
        path = Path(samples.dockerfile("broken-build"))
        with self.assertRaises(LifecycleError) as ctx:
            samples.build("broken-build")
        e = ctx.exception
        self.assertEqual(e.code, "build_failed")
        self.assertIn(str(path), e.message)
        self.assertIn("about to fail", e.message)            # the RUN line's own output
        self.assertRegex(e.message, r"exit code:? 7|non-zero code: 7")   # BuildKit's or the classic builder's wording
        self.assertIn("Dockerization", e.hint)

    def test_failed_build_leaves_no_image(self):
        path = Path(samples.dockerfile("broken-build"))
        with self.assertRaises(LifecycleError):
            samples.build("broken-build")
        self.assertIsNone(image.find_image(image_tag(path.read_text(), path.parent)))

    def test_missing_image(self):
        with self.assertRaises(LifecycleError) as ctx:
            resolve_image("tml-local/never-built:000000000000")
        self.assertEqual(ctx.exception.code, "image_not_found")


@requires_docker
class TestClassicBuilder(unittest.TestCase):
    """The daemon's classic builder, forced: it works with neither the docker command nor buildx."""

    def setUp(self):
        builder_env(self, "legacy")

    def write(self, text: str, extra: dict | None = None) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, content in {"Dockerfile": text, **(extra or {})}.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(content)
        return root / "Dockerfile"

    def test_builds(self):
        path = self.write(samples.DOCKERFILES["idle"] + f"LABEL classic={samples.RUN_ID}\n")
        ref = ensure_image(path, labels=samples.TEST_LABEL)
        self.assertEqual((ref.built, ref.builder), (True, "legacy"))
        labels = client.get_client().images.get(ref.tag).labels
        self.assertEqual((labels[LABEL_BUILT_BY], labels["classic"]), ("lifecycle", samples.RUN_ID))
        self.assertFalse(ensure_image(path, labels=samples.TEST_LABEL).built)

    def test_context_files_are_sent(self):
        text = f"FROM {samples.BASE_IMAGE}\nCOPY data/hello.txt /hello.txt\nRUN grep -q classic /hello.txt\n"
        ref = ensure_image(self.write(text, {"data/hello.txt": "classic builder"}), labels=samples.TEST_LABEL)
        self.assertTrue(ref.built)

    def test_dockerfile_outside_the_context(self):
        context = self.write("unused", {"data/hello.txt": "outside"}).parent
        outside = self.write(f"FROM {samples.BASE_IMAGE}\nCOPY data/hello.txt /hello.txt\n")
        ref = ensure_image(outside, context=context, labels=samples.TEST_LABEL)
        self.assertEqual(ref.builder, "legacy")

    def test_buildkit_only_syntax_names_the_missing_plugin(self):
        path = self.write(f"FROM {samples.BASE_IMAGE}\nRUN --mount=type=cache,target=/root/.cache true\n")
        with self.assertRaises(LifecycleError) as ctx:
            ensure_image(path, labels=samples.TEST_LABEL)
        self.assertEqual(ctx.exception.code, "build_failed")
        self.assertIn("requires BuildKit", ctx.exception.message)
        self.assertIn("docker-buildx", ctx.exception.hint)

    def test_failed_step(self):
        with self.assertRaises(LifecycleError) as ctx:
            samples.build("broken-build")
        self.assertIn("non-zero code: 7", ctx.exception.message)


@requires_docker
@unittest.skipUnless(samples.DOCKER_READY and buildkit_available(), "docker buildx is not installed")
class TestBuildKitBuilder(unittest.TestCase):
    def setUp(self):
        builder_env(self, "buildkit")

    def test_buildkit_only_syntax_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Dockerfile"
            path.write_text(f"FROM {samples.BASE_IMAGE}\nRUN --mount=type=cache,target=/root/.cache true\n")
            ref = ensure_image(path, labels=samples.TEST_LABEL)
        self.assertEqual((ref.built, ref.builder), (True, "buildkit"))


if __name__ == "__main__":
    unittest.main()
