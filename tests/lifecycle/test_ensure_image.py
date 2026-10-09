import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.lifecycle import client, image
from src.lifecycle.image import build_command, ensure_image, image_tag, resolve_image
from src.lifecycle.models import LABEL_BUILT_BY, LABEL_TEST, LifecycleError
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeImage, use_fake
from tests.lifecycle.samples import requires_docker


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


class TestBuildWithoutDaemon(unittest.TestCase):
    """The build step with `docker build` replaced by a mock."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dockerfile = Path(self._tmp.name) / "repo" / "Dockerfile"
        self.dockerfile.parent.mkdir()
        self.dockerfile.write_text("FROM scratch\n")
        self.tag = image_tag("FROM scratch\n", self.dockerfile.parent)

    def run_build(self, images=(), **outcome) -> mock.Mock:
        use_fake(self, FakeClient(images=images))
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
        self.assertTrue(ref.built)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], ["docker", "build"])

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

    def test_docker_command_not_installed(self):
        self.run_build(side_effect=FileNotFoundError("docker"))
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
        self.assertRegex(e.message, r"exit code:? 7")        # BuildKit's report of the failing step
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


if __name__ == "__main__":
    unittest.main()
