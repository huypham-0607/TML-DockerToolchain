import tempfile
import time
import unittest
from pathlib import Path

import requests
from docker import errors as docker_errors

from src.lifecycle import client
from src.lifecycle.image import find_image
from src.lifecycle.models import LABEL_BUILT_BY, LABEL_TEST, ContainerSpec, LifecycleError
from src.lifecycle.provision import provision
from src.lifecycle.status import get_status, list_managed
from src.lifecycle.teardown import STOP_TIMEOUT, destroy, destroy_all_managed, remove, remove_built_images
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, FakeImage, api_error, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker


def tearDownModule():
    samples.cleanup()


def not_found(name: str) -> docker_errors.NotFound:
    return docker_errors.NotFound("404 Client Error: Not Found", explanation=f"No such container: {name}")


# ---------------------------------------------------------------- pure suites (no daemon)

class TestRemoveLogic(unittest.TestCase):
    """remove() against a fake client."""

    def one(self, **kwargs) -> tuple[FakeContainer, FakeClient]:
        fake_kwargs = {k: kwargs.pop(k) for k in ("remove_error",) if k in kwargs}
        container = FakeContainer(make_attrs(**kwargs), **fake_kwargs)
        return container, use_fake(self, FakeClient(container))

    def test_stopped_container_is_removed_with_its_anonymous_volumes(self):
        container, fake = self.one(status="exited", exit_code=3)
        r = remove("tml-demo")
        self.assertEqual(container.calls, [("remove", False, True)])
        self.assertEqual((r.removed, r.stopped, r.killed, r.exit_code), (True, False, False, 3))
        self.assertEqual((r.status.state, r.status.health, r.status.reason), ("missing", "missing", "removed"))
        self.assertIsNone(r.status.handle)
        self.assertEqual(fake.containers.names(), [])

    def test_running_container_is_refused_without_force(self):
        container, fake = self.one(status="running")
        with self.assertRaises(LifecycleError) as ctx:
            remove("tml-demo")
        self.assertEqual(ctx.exception.code, "still_running")
        self.assertIn("force=True", ctx.exception.hint)
        self.assertEqual(container.calls, [])
        self.assertEqual(fake.containers.names(), ["tml-demo"])

    def test_paused_and_restarting_count_as_running(self):
        for state in ("paused", "restarting"):
            container, _ = self.one(status=state)
            with self.assertRaises(LifecycleError) as ctx:
                remove("tml-demo")
            self.assertEqual(ctx.exception.code, "still_running", state)
            self.assertIn(state, ctx.exception.message)

    def test_force_removes_a_running_container(self):
        container, _ = self.one(status="running")
        r = remove("tml-demo", force=True)
        self.assertEqual(container.calls, [("remove", True, True)])
        self.assertEqual((r.removed, r.stopped, r.killed), (True, True, True))   # killed, not stopped gracefully
        self.assertIsNone(r.exit_code)                                           # its exit was never observed

    def test_force_on_a_stopped_container_kills_nothing(self):
        _, _ = self.one(status="exited", exit_code=0)
        r = remove("tml-demo", force=True)
        self.assertEqual((r.removed, r.stopped, r.killed, r.exit_code), (True, False, False, 0))

    def test_volumes_can_be_kept(self):
        container, _ = self.one(status="exited")
        remove("tml-demo", volumes=False)
        self.assertEqual(container.calls, [("remove", False, False)])

    def test_never_started_container(self):
        _, _ = self.one(status="created")
        r = remove("tml-demo")
        self.assertTrue(r.removed)
        self.assertIsNone(r.exit_code)

    def test_missing_container(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            remove("nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_unmanaged_container_is_not_removed_even_with_force(self):
        container = FakeContainer(make_attrs(name="caddy", status="exited", managed=False))
        fake = use_fake(self, FakeClient(container))
        with self.assertRaises(LifecycleError) as ctx:
            remove("caddy", force=True)
        self.assertEqual(ctx.exception.code, "not_managed")
        self.assertEqual(container.calls, [])
        self.assertEqual(fake.containers.names(), ["caddy"])

    def test_daemon_error_is_translated(self):
        self.one(status="exited", remove_error=requests.exceptions.ConnectionError("Connection refused"))
        with self.assertRaises(LifecycleError) as ctx:
            remove("tml-demo")
        self.assertEqual(ctx.exception.code, "docker_unavailable")


class TestDestroyLogic(unittest.TestCase):
    """destroy() against a fake client."""

    def one(self, **kwargs) -> tuple[FakeContainer, FakeClient]:
        fake_kwargs = {k: kwargs.pop(k) for k in ("stop_exit", "remove_error") if k in kwargs}
        container = FakeContainer(make_attrs(**kwargs), **fake_kwargs)
        return container, use_fake(self, FakeClient(container))

    def test_running_container_is_stopped_and_then_removed(self):
        container, fake = self.one(status="running")
        r = destroy("tml-demo")
        self.assertEqual(container.calls, [("stop", STOP_TIMEOUT), ("remove", False, True)])   # in this order
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (True, False, True, 143))
        self.assertEqual((r.status.state, r.status.health), ("missing", "missing"))
        self.assertEqual(fake.containers.names(), [])

    def test_timeout_is_passed_to_the_stop(self):
        container, _ = self.one(status="running")
        destroy("tml-demo", timeout=3)
        self.assertEqual(container.calls[0], ("stop", 3))

    def test_stop_that_needed_sigkill_is_reported(self):
        _, _ = self.one(status="running", stop_exit=137)
        r = destroy("tml-demo")
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (True, True, True, 137))

    def test_force_kills_without_a_graceful_stop(self):
        container, _ = self.one(status="running")
        r = destroy("tml-demo", force=True)
        self.assertEqual(container.calls, [("remove", True, True)])
        self.assertEqual((r.stopped, r.killed, r.removed), (True, True, True))

    def test_stopped_container_is_only_removed(self):
        container, _ = self.one(status="exited", exit_code=3)
        r = destroy("tml-demo")
        self.assertEqual(container.calls, [("remove", False, True)])
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (False, False, True, 3))

    def test_missing_container_is_not_an_error(self):
        use_fake(self, FakeClient())
        r = destroy("nope")
        self.assertEqual((r.container, r.stopped, r.killed, r.removed), ("nope", False, False, False))
        self.assertEqual((r.status.state, r.status.reason), ("missing", "no such container"))
        self.assertIsNone(r.exit_code)

    def test_container_removed_by_someone_else_meanwhile(self):
        container, _ = self.one(status="running", remove_error=not_found("tml-demo"))
        r = destroy("tml-demo")
        self.assertEqual((r.stopped, r.removed, r.exit_code), (True, False, 143))     # we stopped it; we did not remove it
        self.assertEqual(r.status.state, "missing")

    def test_unmanaged_container_is_not_touched_even_with_force(self):
        container = FakeContainer(make_attrs(name="caddy", status="running", managed=False))
        fake = use_fake(self, FakeClient(container))
        for force in (False, True):
            with self.assertRaises(LifecycleError) as ctx:
                destroy("caddy", force=force)
            self.assertEqual(ctx.exception.code, "not_managed")
        self.assertEqual(container.calls, [])
        self.assertEqual(fake.containers.names(), ["caddy"])

    def test_invalid_timeout(self):
        container, _ = self.one(status="running")
        with self.assertRaises(LifecycleError) as ctx:
            destroy("tml-demo", timeout=-1)
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertEqual(container.calls, [])

    def test_other_removal_errors_are_raised(self):
        self.one(status="exited", remove_error=api_error(500, "driver failed to remove root filesystem"))
        with self.assertRaises(LifecycleError) as ctx:
            destroy("tml-demo")
        self.assertEqual(ctx.exception.code, "docker_error")

    def test_empty_name(self):
        use_fake(self, FakeClient())
        with self.assertRaises(LifecycleError) as ctx:
            destroy("")
        self.assertEqual(ctx.exception.code, "invalid_argument")


class TestDestroyAllLogic(unittest.TestCase):
    def fake(self, *containers: FakeContainer) -> FakeClient:
        return use_fake(self, FakeClient(*containers))

    def test_destroys_managed_containers_only(self):
        fake = self.fake(
            FakeContainer(make_attrs(name="tml-a", status="running")),
            FakeContainer(make_attrs(name="tml-b", status="exited", exit_code=3)),
            FakeContainer(make_attrs(name="caddy", status="running", managed=False)),
        )
        results = destroy_all_managed()
        self.assertEqual({r.container: (r.stopped, r.removed) for r in results}, {"tml-a": (True, True), "tml-b": (False, True)})
        self.assertEqual(fake.containers.names(), ["caddy"])

    def test_label_filter(self):
        fake = self.fake(
            FakeContainer(make_attrs(name="tml-a", labels={LABEL_TEST: "run1"})),
            FakeContainer(make_attrs(name="tml-b", labels={LABEL_TEST: "run2"})),
            FakeContainer(make_attrs(name="tml-c")),
        )
        results = destroy_all_managed(labels={LABEL_TEST: "run1"})
        self.assertEqual([r.container for r in results], ["tml-a"])
        self.assertEqual(fake.containers.names(), ["tml-b", "tml-c"])

    def test_force_and_timeout_reach_every_container(self):
        a, b = FakeContainer(make_attrs(name="tml-a")), FakeContainer(make_attrs(name="tml-b"))
        self.fake(a, b)
        destroy_all_managed(timeout=3)
        self.assertEqual(a.calls, [("stop", 3), ("remove", False, True)])
        c, d = FakeContainer(make_attrs(name="tml-c")), FakeContainer(make_attrs(name="tml-d"))
        self.fake(c, d)
        destroy_all_managed(force=True)
        self.assertEqual((c.calls, d.calls), ([("remove", True, True)], [("remove", True, True)]))

    def test_no_readiness_probe_is_run(self):
        a = FakeContainer(make_attrs(name="tml-a"))
        self.fake(a)
        destroy_all_managed()
        self.assertEqual(a.exec_calls, [])

    def test_nothing_to_destroy(self):
        self.fake(FakeContainer(make_attrs(name="caddy", managed=False)))
        self.assertEqual(destroy_all_managed(), [])

    def test_one_failure_does_not_stop_the_others(self):
        broken = FakeContainer(make_attrs(name="tml-broken", status="exited"), remove_error=api_error(500, "device or resource busy"))
        fake = self.fake(
            FakeContainer(make_attrs(name="tml-a", created="2026-10-09T10:00:00Z")),
            broken,
            FakeContainer(make_attrs(name="tml-z", created="2026-10-09T10:20:00Z")),
        )
        with self.assertRaises(LifecycleError) as ctx:
            destroy_all_managed()
        e = ctx.exception
        self.assertEqual(e.code, "docker_error")
        self.assertIn("1 of 3", e.message)
        self.assertIn("tml-broken", e.message)
        self.assertIn("device or resource busy", e.message)
        self.assertEqual(fake.containers.names(), ["tml-broken"])      # the two others are gone

    def test_invalid_timeout(self):
        a = FakeContainer(make_attrs(name="tml-a"))
        self.fake(a)
        with self.assertRaises(LifecycleError) as ctx:
            destroy_all_managed(timeout=-1)
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertEqual(a.calls, [])


class TestRemoveBuiltImagesLogic(unittest.TestCase):
    BUILT = {LABEL_BUILT_BY: "lifecycle"}

    def image(self, tag: str, n: int, **labels) -> FakeImage:
        return FakeImage(tag, image_id="sha256:" + f"{n:02x}" * 32, labels=labels)

    def test_only_images_that_lifecycle_built(self):
        fake = use_fake(self, FakeClient(images=[
            self.image("tml-local/a:1", 1, **self.BUILT),
            self.image("tml-local/b:2", 2, **self.BUILT),
            self.image("python:3.14-slim", 3),
            self.image("team/robust:latest", 4, **{LABEL_BUILT_BY: "dockerization"}),
        ]))
        self.assertEqual(remove_built_images(), ["tml-local/a:1", "tml-local/b:2"])
        self.assertEqual(fake.images.tags(), ["python:3.14-slim", "team/robust:latest"])

    def test_label_filter(self):
        fake = use_fake(self, FakeClient(images=[
            self.image("tml-local/a:1", 1, **self.BUILT, **{LABEL_TEST: "run1"}),
            self.image("tml-local/b:2", 2, **self.BUILT, **{LABEL_TEST: "run2"}),
            self.image("tml-local/c:3", 3, **self.BUILT),
        ]))
        self.assertEqual(remove_built_images(labels={LABEL_TEST: "run1"}), ["tml-local/a:1"])
        self.assertEqual(fake.images.tags(), ["tml-local/b:2", "tml-local/c:3"])

    def test_image_in_use_is_skipped(self):
        in_use = api_error(409, "conflict: unable to remove repository reference \"tml-local/a:1\" (must force) - container d7d9 is using its referenced image")
        fake = use_fake(self, FakeClient(
            images=[self.image("tml-local/a:1", 1, **self.BUILT), self.image("tml-local/b:2", 2, **self.BUILT)],
            image_remove_errors={"tml-local/a:1": in_use},
        ))
        self.assertEqual(remove_built_images(), ["tml-local/b:2"])
        self.assertEqual(fake.images.tags(), ["tml-local/a:1"])

    def test_force_is_passed_on(self):
        fake = use_fake(self, FakeClient(images=[self.image("tml-local/a:1", 1, **self.BUILT)]))
        remove_built_images()
        self.assertEqual(fake.images.remove_calls, [("tml-local/a:1", False)])
        fake = use_fake(self, FakeClient(images=[self.image("tml-local/a:1", 1, **self.BUILT)]))
        remove_built_images(force=True)
        self.assertEqual(fake.images.remove_calls, [("tml-local/a:1", True)])

    def test_every_tag_of_an_image_is_removed(self):
        image = self.image("tml-local/a:1", 1, **self.BUILT)
        image.tags.append("tml-local/a:latest")
        fake = use_fake(self, FakeClient(images=[image]))
        self.assertEqual(remove_built_images(), ["tml-local/a:1", "tml-local/a:latest"])
        self.assertEqual(fake.images.tags(), [])

    def test_untagged_image_is_removed_by_id(self):
        image = self.image("gone", 7, **self.BUILT)
        image.tags = []
        fake = use_fake(self, FakeClient(images=[image]))
        self.assertEqual(remove_built_images(), [image.id])
        self.assertEqual(fake.images.remove_calls, [(image.id, False)])

    def test_image_that_disappeared_meanwhile_is_skipped(self):
        gone = docker_errors.ImageNotFound("404", explanation="No such image: tml-local/a:1")
        use_fake(self, FakeClient(
            images=[self.image("tml-local/a:1", 1, **self.BUILT), self.image("tml-local/b:2", 2, **self.BUILT)],
            image_remove_errors={"tml-local/a:1": gone},
        ))
        self.assertEqual(remove_built_images(), ["tml-local/b:2"])

    def test_other_errors_are_raised(self):
        use_fake(self, FakeClient(
            images=[self.image("tml-local/a:1", 1, **self.BUILT)],
            image_remove_errors={"tml-local/a:1": api_error(500, "layer does not exist")},
        ))
        with self.assertRaises(LifecycleError) as ctx:
            remove_built_images()
        self.assertEqual(ctx.exception.code, "docker_error")
        self.assertIn("remove image 'tml-local/a:1'", ctx.exception.message)

    def test_nothing_built(self):
        use_fake(self, FakeClient(images=[self.image("python:3.14-slim", 3)]))
        self.assertEqual(remove_built_images(), [])


# ---------------------------------------------------------------- live suites (real daemon)

class LiveCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends

    def assertGone(self, name: str):
        self.assertEqual(get_status(name).health, "missing")
        with self.assertRaises(docker_errors.NotFound):
            samples.sdk_container(name)


@requires_docker
class TestDestroyRunning(LiveCase):
    def test_graceful(self):
        name = samples.start("idle", "destroy-running").handle.name
        r = destroy(name)
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (True, False, True, 143))
        self.assertEqual((r.status.state, r.status.health), ("missing", "missing"))
        self.assertGone(name)

    def test_force(self):
        name = samples.start("stubborn", "destroy-force", mode="native").handle.name
        started = time.monotonic()
        r = destroy(name, force=True)
        self.assertLess(time.monotonic() - started, 3)                # no wait for a stop timeout
        self.assertEqual((r.stopped, r.killed, r.removed), (True, True, True))
        self.assertGone(name)

    def test_stubborn_process_is_killed_after_the_timeout(self):
        name = samples.start("stubborn", "destroy-stubborn", mode="native").handle.name
        started = time.monotonic()
        r = destroy(name, timeout=1)
        self.assertGreaterEqual(time.monotonic() - started, 1)
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (True, True, True, 137))
        self.assertGone(name)

    def test_by_id(self):
        handle = samples.start("idle", "destroy-by-id").handle
        self.assertTrue(destroy(handle.id).removed)
        self.assertGone(handle.name)

    def test_name_is_free_again(self):
        first = samples.start("idle", "destroy-reuse-name")
        destroy(first.handle.name)
        second = samples.start("idle", "destroy-reuse-name")
        self.assertEqual(second.action, "created")
        self.assertNotEqual(second.handle.id, first.handle.id)


@requires_docker
class TestDestroyStopped(LiveCase):
    def test_stopped_container(self):
        name = samples.start("idle", "destroy-stopped").handle.name
        samples.sdk_container(name).stop(timeout=5)
        r = destroy(name)
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (False, False, True, 143))
        self.assertGone(name)

    def test_crashed_container_reports_its_exit_code(self):
        name = samples.start("crash", "destroy-crashed", mode="native").handle.name
        r = destroy(name)
        self.assertEqual((r.stopped, r.removed, r.exit_code), (False, True, 3))
        self.assertGone(name)

    def test_finished_container(self):
        name = samples.start("idle", "destroy-finished", mode="native").handle.name
        r = destroy(name)
        self.assertEqual((r.stopped, r.removed, r.exit_code), (False, True, 0))

    def test_container_that_could_not_start(self):
        name = samples.start("empty", "destroy-empty").handle.name      # FROM scratch: idle mode fails with 127
        r = destroy(name)
        self.assertEqual((r.removed, r.exit_code), (True, 127))
        self.assertGone(name)


@requires_docker
class TestIdempotent(LiveCase):
    def test_second_destroy_is_not_an_error(self):
        name = samples.start("idle", "destroy-twice").handle.name
        first, second = destroy(name), destroy(name)
        self.assertEqual((first.removed, second.removed), (True, False))
        self.assertEqual((second.stopped, second.killed, second.exit_code), (False, False, None))
        self.assertEqual((second.status.state, second.status.reason), ("missing", "no such container"))

    def test_container_that_never_existed(self):
        r = destroy(samples.container_name("destroy-never-made"))
        self.assertFalse(r.removed)
        self.assertEqual(r.status.health, "missing")


@requires_docker
class TestNotManaged(LiveCase):
    def test_unmanaged_container_is_refused_and_keeps_running(self):
        other = samples.start_unmanaged("destroy-unmanaged")
        for call in (lambda: destroy(other.name), lambda: destroy(other.name, force=True), lambda: remove(other.name, force=True)):
            with self.assertRaises(LifecycleError) as ctx:
                call()
            self.assertEqual(ctx.exception.code, "not_managed")
        other.reload()
        self.assertEqual(other.status, "running")

    def test_destroy_all_leaves_it_alone(self):
        other = samples.start_unmanaged("destroy-all-unmanaged")
        samples.start("idle", "destroy-all-managed")
        destroy_all_managed(labels=samples.TEST_LABEL, force=True)
        other.reload()
        self.assertEqual(other.status, "running")


@requires_docker
class TestImageKept(LiveCase):
    def test_image_is_still_there(self):
        r = samples.start("idle", "destroy-image-kept")
        destroy(r.handle.name)
        self.assertIsNotNone(find_image(r.image.tag))
        self.assertTrue(samples.start("idle", "destroy-image-kept-again").ok)    # and it still runs

    def test_mounted_directory_is_kept(self):
        data = samples.mount_dir()
        (data / "results.txt").write_text("accuracy 0.93")
        name = samples.start("idle", "destroy-mount-kept", mounts=[f"{data}:/data"]).handle.name
        destroy(name)
        self.assertEqual((data / "results.txt").read_text(), "accuracy 0.93")

    def test_anonymous_volume_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Dockerfile"
            path.write_text(samples.DOCKERFILES["idle"] + "VOLUME /scratch\n")
            spec = ContainerSpec(dockerfile=str(path), name=samples.container_name("destroy-volume"), labels=samples.TEST_LABEL)
            name = provision(spec).handle.name
        volumes = [m["Name"] for m in samples.sdk_container(name).attrs["Mounts"] if m["Type"] == "volume"]
        self.assertEqual(len(volumes), 1)
        sdk = client.get_client()
        sdk.volumes.get(volumes[0])                                     # exists while the container does
        destroy(name)
        with self.assertRaises(docker_errors.NotFound):
            sdk.volumes.get(volumes[0])


@requires_docker
class TestRemove(LiveCase):
    def test_running_container_is_refused(self):
        name = samples.start("idle", "remove-running").handle.name
        with self.assertRaises(LifecycleError) as ctx:
            remove(name)
        self.assertEqual(ctx.exception.code, "still_running")
        self.assertEqual(samples.sdk_container(name).status, "running")

    def test_stopped_container(self):
        name = samples.start("crash", "remove-stopped", mode="native").handle.name
        r = remove(name)
        self.assertEqual((r.removed, r.stopped, r.exit_code), (True, False, 3))
        self.assertGone(name)

    def test_force(self):
        name = samples.start("idle", "remove-force").handle.name
        r = remove(name, force=True)
        self.assertEqual((r.removed, r.stopped, r.killed), (True, True, True))
        self.assertGone(name)

    def test_missing_container(self):
        with self.assertRaises(LifecycleError) as ctx:
            remove(samples.container_name("remove-never-made"))
        self.assertEqual(ctx.exception.code, "not_found")


@requires_docker
class TestDestroyAll(LiveCase):
    def test_destroys_every_managed_container_of_this_run(self):
        names = {
            samples.start("idle", "all-running").handle.name,
            samples.start("crash", "all-crashed", mode="native").handle.name,
            samples.start("stubborn", "all-stubborn", mode="native").handle.name,
        }
        results = destroy_all_managed(labels=samples.TEST_LABEL, timeout=1)
        self.assertEqual({r.container for r in results}, names)
        self.assertTrue(all(r.removed for r in results))
        by_suffix = {r.container.rsplit("-", 1)[-1]: r for r in results}
        self.assertEqual((by_suffix["running"].stopped, by_suffix["running"].killed), (True, False))
        self.assertEqual((by_suffix["crashed"].stopped, by_suffix["crashed"].exit_code), (False, 3))
        self.assertEqual((by_suffix["stubborn"].stopped, by_suffix["stubborn"].killed), (True, True))
        self.assertEqual(list_managed(labels=samples.TEST_LABEL), [])

    def test_nothing_left_to_destroy(self):
        destroy_all_managed(labels=samples.TEST_LABEL, force=True)
        self.assertEqual(destroy_all_managed(labels=samples.TEST_LABEL), [])

    def test_label_filter_limits_what_is_destroyed(self):
        name = samples.start("idle", "all-other-label").handle.name
        self.assertEqual(destroy_all_managed(labels={LABEL_TEST: "some-other-run"}), [])
        self.assertEqual(get_status(name).health, "healthy")


@requires_docker
class TestRemoveBuiltImages(LiveCase):
    def setUp(self):
        samples.remove_containers()      # nothing of this run may hold an image at the start

    def test_removes_the_images_of_this_run(self):
        tags = {samples.build("idle").tag, samples.build("entrypoint").tag}
        removed = remove_built_images(labels=samples.TEST_LABEL)
        self.assertLessEqual(tags, set(removed))
        for tag in tags:
            self.assertIsNone(find_image(tag))
        self.assertEqual(remove_built_images(labels=samples.TEST_LABEL), [])       # a second call finds nothing

    def test_image_in_use_is_skipped_until_its_container_is_destroyed(self):
        r = samples.start("idle", "images-in-use")
        self.assertNotIn(r.image.tag, remove_built_images(labels=samples.TEST_LABEL))
        self.assertIsNotNone(find_image(r.image.tag))
        self.assertEqual(get_status(r.handle.name).health, "healthy")              # the container was not disturbed
        destroy(r.handle.name)
        self.assertIn(r.image.tag, remove_built_images(labels=samples.TEST_LABEL))
        self.assertIsNone(find_image(r.image.tag))

    def test_other_images_are_never_removed(self):
        samples.build("idle")
        remove_built_images(labels=samples.TEST_LABEL, force=True)
        self.assertIsNotNone(find_image(samples.BASE_IMAGE))                       # the base image is not ours

    def test_label_filter_limits_what_is_removed(self):
        tag = samples.build("idle").tag
        self.assertEqual(remove_built_images(labels={LABEL_TEST: "some-other-run"}), [])
        self.assertIsNotNone(find_image(tag))


if __name__ == "__main__":
    unittest.main()
