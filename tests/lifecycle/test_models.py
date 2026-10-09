import json
import unittest

from src.lifecycle.models import (
    ERROR_CODES, HEALTH,
    ContainerHandle, ContainerLogs, ContainerSpec, ContainerStatus,
    Diagnosis, Hint, ImageRef, LifecycleError, ProvisionResult, TeardownResult,
)


class TestSpecValidation(unittest.TestCase):
    def assertInvalid(self, spec: ContainerSpec, fragment: str):
        with self.assertRaises(LifecycleError) as ctx:
            spec.validate()
        self.assertEqual(ctx.exception.code, "invalid_argument")
        self.assertIn(fragment, ctx.exception.message)

    def test_image_only_is_valid(self):
        spec = ContainerSpec(image="demo:latest")
        self.assertIs(spec.validate(), spec)

    def test_dockerfile_only_is_valid(self):
        ContainerSpec(dockerfile="docker/Dockerfile", context=".").validate()

    def test_defaults(self):
        spec = ContainerSpec(image="demo:latest")
        self.assertEqual((spec.mode, spec.gpu, spec.shm_size, spec.network), ("idle", "auto", "2g", "bridge"))
        self.assertEqual((spec.env, spec.mounts, spec.ports, spec.labels), ({}, [], [], {}))
        self.assertEqual((spec.memory, spec.cpus), ("", 0.0))       # default memory limit, no CPU limit

    def test_needs_image_or_dockerfile(self):
        self.assertInvalid(ContainerSpec(), "exactly one of image / dockerfile")

    def test_rejects_image_and_dockerfile_together(self):
        self.assertInvalid(ContainerSpec(image="demo", dockerfile="Dockerfile"), "exactly one of image / dockerfile")

    def test_context_needs_dockerfile(self):
        self.assertInvalid(ContainerSpec(image="demo", context="."), "context")

    def test_unknown_mode(self):
        self.assertInvalid(ContainerSpec(image="demo", mode="daemon"), "mode")

    def test_unknown_gpu_option(self):
        self.assertInvalid(ContainerSpec(image="demo", gpu="yes"), "gpu")

    def test_all_gpu_options_are_valid(self):
        for gpu in ("auto", "none", "all"):
            ContainerSpec(image="demo", gpu=gpu).validate()

    def test_command_needs_native_mode(self):
        self.assertInvalid(ContainerSpec(image="demo", command="python train.py"), "native")
        ContainerSpec(image="demo", mode="native", command="python train.py").validate()

    def test_container_name(self):
        ContainerSpec(image="demo", name="tml-robust_train.1").validate()
        for name in ("x", "-leading", "has space", "semi;colon", "a/b"):
            self.assertInvalid(ContainerSpec(image="demo", name=name), "name")

    def test_unknown_network(self):
        self.assertInvalid(ContainerSpec(image="demo", network="host"), "network")

    def test_negative_cpus(self):
        self.assertInvalid(ContainerSpec(image="demo", cpus=-1), "cpus")


class TestSerialization(unittest.TestCase):
    def roundtrip(self, obj) -> dict:
        data = json.loads(obj.to_json())
        self.assertEqual(data, obj.to_dict())
        return data

    def handle(self) -> ContainerHandle:
        return ContainerHandle(
            id="d7d9d9bb3b41", name="tml-demo", image="demo:latest", image_id="51dafde81dbd", mode="idle",
            ports={"8888/tcp": "8888"}, mounts=["/data:/data:ro"], labels={"tml.managed": "true"},
        )

    def test_spec(self):
        data = self.roundtrip(ContainerSpec(image="demo", env={"A": "1"}, mounts=["/a:/b"], ports=["8888"]))
        self.assertEqual(data["env"], {"A": "1"})
        self.assertEqual(data["mounts"], ["/a:/b"])

    def test_image_ref(self):
        self.assertEqual(self.roundtrip(ImageRef(id="51dafde81dbd", tag="tml-local/demo:abc", built=True))["built"], True)

    def test_handle(self):
        data = self.roundtrip(self.handle())
        self.assertEqual(data["ports"], {"8888/tcp": "8888"})
        self.assertFalse(data["gpu"])

    def test_status_adds_running_and_usable(self):
        status = ContainerStatus(container="tml-demo", state="running", health="healthy", handle=self.handle())
        data = self.roundtrip(status)
        self.assertTrue(data["running"])
        self.assertTrue(data["usable"])
        self.assertEqual(data["handle"]["name"], "tml-demo")
        self.assertIsNone(data["exit_code"])

    def test_missing_status_has_no_handle(self):
        data = self.roundtrip(ContainerStatus(container="gone", state="missing", health="missing"))
        self.assertIsNone(data["handle"])
        self.assertFalse(data["running"])
        self.assertFalse(data["usable"])

    def test_running_but_unhealthy_is_not_usable(self):
        status = ContainerStatus(container="c", state="running", health="unhealthy")
        self.assertTrue(status.running)
        self.assertFalse(status.usable)

    def test_logs_and_hint(self):
        self.assertEqual(self.roundtrip(ContainerLogs(container="c", stderr="boom", lines=1))["stderr"], "boom")
        hint = Hint(code="oom", evidence="OOMKilled", suggestion="raise memory", owner="lifecycle")
        self.assertEqual(self.roundtrip(hint)["owner"], "lifecycle")

    def test_diagnosis_nests_everything(self):
        diagnosis = Diagnosis(
            status=ContainerStatus(container="c", state="exited", health="failed", exit_code=3),
            logs=ContainerLogs(container="c", stderr="boom", lines=1),
            hints=[Hint(code="x", evidence="e", suggestion="s", owner="dockerization")],
        )
        data = self.roundtrip(diagnosis)
        self.assertEqual(data["status"]["exit_code"], 3)
        self.assertFalse(data["status"]["usable"])       # status keeps its derived fields when nested
        self.assertEqual(data["logs"]["stderr"], "boom")
        self.assertEqual(data["hints"][0]["owner"], "dockerization")

    def test_health_verdicts(self):
        self.assertEqual(set(HEALTH), {"starting", "healthy", "unhealthy", "completed", "stopped", "failed", "missing"})


class TestProvisionResult(unittest.TestCase):
    def result(self, mode: str, state: str, health: str, **kwargs) -> ProvisionResult:
        handle = ContainerHandle(id="d7d9d9bb3b41", name="tml-demo", image="demo:1", image_id="51dafde81dbd", mode=mode)
        status = ContainerStatus(container="tml-demo", state=state, health=health, handle=handle)
        return ProvisionResult(status=status, image=ImageRef(id="51dafde81dbd", tag="demo:1"), action="created", **kwargs)

    def test_healthy_is_ok_in_both_modes(self):
        self.assertTrue(self.result("idle", "running", "healthy").ok)
        self.assertTrue(self.result("native", "running", "healthy").ok)

    def test_completed_is_ok_only_in_native_mode(self):
        self.assertTrue(self.result("native", "exited", "completed").ok)      # the command ran to a clean end
        self.assertFalse(self.result("idle", "exited", "completed").ok)       # an idle container must stay up

    def test_other_verdicts_are_not_ok(self):
        for health in ("starting", "unhealthy", "stopped", "failed"):
            self.assertFalse(self.result("native", "running", health).ok, health)

    def test_missing_container_is_not_ok(self):
        status = ContainerStatus(container="gone", state="missing", health="missing")
        result = ProvisionResult(status=status, image=ImageRef(id="5", tag="demo:1"), action="created")
        self.assertFalse(result.ok)
        self.assertIsNone(result.handle)

    def test_handle_shortcut(self):
        self.assertEqual(self.result("idle", "running", "healthy").handle.name, "tml-demo")

    def test_to_dict(self):
        data = json.loads(self.result("idle", "running", "healthy").to_json())
        self.assertEqual(set(data), {"ok", "action", "image", "status", "diagnosis"})
        self.assertIs(data["ok"], True)
        self.assertEqual(data["action"], "created")
        self.assertEqual(data["image"], {"id": "51dafde81dbd", "tag": "demo:1", "built": False})
        self.assertTrue(data["status"]["usable"])                # status keeps its derived fields
        self.assertIsNone(data["diagnosis"])

    def test_to_dict_with_a_diagnosis(self):
        diagnosis = Diagnosis(status=ContainerStatus(container="tml-demo", state="exited", health="failed", exit_code=3))
        data = self.result("native", "exited", "failed", diagnosis=diagnosis).to_dict()
        self.assertIs(data["ok"], False)
        self.assertEqual(data["diagnosis"]["status"]["exit_code"], 3)


class TestTeardownResult(unittest.TestCase):
    def test_defaults_mean_nothing_was_done(self):
        r = TeardownResult(container="c", status=ContainerStatus(container="c", state="exited", health="stopped"))
        self.assertEqual((r.stopped, r.killed, r.removed, r.exit_code), (False, False, False, None))

    def test_to_dict(self):
        status = ContainerStatus(container="tml-demo", state="missing", health="missing", reason="removed")
        result = TeardownResult(container="tml-demo", status=status, stopped=True, killed=True, removed=True, exit_code=137)
        data = json.loads(result.to_json())
        self.assertEqual(set(data), {"container", "status", "stopped", "killed", "removed", "exit_code"})
        self.assertEqual((data["stopped"], data["killed"], data["removed"], data["exit_code"]), (True, True, True, 137))
        self.assertEqual(data["status"]["reason"], "removed")
        self.assertIs(data["status"]["usable"], False)           # status keeps its derived fields
        self.assertIsNone(data["status"]["handle"])

    def test_still_running_is_an_error_code(self):
        self.assertIn("still_running", ERROR_CODES)


class TestLifecycleError(unittest.TestCase):
    def test_fields_and_str(self):
        e = LifecycleError("not_found", "no such container: x", "list them first")
        self.assertEqual((e.code, e.message, e.hint), ("not_found", "no such container: x", "list them first"))
        self.assertEqual(str(e), "not_found: no such container: x")
        self.assertIsInstance(e, Exception)

    def test_hint_is_optional(self):
        self.assertEqual(LifecycleError("timeout", "too slow").hint, "")

    def test_to_dict(self):
        self.assertEqual(
            LifecycleError("oops", "m", "h").to_dict(),
            {"code": "oops", "message": "m", "hint": "h"},
        )

    def test_envelope_is_the_tool_failure_shape(self):
        env = LifecycleError("not_managed", "not ours", "only tml.managed containers").envelope()
        self.assertIs(env["ok"], False)
        self.assertEqual(env["error"]["code"], "not_managed")
        json.dumps(env)  # must be JSON serializable

    def test_error_codes_are_unique(self):
        self.assertEqual(len(ERROR_CODES), len(set(ERROR_CODES)))
        for code in ("docker_unavailable", "image_not_found", "not_found", "not_managed", "not_running", "name_conflict"):
            self.assertIn(code, ERROR_CODES)


if __name__ == "__main__":
    unittest.main()
