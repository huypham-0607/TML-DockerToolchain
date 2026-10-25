import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

from src.lifecycle import __main__ as cli
from src.lifecycle import tools
from src.lifecycle.models import (
    ContainerHandle, ContainerLogs, ContainerSpec, ContainerStatus, Diagnosis, Hint,
    ImageRef, LifecycleError, ProvisionResult, TeardownResult,
)
from src.lifecycle.tools import (
    TOOL_FUNCTIONS, TOOLS, compact, container_logs, container_status, destroy_container,
    diagnose_container, diagnosis_view, provision_container, status_view, stop_container, teardown_view,
)
from tests.lifecycle import samples
from tests.lifecycle.fakes import FakeClient, FakeContainer, make_attrs, use_fake
from tests.lifecycle.samples import requires_docker

REPO = Path(__file__).resolve().parents[2]
TOOL_NAMES = ["provision_container", "container_status", "container_logs", "diagnose_container", "stop_container", "destroy_container"]


def tearDownModule():
    samples.cleanup()


def handle(mode: str = "idle") -> ContainerHandle:
    return ContainerHandle(
        id="d7d9d9bb3b41", name="tml-demo", image="demo:1", image_id="51dafde81dbd", mode=mode,
        created_at="2026-10-09T10:00:00Z", labels={"tml.managed": "true"},
    )


def status(state="running", health="healthy", **fields) -> ContainerStatus:
    return ContainerStatus(container="tml-demo", state=state, health=health, handle=handle(), **fields)


def reply(text: str) -> dict:
    data = json.loads(text)
    assert isinstance(data.get("ok"), bool), data
    return data


# ---------------------------------------------------------------- pure suites (no daemon)

class TestToolSchemas(unittest.TestCase):
    """What the LLM sees of each tool."""

    def test_six_tools_in_order(self):
        self.assertEqual([t.name for t in TOOLS], TOOL_NAMES)
        self.assertEqual([fn.__name__ for fn in TOOL_FUNCTIONS], TOOL_NAMES)

    def test_tools_call_the_plain_functions(self):
        for t, fn in zip(TOOLS, TOOL_FUNCTIONS):
            self.assertIs(t.func, fn)

    def test_every_argument_is_explained(self):
        for t in TOOLS:
            for arg, schema in t.args.items():
                self.assertTrue(schema.get("description"), f"{t.name}.{arg}")

    def test_every_tool_has_a_description(self):
        for t in TOOLS:
            self.assertGreater(len(t.description), 60, t.name)
            self.assertNotIn("Args:", t.description, t.name)        # argument help went into the schema

    def test_required_arguments(self):
        required = {t.name: sorted(a for a, s in t.args.items() if "default" not in s) for t in TOOLS}
        self.assertEqual(required, {
            "provision_container": [], "container_status": [], "container_logs": ["container"],
            "diagnose_container": ["container"], "stop_container": ["container"], "destroy_container": ["container"],
        })

    def test_provision_arguments(self):
        (t,) = [t for t in TOOLS if t.name == "provision_container"]
        self.assertEqual(list(t.args), [
            "image", "dockerfile", "context", "name", "mode", "command", "gpu", "env", "mounts", "ports",
            "memory", "shm_size", "user", "ready_timeout", "replace",
        ])
        self.assertNotIn("labels", t.args)                          # the LLM does not choose labels

    def test_options_named_by_hints_can_be_set(self):
        # every lifecycle hint suggests an option; the agent must be able to pass it
        (t,) = [t for t in TOOLS if t.name == "provision_container"]
        for option in ("gpu", "memory", "shm_size", "ports", "mode", "user", "ready_timeout"):
            self.assertIn(option, t.args)

    def test_owners_are_explained_where_hints_appear(self):
        for name in ("provision_container", "diagnose_container"):
            (t,) = [t for t in TOOLS if t.name == name]
            for owner in ("lifecycle", "dockerization", "execution", "host"):
                self.assertIn(owner, t.description, f"{name}: {owner}")

    def test_tool_invoke_runs_the_function(self):
        (t,) = [t for t in TOOLS if t.name == "container_status"]
        with mock.patch.object(tools, "get_status", return_value=status()) as get:
            data = reply(t.invoke({"container": "tml-demo"}))
        get.assert_called_once_with("tml-demo")
        self.assertTrue(data["ok"])


class TestCompactViews(unittest.TestCase):
    def test_compact(self):
        value = {"a": None, "b": "", "c": [], "d": {}, "e": False, "f": 0, "g": {"h": None, "i": 1}, "j": [{"k": None}]}
        self.assertEqual(compact(value), {"e": False, "f": 0, "g": {"i": 1}, "j": [{}]})

    def test_status_view(self):
        view = status_view(status(health="unhealthy", reason="running; the image HEALTHCHECK fails"))
        self.assertEqual(view["state"], "running")
        self.assertIs(view["usable"], False)                        # a false that matters stays
        self.assertNotIn("running", view)                           # state already says it
        self.assertNotIn("oom_killed", view)
        self.assertNotIn("restart_count", view)
        self.assertNotIn("exit_code", view)
        self.assertEqual(set(view["handle"]), {"name", "id", "image", "mode", "gpu"})   # empty workdir/ports/mounts left out
        self.assertNotIn("labels", view["handle"])
        self.assertNotIn("created_at", view["handle"])

    def test_status_view_keeps_what_explains_a_failure(self):
        view = status_view(status(state="exited", health="failed", exit_code=137, oom_killed=True, restart_count=2))
        self.assertEqual((view["exit_code"], view["oom_killed"], view["restart_count"]), (137, True, 2))

    def test_status_view_of_a_missing_container(self):
        view = status_view(ContainerStatus(container="gone", state="missing", health="missing", reason="no such container"))
        self.assertEqual(view, {"container": "gone", "state": "missing", "health": "missing", "reason": "no such container", "usable": False})

    def test_diagnosis_view(self):
        diagnosis = Diagnosis(
            status=status(state="exited", health="failed", exit_code=1),
            logs=ContainerLogs(container="tml-demo", stderr="ModuleNotFoundError: No module named 'torch'", lines=1),
            hints=[Hint(code="missing_python_module", evidence="ModuleNotFoundError: No module named 'torch'", suggestion="add it", owner="dockerization")],
        )
        view = diagnosis_view(diagnosis)
        self.assertEqual(set(view), {"status", "logs", "hints"})                 # empty parts left out
        self.assertEqual(view["logs"], {"stderr": "ModuleNotFoundError: No module named 'torch'", "lines": 1})
        self.assertEqual(view["hints"][0]["owner"], "dockerization")
        self.assertNotIn("status", diagnosis_view(diagnosis, with_status=False))

    def test_truncation_is_shown_only_when_it_happened(self):
        diagnosis = Diagnosis(status=status(), logs=ContainerLogs(container="c", stdout="x", lines=1, truncated=True))
        self.assertIs(diagnosis_view(diagnosis)["logs"]["truncated"], True)

    def test_teardown_view(self):
        result = TeardownResult(
            container="tml-demo", stopped=True, killed=False, removed=True, exit_code=143,
            status=ContainerStatus(container="tml-demo", state="missing", health="missing", reason="removed"),
        )
        view = teardown_view(result)
        self.assertEqual((view["stopped"], view["killed"], view["removed"], view["exit_code"]), (True, False, True, 143))
        self.assertEqual(view["status"]["health"], "missing")


class TestEnvelopes(unittest.TestCase):
    """Every tool returns JSON with "ok", and never raises."""

    IMAGE = ImageRef(id="51dafde81dbd", tag="tml-local/demo:abc", built=True)

    def patch(self, name: str, **kwargs) -> mock.Mock:
        patcher = mock.patch.object(tools, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def provision_reply(self, result: ProvisionResult, **arguments) -> tuple[dict, ContainerSpec, dict]:
        prov = self.patch("provision", return_value=result)
        data = reply(provision_container(**arguments))
        (spec,), kwargs = prov.call_args
        return data, spec, kwargs

    def test_provision_success(self):
        data, _, _ = self.provision_reply(ProvisionResult(status=status(), image=self.IMAGE, action="created"), image="demo:1")
        self.assertEqual(set(data), {"ok", "action", "image", "status"})
        self.assertIs(data["ok"], True)
        self.assertEqual(data["action"], "created")
        self.assertEqual(data["image"], {"id": "51dafde81dbd", "tag": "tml-local/demo:abc", "built": True})
        self.assertEqual(data["status"]["handle"]["name"], "tml-demo")

    def test_provision_notes_reach_the_llm(self):
        result = ProvisionResult(status=status(), image=self.IMAGE, action="created", notes=["gpu='auto': started without a GPU"])
        data, _, _ = self.provision_reply(result, image="demo:1")
        self.assertEqual(data["notes"], ["gpu='auto': started without a GPU"])

    def test_provision_failure_carries_the_diagnosis(self):
        failed = status(state="exited", health="failed", exit_code=1)
        diagnosis = Diagnosis(
            status=failed, logs=ContainerLogs(container="tml-demo", stderr="No module named 'torch'", lines=1),
            hints=[Hint(code="missing_python_module", evidence="No module named 'torch'", suggestion="add it", owner="dockerization")],
        )
        result = ProvisionResult(status=failed, image=self.IMAGE, action="created", diagnosis=diagnosis)
        data, _, _ = self.provision_reply(result, image="demo:1")
        self.assertIs(data["ok"], False)
        self.assertNotIn("error", data)                             # the request worked; the container did not
        self.assertEqual(data["diagnosis"]["hints"][0]["code"], "missing_python_module")
        self.assertNotIn("status", data["diagnosis"])               # it is already at the top
        self.assertEqual(data["status"]["exit_code"], 1)

    def test_provision_arguments_become_the_spec(self):
        _, spec, kwargs = self.provision_reply(
            ProvisionResult(status=status(), image=self.IMAGE, action="created"),
            dockerfile="repo/Dockerfile", context="repo", name="tml-x", mode="native", command="python train.py",
            gpu="none", env={"SEED": "1"}, mounts=["/data:/data:ro"], ports=["8888"], memory="8g",
            shm_size="4g", user="1000:1000", ready_timeout=60, replace=True,
        )
        self.assertEqual(
            (spec.dockerfile, spec.context, spec.name, spec.mode, spec.command, spec.gpu),
            ("repo/Dockerfile", "repo", "tml-x", "native", "python train.py", "none"),
        )
        self.assertEqual((spec.env, spec.mounts, spec.ports), ({"SEED": "1"}, ["/data:/data:ro"], ["8888"]))
        self.assertEqual((spec.memory, spec.shm_size, spec.user), ("8g", "4g", "1000:1000"))
        self.assertEqual(kwargs, {"replace": True, "ready_timeout": 60})

    def test_provision_defaults(self):
        _, spec, kwargs = self.provision_reply(ProvisionResult(status=status(), image=self.IMAGE, action="created"), image="demo:1", shm_size="")
        self.assertEqual((spec.env, spec.mounts, spec.ports, spec.labels), ({}, [], [], {}))
        self.assertEqual((spec.mode, spec.gpu, spec.shm_size), ("idle", "auto", "2g"))       # an empty shm_size is the default
        self.assertEqual(kwargs, {"replace": False, "ready_timeout": None})                 # 0 means the default timeout

    def test_session_labels_go_on_every_container(self):
        with mock.patch.dict(tools.SESSION_LABELS, {"tml.session": "abc"}):
            _, spec, _ = self.provision_reply(ProvisionResult(status=status(), image=self.IMAGE, action="created"), image="demo:1")
        self.assertEqual(spec.labels, {"tml.session": "abc"})

    def test_container_status_of_one(self):
        get = self.patch("get_status", return_value=status())
        data = reply(container_status("tml-demo"))
        get.assert_called_once_with("tml-demo")
        self.assertEqual((data["ok"], data["status"]["health"]), (True, "healthy"))

    def test_container_status_lists_all_without_an_argument(self):
        listing = self.patch("list_managed", return_value=[status(), status(state="exited", health="failed", exit_code=3)])
        data = reply(container_status())
        listing.assert_called_once_with()
        self.assertEqual([c["health"] for c in data["containers"]], ["healthy", "failed"])

    def test_container_status_of_none_is_an_empty_list(self):
        self.patch("list_managed", return_value=[])
        self.assertEqual(reply(container_status()), {"ok": True, "containers": []})

    def test_container_logs(self):
        logs = self.patch("get_logs", return_value=ContainerLogs(container="tml-demo", stdout="a\nb", lines=2, since="10m"))
        data = reply(container_logs("tml-demo", tail=5, since="10m", grep="a|b"))
        logs.assert_called_once_with("tml-demo", tail=5, since="10m", grep="a|b")
        self.assertEqual(data, {"ok": True, "logs": {"container": "tml-demo", "stdout": "a\nb", "lines": 2, "since": "10m"}})

    def test_diagnose_container(self):
        diag = self.patch("diagnose", return_value=Diagnosis(status=status(), resources={"memory_bytes": 10}))
        data = reply(diagnose_container("tml-demo"))
        diag.assert_called_once_with("tml-demo")
        self.assertEqual(data["diagnosis"]["resources"], {"memory_bytes": 10})
        self.assertEqual(data["diagnosis"]["status"]["health"], "healthy")

    def test_stop_and_destroy(self):
        gone = TeardownResult(container="tml-demo", status=ContainerStatus("tml-demo", "missing", "missing"), removed=True)
        stopped = TeardownResult(container="tml-demo", status=status(state="exited", health="stopped", exit_code=143), stopped=True, exit_code=143)
        stop_mock = self.patch("stop", return_value=stopped)
        destroy_mock = self.patch("destroy", return_value=gone)
        self.assertTrue(reply(stop_container("tml-demo", timeout=3))["result"]["stopped"])
        self.assertTrue(reply(destroy_container("tml-demo", force=True))["result"]["removed"])
        stop_mock.assert_called_once_with("tml-demo", timeout=3)
        destroy_mock.assert_called_once_with("tml-demo", force=True)

    def test_errors_become_the_error_envelope(self):
        error = LifecycleError("not_managed", "container 'caddy' was not created by this toolchain", "only tml.managed containers")
        for name in ("provision", "get_status", "list_managed", "get_logs", "diagnose", "stop", "destroy"):
            self.patch(name, side_effect=error)
        calls = [
            lambda: provision_container(image="demo:1"), lambda: container_status("caddy"), lambda: container_status(),
            lambda: container_logs("caddy"), lambda: diagnose_container("caddy"),
            lambda: stop_container("caddy"), lambda: destroy_container("caddy"),
        ]
        for call in calls:
            self.assertEqual(reply(call()), {"ok": False, "error": error.to_dict()})

    def test_a_bug_is_reported_not_raised(self):
        self.patch("get_status", side_effect=KeyError("State"))
        data = reply(container_status("tml-demo"))
        self.assertEqual(data["error"]["code"], "internal_error")
        self.assertIn("KeyError", data["error"]["message"])
        self.assertIn("toolchain", data["error"]["hint"])

    def test_invalid_arguments_from_the_llm(self):
        use_fake(self, FakeClient())
        for text in (
            provision_container(),                                            # neither image nor dockerfile
            provision_container(image="demo:1", mode="daemon"),
            provision_container(image="demo:1", mounts=["/etc:/host-etc"]),
            container_logs("tml-demo", tail=0),
            stop_container("tml-demo", timeout=-1),
        ):
            self.assertEqual(reply(text)["error"]["code"], "invalid_argument")

    def test_daemon_down(self):
        use_fake(self, FakeClient(error=requests.exceptions.ConnectionError("Connection refused")))
        for text in (container_status("tml-demo"), container_status(), destroy_container("tml-demo")):
            self.assertEqual(reply(text)["error"]["code"], "docker_unavailable")

    def test_with_a_fake_daemon(self):
        use_fake(self, FakeClient(FakeContainer(make_attrs(name="tml-demo", status="exited", exit_code=3))))
        data = reply(container_status("tml-demo"))
        self.assertEqual((data["status"]["health"], data["status"]["exit_code"]), ("failed", 3))
        self.assertEqual(reply(container_status("nope"))["status"]["health"], "missing")       # missing is an answer
        self.assertTrue(reply(destroy_container("tml-demo"))["result"]["removed"])
        self.assertIs(reply(destroy_container("tml-demo"))["result"]["removed"], False)


class TestCli(unittest.TestCase):
    """python -m src.lifecycle, run in this process."""

    def main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def patch_tool(self, name: str, answer: dict) -> mock.Mock:
        patcher = mock.patch.object(tools, name, return_value=json.dumps(answer))
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_exit_code_follows_ok(self):
        self.patch_tool("container_status", {"ok": True, "status": {"health": "healthy"}})
        self.assertEqual(self.main("status", "tml-demo")[0], 0)
        self.patch_tool("container_status", {"ok": False, "error": {"code": "not_found"}})
        self.assertEqual(self.main("status", "tml-demo")[0], 1)

    def test_prints_the_tool_json(self):
        answer = {"ok": True, "containers": [{"container": "tml-a", "health": "healthy"}]}
        status_mock = self.patch_tool("container_status", answer)
        code, out, _ = self.main("status")
        status_mock.assert_called_once_with("")
        self.assertEqual(json.loads(out), answer)
        self.assertGreater(len(out.splitlines()), 1)                # indented by default

    def test_compact_output(self):
        self.patch_tool("container_status", {"ok": True, "containers": []})
        _, out, _ = self.main("--compact", "status")
        self.assertEqual(out, '{"ok":true,"containers":[]}\n')

    def test_provision_options(self):
        prov = self.patch_tool("provision_container", {"ok": True})
        code, _, _ = self.main(
            "provision", "--dockerfile", "repo/Dockerfile", "--context", "repo", "--name", "tml-x",
            "--mode", "native", "--command", "python train.py --epochs 1", "--gpu", "none",
            "--env", "SEED=1", "--env", "MODE=a=b", "--mount", "/a:/a", "--mount", "/b:/b:ro",
            "--port", "8888", "--memory", "8g", "--shm-size", "4g", "--user", "1000:1000",
            "--ready-timeout", "60", "--replace",
        )
        self.assertEqual(code, 0)
        prov.assert_called_once_with(
            image="", dockerfile="repo/Dockerfile", context="repo", name="tml-x", mode="native",
            command="python train.py --epochs 1", gpu="none", env={"SEED": "1", "MODE": "a=b"},
            mounts=["/a:/a", "/b:/b:ro"], ports=["8888"], memory="8g", shm_size="4g",
            user="1000:1000", ready_timeout=60.0, replace=True,
        )

    def test_provision_defaults(self):
        prov = self.patch_tool("provision_container", {"ok": True})
        self.main("provision", "--image", "demo:1")
        self.assertEqual(prov.call_args.kwargs, {
            "image": "demo:1", "dockerfile": "", "context": "", "name": "", "mode": "idle", "command": "",
            "gpu": "auto", "env": {}, "mounts": [], "ports": [], "memory": "", "shm_size": "2g",
            "user": "", "ready_timeout": 0.0, "replace": False,
        })

    def test_labels_apply_to_this_call_only(self):
        seen = {}

        def fake_provision(**kwargs):
            seen.update(tools.SESSION_LABELS)
            return json.dumps({"ok": True})

        with mock.patch.object(tools, "provision_container", side_effect=fake_provision):
            self.main("provision", "--image", "demo:1", "--label", "tml.test=run1", "--label", "team=1")
        self.assertEqual(seen, {"tml.test": "run1", "team": "1"})
        self.assertEqual(tools.SESSION_LABELS, {})

    def test_other_commands(self):
        cases = [
            (("logs", "tml-demo", "--tail", "5", "--since", "10m", "--grep", "Err"), "container_logs", (("tml-demo",), {"tail": 5, "since": "10m", "grep": "Err"})),
            (("diagnose", "tml-demo"), "diagnose_container", (("tml-demo",), {})),
            (("stop", "tml-demo", "--timeout", "3"), "stop_container", (("tml-demo",), {"timeout": 3})),
            (("destroy", "tml-demo", "--force"), "destroy_container", (("tml-demo",), {"force": True})),
            (("destroy", "tml-demo"), "destroy_container", (("tml-demo",), {"force": False})),
        ]
        for argv, name, (args, kwargs) in cases:
            called = self.patch_tool(name, {"ok": True})
            self.assertEqual(self.main(*argv)[0], 0, argv)
            called.assert_called_once_with(*args, **kwargs)

    def test_destroy_all(self):
        gone = TeardownResult(container="tml-a", status=ContainerStatus("tml-a", "missing", "missing"), removed=True)
        with mock.patch.object(cli, "destroy_all_managed", return_value=[gone]) as everything:
            code, out, _ = self.main("destroy-all", "--force", "--label", "tml.test=run1")
        everything.assert_called_once_with(labels={"tml.test": "run1"}, force=True)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["results"][0]["container"], "tml-a")

    def test_destroy_all_without_labels(self):
        with mock.patch.object(cli, "destroy_all_managed", return_value=[]) as everything:
            self.main("destroy-all")
        everything.assert_called_once_with(labels=None, force=False)

    def test_destroy_all_failure(self):
        with mock.patch.object(cli, "destroy_all_managed", side_effect=LifecycleError("docker_error", "could not destroy 1 of 2")):
            code, out, _ = self.main("destroy-all")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["error"]["code"], "docker_error")

    def test_usage_errors(self):
        for argv in ((), ("provision",), ("provision", "--image", "a", "--dockerfile", "b"), ("logs",), ("status", "a", "b"), ("frobnicate",), ("provision", "--image", "a", "--mode", "daemon")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                cli.main(list(argv))
            self.assertEqual(ctx.exception.code, 2, argv)

    def test_bad_pairs(self):
        for argv in (("provision", "--image", "a", "--env", "SEED"), ("provision", "--image", "a", "--label", "=x"), ("destroy-all", "--label", "x")):
            code, out, err = self.main(*argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(out, "")
            self.assertIn("KEY=VALUE", err)

    def test_runs_as_a_module(self):
        done = subprocess.run([sys.executable, "-m", "src.lifecycle", "--help"], cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        for command in ("provision", "status", "logs", "diagnose", "stop", "destroy", "destroy-all"):
            self.assertIn(command, done.stdout)


class TestAgentWiring(unittest.TestCase):
    """src/agent.py hands the lifecycle tools to the agent, with rules on how to use them."""

    @classmethod
    def setUpClass(cls):
        # building the agent makes the chat model; a dummy key is enough, nothing is sent
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "rcac", "GENAI_API_KEY": "dummy-key-for-tests"}):
            cls.agent = importlib.import_module("src.agent")

    def test_lifecycle_tools_are_given_to_the_agent(self):
        names = [t.name for t in self.agent.TOOLS]
        for name in TOOL_NAMES + ["inspect_repo"]:
            self.assertIn(name, names)
        self.assertEqual(len(names), len(set(names)))

    def test_the_agent_gets_the_same_tool_objects(self):
        for t in TOOLS:
            self.assertIn(t, self.agent.TOOLS)

    def test_prompt_names_every_tool(self):
        for name in TOOL_NAMES + ["inspect_repo"]:
            self.assertIn(name, self.agent.SYSTEM_PROMPT)

    def test_prompt_rules(self):
        prompt = self.agent.SYSTEM_PROMPT
        for owner in ("lifecycle", "dockerization", "execution", "host"):
            self.assertIn(owner, prompt)
        self.assertIn("at most 3 provision attempts", prompt)
        self.assertIn("Destroy a failed container before", prompt)
        self.assertIn("A successful build is not a success", prompt)
        self.assertIn('"ok": true', prompt)

    def test_agent_is_built(self):
        self.assertTrue(hasattr(self.agent.agent, "invoke"))


# ---------------------------------------------------------------- live suites (real daemon)

class LiveCase(unittest.TestCase):
    """Containers made through the tools carry the test label, so cleanup finds them."""

    def setUp(self):
        patcher = mock.patch.dict(tools.SESSION_LABELS, samples.TEST_LABEL)
        patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def tearDownClass(cls):
        samples.remove_containers()      # the images stay until the module ends


@requires_docker
class TestHappyPath(LiveCase):
    def test_whole_lifecycle_through_the_tools(self):
        name = samples.container_name("flow-happy")
        dockerfile = samples.dockerfile("idle")

        started = reply(provision_container(dockerfile=dockerfile, name=name))
        self.assertIs(started["ok"], True)
        self.assertEqual((started["action"], started["status"]["handle"]["name"]), ("created", name))
        self.assertNotIn("diagnosis", started)

        now = reply(container_status(name))["status"]
        self.assertEqual((now["health"], now["usable"]), ("healthy", True))

        self.assertEqual(reply(container_logs(name))["logs"]["lines"], 0)          # idle: no output

        healthy = reply(diagnose_container(name))["diagnosis"]
        self.assertNotIn("hints", healthy)
        self.assertGreater(healthy["resources"]["memory_bytes"], 0)

        stopped = reply(stop_container(name))["result"]
        self.assertEqual((stopped["stopped"], stopped["killed"], stopped["status"]["health"]), (True, False, "stopped"))

        again = reply(provision_container(dockerfile=dockerfile, name=name))
        self.assertEqual((again["ok"], again["action"]), (True, "restarted"))
        self.assertIs(again["image"]["built"], False)

        gone = reply(destroy_container(name))["result"]
        self.assertEqual((gone["stopped"], gone["removed"]), (True, True))
        self.assertEqual(reply(container_status(name))["status"]["health"], "missing")
        self.assertIs(reply(destroy_container(name))["result"]["removed"], False)    # again: not an error

    def test_through_the_langchain_tools(self):
        by_name = {t.name: t for t in TOOLS}
        name = samples.container_name("flow-langchain")
        started = reply(by_name["provision_container"].invoke({"dockerfile": samples.dockerfile("idle"), "name": name}))
        self.assertIs(started["ok"], True)
        self.assertEqual(reply(by_name["container_status"].invoke({"container": name}))["status"]["health"], "healthy")
        self.assertTrue(reply(by_name["destroy_container"].invoke({"container": name}))["result"]["removed"])

    def test_listing(self):
        name = samples.container_name("flow-listed")
        reply(provision_container(dockerfile=samples.dockerfile("idle"), name=name))
        other = samples.start_unmanaged("flow-unlisted")
        names = [c["container"] for c in reply(container_status())["containers"]]
        self.assertIn(name, names)
        self.assertNotIn(other.name, names)


@requires_docker
class TestFailurePath(LiveCase):
    def test_failed_start_names_the_owner(self):
        name = samples.container_name("flow-failed")
        failed = reply(provision_container(dockerfile=samples.dockerfile("missing-module"), name=name, mode="native"))
        self.assertIs(failed["ok"], False)
        self.assertNotIn("error", failed)
        self.assertEqual((failed["status"]["health"], failed["status"]["exit_code"]), ("failed", 1))
        (hint,) = failed["diagnosis"]["hints"]
        self.assertEqual((hint["code"], hint["owner"]), ("missing_python_module", "dockerization"))
        self.assertIn("No module named 'torch'", failed["diagnosis"]["logs"]["stderr"])

        later = reply(diagnose_container(name))["diagnosis"]
        self.assertEqual(later["hints"][0]["code"], "missing_python_module")
        self.assertTrue(reply(destroy_container(name))["result"]["removed"])

    def test_lifecycle_hint_then_a_better_provision(self):
        # the hint says: more memory. The agent destroys and provisions again with a larger limit.
        name = samples.container_name("flow-oom")
        dockerfile = samples.dockerfile("oom")
        first = reply(provision_container(dockerfile=dockerfile, name=name, mode="native", memory="64m"))
        self.assertEqual([(h["code"], h["owner"]) for h in first["diagnosis"]["hints"]], [("out_of_memory", "lifecycle")])
        reply(destroy_container(name))
        second = reply(provision_container(dockerfile=dockerfile, name=name, mode="native", memory="128m"))
        self.assertEqual(second["action"], "created")
        self.assertEqual(second["diagnosis"]["hints"][0]["code"], "out_of_memory")   # this sample never has enough

    def test_conflict_without_destroy(self):
        name = samples.container_name("flow-conflict")
        reply(provision_container(dockerfile=samples.dockerfile("idle"), name=name))
        clash = reply(provision_container(dockerfile=samples.dockerfile("idle"), name=name, memory="256m"))
        self.assertEqual(clash["error"]["code"], "name_conflict")
        self.assertIn("replace=True", clash["error"]["hint"])
        replaced = reply(provision_container(dockerfile=samples.dockerfile("idle"), name=name, memory="256m", replace=True))
        self.assertEqual((replaced["ok"], replaced["action"]), (True, "replaced"))


@requires_docker
class TestToolErrors(LiveCase):
    def test_unmanaged_container_is_refused_by_every_tool(self):
        other = samples.start_unmanaged("flow-refused")
        for text in (
            container_status(other.name), container_logs(other.name), diagnose_container(other.name),
            stop_container(other.name), destroy_container(other.name, force=True),
        ):
            self.assertEqual(reply(text)["error"]["code"], "not_managed")
        other.reload()
        self.assertEqual(other.status, "running")

    def test_missing_image(self):
        data = reply(provision_container(image="tml-local/never-built:000000000000", name=samples.container_name("flow-noimage")))
        self.assertEqual(data["error"]["code"], "image_not_found")

    def test_failed_build(self):
        data = reply(provision_container(dockerfile=samples.dockerfile("broken-build"), name=samples.container_name("flow-broken")))
        self.assertEqual(data["error"]["code"], "build_failed")
        self.assertIn("about to fail", data["error"]["message"])

    def test_missing_container(self):
        missing = samples.container_name("flow-never-made")
        self.assertEqual(reply(container_logs(missing))["error"]["code"], "not_found")
        self.assertEqual(reply(stop_container(missing))["error"]["code"], "not_found")
        self.assertEqual(reply(diagnose_container(missing))["diagnosis"]["status"]["health"], "missing")

    def test_mount_outside_the_allowed_roots(self):
        data = reply(provision_container(dockerfile=samples.dockerfile("idle"), mounts=["/etc:/host-etc:ro"]))
        self.assertEqual(data["error"]["code"], "invalid_argument")


@requires_docker
class TestCliLive(LiveCase):
    """The command line in its own process, against the daemon."""

    def cli(self, *argv: str) -> tuple[int, dict]:
        done = subprocess.run(
            [sys.executable, "-m", "src.lifecycle", "--compact", *argv],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        return done.returncode, (json.loads(done.stdout) if done.stdout.strip() else {"stderr": done.stderr})

    def test_whole_lifecycle(self):
        name = samples.container_name("cli-flow")
        label = f"tml.test={samples.RUN_ID}"
        code, data = self.cli("provision", "--dockerfile", samples.dockerfile("idle"), "--name", name, "--label", label)
        self.assertEqual((code, data["ok"], data["status"]["health"]), (0, True, "healthy"))
        self.assertEqual(samples.sdk_container(name).labels["tml.test"], samples.RUN_ID)

        self.assertEqual(self.cli("status", name)[0], 0)
        self.assertEqual(self.cli("logs", name, "--tail", "5")[0], 0)
        self.assertEqual(self.cli("diagnose", name)[0], 0)
        code, data = self.cli("stop", name, "--timeout", "2")
        self.assertEqual((code, data["result"]["stopped"]), (0, True))
        code, data = self.cli("destroy", name)
        self.assertEqual((code, data["result"]["removed"]), (0, True))
        code, data = self.cli("status", name)
        self.assertEqual((code, data["status"]["health"]), (0, "missing"))

    def test_failed_start_exits_with_1(self):
        name = samples.container_name("cli-crash")
        code, data = self.cli("provision", "--dockerfile", samples.dockerfile("crash"), "--name", name,
                              "--mode", "native", "--label", f"tml.test={samples.RUN_ID}")
        self.assertEqual((code, data["ok"], data["status"]["exit_code"]), (1, False, 3))
        self.assertEqual(data["diagnosis"]["logs"]["stderr"], "boom")

    def test_destroy_all_of_one_label(self):
        label = f"tml.test={samples.RUN_ID}"
        for suffix in ("cli-all-a", "cli-all-b"):
            self.cli("provision", "--dockerfile", samples.dockerfile("idle"), "--name", samples.container_name(suffix), "--label", label)
        code, data = self.cli("destroy-all", "--label", label)
        self.assertEqual(code, 0)
        self.assertLessEqual({samples.container_name("cli-all-a"), samples.container_name("cli-all-b")}, {r["container"] for r in data["results"]})
        self.assertEqual(self.cli("status", samples.container_name("cli-all-a"))[1]["status"]["health"], "missing")

    def test_refused_container_exits_with_1(self):
        other = samples.start_unmanaged("cli-refused")
        code, data = self.cli("destroy", other.name)
        self.assertEqual((code, data["error"]["code"]), (1, "not_managed"))


if __name__ == "__main__":
    unittest.main()
