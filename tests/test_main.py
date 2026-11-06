"""src/main.py: the step limit, and a clear message instead of a traceback when it runs out."""

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from langgraph.errors import GraphRecursionError

from src import main as entry


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.config = result, error, None

    def invoke(self, inp, config):
        self.config = config
        if self.error:
            raise self.error
        return self.result


def run(agent, argv=("x",), env=None):
    fake_module = types.ModuleType("src.agent")
    fake_module.agent = agent
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(sys.modules, {"src.agent": fake_module}), mock.patch.object(sys, "argv", ["main", *argv]), \
            mock.patch.dict(os.environ, env or {}), redirect_stdout(out), redirect_stderr(err):
        code = entry.main()
    return code, out.getvalue(), err.getvalue()


class TestMain(unittest.TestCase):
    def test_default_limit_is_far_above_50(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(entry.step_limit(), 150)

    def test_limit_from_env(self):
        agent = FakeAgent(result={"messages": [types.SimpleNamespace(content="done")]})
        code, out, _ = run(agent, env={"TML_STEP_LIMIT": "300"})
        self.assertEqual((code, out.strip(), agent.config["recursion_limit"]), (0, "done", 300))

    def test_running_out_of_steps_is_a_message_not_a_traceback(self):
        code, _, err = run(FakeAgent(error=GraphRecursionError("Recursion limit of 150 reached")))
        self.assertEqual(code, 1)
        self.assertIn("TML_STEP_LIMIT", err)


    def test_missing_api_key_is_a_message(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("builtins.__import__", side_effect=self._importer(RuntimeError("GENAI_API_KEY is not set"))), \
                mock.patch.object(sys, "argv", ["main", "x"]), redirect_stdout(out), redirect_stderr(err):
            code = entry.main()
        self.assertEqual(code, 2)
        self.assertIn("GENAI_API_KEY is not set", err.getvalue())

    @staticmethod
    def _importer(error):
        real = __import__

        def fake(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "agent" and level == 1 or name == "src.agent":
                raise error
            return real(name, globals, locals, fromlist, level)
        return fake


if __name__ == "__main__":
    unittest.main()
