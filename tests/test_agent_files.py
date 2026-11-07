"""
    The agent's built-in file tools (src/fs_backend.py).

    Bug this covers: deepagents' default backend is an in-memory filesystem, so read_file / ls / glob
    answered "not found" for the repository that inspect_repo had just listed, and the agent gave up.

    The direct tests call the backend. The agent tests run a real deepagents graph, with the same
    backend as src/agent.py, driven by a scripted fake chat model: no API key, no network.
"""

import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.fs_backend import SCRATCH_ROOT, is_secret, make_backend
from tests.test_structure import make_repo


class ScriptedModel(BaseChatModel):
    """Answers with the next scripted AIMessage on every call."""

    script: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


class TestBackendDirect(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "toy"
        make_repo(self.repo, {
            "requirements.txt": "torch==1.13.1+cu116\nnumpy==1.24.2\n",
            "src/train.py": "import torch\n",
            ".env": "GENAI_API_KEY=sk-do-not-leak\n",
        })
        self.b = make_backend()

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_the_real_repository(self):
        r = self.b.read(str(self.repo / "requirements.txt"))
        self.assertIsNone(r.error)
        self.assertIn("torch==1.13.1+cu116", str(r.file_data))
        names = {Path(e["path"]).name for e in self.b.ls(str(self.repo)).entries}
        self.assertTrue({"requirements.txt", "src"} <= names)
        self.assertEqual([Path(m["path"]).name for m in self.b.glob("**/*.py", str(self.repo)).matches], ["train.py"])

    def test_writes_edits_and_deletes_are_refused(self):
        self.assertIn("read-only", self.b.write(str(self.repo / "new.txt"), "x").error)
        self.assertFalse((self.repo / "new.txt").exists())
        self.assertIn("read-only", self.b.edit(str(self.repo / "requirements.txt"), "numpy", "jax").error)
        self.assertIn("numpy", (self.repo / "requirements.txt").read_text())
        self.assertIsNotNone(self.b.delete(str(self.repo / "requirements.txt")).error)
        self.assertTrue((self.repo / "requirements.txt").exists())

    def test_secrets_are_not_served(self):
        self.assertIn("secrets", self.b.read(str(self.repo / ".env")).error)
        self.assertEqual(self.b.grep("sk-do-not-leak", str(self.repo)).matches, [])
        responses = self.b.download_files([str(self.repo / ".env"), str(self.repo / "requirements.txt")])
        self.assertEqual([r.error for r in responses], ["permission_denied", None])

    def test_secret_names(self):
        for p in ["/a/.env", "/a/.env.local", "/a/prod.env", "/h/.ssh/config", "/a/key.pem", "/h/.aws/credentials"]:
            self.assertTrue(is_secret(p), p)
        for p in ["/a/train.py", "/a/environment.yml", "/a/requirements.txt", "/a/envs/x.py"]:
            self.assertFalse(is_secret(p), p)


class TestBackendInAgent(unittest.TestCase):
    """The same backend inside a real deepagents graph: the bug happened there, not in the backend."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "toy"
        make_repo(self.repo, {"requirements.txt": "torch==1.13.1+cu116\n", ".env": "KEY=sk-do-not-leak\n"})

    def tearDown(self):
        self._tmp.cleanup()

    def run_agent(self, script):
        from deepagents import create_deep_agent

        model = ScriptedModel(script=script)
        agent = create_deep_agent(model=model, tools=[], system_prompt="test", backend=make_backend())
        out = agent.invoke({"messages": [{"role": "user", "content": "go"}]}, config={"recursion_limit": 30})
        return {m.tool_call_id: m.content for m in out["messages"] if isinstance(m, ToolMessage)}

    def test_agent_reads_real_files_and_cannot_write_them(self):
        req, env, new = (str(self.repo / n) for n in ("requirements.txt", ".env", "Dockerfile"))
        replies = self.run_agent([
            call("read_file", {"file_path": req}, 1),
            call("ls", {"path": str(self.repo)}, 2),
            call("read_file", {"file_path": env}, 3),
            call("write_file", {"file_path": new, "content": "FROM scratch\n"}, 4),
            call("write_file", {"file_path": f"{SCRATCH_ROOT}/notes.md", "content": "plan\n"}, 5),
            call("read_file", {"file_path": f"{SCRATCH_ROOT}/notes.md"}, 6),
            AIMessage(content="done"),
        ])
        self.assertIn("torch==1.13.1+cu116", replies["c1"])                 # the bug: was "not found"
        self.assertIn("requirements.txt", replies["c2"])
        self.assertNotIn("sk-do-not-leak", replies["c3"])
        self.assertIn("read-only", replies["c4"])
        self.assertFalse(os.path.exists(new))
        self.assertIn("plan", replies["c6"])                                  # scratch area works in memory
        self.assertFalse(os.path.exists(f"{SCRATCH_ROOT}/notes.md"))         # and never touches the disk


if __name__ == "__main__":
    unittest.main()
