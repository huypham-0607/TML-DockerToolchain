"""
    src/llm.py: model settings and the retries on rate limits. No network: the parent's
    _generate is replaced, so the tests check only what RetryingChatOpenAI adds.
"""

import os
import unittest
from unittest import mock

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from src import llm

# What RCAC really answered (Mon 2 Nov, the agent run on FFB)
RCAC_RATE_LIMIT = "Error code: 400 - {'detail': 'Rate limit exceeded. Please try again later.'}"
OK = ChatResult(generations=[ChatGeneration(message=AIMessage(content="pong"))])


class FakeBadRequest(Exception):
    pass


def model(retries=3):
    with mock.patch.dict(os.environ, {"GENAI_API_KEY": "test-key", "LLM_RATE_RETRIES": str(retries)}):
        m = llm.make_model("rcac")
    m.wait_scale = 0.0
    return m


class TestIsRetryable(unittest.TestCase):
    def test_rcac_rate_limit_400(self):
        self.assertTrue(llm.is_retryable(FakeBadRequest(RCAC_RATE_LIMIT)))

    def test_rcac_server_connection_error(self):
        # ended the UAP agent run after 26 minutes (Sat 7 Nov)
        self.assertTrue(llm.is_retryable(FakeBadRequest("Error code: 400 - {'detail': 'Open WebUI: Server Connection Error'}")))
        self.assertTrue(llm.is_retryable(Exception("Error code: 503 - Service Unavailable")))

    def test_429(self):
        self.assertTrue(llm.is_retryable(Exception("Error code: 429 - Too Many Requests")))

    def test_null_body(self):
        self.assertTrue(llm.is_retryable(TypeError("'NoneType' object is not subscriptable")))

    def test_real_errors_are_not_retried(self):
        for e in [FakeBadRequest("Error code: 400 - invalid model"), ValueError("bad"), TypeError("unexpected keyword")]:
            self.assertFalse(llm.is_retryable(e), e)

    def test_waits_grow_then_cap(self):
        self.assertEqual([llm.retry_wait(i) for i in range(5)], [15.0, 30.0, 60.0, 60.0, 60.0])


class TestRetries(unittest.TestCase):
    def test_retries_a_rate_limit_then_succeeds(self):
        m = model()
        answers = [FakeBadRequest(RCAC_RATE_LIMIT), FakeBadRequest(RCAC_RATE_LIMIT), OK]

        def fake(*a, **k):
            x = answers.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

        with mock.patch.object(ChatOpenAI, "_generate", side_effect=fake) as gen, mock.patch("time.sleep") as sleep:
            out = m._generate([])
        self.assertEqual(out.generations[0].message.content, "pong")
        self.assertEqual(gen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_gives_up_after_the_retry_limit(self):
        m = model(retries=2)
        with mock.patch.object(ChatOpenAI, "_generate", side_effect=FakeBadRequest(RCAC_RATE_LIMIT)) as gen:
            with self.assertRaises(FakeBadRequest):
                m._generate([])
        self.assertEqual(gen.call_count, 3)   # first try + 2 retries

    def test_other_errors_are_raised_at_once(self):
        m = model()
        with mock.patch.object(ChatOpenAI, "_generate", side_effect=FakeBadRequest("Error code: 400 - bad request")) as gen:
            with self.assertRaises(FakeBadRequest):
                m._generate([])
        self.assertEqual(gen.call_count, 1)

    def test_async_path_retries_too(self):
        import asyncio
        m = model()
        answers = [FakeBadRequest(RCAC_RATE_LIMIT), OK]

        async def fake(*a, **k):
            x = answers.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

        with mock.patch.object(ChatOpenAI, "_agenerate", side_effect=fake):
            out = asyncio.run(m._agenerate([]))
        self.assertEqual(out.generations[0].message.content, "pong")


class TestSettings(unittest.TestCase):
    def test_rcac_stays_under_20_per_minute(self):
        m = model()
        self.assertLess(m.rate_limiter.requests_per_second * 60, 20)
        self.assertEqual(m.openai_api_base, "https://genai.rcac.purdue.edu/api")
        self.assertEqual(m.model_name, "gpt-oss:120b")

    def test_overrides(self):
        env = {"GENAI_API_KEY": "k", "LLM_REQUESTS_PER_MINUTE": "6", "LLM_MODEL": "other-model"}
        with mock.patch.dict(os.environ, env):
            m = llm.make_model("rcac")
        self.assertAlmostEqual(m.rate_limiter.requests_per_second, 0.1)
        self.assertEqual(m.model_name, "other-model")

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(llm, "load_dotenv"):
            with self.assertRaises(RuntimeError) as ctx:
                llm.make_model("rcac")
        self.assertIn("GENAI_API_KEY", str(ctx.exception))

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            llm.make_model("nope")

    def test_model_still_binds_tools(self):
        # deepagents calls bind_tools; the subclass must keep that working
        from langchain_core.tools import tool

        @tool
        def ping() -> str:
            """ping"""
            return "pong"

        self.assertIsNotNone(model().bind_tools([ping]))


if __name__ == "__main__":
    unittest.main()



class TestToolNames(unittest.TestCase):
    def test_leaked_chat_format_tokens_are_cut_from_tool_names(self):
        self.assertEqual(llm.clean_tool_name("container_status<|channel|>commentary"), "container_status")
        self.assertEqual(llm.clean_tool_name("build_image"), "build_image")
        msg = AIMessage(content="", tool_calls=[
            {"name": "container_status<|channel|>commentary", "args": {"container": ""}, "id": "1"},
            {"name": "ls", "args": {"path": "/"}, "id": "2"}])
        fixed = llm.fix_tool_names(ChatResult(generations=[ChatGeneration(message=msg)]))
        self.assertEqual([c["name"] for c in fixed.generations[0].message.tool_calls], ["container_status", "ls"])
