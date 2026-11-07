"""
    Chat model selection

    Pick the provider with LLM_PROVIDER (default: rcac) and optionally the model with LLM_MODEL.
    Both providers speak the OpenAI API; only the endpoint, key and default model differ.

      LLM_PROVIDER=rcac    -> GENAI_API_KEY,  default model gpt-oss:120b (Purdue RCAC GenAI)
      LLM_PROVIDER=openai  -> OPENAI_API_KEY, default model gpt-5-mini

    To add a provider, add an entry to PROVIDERS.

    Rate limits. Purdue GenAI allows 20 requests per minute per user
    (https://docs.rcac.purdue.edu/services/genai/api/). Two things keep a run going:
      - a client-side limiter, set to LLM_REQUESTS_PER_MINUTE (default: the provider's value);
      - retries with waiting when the server still refuses. RCAC answers a rate limit with HTTP 400
        ("Rate limit exceeded"), not 429, so the OpenAI client's own retries never fire; a busy backend
        also comes as HTTP 400 ("Open WebUI: Server Connection Error"); and it may answer with JSON null
        instead of an error. All of these are retried LLM_RATE_RETRIES times (default 6),
        waiting 15 s, 30 s, then 60 s between tries.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

# Reads .env from the repository root (existing environment variables win)
load_dotenv()


PROVIDERS = {
    "openai": {
        "model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "requests_per_minute": 120,
    },
    "rcac": {
        "model": "gpt-oss:120b",
        "api_key_env": "GENAI_API_KEY",
        "base_url": "https://genai.rcac.purdue.edu/api",
        "requests_per_minute": 18,   # the limit is 20/min per user; keep a margin
    },
}

DEFAULT_RATE_RETRIES = 6
RETRY_WAITS = (15.0, 30.0, 60.0)   # seconds before retry 1, 2, and 3 onwards


# Answers that mean "not now" rather than "wrong request". RCAC sends several of them as HTTP 400:
# "Rate limit exceeded", and "Open WebUI: Server Connection Error" when its model backend is busy.
RETRYABLE_TEXT = (
    "rate limit", "too many requests", "error code: 429",
    "server connection error", "temporarily unavailable", "service unavailable", "bad gateway",
    "gateway timeout", "error code: 502", "error code: 503", "error code: 504",
)


def is_retryable(error: BaseException) -> bool:
    """True for a refused-for-now answer: a rate limit or a busy server, or the null body RCAC sends instead of an error."""
    text = str(error).lower()
    if any(t in text for t in RETRYABLE_TEXT):
        return True
    # a JSON null body breaks response parsing with a NoneType error deep in the client
    return isinstance(error, (TypeError, AttributeError)) and "nonetype" in text


def retry_wait(attempt: int) -> float:
    """Seconds to wait before retry number `attempt` (0-based)."""
    return RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]


class RetryingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that waits and retries when the server refuses for now (see is_retryable)."""

    rate_limit_retries: int = DEFAULT_RATE_RETRIES
    wait_scale: float = 1.0   # tests set 0 so that nothing sleeps

    def _wait(self, attempt: int, error: BaseException) -> float:
        seconds = retry_wait(attempt) * self.wait_scale
        print(f"[llm] server refused ({str(error)[:80]}); waiting {seconds:.0f}s, "
              f"retry {attempt + 1}/{self.rate_limit_retries}", file=sys.stderr, flush=True)
        return seconds

    def _generate(self, *args, **kwargs):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - re-raised unless it is a retryable refusal
                if attempt == self.rate_limit_retries or not is_retryable(e):
                    raise
                time.sleep(self._wait(attempt, e))
        raise AssertionError("unreachable")

    async def _agenerate(self, *args, **kwargs):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if attempt == self.rate_limit_retries or not is_retryable(e):
                    raise
                await asyncio.sleep(self._wait(attempt, e))
        raise AssertionError("unreachable")


def make_model(provider: str | None = None, model: str | None = None) -> RetryingChatOpenAI:
    """Builds the chat model. Arguments override LLM_PROVIDER / LLM_MODEL."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "rcac")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; choose one of {sorted(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set (needed for LLM_PROVIDER={provider}); add it to .env")

    per_minute = float(os.environ.get("LLM_REQUESTS_PER_MINUTE") or cfg["requests_per_minute"])
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    return RetryingChatOpenAI(
        model=model or os.environ.get("LLM_MODEL") or cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
        rate_limiter=rate_limiter,
        max_retries=3,   # the OpenAI client's own retries: connection errors, 429, 5xx
        rate_limit_retries=int(os.environ.get("LLM_RATE_RETRIES") or DEFAULT_RATE_RETRIES),
    )
