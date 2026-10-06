"""
    Chat model selection

    Pick the provider with LLM_PROVIDER (default: rcac) and optionally the model with LLM_MODEL.
    Both providers speak the OpenAI API; only the endpoint, key and default model differ.

      LLM_PROVIDER=rcac    -> GENAI_API_KEY,  default model gpt-oss:120b (Purdue RCAC GenAI)
      LLM_PROVIDER=openai  -> OPENAI_API_KEY, default model gpt-5-mini

    To add a provider, add an entry to PROVIDERS.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

# Reads .env from the repository root (existing environment variables win)
load_dotenv()


PROVIDERS = {
    "openai": {
        "model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "rcac": {
        "model": "gpt-oss:120b",
        "api_key_env": "GENAI_API_KEY",
        "base_url": "https://genai.rcac.purdue.edu/api",
    },
}


def make_model(provider: str | None = None, model: str | None = None):
    """Builds the chat model. Arguments override LLM_PROVIDER / LLM_MODEL."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "rcac")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; choose one of {sorted(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set (needed for LLM_PROVIDER={provider}); add it to .env")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2,   # 120/min, comfortable for Purdue GenAI
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    return init_chat_model(
        model or os.environ.get("LLM_MODEL") or cfg["model"],
        model_provider="openai",
        base_url=cfg["base_url"],
        api_key=api_key,
        rate_limiter=rate_limiter,
        max_retries=3,
    )
