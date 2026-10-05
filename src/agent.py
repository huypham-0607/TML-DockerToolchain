import os

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from deepagents import create_deep_agent
from .tools import meow, woof


# System prompt to steer the agent to trace symbols through the local file tree
navigator_instructions = r"""You are a helpful agent, you task is to either meow or woof"""

rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,   # 120/min, comfortable for Purdue GenAI
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

model = init_chat_model(
    "gpt-oss:120b",
    model_provider="openai",
    base_url="https://genai.rcac.purdue.edu/api",
    api_key=os.environ["GENAI_API_KEY"],
    rate_limiter=rate_limiter,
    max_retries=3,
)

agent = create_deep_agent(
    model = model,
    tools = [meow,woof],
    system_prompt = navigator_instructions,
)
