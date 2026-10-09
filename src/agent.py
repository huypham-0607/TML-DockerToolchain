from deepagents import create_deep_agent
from .llm import make_model
from .tools import meow, woof, inspect_repo, build_image


# System prompt to steer the agent to trace symbols through the local file tree
navigator_instructions = r"""You are a helpful agent. Use inspect_repo to inspect a local repository
when asked about its structure, dependencies, setup steps, or Python/CUDA/framework versions.
To containerize a repository: inspect it, write a Dockerfile, and call build_image. If the build fails,
read log_tail, fix the Dockerfile, and call build_image again (at most 5 attempts)."""

# Provider and model come from LLM_PROVIDER / LLM_MODEL (see src/llm.py)
model = make_model()

agent = create_deep_agent(
    model = model,
    tools = [meow, woof, inspect_repo, build_image],
    system_prompt = navigator_instructions,
)
