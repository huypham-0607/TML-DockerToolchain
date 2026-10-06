# TML-DockerToolchain

Automated containerization infrastructure for trustworthy machine learning (TML) research repositories.
The toolchain changes a research repository into a validated Docker environment.

> **Note:** Development is ongoing. Most of the design in this document is not implemented at this time. Refer to [Status](#status).

## Contents

- [Background: Landseer](#background-landseer)
- [Project Scope](#project-scope)
- [Setup](#setup)
- [Status](#status)
- [Agent Toolchain Categories](#agent-toolchain-categories)
- [Agent Workflow](#agent-workflow)
- [Project Layout](#project-layout)
- [Important Design Decisions](#important-design-decisions)
- [Next Steps](#next-steps)

## Background: Landseer

- Broader motivation: it is difficult to containerize TML repositories.
- Manual reconstruction of an environment is slow and causes errors.
- The goal is automated containerization.
- Development is ongoing.

## Project Scope

- Team 1 is responsible only for the automated containerization infrastructure for TML research repositories.
- Goal: change a repository into a validated Docker environment for the downstream teams.
- The downstream teams use the environment for these tasks:
  - Benchmark modification
  - Experiment reproduction
  - Metric validation
- A generated Dockerfile is successful only if the image builds and the container starts.

### Core Pipeline

1. Research repository
2. Dockerfile
3. Docker image
4. Running container
5. Generic command-execution interface

## Setup

### Prerequisites

| Item | Requirement | Purpose |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | A recent version | Installs Python and the dependencies. |
| Python | 3.14 or later | uv installs this version automatically if it is not on your computer. |
| LLM API key | A Purdue GenAI key for `genai.rcac.purdue.edu`, or an OpenAI key from [platform.openai.com](https://platform.openai.com/api-keys) | The agent uses this service for the LLM. Refer to [Select the LLM](#select-the-llm). |
| Docker | A recent version | Necessary for the planned pipeline. The current code does not call Docker. |

### Install

1. Clone the repository.

   ```bash
   git clone https://github.com/huypham-0607/TML-DockerToolchain.git
   cd TML-DockerToolchain
   ```

2. Install the dependencies. This command makes the `.venv/` directory.

   ```bash
   uv sync
   ```

### Set the Environment Variables

1. Make a file with the name `.env` in the repository root.
2. Add the variables to the file.

   ```bash
   export GENAI_API_KEY="<your Purdue GenAI API key>"
   # Optional: use OpenAI instead of Purdue GenAI
   # export LLM_PROVIDER="openai"
   # export OPENAI_API_KEY="<your OpenAI API key>"
   export LANGSMITH_TRACING="true"
   export LANGSMITH_API_KEY="<your LangSmith API key>"
   ```

3. The agent loads `.env` automatically (`python-dotenv`). Variables that are already set in your shell take priority.

| Variable | Necessary | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | No | `rcac` (default) or `openai`. |
| `LLM_MODEL` | No | Overrides the default model of the provider. |
| `GENAI_API_KEY` | If `LLM_PROVIDER=rcac` (default) | The key for the Purdue GenAI API. |
| `OPENAI_API_KEY` | If `LLM_PROVIDER=openai` | The key for the OpenAI API. |
| `LANGSMITH_TRACING` | No | Set to `true` to record LangSmith traces. |
| `LANGSMITH_API_KEY` | No | Necessary only if LangSmith tracing is on. |

> **Caution:** Do not commit `.env`. Git ignores this file.

### Select the LLM

`src/llm.py` makes the chat model. Set `LLM_PROVIDER` in `.env` to change the provider. You do not have to change the code.

| `LLM_PROVIDER` | Default model | Endpoint | Key |
| --- | --- | --- | --- |
| `rcac` (default) | `gpt-oss:120b` | `https://genai.rcac.purdue.edu/api` | `GENAI_API_KEY` |
| `openai` | `gpt-5-mini` | OpenAI | `OPENAI_API_KEY` |

To add a provider, add an entry to `PROVIDERS` in `src/llm.py`.

### Run

- Do a check of the installation. This command prints `Hello from tml-dockertoolchain!`.

  ```bash
  uv run main.py
  ```

- Start the placeholder agent with a question. If you give no question, the agent uses `Hello world!`.
  The agent prints its last reply.

  ```bash
  uv run python -m src.main "your question"
  ```

## Status

Current work: repository exploration (repository inspection).

### Repository Inspection

Abstract: inspect these items in a repository:

- Repository structure
- Dependency files
- README setup instructions
- Python, CUDA, and framework versions
- Existing Docker configuration
- Relevant environment metadata

Sections:

| Section | Name | Scope | Module |
| --- | --- | --- | --- |
| A | Repository Structure | File tree, languages, important files, existing Docker configuration | `repo/structure.py` |
| B | Dependency Detection | `requirements.txt`, `pyproject.toml`, `environment.yml`, `setup.py`, package versions | `repo/dependencies.py` |
| C | Documentation / Search | README setup extraction, installation commands, optional Google search fallback | `repo/documentation.py` |
| D | Environment Detection | Python, CUDA, and PyTorch/TensorFlow versions. Combine the evidence from code, configuration, and documentation. | `repo/environment.py` |

### Current Code

The agent and the tools in `src/` are placeholders at this time.

| File | State |
| --- | --- |
| `main.py` | Placeholder. Prints a greeting. |
| `src/main.py` | Command-line entry point. Sends one question to the agent and prints the last reply. |
| `src/agent.py` | Makes the agent with `deepagents`. Gets the model from `src/llm.py`. |
| `src/llm.py` | Makes the chat model for the provider in `LLM_PROVIDER` (OpenAI or Purdue GenAI). The rate limit is 2 requests each second. |
| `src/tools.py` | Placeholder. Contains two temporary tools: `meow` and `woof`. Each tool returns a fixed string. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Project metadata, locked dependencies, and the Python version. |

These parts of the design are not in the repository at this time:

- Repository inspection modules (sections A to D)
- Dockerization, container lifecycle, and container execution
- Dockerfile templates
- Schemas for the structured return objects
- Tests

## Agent Toolchain Categories

| Category | Functions |
| --- | --- |
| Repository inspection | Inspect the repository structure, dependency files, and README setup instructions. Inspect the Python, CUDA, and framework versions, the existing Docker configuration, and the environment metadata. |
| Dockerization | Generate or select a Dockerfile template. Build the Docker image. Inspect build failures. Repair and try again until the build is successful or the retry limit is reached. |
| Container lifecycle | Provision and start a container from an image. Check the status. Stop the container. Remove the container. |
| Container execution | Execute arbitrary commands in a provisioned container. Return structured results: stdout, stderr, exit code, and timeout status. |

### LLM-Facing Tools

Keep the main LLM-facing tools high-level:

- `dockerize_repo(repo)`
- `build_image(dockerfile)` (possibly internal to `dockerize_repo()`)
- `provision_container(image, name)`
- `run_command_in_container(container, cmd)`
- `destroy_container(container)`

### Lower-Level Helpers

Lower-level helpers stay as ordinary Python functions. They are not agent tools.
Examples: dependency detection and log parsing.

## Agent Workflow

1. Inspect the repository.
2. Infer the software and environment requirements.
3. Select a Dockerfile template.
4. Generate the Dockerfile.
5. Build the Docker image.
6. If the build fails, do these steps again until the fixed retry limit:
   1. Read the logs.
   2. Diagnose the failure.
   3. Modify the Dockerfile.
   4. Build the image again.
7. Provision the container.
8. Run basic smoke tests and commands.
9. Return the validated container artifact.

### Key Split

- The LLM does the reasoning and makes the repair decisions.
- Deterministic Python and Docker code does the container operations.

### Smoke Tests

- Smoke tests are essential.
- A successful build is not sufficient. The tests must also show that the basic functions run.
- Example risk: IPmix hardcoded CUDA. This broke environments that have no GPU. Similar problems can occur again.

## Project Layout

### Current Layout

```text
TML-DockerToolchain/
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
└── src/
    ├── __init__.py
    ├── agent.py
    ├── main.py
    └── tools.py
```

### Proposed Layout

Proposed repository structure for `container-agent/`:

```text
container-agent/
├── src/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── state.py
│   ├── repo/
│   │   ├── inspect.py
│   │   ├── dependencies.py
│   │   └── setup_parser.py
│   ├── docker/
│   │   ├── dockerize.py
│   │   ├── build.py
│   │   ├── provision.py
│   │   ├── exec.py
│   │   └── lifecycle.py
│   └── schemas/
│       ├── repository.py
│       ├── build.py
│       └── container.py
├── templates/
│   ├── python.dockerfile
│   ├── conda.dockerfile
│   └── pytorch-cuda.dockerfile
├── tests/
├── runs/
└── pyproject.toml
```

## Important Design Decisions

### The Team Boundary Is Explicit

- Team 1 produces the environment.
- Team 1 does not decide which experiment to run.
- Team 1 does not decide if a paper metric was reproduced.
- `run_command_in_container()` is a generic execution primitive for the other teams.

### Use Structured Return Objects, Not Raw Text

- Objects: `RepoProfile`, `BuildResult`, `ContainerHandle`, `CommandResult`
- Structured objects make LangChain tool calls more reliable.
- Structured objects give the downstream teams stable interfaces.

### First Milestone Success Definition (Simple)

Repository → Dockerfile → successful image build → container starts → arbitrary command executes

### Docker First, Apptainer Later

- Do all the work in Docker first.
- Then migrate to Apptainer for RCAC/Purdue SLURM support.
- Future path: Repository → Dockerfile → Docker/OCI image → Apptainer/SIF → RCAC
- Team 1 is not responsible for SLURM jobs, experiment reproduction, or metric validation.

### Inter-Team Coordination

- Container quality will be significantly different from team to team.
- Inter-team communication and the split of responsibilities must become stronger.

## Next Steps

- **Write smoke tests for container validation.**
- **Implement the first set of toolchains for repository inspection.**
- **Share the Claude chat link for review.**
