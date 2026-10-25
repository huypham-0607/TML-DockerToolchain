# TML-DockerToolchain

Automated containerization infrastructure for trustworthy machine learning (TML) research repositories.
The toolchain changes a research repository into a validated Docker environment.

> **Note:** Development is ongoing. The four toolchains are implemented and connected to one agent. Refer to [Status](#status).

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
| Docker | A recent version, with the daemon running | Builds images and runs containers. The Docker-backed tests are skipped when the daemon is not reachable. |

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

All four toolchains are implemented and connected to the agent in `src/agent.py`:

| Toolchain | Package | State |
| --- | --- | --- |
| Repository inspection | `src/repo/` | Sections A to D, combined by `inspect.py` |
| Dockerization | `src/docker/` | First version: one `python:<ver>-slim` template (`generate.py`) and a `docker build` wrapper (`build.py`) |
| Container lifecycle | `src/lifecycle/` | Provision, status, logs, diagnosis with hints, stop, destroy |
| Container execution | `src/execution/` | `docker exec` with structured results and a timeout |

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

| File | State |
| --- | --- |
| `main.py` | Placeholder. Prints a greeting. |
| `src/main.py` | Command-line entry point. Sends one question to the agent and prints the last reply. |
| `src/agent.py` | Makes the agent with `deepagents`, with all the tools and one pipeline system prompt. Gets the model from `src/llm.py`. |
| `src/llm.py` | Makes the chat model for the provider in `LLM_PROVIDER` (OpenAI or Purdue GenAI). The rate limit is 2 requests each second. |
| `src/tools.py` | The tools `inspect_repo`, `generate_dockerfile`, and `build_image`. Also still contains the two placeholder tools `meow` and `woof`. |
| `src/lifecycle/tools.py` | The six lifecycle tools: `provision_container`, `container_status`, `container_logs`, `diagnose_container`, `stop_container`, `destroy_container`. |
| `src/execution/tools.py` | The tool `run_command_in_container`. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Project metadata, locked dependencies, and the Python version. |

The lifecycle and execution toolchains also have a command line that does not call the LLM: `uv run python -m src.lifecycle --help` and `uv run python -m src.execution --help`.

### Tests

The tests use `unittest` and do not call the LLM:

```bash
uv run python -m unittest discover -s tests -t .
```

With the Docker daemon running, all 684 tests run (about 5 minutes). One GPU-detection test in `tests/lifecycle/` fails at this time.

These parts of the design are not in the repository at this time:

- Dockerfile templates for conda and PyTorch + CUDA

## Agent Toolchain Categories

| Category | Functions |
| --- | --- |
| Repository inspection | Inspect the repository structure, dependency files, and README setup instructions. Inspect the Python, CUDA, and framework versions, the existing Docker configuration, and the environment metadata. |
| Dockerization | Generate or select a Dockerfile template. Build the Docker image. Inspect build failures. Repair and try again until the build is successful or the retry limit is reached. |
| Container lifecycle | Provision and start a container from an image. Check the status. Stop the container. Remove the container. |
| Container execution | Execute arbitrary commands in a provisioned container. Return structured results: stdout, stderr, exit code, and timeout status. |

### LLM-Facing Tools

Keep the main LLM-facing tools high-level. The agent (`src/agent.py`) has these tools at this time. The docstring of each tool is the description that the LLM reads.

| Tool | Description of input arguments | Description of output arguments | Tool description |
| --- | --- | --- | --- |
| `inspect_repo` | • `repo_path` (str, required): path to a local repository root.<br>• `search` (bool, default `False`): when `True`, run a Google search if the README has no install commands. | JSON string (`RepoProfile.summary()`) with these fields:<br>• Structure: `root`, `primary_language`, `languages`, `total_files`, `tree`, `important_files`, `existing_docker`, `large_files`<br>• Dependencies: `dependency_files`, `python_requires`, `packages` (name → pinned version), `conda_channels`, `index_urls`<br>• Docs: `install_commands`, `setup_sections`, `search_results`<br>• Environment: `environment` (Python/CUDA/cuDNN/framework versions, each with `value`, `confidence`, `constraints` and `conflict`), `frameworks`, `gpu` (`uses_cuda`, `gpu_required`, `needs_nvcc`, `hardcoded_cuda`)<br>• `warnings` | Inspects a local repository without running its code and returns what is needed to containerize it: its structure, dependencies, setup steps from the docs, and the Python, CUDA and framework versions it needs. Fully deterministic (no LLM), apart from the optional search. Wraps `inspect_repository()`. |
| `generate_dockerfile` | • `repo_path` (str, required): path to a local repository root. | JSON string with these fields:<br>• `dockerfile`: the Dockerfile text<br>• `dockerfile_path`: where it was saved (`runs/<repo>/Dockerfile`)<br>• `notes`: hints for the LLM, for example "repo uses CUDA: run the container with --gpus all", "repo compiles CUDA code: base image likely needs to be nvidia/cuda:\<ver\>-devel", or "Python version not found; defaulting to 3.10" | Inspects the repository and makes a starter Dockerfile from one template: `python:<ver>-slim`, framework pins in their own layer, `pip install -r requirements.txt`, and `pip install -e .` if the repository is installable. Does not build. The LLM can edit the text before it calls `build_image`. Wraps `docker.generate.generate_dockerfile()`. |
| `build_image` | • `repo_path` (str, required): path to a local repository root; the build context.<br>• `dockerfile` (str, required): the complete Dockerfile text.<br>• `tag` (str, default `""`): image tag; empty gives `tml/<repo-name>:latest`. | JSON string (`BuildResult`) with these fields:<br>• `success`, `image` (the tag, or null if the build failed), `exit_code`<br>• `log_tail`: the last 4000 characters of the build log<br>• `log_path`, `dockerfile_path` (both in `runs/<repo>/`)<br>• `timed_out` | Runs `docker build` on Dockerfile text, with the repository as the build context. If the build fails, the LLM reads `log_tail`, fixes the Dockerfile, and calls the tool again (at most 5 attempts). Timeout: 1 hour. Wraps `docker.build.build_image()`. |
| `provision_container` | • `image` (str): tag or id of an image that is already on this machine.<br>• `dockerfile` (str): path to a Dockerfile to build and run instead. Give exactly one of `image` or `dockerfile`.<br>• `context` (str): build context; default is the Dockerfile's directory.<br>• `name` (str): container name; generated when empty.<br>• `mode` (str, default `idle`): `idle` keeps the container running for later commands; `native` runs the image's own command.<br>• `command` (str): native mode only, instead of the image's `CMD`.<br>• `gpu` (str, default `auto`): `auto`, `all`, or `none`.<br>• `env` (dict): environment variables.<br>• `mounts` (list): `host:container[:ro]`, only from `runs/` and `TML_MOUNT_ROOTS`.<br>• `ports` (list): `host:container` or `container`, on 127.0.0.1 only.<br>• `memory` (str): limit such as `8g`; default 75% of the host memory.<br>• `shm_size` (str, default `2g`): size of `/dev/shm`.<br>• `user` (str): for example `1000:1000`; default root.<br>• `ready_timeout` (float, default 0): seconds; 0 gives 30, or 120 with a HEALTHCHECK.<br>• `replace` (bool, default `False`): replace a container of this name with different settings. | JSON string with these fields:<br>• `ok`: `true` when the container is ready<br>• `action`: `created`, `replaced`, `reused`, or `restarted`<br>• `image`: `id`, `tag`, `built`, `builder`<br>• `status`: `container`, `state`, `health`, `reason`, `usable`, `exit_code`, `started_at`, and `handle` (`name`, `id`, `image`, `mode`, `gpu`, `workdir`, `ports`, `mounts`)<br>• `diagnosis` (only when the container started but is not usable): `logs`, `hints`, `processes`, `resources`, `events`, `notes`<br>• `error` (only for a failed request): `code`, `message`, `hint` | Starts a Docker container and waits until it is ready. Images are never pulled; a Dockerfile is built once and reused while it does not change. The container is never privileged and never gets the host network or the Docker socket. A container with the same name and settings is reused. Wraps `lifecycle.provision()`. |
| `container_status` | • `container` (str, default `""`): container name or id; empty lists all containers that the toolchain made. | JSON string with these fields:<br>• `ok`<br>• `status` (one container): `container`, `state`, `health`, `reason`, `usable`, `exit_code`, `oom_killed`, `restart_count`, `started_at`, `finished_at`, `handle`<br>• `containers` (no argument): a list of `status` objects | Reports the state and health of a container. Health is one of `starting`, `healthy`, `unhealthy`, `completed`, `stopped`, `failed`, or `missing`. `usable` is `true` when commands can run in it. A missing container is not an error. Wraps `get_status()` and `list_managed()`. |
| `container_logs` | • `container` (str, required): container name or id; it may be stopped.<br>• `tail` (int): lines of each stream, 1 to 2000.<br>• `since` (str, default `""`): a duration such as `10m`, or an ISO timestamp.<br>• `grep` (str, default `""`): regular expression; only matching lines, for example `Error\|Traceback`. | JSON string with these fields:<br>• `ok`<br>• `logs`: `stdout`, `stderr`, `lines`, `truncated`, `since` | Reads the runtime output of a container (not the build output), stdout and stderr apart. The newest lines are kept. Wraps `get_logs()`. |
| `diagnose_container` | • `container` (str, required): container name or id. | JSON string with these fields:<br>• `ok`<br>• `diagnosis`: `status`, `logs`, `hints` (each with `code`, `evidence`, `suggestion`, `owner`), `processes`, `resources` (CPU, memory, process count), `events`, `notes` | Explains the state of a container. Use it when a container is not healthy or a command behaved strangely. The hints are deterministic labels of known failure patterns (for example `oom`, `missing_python_module`, `gpu_unavailable`); the LLM makes the repair decision. No hints means no known pattern: read the logs. Wraps `diagnose()`. |
| `stop_container` | • `container` (str, required): container name or id.<br>• `timeout` (int): seconds to wait after SIGTERM before SIGKILL, 0 to 600. | JSON string with these fields:<br>• `ok`<br>• `result`: `container`, `stopped`, `killed` (`true` if SIGKILL was necessary), `exit_code`, `status` | Stops a container and keeps it, with its files and logs. `provision_container` with the same settings starts it again. Stopping a container that is not running changes nothing. Wraps `teardown.stop()`. |
| `destroy_container` | • `container` (str, required): container name or id.<br>• `force` (bool, default `False`): kill at once instead of a graceful stop. | JSON string with these fields:<br>• `ok`<br>• `result`: `container`, `stopped`, `killed`, `removed`, `exit_code`, `status` (`missing` after the removal) | Stops a container if it runs, then removes it and its anonymous volumes. The image and the mounted host directories stay. A container that does not exist is not an error (`removed: false`). Wraps `teardown.destroy()`. |
| `run_command_in_container` | • `container` (str, required): name or id of a running container (from `provision_container`).<br>• `command` (str, required): the shell command, for example `python --version`.<br>• `cwd` (str, default `/workspace`): working directory in the container.<br>• `timeout` (int, default 300): seconds before the command is stopped. | JSON string (`ExecResult`) with these fields:<br>• `ok`: `true` if `exit_code` is 0 and the command did not time out<br>• `command`, `exit_code`, `stdout`, `stderr`, `duration` (seconds), `timed_out`, `error` | Runs a command in a provisioned container through `docker exec` and returns a structured result. The generic execution primitive for the downstream teams, and the agent's smoke-test tool. Wraps `execute_in_container()`. |
| `meow`, `woof` | None. | The fixed string `Meow` or `Woof`. | Placeholders from the first scaffold. They can be removed. |

The lifecycle tools return JSON and never raise:

- Success: `{"ok": true, ...}`
- Failed request: `{"ok": false, "error": {"code", "message", "hint"}}`
- From `provision_container` only: `{"ok": false, "status": ..., "diagnosis": ...}` when the container started but is not usable.

Each diagnosis hint names an owner, the toolchain that can act on it:

| Owner | Meaning |
| --- | --- |
| `lifecycle` | Destroy the container and provision it again with other options (for example `gpu`, `memory`, `shm_size`). |
| `dockerization` | The image must change. Fix the Dockerfile and build again. |
| `execution` | The container is fine. The command must change. |
| `host` | A person must look at the machine. |

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
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── llm.py
│   ├── main.py
│   ├── tools.py
│   ├── repo/          inspection (sections A to D)
│   ├── docker/        dockerization
│   ├── lifecycle/     container lifecycle
│   └── execution/     container execution
└── tests/
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
