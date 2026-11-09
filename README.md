# TML-DockerToolchain

Automated containerization infrastructure for trustworthy machine learning (TML) research repositories.
The toolchain changes a research repository into a validated Docker environment.

> **Note:** Development is ongoing. The four toolchains are implemented and connected to one agent. Refer to [Status](#status).

## Contents

- [Background: Landseer](#background-landseer)
- [Project Scope](#project-scope)
- [Setup](#setup)
- [Status](#status)
- [Evaluation Against the Reference Dockerfiles](#evaluation-against-the-reference-dockerfiles)
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
| `LLM_REQUESTS_PER_MINUTE` | No | Client-side request limit. Default: 18 for `rcac` (Purdue GenAI allows 20 per minute per user), 120 for `openai`. |
| `LLM_RATE_RETRIES` | No | How often a refused request ("Rate limit exceeded") is retried, waiting 15 s, 30 s, then 60 s. Default 6. |
| `TML_STEP_LIMIT` | No | The agent's step limit (LangGraph recursion limit, 2 steps per agent round). Default 150. |
| `TML_MAX_COMMAND_SECONDS` | No | The longest one `run_command_in_container` call from the agent may run. Default 300. Longer requests are cut to this. |
| `TML_TORCH_BUILD` | No | `cpu`, `cuda` or `auto` (default): which PyTorch build `generate_dockerfile` installs. `auto` uses the CUDA build only if Docker on this machine can attach an NVIDIA GPU. |

> **Caution:** Do not commit `.env`. Git ignores this file.

### Select the LLM

`src/llm.py` makes the chat model. Set `LLM_PROVIDER` in `.env` to change the provider. You do not have to change the code.

| `LLM_PROVIDER` | Default model | Endpoint | Key |
| --- | --- | --- | --- |
| `rcac` (default) | `gpt-oss:120b` | `https://genai.rcac.purdue.edu/api` | `GENAI_API_KEY` |
| `openai` | `gpt-5-mini` | OpenAI | `OPENAI_API_KEY` |

To add a provider, add an entry to `PROVIDERS` in `src/llm.py`.

Purdue GenAI allows 20 requests per minute per user ([RCAC GenAI API](https://docs.rcac.purdue.edu/services/genai/api/)). It answers a rate limit with HTTP 400 ("Rate limit exceeded") instead of 429, so the OpenAI client does not retry it, and it can answer with a JSON `null` instead of an error. `src/llm.py` keeps the request rate under the limit and retries both cases with waits.

### Run

- Do a check of the installation. This command prints `Hello from tml-dockertoolchain!`.

  ```bash
  uv run main.py
  ```

- Check the LLM connection, tool calling, and (with `--repo`) one inspection by the agent:

  ```bash
  uv run python -m scripts.smoke_test
  uv run python -m scripts.smoke_test --repo /abs/path/to/a/repository
  ```

- Containerize a repository with the agent. Give the absolute path of a local clone. The agent prints its last reply.

  ```bash
  uv run python -m src.main "Containerize the repository at /abs/path/to/repo. At the end, show the final Dockerfile you used."
  ```

- The same, but every step is printed while the agent works (each tool call, its arguments and the start of its reply), with a summary at the end, also when the agent crashes or runs out of steps. `--log` saves the full transcript.

  ```bash
  uv run python -m scripts.run_agent --log runs/agent.log "Containerize the repository at /abs/path/to/repo"
  ```

## Status

All four toolchains are implemented and connected to the agent in `src/agent.py`:

| Toolchain | Package | State |
| --- | --- | --- |
| Repository inspection | `src/repo/` | Sections A to D, combined by `inspect.py` |
| Dockerization | `src/docker/` | One `python:<ver>-slim` template (`generate.py`): PyTorch from the PyTorch index (CPU or CUDA build), `+cuNNN` pins, matched torchvision / torchaudio, packages imported but not declared, duplicate pins removed, OS packages for `tkinter`, `WANDB_MODE=disabled` for wandb, a package layout for repositories that import themselves. A `docker build` wrapper (`build.py`). |
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
| `src/agent.py` | Makes the agent with `deepagents`, with all the tools and one pipeline system prompt. Gets the model from `src/llm.py` and the file backend from `src/fs_backend.py`. |
| `src/fs_backend.py` | The backend of the agent's built-in file tools (`ls`, `read_file`, `glob`, `grep`): the real disk, read-only, without secret files such as `.env`. Writes go only to an in-memory `/agent_scratch/`. |
| `src/llm.py` | Makes the chat model for the provider in `LLM_PROVIDER` (OpenAI or Purdue GenAI): 18 requests per minute for Purdue GenAI, and retries when the server refuses for now. |
| `src/tools.py` | The tools `inspect_repo`, `generate_dockerfile`, and `build_image`. |
| `scripts/smoke_test.py` | Checks the configuration, the LLM endpoint, tool calling, and one agent inspection. |
| `scripts/run_agent.py` | Runs the agent and prints every step. |
| `scripts/evaluate.py` | Scores the pipeline against the reference Dockerfiles. Refer to [Evaluation](#evaluation-against-the-reference-dockerfiles). |
| `references/` | Hand-made reference Dockerfiles and CPU smoke tests for the four target repositories. Refer to `references/README.md`. |
| `src/lifecycle/tools.py` | The six lifecycle tools: `provision_container`, `container_status`, `container_logs`, `diagnose_container`, `stop_container`, `destroy_container`. |
| `src/execution/tools.py` | The tool `run_command_in_container`. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Project metadata, locked dependencies, and the Python version. |

The lifecycle and execution toolchains also have a command line that does not call the LLM: `uv run python -m src.lifecycle --help` and `uv run python -m src.execution --help`.

### Tests

The tests use `unittest` and do not call the LLM:

```bash
uv run python -m unittest discover -s tests -t .
```

With the Docker daemon running, all tests run (more than 800, about 5 minutes). Without Docker, the tests that need it are skipped. Set `TML_TEST_BASE_IMAGE` to use another base image for the Docker tests (default `python:3.14-slim`, pulled once).

| Test file | What it covers |
| --- | --- |
| `tests/test_structure.py`, `tests/test_inspection.py` | Repository inspection, sections A to D |
| `tests/test_generate.py`, `tests/test_build.py` | The Dockerfile template (PyTorch builds, `+cuNNN` pins, imports, torchvision pairs) and the build wrapper |
| `tests/test_existing_dockerfiles.py` | Toy repositories that ship their own Dockerfile: inspection, real builds, existing images, build context, Dockerfile to container to command |
| `tests/lifecycle/` | The six lifecycle tools, with a fake Docker client and with the real daemon |
| `tests/test_exec.py` | Command execution |
| `tests/test_agent_files.py` | The agent's file tools: the real repository is readable, writes and secrets are refused (also inside a real `deepagents` graph, with a scripted model) |
| `tests/test_llm.py`, `tests/test_main.py` | Rate-limit retries, model settings, the step limit |
| `tests/test_evaluate.py` | The evaluation script |

These parts of the design are not in the repository at this time:

- Dockerfile templates for conda and for CUDA base images (`nvidia/cuda`) for code that compiles CUDA

## Evaluation Against the Reference Dockerfiles

`references/` has a hand-made Dockerfile for each of the four target repositories (IPMix, Fair Fairness Benchmark, EP/BNP, UAP), with a CPU smoke test (`smoke.py`) that runs the repository's own code without a GPU, a dataset or the network. `references/README.md` lists the problems in each repository that the Dockerfiles had to work around: the cases the pipeline must handle.

1. Build and test the references (once):

   ```bash
   references/run.sh all
   ```

2. Score the pipeline against them:

   ```bash
   uv run python -m scripts.evaluate                 # template step only: no LLM, no API key
   uv run python -m scripts.evaluate --agent ffb     # the full agent on one repository (about 2 minutes)
   uv run python -m scripts.evaluate --agent --model qwen3.6:27b ffb   # the same tools, another LLM
   ```

For each repository, `scripts/evaluate.py` clones it at the pinned commit (`references/repos.txt`), lets the pipeline build an image, and reports:

| Column | Meaning |
| --- | --- |
| Builds | The pipeline produced an image. |
| Python, torch, torchvision | The versions in the pipeline's image, and whether they match the reference image. |
| Smoke test | The reference `smoke.py` passes in the pipeline's image. |

The table is saved to `runs/eval/results.md` and `runs/eval/results.json`. Each run adds or replaces one row per repository, mode and model, and keeps the other rows, so repositories and models can be run one at a time (`--fresh` starts over). `--skip-build` scores the image that an earlier run built, without running the pipeline again. The agent's transcript is saved to `runs/eval/<name>-agent-<model>.log`.

**Other agents.** `--model` drives the same tools with another LLM (it sets `LLM_MODEL`), and `--provider openai` with OpenAI instead of RCAC (it sets `LLM_PROVIDER`; needs `OPENAI_API_KEY`). Each model gets its own rows, labelled for example `agent (qwen3.6:27b)`, so the table compares models. RCAC lists the models your key can use at `https://genai.rcac.purdue.edu/api/models`.

## Agent Toolchain Categories

| Category | Functions |
| --- | --- |
| Repository inspection | Inspect the repository structure, dependency files, and README setup instructions. Inspect the Python, CUDA, and framework versions, the existing Docker configuration, and the environment metadata. |
| Dockerization | Generate or select a Dockerfile template. Build the Docker image. Inspect build failures. Repair and try again until the build is successful or the retry limit is reached. |
| Container lifecycle | Provision and start a container from an image. Check the status. Stop the container. Remove the container. |
| Container execution | Execute arbitrary commands in a provisioned container. Return structured results: stdout, stderr, exit code, and timeout status. |

### LLM-Facing Tools

Keep the main LLM-facing tools high-level. The agent (`src/agent.py`) has these tools at this time. The docstring of each tool is the description that the LLM reads.

`deepagents` also gives the agent a todo list and file tools (`ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`). By default those work on an empty in-memory filesystem, so the agent could not read the repository. `src/fs_backend.py` connects them to the real disk, read-only: the agent reads requirements, READMEs and code with them, never changes the repository, and cannot read secret files (`.env`, SSH and cloud keys).

| Tool | Description of input arguments | Description of output arguments | Tool description |
| --- | --- | --- | --- |
| `inspect_repo` | • `repo_path` (str, required): path to a local repository root.<br>• `search` (bool, default `False`): when `True`, run a Google search if the README has no install commands. | JSON string (`RepoProfile.summary()`) with these fields:<br>• Structure: `root`, `primary_language`, `languages`, `total_files`, `tree`, `important_files`, `existing_docker`, `large_files`<br>• Dependencies: `dependency_files`, `python_requires`, `packages` (name → pinned version), `conda_channels`, `index_urls`<br>• Docs: `install_commands`, `setup_sections`, `search_results`<br>• Environment: `environment` (Python/CUDA/cuDNN/framework versions, each with `value`, `confidence`, `constraints` and `conflict`), `frameworks`, `third_party_imports` (top-level modules the code imports, without the standard library and the repository's own modules), `gpu` (`uses_cuda`, `gpu_required`, `needs_nvcc`, `hardcoded_cuda`)<br>• `warnings` | Inspects a local repository without running its code and returns what is needed to containerize it: its structure, dependencies, setup steps from the docs, and the Python, CUDA and framework versions it needs. Fully deterministic (no LLM), apart from the optional search. Wraps `inspect_repository()`. |
| `generate_dockerfile` | • `repo_path` (str, required): path to a local repository root. | JSON string with these fields:<br>• `dockerfile`: the Dockerfile text<br>• `dockerfile_path`: where it was saved (`runs/<repo>/Dockerfile`)<br>• `notes`: hints for the LLM, for example which PyTorch build was chosen, "code calls CUDA without checking for a GPU (file:line): that code needs an NVIDIA GPU", torchvision re-pinned to match torch, packages installed from imports, or the repository's own Docker configuration | Inspects the repository and makes a starter Dockerfile from one template: `python:<ver>-slim`; torch / torchvision / torchaudio from the PyTorch index in their own layer (CPU build, or the CUDA build when Docker here can attach an NVIDIA GPU; `TML_TORCH_BUILD` overrides), without `+cuNNN` suffixes and with matching versions; `pip install -r` of each requirements file with the torch lines filtered out; `pip install -e .` if the repository is installable; imported packages that no file declares, best effort. Does not build. The LLM can edit the text before it calls `build_image`. Wraps `docker.generate.generate_dockerfile()`. |
| `build_image` | • `repo_path` (str, required): path to a local repository root; the build context.<br>• `dockerfile` (str, required): the complete Dockerfile text.<br>• `tag` (str, default `""`): image tag; empty gives `tml/<repo-name>:latest`. | JSON string (`BuildResult`) with these fields:<br>• `success`, `image` (the tag, or null if the build failed), `exit_code`<br>• `log_tail`: the last 4000 characters of the build log<br>• `log_path`, `dockerfile_path` (both in `runs/<repo>/`)<br>• `timed_out` | Runs `docker build` on Dockerfile text, with the repository as the build context. If the build fails, the LLM reads `log_tail`, fixes the Dockerfile, and calls the tool again (at most 5 attempts). Timeout: 1 hour. Wraps `docker.build.build_image()`. |
| `provision_container` | • `image` (str): tag or id of an image that is already on this machine.<br>• `dockerfile` (str): path to a Dockerfile to build and run instead. Give exactly one of `image` or `dockerfile`.<br>• `context` (str): build context; default is the Dockerfile's directory.<br>• `name` (str): container name; generated when empty.<br>• `mode` (str, default `idle`): `idle` keeps the container running for later commands; `native` runs the image's own command.<br>• `command` (str): native mode only, instead of the image's `CMD`.<br>• `gpu` (str, default `auto`): `auto` (a GPU if Docker can attach one, otherwise none), `all` (fail without a GPU), or `none`.<br>• `env` (dict): environment variables.<br>• `mounts` (list): `host:container[:ro]`, only from `runs/` and `TML_MOUNT_ROOTS`.<br>• `ports` (list): `host:container` or `container`, on 127.0.0.1 only.<br>• `memory` (str): limit such as `8g`; default 75% of the host memory, with no swap.<br>• `shm_size` (str, default `2g`): size of `/dev/shm`.<br>• `user` (str): for example `1000:1000`; default is the image's user (root for most images).<br>• `ready_timeout` (float, default 0): seconds; 0 gives 30, or 120 with a HEALTHCHECK.<br>• `replace` (bool, default `False`): replace a container of this name with different settings. | JSON string with these fields:<br>• `ok`: `true` when the container is ready, or (native mode) when its command ended with code 0<br>• `action`: `created`, `replaced`, `reused`, or `restarted`<br>• `image`: `id`, `tag`, `built`, `builder`<br>• `status`: the same fields as `container_status`<br>• `notes`: what provision had to change, for example a start without the GPU<br>• `diagnosis` (only when the container started but is not usable): `logs`, `hints`, `processes`, `resources`, `events`, `notes`<br>• `error` (only for a failed request): `code`, `message`, `hint` | Starts a Docker container and waits until it is ready. Images are never pulled; a Dockerfile is built once and reused while its text and context path do not change. The container is never privileged and never gets the host network or the Docker socket. A container with the same name and settings is reused; other settings give `name_conflict` unless `replace` is true. A container that starts and then fails is not an error: the reply carries its diagnosis. Wraps `lifecycle.provision.provision()`. |
| `container_status` | • `container` (str, default `""`): container name or id; empty lists all containers that the toolchain made. | JSON string with these fields:<br>• `ok`<br>• `status` (one container): `container`, `state`, `health`, `reason`, `usable`, `exit_code`, `oom_killed`, `restart_count`, `started_at`, `finished_at`, `error`, `healthcheck_output`, `handle` (`name`, `id`, `image`, `mode`, `gpu`, `workdir`, `ports`, `mounts`). Empty fields are left out.<br>• `containers` (no argument): a list of `status` objects, oldest first | Reports the state and health of a container. Health is one of `starting`, `healthy`, `unhealthy`, `completed` (ended with code 0), `stopped`, `failed`, or `missing`. `usable` is `true` when commands can run in it. A missing container is not an error. Containers that the toolchain did not make are refused (`not_managed`). Wraps `get_status()` and `list_managed()`. |
| `container_logs` | • `container` (str, required): container name or id; it may be stopped.<br>• `tail` (int, default 200): lines of each stream, 1 to 2000.<br>• `since` (str, default `""`): a duration such as `10m`, or an ISO timestamp.<br>• `grep` (str, default `""`): regular expression; only matching lines, for example `Error\|Traceback`. | JSON string with these fields:<br>• `ok`<br>• `logs`: `container` (the name or id that was asked for), `stdout`, `stderr`, `lines`, `truncated` (only when older output was cut), `since` (only when given) | Reads the runtime output of a container (not the build output), stdout and stderr apart. The newest lines are kept, at most 8000 characters. Colour codes and redrawn progress bars are cleaned. Wraps `get_logs()`. |
| `diagnose_container` | • `container` (str, required): container name or id. | JSON string with these fields:<br>• `ok`<br>• `diagnosis`: `status`, `logs` (the last 60 lines of each stream), `hints` (each with `code`, `evidence`, `suggestion`, `owner`), `processes` and `resources` (CPU, memory, process count; running containers only), `events`, `notes` | Explains the state of a container. Use it when a container is not healthy or a command behaved strangely. The hints are deterministic labels of known failure patterns (for example `out_of_memory`, `missing_python_module`, `gpu_unavailable`); the LLM makes the repair decision. No hints means no known pattern: read the logs. A missing container gives health `missing`. Wraps `diagnose()`. |
| `stop_container` | • `container` (str, required): container name or id.<br>• `timeout` (int, default 10): seconds to wait after SIGTERM before SIGKILL, 0 to 600. | JSON string with these fields:<br>• `ok`<br>• `result`: `container`, `stopped`, `killed` (`true` if SIGKILL was necessary), `exit_code`, `status` | Stops a container and keeps it, with its files and logs. `provision_container` with the same settings starts it again. Stopping a container that is not running changes nothing. Wraps `teardown.stop()`. |
| `destroy_container` | • `container` (str, required): container name or id.<br>• `force` (bool, default `False`): kill at once instead of a graceful stop (SIGTERM, then SIGKILL after 10 seconds). | JSON string with these fields:<br>• `ok`<br>• `result`: `container`, `stopped`, `killed`, `removed`, `exit_code` (the last exit code), `status` (`missing` after the removal) | Stops a container if it runs, then removes it and its anonymous volumes. The image and the mounted host directories stay. A container that does not exist is not an error (`removed: false`). Containers that the toolchain did not make are refused (`not_managed`). Wraps `teardown.destroy()`. |
| `run_command_in_container` | • `container` (str, required): name or id of a running container (from `provision_container`). The reply gets a `warning` when the container runs an older build of its image (the image was rebuilt after the container started).<br>• `command` (str, required): the shell command, for example `python --version`.<br>• `cwd` (str, default `/workspace`): working directory in the container.<br>• `timeout` (int, default 300): seconds before the command is stopped; at most `TML_MAX_COMMAND_SECONDS` (300). | JSON string (`ExecResult`) with these fields:<br>• `ok`: `true` if `exit_code` is 0 and the command did not time out<br>• `command`, `exit_code`, `stdout`, `stderr`, `duration` (seconds), `timed_out`, `error`<br>• `hint`, after a timeout: do not run it again; test with a small input instead<br>Progress bars (`\r` updates) are collapsed to their last state, and long output keeps its head and tail. | Runs a command in a provisioned container through `docker exec` and returns a structured result. The generic execution primitive for the downstream teams, and the agent's smoke-test tool. Wraps `execute_in_container()`. |

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
├── Dockerfile         a hand-made Dockerfile (PR #8)
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
├── src/
│   ├── __init__.py
│   ├── agent.py       the agent and its system prompt
│   ├── fs_backend.py  read-only real-disk backend for the agent's file tools
│   ├── llm.py         the chat model, rate limit and retries
│   ├── main.py
│   ├── tools.py
│   ├── repo/          inspection (sections A to D)
│   ├── docker/        dockerization
│   ├── lifecycle/     container lifecycle
│   └── execution/     container execution
├── scripts/
│   ├── smoke_test.py  LLM and agent checks
│   ├── run_agent.py   run the agent and print every step
│   └── evaluate.py    score the pipeline against references/
├── references/        hand-made reference Dockerfiles and smoke tests
│   ├── repos.txt      the four repositories and their pinned commits
│   ├── run.sh         build and smoke-test a reference
│   └── ipmix/ ffb/ ep_bnp/ uap/
├── runs/              generated Dockerfiles, build logs, evaluation results (Git ignores it)
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

- **Run the full agent on the four reference repositories** (`scripts/evaluate.py --agent`) and record the results.
- **Test the CUDA builds on a GPU host (RCAC):** IPMix training, the EP/BNP defenses and the FFB HSIC / PR methods call `.cuda()` directly and cannot run on CPU.
- **A CUDA base-image template** (`nvidia/cuda:<ver>-devel`) for repositories that compile CUDA code.
- **Prefer the repository's own Dockerfile** when it has one (the agent is told to try it first; the template still ignores it).
- **Apptainer / SIF** conversion for RCAC.
