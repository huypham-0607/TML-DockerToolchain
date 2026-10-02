# TML-DockerToolchain
## Background: Landseer
- Broader motivation: containerizing trustworthy-ML (TML) repositories is difficult
- Manual environment reconstruction is slow and error-prone; goal is automated containerization
- Development is ongoing
## Project Scope
- Team 1 is responsible only for automated containerization infrastructure for TML research repositories
- Goal: transform a repository into a validated Docker environment for downstream teams
  - Use cases: benchmark modification, experiment reproduction, metric validation
- Core pipeline:
  1. Research Repository
  2. Dockerfile
  3. Docker Image
  4. Running Container
  5. Generic command-execution interface
- A generated Dockerfile is considered successful only if it actually builds and starts
## Agent Toolchain Categories
- Repository inspection
  - Repository structure, dependency files, README setup instructions
  - Python/CUDA/framework versions, existing Docker configuration, environment metadata
- Dockerization
  - Generate/select a Dockerfile template
  - Build the Docker image, inspect build failures, repair and retry until success or retry limit
- Container lifecycle
  - Provision/start a container from an image, check status, stop, and remove
- Container execution
  - Execute arbitrary commands inside a provisioned container
  - Return structured results: stdout, stderr, exit code, timeout status
- Main LLM-facing tools should remain high-level:
  - dockerize_repo(repo)
  - build_image(dockerfile) (possibly internal to dockerize_repo())
  - provision_container(image, name)
  - run_command_in_container(container, cmd)
  - destroy_container(container)
- Lower-level helpers (dependency detection, log parsing) stay as ordinary Python functions, not agent tools
## Agent Workflow
1. Inspect repository
2. Infer software/environment requirements
3. Select Dockerfile template
4. Generate Dockerfile
5. Build Docker image
6. If build fails: read logs → diagnose → modify Dockerfile → rebuild (fixed retry limit)
7. Provision container
8. Run basic smoke tests/commands
9. Return validated container artifact
- Key split: LLM handles reasoning and repair decisions; deterministic Python/Docker code performs actual container operations
- Smoke tests are essential: need to validate not just that the container builds, but that basic functions actually run
  - Example risk: IPmix hardcoded CUDA, breaking environments without a GPU; similar issues could recur
## Project Layout
- Proposed repository structure for container-agent/:
  - src/agent/: https://agent.py, https://prompts.py, https://state.py
  - src/repo/: https://inspect.py, https://dependencies.py, https://setup_parser.py
  - src/docker/: https://dockerize.py, https://build.py, https://provision.py, https://exec.py, https://lifecycle.py
  - src/schemas/: https://repository.py, https://build.py, https://container.py
  - templates/: https://python.do, https://conda.do, https://pytorch-cuda.do
  - tests/, runs/, https://pyproject.to
## Important Design Decisions
- Team boundary is explicit: Team 1 produces the environment; it does not determine which experiment to run or whether a paper metric was reproduced
  - run_command_in_container() is a generic execution primitive for other teams
- Use structured return objects over raw text
  - RepoProfile, BuildResult, ContainerHandle, CommandResult
  - Makes LangChain tool calls more reliable and gives downstream teams stable interfaces
- First milestone success definition (simple):
  - Repository → Dockerfile → successful image build → container starts → arbitrary command executes
- Docker first, Apptainer later
  - Do everything in Docker first, then migrate to Apptainer for RCAC/Purdue SLURM support
  - Future path: Repository → Dockerfile → Docker/OCI Image → Apptainer/SIF → RCAC
  - Team 1 does not take responsibility for SLURM jobs, experiment reproduction, or metric validation
- Container quality will vary significantly across teams; inter-team communication and responsibility splits need strengthening
## Next Steps
- **Write smoke tests for container validation**
- **Implement first set of toolchains for repository inspection**
- **Share Claude chat link for review**
---