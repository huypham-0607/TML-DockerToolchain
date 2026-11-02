# Hand-made reference Dockerfile: Pre-activation Distributions Expose Backdoor Neurons (NeurIPS 2022)
#   repo:   https://github.com/RJ-T/NIPS2022_EP_BNP
#   commit: af5150da52fe55aa0e265352576c435e1d9a39ab
#
# Build context = a clone of the repository at that commit (references/run.sh does this).
#
# The repo has no requirements file. Versions chosen to match its 2022 release:
#   Python 3.8, torch 1.12.1 / torchvision 0.13.1, numpy < 2, tqdm, torchattacks (cla attack only).
#
# CPU torch by default. On an NVIDIA GPU host:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu116
# GPU NOTE: train.py runs on CPU with --device cpu, but defense.py (used by test.py) calls
# .cuda() directly, so the EP / BNP defenses need a GPU.
# train.py downloads CIFAR-10 into --dataset-dir (default ../data/cifar10) on first use.

FROM python:3.8-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        torch==1.12.1 torchvision==0.13.1
RUN pip install numpy==1.24.4 Pillow==9.5.0 tqdm==4.64.1 torchattacks==3.4.0

COPY . /workspace

CMD ["bash"]
