# Hand-made reference Dockerfile: Fair Fairness Benchmark (FFB)
#   repo:   https://github.com/ahxt/fair_fairness_benchmark
#   commit: 394f020e9d93cb195885ca5c67f411b175726e63
#
# Build context = a clone of the repository at that commit (references/run.sh does this).
#
# Python 3.8, as in the README (conda create --name ffb_env python=3.8).
# requirements.txt pins torch==1.13.1+cu116, which plain `pip install -r` cannot find on PyPI,
# so torch / torchvision / torchaudio are installed first from the PyTorch index and filtered
# out of the requirements. pip / setuptools / wheel pins are skipped too.
# folktables is imported by dataset.load_acs_data but missing from requirements.txt.
#
# CPU torch by default (the scripts pick "cuda" only if available). On an NVIDIA GPU host:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu116
#
# Datasets are NOT in the image: download datasets.zip (Dropbox link in datasets/readme.md)
# and mount it at /workspace/datasets. The scripts read ../datasets/<name>/raw from src/.
# wandb runs disabled (no account needed); set WANDB_MODE=online to log.

FROM python:3.8-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=disabled

WORKDIR /workspace

RUN pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

        # src/dataset.py line 1 is a stray `from tkinter import S` (unused, likely an editor auto-import).
# python:3.8-slim has the tkinter module but not the Tk library it loads, so the import crashes.
RUN apt-get update && apt-get install -y --no-install-recommends libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio|pip|setuptools|wheel)==' /tmp/requirements.txt > /tmp/requirements-rest.txt \
    && pip install -r /tmp/requirements-rest.txt \
    && pip install folktables==0.0.12

COPY . /workspace
WORKDIR /workspace/src

CMD ["bash"]
