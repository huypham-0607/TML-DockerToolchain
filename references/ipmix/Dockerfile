# Hand-made reference Dockerfile: IPMix (NeurIPS 2023)
#   repo:   https://github.com/hzlsaber/IPMix
#   commit: 050bcb6cb562e0f45b9caf64b17283ef58b21a41
#
# Build context = a clone of the repository at that commit (references/run.sh does this).
#
# The repo has no requirements file. Versions chosen to match its 2023 release:
#   Python 3.10, torch 2.0.1 / torchvision 0.15.2, numpy < 2, Pillow 9.5.
# cvxpy is left out: calibration_tools.tune_temp imports it only with binary_search=False,
# which cifar.py never uses.
#
# CPU torch by default. On an NVIDIA GPU host:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
# GPU NOTE: cifar.py / imagenet.py call .cuda() directly, so real training needs a GPU.
# Training also needs a mixing set (Google Drive link in the README) and downloads CIFAR.

FROM python:3.10-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        torch==2.0.1 torchvision==0.15.2
RUN pip install numpy==1.24.4 Pillow==9.5.0

COPY . /workspace

CMD ["bash"]
