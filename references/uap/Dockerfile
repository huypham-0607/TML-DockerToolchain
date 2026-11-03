# Hand-made reference Dockerfile: Fingerprinting DNNs via Universal Adversarial Perturbations (CVPR 2022)
#   repo:   https://github.com/faiimea/UAP
#   commit: 9f66dd0912b7c42ad3b74866036ff814a5609b0b
#
# Build context = a clone of the repository at that commit (references/run.sh does this).
#
# Python 3.8 (the repo's __pycache__ is cpython-38). requirements.txt cannot be installed as-is:
#   - `art==6.1` is the ASCII-art package, not the Adversarial Robustness Toolbox. Both install
#     a top-level `art` module and clash; the code needs the toolbox (art.attacks...). Dropped.
#   - Pillow and scikit_learn are pinned twice with different versions. Kept the newer ones.
#   - torch==2.2.0 with torchvision==0.16.0 conflict (0.16 needs torch 2.1). Kept torch 2.2.0,
#     paired with torchvision 0.17.0.
#
# The code imports itself as a package named `uap` (main.py: `from uap.train.model_structure ...`)
# and train/*.py import `models` from the repo root, so the repo goes to /workspace/uap and both
# /workspace and /workspace/uap are on PYTHONPATH.
# main.py still cannot run: it loads checkpoints and pickles from the authors' machine
# (/home/fazhong/studio/uap/...). Those files are not public.
#
# CPU torch by default. On an NVIDIA GPU host:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
# GPU NOTE: many scripts default to device cuda:0.

FROM python:3.8-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace:/workspace/uap

WORKDIR /workspace/uap

RUN pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        torch==2.2.0 torchvision==0.17.0
# setuptools < 70: ART 1.17.1 imports `packaging` from pkg_resources, which setuptools 70 removed
RUN pip install adversarial-robustness-toolbox==1.17.1 numpy==1.24.3 matplotlib==3.7.5 \
        Pillow==10.2.0 scikit-learn==1.3.2 tqdm==4.66.1 "setuptools<70"

COPY . /workspace/uap

CMD ["bash"]
