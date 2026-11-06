# UAP (CVPR 2022): https://github.com/faiimea/UAP (commit 9f66dd0)
# requirements.txt cannot be installed as-is (wrong `art` package, conflicting pins): see references/README.md.
# CPU by default. For a GPU: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
FROM python:3.8-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# the code imports itself as a package named `uap`, and imports `models` from the repo root
ENV PYTHONPATH=/workspace:/workspace/uap

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# setuptools<70: the adversarial-robustness-toolbox needs pkg_resources.packaging
RUN pip install --no-cache-dir --index-url $TORCH_INDEX_URL --extra-index-url https://pypi.org/simple \
    torch==2.2.0 torchvision==0.17.0 && \
    pip install --no-cache-dir adversarial-robustness-toolbox==1.17.1 numpy==1.24.3 matplotlib==3.7.5 \
    Pillow==10.2.0 scikit-learn==1.3.2 tqdm==4.66.1 "setuptools<70"

WORKDIR /workspace/uap
RUN git clone https://github.com/faiimea/UAP . && \
    git checkout 9f66dd0912b7c42ad3b74866036ff814a5609b0b

CMD ["bash"]
