# Fair Fairness Benchmark: https://github.com/ahxt/fair_fairness_benchmark (commit 394f020)
# CPU by default. For a GPU: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu116
FROM python:3.8-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV WANDB_MODE=disabled

# git to clone the repo; libtk8.6 because src/dataset.py imports tkinter
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libtk8.6 && \
    rm -rf /var/lib/apt/lists/*

# torch from the PyTorch index: requirements.txt pins torch==1.13.1+cu116, which is not on PyPI
RUN pip install --no-cache-dir --index-url $TORCH_INDEX_URL --extra-index-url https://pypi.org/simple \
    torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

WORKDIR /workspace
RUN git clone https://github.com/ahxt/fair_fairness_benchmark . && \
    git checkout 394f020e9d93cb195885ca5c67f411b175726e63

# the rest of requirements.txt, plus folktables (imported but not listed)
RUN grep -vE '^(torch|torchvision|torchaudio|pip|setuptools|wheel)==' requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt folktables==0.0.12

WORKDIR /workspace/src
CMD ["bash"]
