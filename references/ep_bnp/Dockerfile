# EP/BNP (NeurIPS 2022): https://github.com/RJ-T/NIPS2022_EP_BNP (commit af5150d)
# The repo has no requirements file; versions match its 2022 release.
# CPU by default. For a GPU: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu116
FROM python:3.8-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url $TORCH_INDEX_URL --extra-index-url https://pypi.org/simple \
    torch==1.12.1 torchvision==0.13.1 && \
    pip install --no-cache-dir numpy==1.24.4 Pillow==9.5.0 tqdm==4.64.1 torchattacks==3.4.0

WORKDIR /workspace
RUN git clone https://github.com/RJ-T/NIPS2022_EP_BNP . && \
    git checkout af5150da52fe55aa0e265352576c435e1d9a39ab

CMD ["bash"]
