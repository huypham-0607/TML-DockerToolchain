# Reference Dockerfiles

Hand-made Dockerfiles for the four trustworthy-ML repositories the toolchain targets.
Each one is a known-good answer to compare the pipeline's generated Dockerfile against.

| Folder | Repository | Commit | Python | torch | Runs on CPU? |
| --- | --- | --- | --- | --- | --- |
| `ipmix/` | [hzlsaber/IPMix](https://github.com/hzlsaber/IPMix) | `050bcb6` | 3.10 | 2.0.1 | Augmentation and models yes. Training no: `cifar.py` calls `.cuda()`. |
| `ffb/` | [ahxt/fair_fairness_benchmark](https://github.com/ahxt/fair_fairness_benchmark) | `394f020` | 3.8 | 1.13.1 | Mostly. ERM and the Diff* methods run on CPU. HSIC and PR do not: `src/loss.py` calls `.cuda()` directly. |
| `ep_bnp/` | [RJ-T/NIPS2022_EP_BNP](https://github.com/RJ-T/NIPS2022_EP_BNP) | `af5150d` | 3.8 | 1.12.1 | Training yes (`--device cpu`). Defenses no: `defense.py` calls `.cuda()`. |
| `uap/` | [faiimea/UAP](https://github.com/faiimea/UAP) | `9f66dd0` | 3.8 | 2.2.0 | Models and ART attacks yes. `main.py` no: it needs files from the authors' machine. |

Each folder has:

- `Dockerfile`: the reference. The header comment says why each version was chosen and what in the repository had to be worked around.
- `smoke.py`: a CPU smoke test that needs no GPU, no dataset and no network. It is piped into the container, so the repository itself is not changed. It prints `SMOKE OK` at the end.

## Run

You need Docker and a few GB of free disk per image. With the CUDA builds of PyTorch the images were 3 to 9 GB each. The default CPU builds are much smaller.

```bash
references/run.sh ffb        # one of: ipmix, ffb, ep_bnp, uap
references/run.sh all        # all four
CLEAN=1 references/run.sh all   # remove each image after its smoke test
```

`run.sh` builds `tml-ref/<name>:latest`, then runs `smoke.py` in a container without a network.

**How the repository gets into the image:**

- **`ffb`, `ep_bnp`, `uap`:** the Dockerfile clones the repository itself (`git clone`, then `git checkout` of the pinned commit). The Dockerfile works on its own, without a local clone:

  ```bash
  docker build -t tml-ref/ffb references/ffb
  docker run --rm -i --network none tml-ref/ffb python - < references/ffb/smoke.py
  ```

- **`ipmix`:** the Dockerfile copies a local clone (`COPY . /workspace`). `run.sh` first clones the repository at the pinned commit into `references/.repos/ipmix` (Git ignores this folder) and uses that clone as the build context.

### Torch build: CPU or GPU

The images install the **CPU** build of PyTorch. It works on any machine and skips the several GB of NVIDIA libraries in the CUDA builds.
For an NVIDIA GPU host, pass the CUDA index:

```bash
BUILD_ARGS="--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118" references/run.sh ipmix
```

Use `cu116` for `ffb` and `ep_bnp`: torch 1.12 and 1.13 have no `cu118` builds.

### Run the real code

These commands use the real datasets. They take longer and download data.

```bash
# FFB: download datasets.zip (Dropbox link in the repo's datasets/readme.md), unzip to ./datasets
docker run --rm -v "$PWD/datasets:/workspace/datasets" tml-ref/ffb \
    python -u ffb_tabular_erm.py --dataset adult --model erm --sensitive_attr sex --target_attr income

# EP/BNP: downloads CIFAR-10 (~170 MB). One epoch on CPU is slow.
docker run --rm tml-ref/ep_bnp python train.py --device cpu --epochs 1 \
    --attack-type badnets --poisoning-rate 0.1 --trigger-size 3 --manual-seed 0

# IPMix: needs an NVIDIA GPU (build with the CUDA index), the mixing set from the README, and CIFAR.
docker run --rm --gpus all -v "$PWD/data:/data" tml-ref/ipmix \
    python cifar.py --dataset cifar10 --data-path /data --mixing-set /data/mixing_set --model wrn --no-jsd
```

## Problems found in the repositories

These problems are the things the agent has to get right, so they are also the test cases.

| Repository | Problem | Fix in the reference Dockerfile |
| --- | --- | --- |
| IPMix | No requirements file. | Versions chosen from the 2023 release date. |
| IPMix | `cifar.py` and `imagenet.py` call `.cuda()` directly. | None possible without changing code. Training needs a GPU. |
| IPMix | `models/ResNet/resnet.py`: `resnet18(num_classes)` and `resnet50(num_classes)` ignore `num_classes`, so the model always has 100 outputs, also on CIFAR-10. | None (code bug). The smoke test reports it. |
| FFB | `torch==1.13.1+cu116` is not on PyPI, so `pip install -r requirements.txt` fails. | torch installed first from the PyTorch index, then filtered out of the requirements. |
| FFB | `src/dataset.py` line 1 is a stray, unused `from tkinter import S`. `python:3.8-slim` has no Tk library, so every script crashes at import. | `apt-get install libtk8.6`. |
| FFB | `src/loss.py`: the `HSIC` and `PRLoss` losses call `.cuda()` directly, so `ffb_*_hsic.py` and `ffb_*_pr.py` need a GPU. | None possible without changing code. The smoke test runs ERM and DiffDP. |
| FFB | `folktables` is imported (ACS dataset) but missing from requirements. | Installed separately. |
| FFB | Scripts call `wandb.init()`, which asks for a login. | `WANDB_MODE=disabled`. |
| FFB | Scripts ignore `--data_path` and read `../datasets/<name>/raw` from the current directory. | `WORKDIR /workspace/src`. Mount the data at `/workspace/datasets`. |
| EP/BNP | No requirements file. | Versions chosen from the 2022 release date. |
| EP/BNP | `defense.py` calls `.cuda()` directly. | None possible without changing code. The defenses need a GPU. |
| UAP | `art==6.1` in requirements is the ASCII-art package. It clashes with the Adversarial Robustness Toolbox, which also installs a top-level `art` module. | Dropped. Only `adversarial-robustness-toolbox` is installed. |
| UAP | Pillow and scikit-learn are each pinned twice with different versions. | Kept the newer pin. |
| UAP | `torch==2.2.0` and `torchvision==0.16.0` conflict. | torchvision 0.17.0. |
| UAP | The code imports itself as a package named `uap`, and `train/*.py` import `models` from the repo root. | Repo copied to `/workspace/uap`, `PYTHONPATH=/workspace:/workspace/uap`. |
| UAP | `main.py` loads checkpoints and pickles from `/home/fazhong/studio/uap/...`. | None. Those files are not public. |
| UAP | `train/model_structure.get_model` lists about 45 networks, but most import names that the repo's `models/` package does not define. Of the 12 tested, only `resnet18` and `resnet34` build. | None (code bug). The smoke test reports which names work. |
| UAP | ART 1.17.1 does `from pkg_resources import packaging`, which setuptools 70 and later removed, so `import art.estimators` fails. | `setuptools<70`. |

## Compare against the pipeline

Compare outcomes, not Dockerfile text: two different Dockerfiles can both be correct.
`scripts/evaluate.py` does the comparison for every repository in `repos.txt`:

```bash
references/run.sh all                              # the reference images, once
uv run python -m scripts.evaluate                  # the pipeline's template step (no LLM)
uv run python -m scripts.evaluate --agent ffb      # the full agent on one repository
```

It checks:

1. The pipeline's image builds.
2. Python major.minor and the torch / torchvision versions match the reference image (`+cpu` / `+cuNNN` suffixes ignored).
3. The same `smoke.py` passes in the pipeline's image, without network.

Results: `runs/eval/results.md` and `runs/eval/results.json`.

The smoke scripts assume the repository is at `/workspace` (`/workspace/uap` for UAP, which imports itself as the package `uap`) and, for FFB, that the scripts are in `/workspace/src`.
