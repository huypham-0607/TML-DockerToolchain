"""
CPU smoke test for the EP/BNP reference image. No GPU, no network.

    docker run --rm -i tml-ref/ep_bnp python - < references/ep_bnp/smoke.py

Runs the repository's own training loop (train.main) for 10 short epochs on a tiny RANDOM
stand-in for CIFAR-10 (so nothing is downloaded), with a BadNets trigger, and checks that the
checkpoint is written. test.py / defense.py are only imported: the defenses call .cuda().

With the real data (downloads CIFAR-10, ~170 MB, slow on CPU):
    docker run --rm tml-ref/ep_bnp python train.py --device cpu --epochs 1 \
        --attack-type badnets --poisoning-rate 0.1 --trigger-size 3 --manual-seed 0
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torchvision

print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", np.__version__, "| cuda available", torch.cuda.is_available())


class TinyCIFAR10(torch.utils.data.Dataset):
    """Random 32x32 RGB images with CIFAR-10 labels; replaces torchvision's CIFAR10 here."""

    def __init__(self, root, train=True, download=False, transform=None, **_):
        n = 200 if train else 100
        g = torch.Generator().manual_seed(1 if train else 2)
        self.data = (torch.rand(n, 32, 32, 3, generator=g) * 255).to(torch.uint8).numpy()
        self.targets = [i % 10 for i in range(n)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


torchvision.datasets.CIFAR10 = TinyCIFAR10

import train  # noqa: E402
import test  # noqa: E402,F401  (imports defense.py)
import torchattacks  # noqa: E402,F401  (used by the 'cla' attack)

ckpt = tempfile.mkdtemp(prefix="ep-bnp-ckpt-")
args = argparse.Namespace(
    model="resnet18", workers=0, epochs=10, batch_size=64, lr=0.01, momentum=0.9, weight_decay=5e-4,
    checkpoint=ckpt, manual_seed=0, device="cpu", dataset="cifar10", dataset_dir="/tmp/unused",
    attack_type="badnets", target_label=0, poisoning_rate=0.1, trigger_size=3,
)
torch.manual_seed(0)
train.args = args   # train() and val() read a module-level `args` that `python train.py` creates
train.main(args)

saved = os.listdir(ckpt)
assert saved, "no checkpoint written"
state = torch.load(os.path.join(ckpt, saved[0]), map_location="cpu")
assert "state_dict" in state and "trigger" in state
print("checkpoint ok:", saved[0])
print("SMOKE OK")
