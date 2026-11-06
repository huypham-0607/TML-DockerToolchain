"""
CPU smoke test for the UAP reference image. No datasets, no GPU, no network.

    docker run --rm -i tml-ref/uap python - < references/uap/smoke.py

Checks: `art` is the Adversarial Robustness Toolbox (not the ASCII-art package that
requirements.txt also lists), the package-style import main.py uses works, models build, and
ART's FGSM and universal perturbation run against a repository model. main.py itself cannot
run: it loads files from the authors' machine (/home/fazhong/studio/uap/...).
"""
import sys

# The code imports itself as the package `uap` (main.py: from uap.train.model_structure import ...),
# and train/*.py import `models` from the repo root. Say so plainly when an image lacks that layout.
try:
    import uap.train.model_structure  # noqa: F401
except ModuleNotFoundError as e:
    sys.exit(f"FAIL: {e}. UAP must be importable as the package `uap`: put the repository at <dir>/uap "
             "and add <dir> and <dir>/uap to PYTHONPATH (the reference uses /workspace/uap)")

import numpy as np
import sklearn
import torch
import torchvision

import art

print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", np.__version__, "| scikit-learn", sklearn.__version__, "| ART", getattr(art, "__version__", "?"),
      "| cuda available", torch.cuda.is_available())
assert getattr(art, "__version__", "") == "1.17.1", "the `art` module is not the Adversarial Robustness Toolbox"

from art.attacks.evasion import CarliniL2Method, FastGradientMethod, UniversalPerturbation  # noqa: E402,F401
from art.estimators.classification import PyTorchClassifier  # noqa: E402

from encoder import classifier  # noqa: E402
from uap.train.model_structure import get_model  # noqa: E402  (the import main.py uses)

x = np.random.rand(8, 3, 32, 32).astype(np.float32)

# train/model_structure.get_model lists ~45 names, but most import functions that the repo's
# models/ package does not define. Report which ones build; resnet18 (used by main.py) must.
NAMES = ["resnet18", "resnet34", "resnet50", "vgg16", "densenet121", "googlenet", "mobilenet",
         "mobilenetv2", "shufflenetv2", "preactresnet18", "resnext50", "seresnet18"]
working, broken = [], []
for name in NAMES:
    try:
        net = get_model(name, False, 10).eval()
        with torch.no_grad():
            out = net(torch.from_numpy(x[:2]))
        assert out.shape == (2, 10), out.shape
        working.append(name)
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        broken.append(f"{name} ({type(e).__name__})")
print("get_model works for:", ", ".join(working))
print("get_model broken for:", ", ".join(broken) or "none")
assert "resnet18" in working, "resnet18 must build"

encoder = classifier(input_channel=3000)
print("encoder classifier builds ok")

net = get_model("resnet18", False, 10).eval()
clf = PyTorchClassifier(model=net, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 32, 32),
                        nb_classes=10, clip_values=(0.0, 1.0), device_type="cpu")
adv = FastGradientMethod(clf, eps=0.03).generate(x)
assert adv.shape == x.shape
print("ART FGSM ok, max change %.3f" % float(np.abs(adv - x).max()))
uap = UniversalPerturbation(clf, attacker="fgsm", max_iter=1, eps=0.1, batch_size=8, verbose=False)
uap.generate(x)
assert uap.noise is not None
print("ART universal perturbation ok, noise shape", tuple(uap.noise.shape))

print("SMOKE OK")
