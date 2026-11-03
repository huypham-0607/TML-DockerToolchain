"""
CPU smoke test for the IPMix reference image. No datasets, no GPU, no network.

    docker run --rm -i tml-ref/ipmix python - < references/ipmix/smoke.py

Checks: versions, the IPMix augmentation on a random image, every model in cifar.py builds and
runs forward, and one training step through IPMixDataset. cifar.main() itself is not run: it
calls .cuda() and needs the CIFAR data and the mixing set.
"""
import sys

# cifar.py parses its arguments at import time and requires these two
sys.argv = ["cifar.py", "--data-path", "/tmp", "--mixing-set", "/tmp"]

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torchvision import transforms

print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", np.__version__, "| Pillow", PIL.__version__, "| cuda available", torch.cuda.is_available())

import cifar  # noqa: E402
import IPMix_utils  # noqa: E402
from models.ResNeXt_DenseNet.models.densenet import densenet  # noqa: E402
from models.ResNeXt_DenseNet.models.resnext import resnext29  # noqa: E402
from models.ResNet.resnet import resnet18  # noqa: E402
from models.WideResNet_pytorch.wideresnet import WideResNet  # noqa: E402


def random_image():
    return Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))


preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

# every image-level augmentation, then the full IPMix mix
for op in IPMix_utils.augmentations:
    op(random_image(), 3)
mixed = cifar.ipmix(random_image(), random_image(), preprocess)
assert mixed.shape == (3, 32, 32), mixed.shape
print("ipmix augmentation ok:", tuple(mixed.shape))

x = torch.randn(2, 3, 32, 32)
for name, net in [("wrn", WideResNet(40, 10, 4, 0.3)), ("resnet18", resnet18(num_classes=10)),
                  ("resnext29", resnext29(num_classes=10)), ("densenet", densenet(num_classes=10))]:
    net.eval()
    with torch.no_grad():
        out = net(x)
    if name == "resnet18" and out.shape == (2, 100):
        # KNOWN REPO BUG: models/ResNet/resnet.py resnet18(num_classes) ignores num_classes,
        # so the head always has 100 outputs (CIFAR-10 runs train a 100-way classifier)
        print("model resnet18 forward ok (KNOWN REPO BUG: num_classes ignored, 100 outputs)")
        continue
    assert out.shape == (2, 10), (name, out.shape)
    print(f"model {name} forward ok")

# one training step on CPU through the repo's dataset wrapper (no JSD, like --no-jsd)
data = [(random_image(), i % 10) for i in range(8)]
mixing = [(random_image(), 0) for _ in range(4)]
loader = torch.utils.data.DataLoader(cifar.IPMixDataset(data, mixing, preprocess, no_jsd=True), batch_size=4)
net = WideResNet(16, 10, 1, 0.0)   # small WRN: keeps the CPU step fast
opt = torch.optim.SGD(net.parameters(), lr=0.1)
images, targets = next(iter(loader))
loss = torch.nn.functional.cross_entropy(net(images), targets)
opt.zero_grad()
loss.backward()
opt.step()
print("training step ok, loss %.3f" % loss.item())

print("SMOKE OK")
