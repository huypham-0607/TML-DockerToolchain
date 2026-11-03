"""
CPU smoke test for the FFB reference image. No GPU, no network, no wandb account.

    docker run --rm -i tml-ref/ffb python - < references/ffb/smoke.py

Writes a small SYNTHETIC dataset in the UCI Adult format to a temporary directory, then runs
the repository's own scripts on it, end to end: ffb_tabular_erm.py and ffb_tabular_diffdp.py.
A mounted real dataset at /workspace/datasets is not touched.

With the real data (datasets.zip from the Dropbox link in datasets/readme.md):
    docker run --rm -v "$PWD/datasets:/workspace/datasets" tml-ref/ffb \
        python -u ffb_tabular_erm.py --dataset adult --model erm --sensitive_attr sex \
        --target_attr income --num_training_steps 150
"""
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import pandas
import sklearn
import torch

print("python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", np.__version__,
      "| pandas", pandas.__version__, "| scikit-learn", sklearn.__version__,
      "| cuda available", torch.cuda.is_available())

SRC = "/workspace/src"
WORKCLASS = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov"]
EDUCATION = [("Bachelors", 13), ("HS-grad", 9), ("Masters", 14), ("Some-college", 10)]
MARITAL = ["Never-married", "Married-civ-spouse", "Divorced"]
OCCUPATION = ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Adm-clerical"]
RELATIONSHIP = ["Husband", "Wife", "Not-in-family", "Own-child"]
RACE = ["White", "Black", "Asian-Pac-Islander"]
COUNTRY = ["United-States", "Mexico", "India"]


def adult_rows(n, test):
    rng = random.Random(0 if not test else 1)
    rows = []
    for _ in range(n):
        edu, edu_num = rng.choice(EDUCATION)
        sex = rng.choice(["Male", "Female"])
        high = rng.random() < (0.35 if sex == "Male" else 0.15)
        label = (">50K" if high else "<=50K") + ("." if test else "")
        rows.append(", ".join(map(str, [
            rng.randint(18, 70), rng.choice(WORKCLASS), rng.randint(20000, 400000), edu, edu_num,
            rng.choice(MARITAL), rng.choice(OCCUPATION), rng.choice(RELATIONSHIP), rng.choice(RACE), sex,
            rng.choice([0, 0, 0, 5000]), 0, rng.randint(20, 60), rng.choice(COUNTRY), label,
        ])))
    return rows


tmp = tempfile.mkdtemp(prefix="ffb-smoke-")
raw = os.path.join(tmp, "datasets", "adult", "raw")
os.makedirs(raw)
os.makedirs(os.path.join(tmp, "work"))
with open(os.path.join(raw, "adult.data"), "w") as f:
    f.write("\n".join(adult_rows(600, test=False)) + "\n")
with open(os.path.join(raw, "adult.test"), "w") as f:
    f.write("|1x3 Cross validator\n" + "\n".join(adult_rows(300, test=True)) + "\n")

common = ["--dataset", "adult", "--sensitive_attr", "sex", "--target_attr", "income",
          "--num_training_steps", "5", "--batch_size", "64", "--evaluation_batch_size", "4096", "--log_freq", "5"]
runs = [
    ["ffb_tabular_erm.py", "--model", "erm"],
    ["ffb_tabular_diffdp.py", "--model", "diffdp", "--lam", "1.0"],
]
for script, *extra in runs:
    # the scripts read ../datasets/adult/raw relative to the working directory
    cmd = [sys.executable, "-u", os.path.join(SRC, script)] + extra + common
    p = subprocess.run(cmd, cwd=os.path.join(tmp, "work"), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        sys.exit(f"{script} failed with exit code {p.returncode}")
    print(f"{script} ok ({len(p.stdout.splitlines())} lines of output)")

print("SMOKE OK")
