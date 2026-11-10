"""Build notebooks/02_multiseed_robustness_kaggle.ipynb.

The notebook is self-contained (it embeds src/*.py with %writefile) so it runs on Kaggle
without the GitHub repo being public. src/ stays the single source of truth: after
editing a file there, run

    python tools/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "02_multiseed_robustness_kaggle.ipynb")
SRC_FILES = ["patch_upstream.py", "train_seeded.py", "eval_ensemble.py", "aggregate.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# EpiGraph - multi-seed robustness study (Kaggle)

Requirements: **GPU** accelerator, **Internet** enabled, and an input dataset containing the two `.pkl` files.
Cells are run top to bottom. For the long run, **Save Version -> Save & Run All (Commit)** keeps the notebook running in the background.

Contents
1. Evaluates the authors' 10 checkpoints with 4-decimal metrics + 95% bootstrap CI over test antigens.
2. **Experiment A** - retrains the full 10-fold ensemble for several *weight-initialisation* seeds (fold split fixed at the paper's seed 30).
3. **Experiment B** - retrains with several *fold-split* seeds (split seed and init seed both change).
4. Aggregates everything into mean +- std tables (`results/summary.md`).

One full ensemble = 10 folds x 50 epochs ~ 33 min on a Kaggle GPU. The notebook is resumable:
runs that already have a `metrics.json` are skipped.
""")

code("""
# ---- configuration ---------------------------------------------------------
INIT_SEEDS  = [1, 2, 3, 4, 5]   # experiment A: torch seeds, split seed stays BASE_SPLIT_SEED
SPLIT_SEEDS = [31, 32, 33]      # experiment B: split seed = torch seed = value ([] to skip)
BASE_SPLIT_SEED = 30            # upstream default (--seed)
EPOCHS = 50                     # upstream default
KEEP_CHECKPOINTS = False        # True keeps ~120 MB of .pt files per run in the notebook output
""")

code("""
# ---- 1. clone upstream repo, copy data, install --------------------------------
import os, glob, shutil, subprocess, sys
W = "/kaggle/working"
os.chdir(W)
if not os.path.exists("EpiGraph"):
    subprocess.run(["git", "clone", "https://github.com/sj584/EpiGraph"], check=True)
pkls = glob.glob("/kaggle/input/**/*.pkl", recursive=True)
assert pkls, "No .pkl found - add the dataset with epitope3d_*_data.pkl as notebook input"
for p in pkls:
    shutil.copy(p, f"{W}/EpiGraph/")
os.makedirs(f"{W}/src", exist_ok=True)
print(pkls)
""")
code("!pip install -q torch_geometric")

for fn in SRC_FILES:
    with open(os.path.join(ROOT, "src", fn), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile /kaggle/working/src/{fn}\n{body}")

code("""
# ---- 2. patch upstream for the new torch_geometric (idempotent) ---------------------
!python /kaggle/working/src/patch_upstream.py --upstream /kaggle/working/EpiGraph
""")

code("""
# ---- 3. author checkpoints: 4-decimal metrics + bootstrap CI ------------------------
import os, subprocess, time, json
W = "/kaggle/working"; RUNS = f"{W}/runs"
os.makedirs(f"{RUNS}/pretrained", exist_ok=True)

def sh(cmd, log=None):
    t = time.time()
    if log:
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd)
    assert r.returncode == 0, f"failed: {' '.join(cmd)} (see {log})"
    return time.time() - t

def evaluate(name, ckpt_dir):
    sh(["python", f"{W}/src/eval_ensemble.py", "--upstream", f"{W}/EpiGraph",
        "--ckpt_dir", ckpt_dir, "--out", f"{RUNS}/{name}/metrics.json"])

if not os.path.exists(f"{RUNS}/pretrained/metrics.json"):
    evaluate("pretrained", f"{W}/EpiGraph/checkpoint")
""")

code("""
# ---- 4. retrain (resumable) ------------------------------------------------------
import shutil, glob
jobs = [(f"init_s{s}", s, BASE_SPLIT_SEED) for s in INIT_SEEDS] + [(f"split_s{s}", s, s) for s in SPLIT_SEEDS]

for name, torch_seed, split_seed in jobs:
    run = f"{RUNS}/{name}"
    if os.path.exists(f"{run}/metrics.json"):
        print(name, "already done, skipping"); continue
    shutil.rmtree(run, ignore_errors=True)       # drop half-finished runs
    os.makedirs(run)
    dt = sh(["python", "-u", f"{W}/src/train_seeded.py", "--upstream", f"{W}/EpiGraph",
             "--torch_seed", str(torch_seed), "--seed", str(split_seed), "--epochs", str(EPOCHS),
             "--model_save_dir", f"{run}/models"], log=f"{run}/train.log")
    assert len(glob.glob(f"{run}/models/*.pt")) == 10, "expected 10 checkpoints"
    evaluate(name, f"{run}/models")
    if not KEEP_CHECKPOINTS:
        shutil.rmtree(f"{run}/models")
    m = json.load(open(f"{run}/metrics.json"))["ensemble"]
    print(f"{name}: train {dt/60:.1f} min | AUC-ROC {m['auc_roc']:.4f} AUC-PR {m['auc_pr']:.4f} "
          f"F1 {m['f1']:.4f} BACC {m['bacc']:.4f} MCC {m['mcc']:.4f}")
""")

code("""
# ---- 5. aggregate + package small outputs ------------------------------------------
!python /kaggle/working/src/aggregate.py --runs /kaggle/working/runs --out /kaggle/working/results
!cd /kaggle/working && zip -qr results.zip results runs -x "runs/*/models/*" && ls -la results.zip
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", OUT, f"({len(cells)} cells)")
