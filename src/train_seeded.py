"""Run the upstream train.py with the *PyTorch* random number generators seeded.

Upstream train.py only seeds Python's `random` (which controls the order of the 200
antigens before the consecutive 10-fold split, `--seed`, default 30). It never seeds
PyTorch, so the initial weights of every model differ from run to run.

This wrapper seeds torch / numpy first and then executes the unmodified train.py, so
that two sources of randomness can be controlled independently:

    --torch_seed N   weight initialisation (and any torch RNG use)
    --seed N         fold split (upstream argument, forwarded untouched)

Usage (all other arguments are forwarded to train.py):

    python src/train_seeded.py --upstream EpiGraph --torch_seed 1 --seed 30 \
        --model_save_dir runs/init_s1/models

Note: even with identical seeds, GPU runs are not bit-for-bit reproducible, because
scatter/atomic-add kernels used by GATConv are non-deterministic. Seeds make runs
*comparable* (same initialisation, same split), not identical.
"""
import argparse
import os
import runpy
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--upstream", required=True, help="path to the (patched) upstream repo")
    ap.add_argument("--torch_seed", type=int, required=True)
    known, rest = ap.parse_known_args()

    upstream = os.path.abspath(known.upstream)
    # make relative --model_save_dir independent of the chdir below
    for i, tok in enumerate(rest):
        if tok == "--model_save_dir" and i + 1 < len(rest):
            rest[i + 1] = os.path.abspath(rest[i + 1])

    torch.manual_seed(known.torch_seed)
    torch.cuda.manual_seed_all(known.torch_seed)
    np.random.seed(known.torch_seed)

    # train.py reads the .pkl files and imports `model` relative to the repo root
    os.chdir(upstream)
    sys.path.insert(0, upstream)
    sys.argv = ["train.py"] + rest
    runpy.run_path("train.py", run_name="__main__")


if __name__ == "__main__":
    main()
