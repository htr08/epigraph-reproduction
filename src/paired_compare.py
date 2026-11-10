"""Paired comparison of the authors' checkpoints with the retrained ensembles.

Uses the per-residue predictions saved by eval_ensemble.py (runs/<name>/metrics_preds.npz).
All ensembles are evaluated on the same test residues, so the difference of a metric can be
bootstrapped over antigens *jointly* (same resampled antigens for both models). This gives
a paired confidence interval that accounts for test-set sampling noise only; it does not
include the training variance of the authors' checkpoints (a single set of 10 models).

    python src/paired_compare.py --runs results/02_multiseed/runs --out results/02_multiseed/paired_comparison.md
"""
import argparse
import glob
import os

import numpy as np
from sklearn import metrics as skm


def auc_pr(y, p):
    prec, rec, _ = skm.precision_recall_curve(y, p)
    return skm.auc(rec, prec)


def auc_roc(y, p):
    return skm.roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    def load(name):
        return np.load(os.path.join(a.runs, name, "metrics_preds.npz"))

    ref = load("pretrained")
    y, ant, p_ref = ref["y"], ref["antigen"], ref["ensemble"]
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.runs, "*", "metrics_preds.npz")))
    names = [n for n in names if n != "pretrained"]
    cands = {}
    for n in names:
        d = load(n)
        assert (d["y"] == y).all(), "runs were evaluated on different residues"
        cands[n] = d["ensemble"]
    cands[f"mean of all {len(names)} ensembles"] = np.mean(list(cands.values()), axis=0)

    idx = [np.where(ant == i)[0] for i in range(ant.max() + 1)]
    rng = np.random.default_rng(a.seed)
    draws = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(a.n_boot)]

    def row(fn, p):
        pt = fn(y, p_ref) - fn(y, p)
        d = [fn(y[ix], p_ref[ix]) - fn(y[ix], p[ix]) for ix in draws]
        return f"{pt:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"

    lines = [
        "# Authors' checkpoints minus retrained ensembles (paired bootstrap over test antigens)", "",
        f"Difference = metric(authors' 10 checkpoints) - metric(retrained ensemble); point estimate and 95% CI "
        f"({a.n_boot} antigen-level bootstrap draws, same antigens for both models).", "",
        "| Retrained ensemble | Diff. AUC-PR | Diff. AUC-ROC |", "|---|---|---|",
    ]
    for n, p in cands.items():
        lines.append(f"| {n} | {row(auc_pr, p)} | {row(auc_roc, p)} |")
    text = "\n".join(lines) + "\n"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
