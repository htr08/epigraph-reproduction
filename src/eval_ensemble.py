"""Evaluate an ensemble of EpiGraph checkpoints on the epitope3D test set.

Differs from upstream evaluate.py in three ways (the ensemble definition is the same:
mean of the per-model probabilities over surface residues, fixed decision threshold):

  * metrics are written to JSON with full precision (upstream prints 2 decimals);
  * metrics of every single model are reported next to the ensemble;
  * a bootstrap over *antigens* gives 95% confidence intervals for the ensemble.

    python src/eval_ensemble.py --upstream EpiGraph --ckpt_dir runs/init_s1/models \
        --out runs/init_s1/metrics.json

Requires the upstream repo to be patched first (src/patch_upstream.py).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn import metrics as skm


def load_state(path):
    sd = torch.load(path, map_location="cpu")
    # checkpoints written by an old torch_geometric use lin_src/lin_dst
    return {k.replace("lin_src", "lin"): v for k, v in sd.items() if "lin_dst" not in k}


@torch.no_grad()
def predict_all(model, data_list, device):
    """Per-antigen arrays of predicted probability on surface residues (train_mask)."""
    model.eval()
    out = []
    for d in data_list:
        d = d.to(device)
        p = model(d)[d.train_mask].reshape(-1).cpu().numpy()
        out.append(p)
    return out


def threshold_free(y, p):
    fpr, tpr, _ = skm.roc_curve(y, p)
    prec, rec, _ = skm.precision_recall_curve(y, p)
    return {
        "auc_roc": float(skm.auc(fpr, tpr)),
        # same definition as upstream evaluate.py (trapezoid under the PR curve)
        "auc_pr": float(skm.auc(rec, prec)),
        "avg_precision": float(skm.average_precision_score(y, p)),
    }


def all_metrics(y, p, thr):
    m = threshold_free(y, p)
    yhat = (p > thr).astype(int)
    m["f1"] = float(skm.f1_score(y, yhat))
    m["bacc"] = float(skm.balanced_accuracy_score(y, yhat))
    m["mcc"] = float(skm.matthews_corrcoef(y, yhat))
    return m


def summarize(values):
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "values": [float(x) for x in v],
    }


def bootstrap_ci(y_list, p_list, thr, n_boot, seed):
    """Resample antigens (not residues) with replacement; residues of one antigen are correlated."""
    rng = np.random.default_rng(seed)
    n = len(y_list)
    keys = ["auc_roc", "auc_pr", "f1", "bacc", "mcc"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = np.concatenate([y_list[i] for i in idx])
        p = np.concatenate([p_list[i] for i in idx])
        if y.min() == y.max():
            continue
        m = all_metrics(y, p, thr)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", required=True, help="metrics JSON to write")
    ap.add_argument("--threshold", type=float, default=0.1481, help="upstream default")
    ap.add_argument("--hid_dim", type=int, default=128)
    ap.add_argument("--num_head", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.upstream))
    from model import GAT  # noqa: E402  (needs the patched upstream model.py)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    with open(os.path.join(a.upstream, "epitope3d_test_data.pkl"), "rb") as f:
        test = pickle.load(f)

    ckpts = sorted(f for f in os.listdir(a.ckpt_dir) if f.endswith(".pt"))
    assert ckpts, f"no .pt files in {a.ckpt_dir}"

    y_list = [d.y[d.train_mask].cpu().numpy().astype(int) for d in test]
    y = np.concatenate(y_list)

    per_model = []  # list over models of list over antigens
    for name in ckpts:
        model = GAT(in_dim=1792, hid_dim=a.hid_dim, out_dim=1, num_head=a.num_head, out_head=1)
        model.load_state_dict(load_state(os.path.join(a.ckpt_dir, name)))
        model.to(device)
        per_model.append(predict_all(model, test, device))

    ens_list = [np.mean([pm[i] for pm in per_model], axis=0) for i in range(len(test))]
    p_ens = np.concatenate(ens_list)

    single = [all_metrics(y, np.concatenate(pm), a.threshold) for pm in per_model]
    keys = ["auc_roc", "auc_pr", "f1", "bacc", "mcc"]

    per_antigen = [
        skm.roc_auc_score(yy, pp) for yy, pp in zip(y_list, ens_list) if 0 < yy.sum() < len(yy)
    ]

    result = {
        "ckpt_dir": a.ckpt_dir,
        "checkpoints": ckpts,
        "n_models": len(ckpts),
        "n_antigens": len(test),
        "n_surface_residues": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": a.threshold,
        "ensemble": all_metrics(y, p_ens, a.threshold),
        "ensemble_ci95_bootstrap_antigens": bootstrap_ci(
            y_list, ens_list, a.threshold, a.n_boot, a.boot_seed
        ),
        "single_models": {k: summarize([m[k] for m in single]) for k in keys},
        "per_antigen_auc_roc": {
            "n_antigens_with_both_classes": len(per_antigen),
            "mean": float(np.mean(per_antigen)),
            "median": float(np.median(per_antigen)),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        os.path.splitext(a.out)[0] + "_preds.npz",
        y=y,
        antigen=np.concatenate([np.full(len(v), i) for i, v in enumerate(y_list)]),
        ensemble=p_ens,
        per_model=np.stack([np.concatenate(pm) for pm in per_model]),
    )

    e = result["ensemble"]
    ci = result["ensemble_ci95_bootstrap_antigens"]
    print(f"[{a.ckpt_dir}] {len(ckpts)} models, {len(test)} antigens, "
          f"{len(y)} surface residues, positive rate {y.mean():.3f}")
    for k in keys:
        print(f"  {k:8s} {e[k]:.4f}   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]   "
              f"single-model mean {result['single_models'][k]['mean']:.4f} "
              f"± {result['single_models'][k]['std']:.4f}")


if __name__ == "__main__":
    main()
