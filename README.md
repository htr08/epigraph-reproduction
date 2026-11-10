# epigraph-reproduction

Independent reproduction and robustness study of **EpiGraph** (Choi & Kim, *Sci Rep* 2024, 14:27496,
[doi:10.1038/s41598-024-78506-z](https://doi.org/10.1038/s41598-024-78506-z)),
a graph-attention model for B cell epitope prediction. Original code: <https://github.com/sj584/EpiGraph>.

This repository contains only original material (notebooks, scripts, reports). The upstream code and the
data are not copied here; the notebooks clone upstream at run time.

## Main findings

| Metric | Paper | Authors' checkpoints | Retrained, 5 init seeds (mean ± std) | Retrained, 3 split seeds (mean ± std) |
|---|---|---|---|---|
| AUC-ROC | 0.73 | 0.725 | 0.726 ± 0.005 | 0.727 ± 0.004 |
| AUC-PR | 0.23 | 0.232 | 0.215 ± 0.003 | 0.219 ± 0.012 |
| F1 | 0.29 | 0.287 | 0.281 ± 0.010 | 0.278 ± 0.005 |
| BACC | 0.62 | 0.624 | 0.621 ± 0.006 | 0.617 ± 0.006 |
| MCC | 0.19 | 0.193 | 0.184 ± 0.012 | 0.182 ± 0.007 |

- Reproduced: exactly with the released checkpoints; AUC-ROC, BACC, F1 and MCC also when retraining (8 independent 10-fold ensembles).
- Exception: retrained AUC-PR (mean 0.215 and 0.219) is below the released checkpoints' 0.232 in all 8 runs; the cause cannot be determined from these data.
- 95% bootstrap CI of the ensemble AUC-ROC on the 45-antigen test set: 0.655 - 0.780 (about ten times wider than the retraining noise, std 0.005).
- Full write-up: [reports/analysis_en.md](reports/analysis_en.md) (Vietnamese: [reports/analysis_vi.md](reports/analysis_vi.md)).

## Layout

```
.
├── notebooks/
│   ├── 01_reproduce_epigraph_kaggle.ipynb    clone upstream, patch, evaluate checkpoints, retrain once
│   └── 02_multiseed_robustness_kaggle.ipynb  multi-seed retraining, bootstrap CI, aggregation (self-contained)
├── src/
│   ├── patch_upstream.py    compatibility patches for new torch_geometric
│   ├── train_seeded.py      seeds PyTorch, then runs upstream train.py unchanged
│   ├── eval_ensemble.py     4-decimal metrics, per-model metrics, bootstrap CI over antigens
│   ├── aggregate.py         mean +- std tables over runs
│   └── paired_compare.py    paired bootstrap: authors' checkpoints vs retrained ensembles
├── tools/
│   └── build_notebook.py    regenerates notebook 02 from src/
├── reports/
│   ├── analysis_en.md       full analysis (English)
│   └── analysis_vi.md       analysis of the first retraining run (Vietnamese)
├── results/
│   └── 02_multiseed/        output of notebook 02
│       ├── summary.md, summary.json, paired_comparison.md
│       └── runs/<name>/     metrics.json, train.log, metrics_preds.npz
├── logs/                    raw Kaggle logs, named after the notebook that produced them
│   ├── 01_reproduce_epigraph_kaggle.log
│   └── 02_multiseed_robustness_kaggle.log
└── data/                    git-ignored: epitope3d_{train,test}_data.pkl
```

## How to run

Data: download the two preprocessed pickles (`epitope3d_train_data.pkl`, `epitope3d_test_data.pkl`) from the
upstream README's Google Drive link and upload them as a Kaggle dataset.

1. Kaggle -> New Notebook -> File -> Import notebook -> `notebooks/02_multiseed_robustness_kaggle.ipynb`.
2. Settings: Accelerator **GPU**, Internet **on**, Add Input -> the dataset containing the two `.pkl` files.
3. *Save Version -> Save & Run All (Commit)*. About 33 min per 10-model ensemble; the notebook is resumable.
4. Download the notebook output (`results/` and `runs/`, checkpoints excluded). The small files from both are kept in this repository under `results/02_multiseed/`.

Edit the `INIT_SEEDS` / `SPLIT_SEEDS` lists in the first code cell to change the number of runs.
After editing anything in `src/`, run `python tools/build_notebook.py` to refresh notebook 02.

Local check (CPU, no GPU needed for evaluation): clone upstream, `python src/patch_upstream.py --upstream EpiGraph`,
copy the `.pkl` files into it, then `python src/eval_ensemble.py --upstream EpiGraph --ckpt_dir EpiGraph/checkpoint --out out/m.json --device cpu`.

## Notes

- The paper is CC BY-NC-ND 4.0 and the upstream repository has no license file, so neither is redistributed here.
- GPU training is not bit-for-bit reproducible even with fixed seeds (GPU scatter kernels can be non-deterministic).
