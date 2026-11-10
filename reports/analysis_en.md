# Reproducing EpiGraph: Results Analysis

Reproduction of *B cell epitope prediction by capturing spatial clustering property of the epitopes using graph attention network* (Choi & Kim, Sci Rep 2024, 14:27496; code: [sj584/EpiGraph](https://github.com/sj584/EpiGraph)).
All runs were done on Kaggle (GPU) with the authors' unmodified model and training code, apart from three compatibility patches for a newer `torch_geometric` (see [Section 2](#2-setup)).

## 1. Summary

- **The paper's headline numbers are reproduced.** The authors' 10 released checkpoints give exactly the published test metrics; retraining the full 10-fold ensemble from scratch gives numbers within 0.01 of them.
- **The test set is small, so the numbers are uncertain.** The epitope3D test set has only 45 antigens (7,747 surface residues, 10.0% epitopes). A bootstrap over antigens gives a 95% CI of **0.655–0.780** for the ensemble AUC-ROC, so the 0.02 gap to BepiPred-3.0 / DiscoTope-3.0 is not resolvable on this benchmark.
- **The ensemble matters.** Single models reach AUC-ROC 0.709 ± 0.014; the 10-model average reaches 0.725.
- **Training overfits early.** The best checkpoint is usually found at epoch 8–17 of 50.
- **Pending:** a multi-seed study (Section 6) will replace the single retraining run by mean ± std over several runs. Its code is ready; results are not yet included here.

## 2. Setup

| Item | Value |
|---|---|
| Model | 8-layer GAT (hidden 128, 8 heads) with residual connections; input = ESM-IF1 (512) + ESM-2 (1280) = 1792 features per residue |
| Data | epitope3D: 199 antigens in the provided training pickle (paper: 200; see Section 5, item 5), 45 in the test set |
| Training | 10-fold cross-validation (consecutive folds after a seeded shuffle), 50 epochs, batch 8, Adam, lr 1e-5; one model saved per fold |
| Checkpoint rule | a checkpoint is updated only when validation loss **and** validation AUC-ROC both improve |
| Prediction | mean of the 10 models' probabilities on surface residues (RSA ≥ 0.15); decision threshold 0.1481 for F1 / BACC / MCC |
| Patches | `data.node_attrs` → `data['node_attrs']`; checkpoint key `lin_src` → `lin`; `set()` of `Data` objects replaced by an index-based split (same samples, original order) |

Two runs were made: **(i)** inference with the authors' checkpoints, **(ii)** full retraining (10 folds × 50 epochs, ≈ 33 min on a Kaggle GPU) followed by the same evaluation.

## 3. Results on the epitope3D test set

| Metric | Paper | Authors' checkpoints | Retrained (1 run) |
|---|---|---|---|
| AUC-ROC | 0.73 | 0.73 (0.725) | 0.72 |
| AUC-PR | 0.23 | 0.23 (0.232) | 0.22 |
| F1 | 0.29 | 0.29 (0.287) | 0.29 |
| BACC | 0.62 | 0.62 (0.624) | 0.63 |
| MCC | 0.19 | 0.19 (0.193) | 0.19 |

Values in parentheses are the unrounded results from `src/eval_ensemble.py`; the upstream `evaluate.py` only prints two decimals.

Paper benchmark for context (Table 1): GraphBepi AUC-ROC 0.64, F1 0.28, MCC 0.16, BACC 0.62; BepiPred-3.0 and DiscoTope-3.0 AUC-ROC 0.71, F1 0.19–0.20, MCC 0.08–0.09, BACC 0.57. The retrained model (0.72) is therefore above GraphBepi in AUC-ROC and level with BepiPred-3.0 / DiscoTope-3.0.

**Reading the numbers**

- *Authors' checkpoints* match the paper to the printed precision: data, features and inference pipeline are correct.
- *Retraining* deviates by at most 0.01. The fold split is fixed by `random.seed(30)`, but upstream never seeds PyTorch, so each training run starts from different random weights; the sample order inside a fold also differs slightly because of the `set()` patch. A deviation of this size is what run-to-run noise would produce (quantified in Section 6).
- *AUC-PR is low in absolute terms but not poor.* With 10.0% positives, a random predictor has AUC-PR ≈ 0.10, so 0.23 is about 2.3× the chance level.

## 4. How certain are these numbers? (bootstrap over test antigens)

Residues from one antigen are strongly correlated, so antigens, not residues, were resampled (1,000 draws, pooled metrics recomputed on each draw). Authors' checkpoints, 10-model ensemble:

| Metric | Point estimate | 95% CI | Single model, mean ± std (n = 10) |
|---|---|---|---|
| AUC-ROC | 0.725 | 0.655 – 0.780 | 0.709 ± 0.014 |
| AUC-PR | 0.232 | 0.180 – 0.293 | 0.214 ± 0.014 |
| F1 | 0.287 | 0.219 – 0.338 | 0.272 ± 0.010 |
| BACC | 0.624 | 0.574 – 0.672 | 0.614 ± 0.012 |
| MCC | 0.193 | 0.123 – 0.252 | 0.175 ± 0.011 |

1. **The test-set uncertainty (±0.06 AUC-ROC) is several times larger than the retraining noise (≤ 0.01).** "Reproduced within 0.01" is therefore a statement about run-to-run stability, not evidence that the method beats others by a given margin.
2. **Competitor comparison.** BepiPred-3.0 / DiscoTope-3.0 (0.71) lie well inside the EpiGraph CI; GraphBepi (0.64) lies just below its lower bound. This is an unpaired comparison (the competitors' per-residue scores are not part of the released data), so a significance test is not possible; the CI only shows that the benchmark cannot separate EpiGraph from the two 0.71 methods.
3. **Ensembling helps consistently:** +0.016 AUC-ROC, +0.018 AUC-PR, +0.015 F1, +0.010 BACC, +0.018 MCC over the average single model. The validation-AUC / test-AUC gap noted in Section 5 is partly this ensemble effect.
4. **Pooled AUC overstates per-antigen discrimination.** The mean per-antigen AUC-ROC of the ensemble is 0.624 (median 0.643), clearly lower than the pooled 0.725. Pooling lets the model be rewarded for ranking residues of "epitope-rich" antigens above those of "epitope-poor" ones, which is a weaker claim than locating the epitope within one antigen. The paper's Fig. 3 (per-antigen AUC from 0.17 to 0.92) is consistent with this.

## 5. Training dynamics (retrained run)

Validation = the 20 antigens held out in each fold. The AUC-ROC reported by `train.py` is computed per batch (8 + 8 + 4 antigens) and averaged over batches, so it is a noisy selection signal, not an unbiased performance estimate; the paper does not report it.

| Fold | Held-out indices | Best epoch | Val loss | Val AUC-ROC |
|---|---|---|---|---|
| 1 | 0–19 | 14 | 0.327 | 0.628 |
| 2 | 20–39 | 12 | 0.277 | 0.705 |
| 3 | 40–59 | 17 | 0.274 | 0.650 |
| 4 | 60–79 | 16 | 0.301 | 0.713 |
| 5 | 80–99 | 9 | 0.351 | 0.686 |
| 6 | 100–119 | 32 | 0.278 | 0.761 |
| 7 | 120–139 | 8 | 0.299 | 0.657 |
| 8 | 140–159 | 11 | 0.330 | 0.630 |
| 9 | 160–179 | 11 | 0.381 | 0.701 |
| 10 | 180–198 | 17 | 0.264 | 0.662 |

Mean validation AUC-ROC ≈ 0.68.

1. **Validation AUC (0.68) is lower than test AUC (0.72) and varies from 0.63 to 0.76 across folds.** Each fold validates on only 20 antigens, so the estimate is noisy; the test ensemble additionally benefits from averaging 10 models.
2. **Early overfitting.** Training loss falls steadily to ≈ 0.23 while validation loss rises again after epoch ≈ 12–17; the selected epoch is 8–17 in nine of ten folds (fold 6: 32). The remaining epochs are wasted compute.
3. **The saved checkpoint is not the best-AUC epoch.** The two-metric rule only accepts an epoch when loss and AUC improve together (e.g. fold 1 reaches ≈ 0.635 AUC at epoch ≈ 22 but the model saved is epoch 14 with 0.6275).
4. **Late instability.** Fold 2 (epochs 46–49) and fold 4 (epochs 44–47) show validation-loss spikes to 0.37–0.38. They do not affect results because the selected checkpoint is much earlier.
5. **Antigen count (199 vs 200).** Per-fold counts are 179 + 20 = 199 antigens, whereas the paper trains on 200. The authors' `csv files/` list 200 antigens for the epitope3D training data (180 in `epitope3d_dataset_180_Train.csv`, 20 in `epitope3d_dataset_20_Test.csv`, no duplicates). Comparing PDB identifiers shows that exactly one of them, **3FCS**, is absent from the released `epitope3d_train_data.pkl`; every other entry matches, and the 45 test antigens match the blind-test list exactly. The released pickle therefore omits one antigen of the described training set. The reason is not documented in the repository (a failure during structure or embedding preprocessing is a possible but unverified explanation). The effect of one missing antigen out of 200 is negligible next to the noise discussed above.
6. **Timing.** The "training time" printed in the log is cumulative (≈ 190 s per fold; 1,920 s at fold 10).

## 6. Robustness study: what is being tested and why

Section 3 rests on a single retraining run, so it cannot separate "the method reproduces" from "this run was lucky". Upstream `train.py` has two independent sources of randomness:

| Source | Controlled by | Effect |
|---|---|---|
| Fold assignment | `--seed` (default 30; `random.seed` + `random.shuffle` before the consecutive `KFold`) | which 20 antigens validate each model, and which 179 train it |
| Weight initialisation (and any torch RNG) | not seeded upstream | starting point of each model |

`src/train_seeded.py` seeds PyTorch/NumPy and then runs the unmodified `train.py`, so both sources can be varied separately:

- **Experiment A – initialisation only.** Fold split fixed at seed 30; torch seeds 1–5 → five independent 10-model ensembles. Isolates training noise.
- **Experiment B – fold split + initialisation.** Split seeds 31–33 (torch seed equal to split seed) → three ensembles. Captures how sensitive the result is to the cross-validation partition.

Each run is evaluated with `src/eval_ensemble.py`, and `src/aggregate.py` reports mean ± std, min–max, a 95% t-interval of the mean, and whether the paper's value falls inside that interval (allowing ±0.005 for the paper's rounding), plus single-model vs ensemble gains. The study is executed by `notebooks/02_multiseed_robustness_kaggle.ipynb` on Kaggle (≈ 33 min per ensemble; 8 ensembles ≈ 4.5 h).

Caveats that remain by design: (i) GPU runs are not bit-for-bit reproducible even with fixed seeds, because the scatter operations in `GATConv` use non-deterministic atomics, so seeds make runs comparable rather than identical; (ii) the decision threshold 0.1481 comes from the authors' validation procedure (average of per-fold MCC-maximising thresholds) and is reused for retrained models, so F1/BACC/MCC of retrained models also depend on how well that threshold transfers, whereas AUC-ROC and AUC-PR do not.

**Results:** _to be added after the Kaggle run (paste `results/summary.md` here)._

## 7. Limitations

- Only one test set of 45 antigens; CIs are wide (Section 4). The DiscoTope-3 benchmark in the paper (Table 2) was not reproduced here.
- Competitor numbers are taken from the paper, not re-run, and cannot be compared with paired statistics.
- The pooled-residue metrics reward between-antigen score calibration; per-antigen AUC is substantially lower (Section 4).
- Training-set labels come from antibody–antigen complexes and are incomplete (unlabelled residues may be true epitopes), a limitation the paper acknowledges.

## 8. Conclusion

The published epitope3D results of EpiGraph can be reproduced: exactly with the released checkpoints, and within 0.01 when retraining. The model reliably learns a useful signal (AUC-ROC ≈ 0.72, AUC-PR ≈ 2.3× chance), benefits clearly from ensembling, and overfits early. However, with 45 test antigens the 95% interval for AUC-ROC is about ±0.06, which is too wide to rank EpiGraph against BepiPred-3.0 or DiscoTope-3.0; stronger claims need larger test sets or paired comparisons. The multi-seed study in Section 6 will quantify the retraining variance directly.

## Glossary

- **Seed**: the starting value of a pseudo-random number generator. The same seed gives the same sequence of "random" numbers, so a seed is what makes a random choice (shuffling the data, initialising weights) repeatable.
- **Ensemble**: here, the average of the probabilities of the 10 fold models.
- **AUC-ROC / AUC-PR**: threshold-free ranking metrics; AUC-PR is sensitive to the 10% positive rate.
- **BACC / MCC / F1**: threshold-dependent metrics, computed at 0.1481.
- **RSA**: relative solvent accessibility; residues with RSA ≥ 0.15 are treated as surface residues.
