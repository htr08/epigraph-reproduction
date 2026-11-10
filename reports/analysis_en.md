# Reproducing EpiGraph: Results Analysis

Reproduction of *B cell epitope prediction by capturing spatial clustering property of the epitopes using graph attention network* (Choi & Kim, Sci Rep 2024, 14:27496; code: [sj584/EpiGraph](https://github.com/sj584/EpiGraph)).
All runs were done on Kaggle (GPU) with the authors' model and training code plus three small compatibility patches for a newer `torch_geometric` (see [Section 2](#2-setup)).

## 1. Summary

- **The paper's headline numbers are reproduced.** The authors' 10 released checkpoints reproduce the published test metrics to the printed two-decimal precision. Retraining the full 10-fold ensemble from scratch (8 independent runs) gives AUC-ROC 0.726 ± 0.005 (5 initialisation seeds) and 0.727 ± 0.004 (3 fold-split seeds), versus 0.73 in the paper.
- **One small, consistent shortfall.** Mean AUC-PR of the retrained ensembles is 0.215 (A) and 0.219 (B), and every one of the 8 runs (0.207–0.230) is below the 0.232 of the released checkpoints. The ≈ 0.015 gap is small relative to the test-set uncertainty (95% CI of AUC-PR: 0.180–0.293) and its cause cannot be determined from these data (Section 6).
- **The test set is small, so the numbers are uncertain.** The epitope3D test set has only 45 antigens (7,747 surface residues, 10.0% epitopes). A bootstrap over antigens gives a 95% CI of **0.655–0.780** for the ensemble AUC-ROC, so the 0.02 gap between the published AUC-ROC values (0.73 vs 0.71 for BepiPred-3.0 / DiscoTope-3.0) is not resolvable on this benchmark.
- **The ensemble matters.** Single models reach AUC-ROC 0.709 ± 0.014; the 10-model average reaches 0.725.
- **Validation loss rises early.** The best checkpoint is usually found at epoch 8–17 of 50 (first retraining run; the 8 later runs give a median of 14–15).
- **Retraining noise is small compared with test-set noise.** The standard deviation of the ensemble AUC-ROC across initialisation seeds is 0.005, about an order of magnitude below the ±0.06 half-width of the bootstrap interval.

## 2. Setup

| Item | Value |
|---|---|
| Model | 8-layer GAT (hidden 128, 8 heads) with residual connections; input = ESM-IF1 (512) + ESM-2 (1280) = 1792 features per residue |
| Data | epitope3D: 199 antigens in the provided training pickle (paper: 200; see Section 5, item 5), 45 in the test set |
| Training | 10-fold cross-validation (consecutive folds after a seeded shuffle), 50 epochs, batch 8, Adam, lr 1e-5; one model saved per fold |
| Checkpoint rule | the saved checkpoint is replaced only when validation loss **and** validation AUC-ROC are both better than the best values so far |
| Prediction | mean of the 10 models' probabilities on surface residues (RSA ≥ 0.15); decision threshold 0.1481 for F1 / BACC / MCC |
| Patches | `data.node_attrs` → `data['node_attrs']`; checkpoint key `lin_src` → `lin`; `set()` of `Data` objects replaced by an index-based split (same samples, original order) |

Two runs were made: **(i)** inference with the authors' checkpoints, **(ii)** full retraining (10 folds × 50 epochs, ≈ 33 min on a Kaggle GPU) followed by the same evaluation.

## 3. Results on the epitope3D test set

| Metric | Paper | Authors' checkpoints | Retrained, first run (notebook 01) | Retrained, 5 init seeds (mean ± std) |
|---|---|---|---|---|
| AUC-ROC | 0.73 | 0.73 (0.725) | 0.72 | 0.726 ± 0.005 |
| AUC-PR | 0.23 | 0.23 (0.232) | 0.22 | 0.215 ± 0.003 |
| F1 | 0.29 | 0.29 (0.287) | 0.29 | 0.281 ± 0.010 |
| BACC | 0.62 | 0.62 (0.624) | 0.63 | 0.621 ± 0.006 |
| MCC | 0.19 | 0.19 (0.193) | 0.19 | 0.184 ± 0.012 |

Values in parentheses are the unrounded results from `src/eval_ensemble.py`; the upstream `evaluate.py` only prints two decimals. The last column is Experiment A of Section 6 (fold split fixed at the paper's seed 30).

Paper benchmark for context (Table 1): GraphBepi AUC-ROC 0.64, F1 0.28, MCC 0.16, BACC 0.62; BepiPred-3.0 and DiscoTope-3.0 AUC-ROC 0.71, F1 0.19–0.20, MCC 0.08–0.09, BACC 0.57. The retrained ensembles (AUC-ROC 0.726 on average over five seeds) are therefore above GraphBepi in AUC-ROC and level with BepiPred-3.0 / DiscoTope-3.0.

**Reading the numbers**

- *Authors' checkpoints* match the paper to the printed precision: data, features and inference pipeline are correct.
- *Retraining* reproduces AUC-ROC (0.726 vs 0.725 for the checkpoints) and, within run-to-run noise, BACC, F1 and MCC. AUC-PR is the one metric that is lower in every retrained run (Section 6). The fold split is fixed by `random.seed(30)`, but upstream never seeds PyTorch, so each training run starts from different random weights; the sample order inside a fold also differs slightly because of the `set()` patch.
- *AUC-PR is low in absolute terms but not poor.* With 10.0% positives, a random predictor has AUC-PR ≈ 0.10, so 0.23 is about 2.3× the chance level.

## 4. How certain are these numbers? (bootstrap over test antigens)

Residues from one antigen are correlated, so antigens, not residues, were resampled (1,000 draws, pooled metrics recomputed on each draw). Authors' checkpoints, 10-model ensemble:

| Metric | Point estimate | 95% CI | Single model, mean ± std (n = 10) |
|---|---|---|---|
| AUC-ROC | 0.725 | 0.655 – 0.780 | 0.709 ± 0.014 |
| AUC-PR | 0.232 | 0.180 – 0.293 | 0.214 ± 0.013 |
| F1 | 0.287 | 0.219 – 0.338 | 0.272 ± 0.010 |
| BACC | 0.624 | 0.574 – 0.672 | 0.614 ± 0.012 |
| MCC | 0.193 | 0.123 – 0.252 | 0.175 ± 0.010 |

1. **The test-set uncertainty (±0.06 AUC-ROC) is about ten times larger than the retraining noise (std 0.005, Section 6).** Agreement between retrained and published numbers is therefore a statement about run-to-run stability, not evidence that the method beats others by a given margin.
2. **Competitor comparison.** BepiPred-3.0 / DiscoTope-3.0 (0.71) lie well inside the EpiGraph CI; GraphBepi (0.64) lies just below its lower bound. This is an unpaired comparison: only the aggregate metrics of Table 1 of the paper are available for the competing methods, not their per-residue scores, so a significance test is not possible; the CI only shows that the benchmark cannot separate EpiGraph from the two 0.71 methods.
3. **Ensembling helps consistently:** +0.016 AUC-ROC, +0.018 AUC-PR, +0.015 F1, +0.010 BACC, +0.018 MCC over the average single model. The validation-AUC / test-AUC gap noted in Section 5 is partly this ensemble effect.
4. **Per-antigen AUC is lower than the pooled AUC.** The mean per-antigen AUC-ROC of the ensemble is 0.624 (median 0.643), clearly lower than the pooled 0.725. The pooled value also depends on how comparable the scores are across antigens, whereas the per-antigen value measures only the ranking of residues within one antigen, which is the practical use case. Per-antigen performance varies widely; the paper's Fig. 3 shows four example antigens (selected as good, random and poor cases) with AUC from 0.17 to 0.92.

## 5. Training dynamics (first retraining run, notebook 01)

Validation = the antigens held out in each fold: 20 in folds 1–9 and 19 in fold 10 (the 199 antigens are split into consecutive blocks). The AUC-ROC reported by `train.py` is computed per batch (8 + 8 + 4 antigens; 8 + 8 + 3 in fold 10) and averaged over batches, so it is a noisy selection signal, not an unbiased performance estimate; the main text of the paper does not report it.

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

1. **Validation AUC (0.68) is lower than test AUC (0.72) and varies from 0.63 to 0.76 across folds.** Each fold validates on only 19–20 antigens, so the estimate is noisy; the test ensemble additionally benefits from averaging 10 models.
2. **Early overfitting in the loss.** Training loss reaches 0.22–0.26 at epoch 50, whereas validation loss is lowest at epoch 8–21 in nine folds (fold 6: 32) and ends above its minimum in all ten folds. The selected epoch is 8–17 in nine of ten folds (fold 6: 32), so most of the 50 epochs do not change the saved model. Validation AUC-ROC behaves differently: its maximum lies anywhere between epoch 8 and 46 (e.g. fold 3: epoch 46, fold 9: epoch 40). The overfitting is therefore visible mainly in the loss (calibration of the probabilities), less clearly in the ranking quality.
3. **The saved checkpoint is not the best-AUC epoch.** The two-metric rule only accepts an epoch when loss and AUC are both better than their best values so far (e.g. fold 1 reaches ≈ 0.635 AUC at epoch ≈ 22 but the model saved is epoch 14 with 0.6275).
4. **Isolated validation-loss spikes late in training.** Single-epoch jumps of more than 0.03 occur in four folds: fold 4 (epoch 45: 0.333 → 0.387), fold 9 (epoch 40: 0.385 → 0.442), fold 2 (epoch 48: 0.319 → 0.375) and fold 8 (epoch 48: 0.444 → 0.879, the largest). They do not affect results because the selected checkpoints are much earlier.
5. **Antigen count (199 vs 200).** The folds contain 179 training + 20 validation antigens (fold 10: 180 + 19), i.e. 199 in total, whereas the paper trains on 200. The authors' `csv files/` list 200 antigens for the epitope3D training data (180 in `epitope3d_dataset_180_Train.csv`, 20 in `epitope3d_dataset_20_Test.csv`, no duplicates). Comparing PDB identifiers shows that exactly one of them, **3FCS**, is absent from the released `epitope3d_train_data.pkl`; every other entry matches, and the 45 test antigens match the blind-test list exactly. The released pickle therefore omits one antigen of the described training set. The reason is not documented in the repository (a failure during structure or embedding preprocessing is a possible but unverified explanation). One missing antigen out of 200 is expected to have only a small effect; this was not tested.
6. **Timing.** The "training time" printed in the log is cumulative (≈ 190 s per fold; 1,920 s at fold 10).

## 6. Robustness study: what is being tested and why

The first retraining run (Section 3, notebook 01) is a single sample, so it cannot separate "the method reproduces" from "this run was lucky". Upstream `train.py` has two independent sources of randomness:

| Source | Controlled by | Effect |
|---|---|---|
| Fold assignment | `--seed` (default 30; `random.seed` + `random.shuffle` before the consecutive `KFold`) | which 19–20 antigens validate each model, and which 179–180 train it |
| Weight initialisation (and any torch RNG) | not seeded upstream | starting point of each model |

`src/train_seeded.py` seeds PyTorch/NumPy and then runs the unmodified `train.py`, so both sources can be varied separately:

- **Experiment A – initialisation only.** Fold split fixed at seed 30; torch seeds 1–5 → five independent 10-model ensembles. Isolates training noise.
- **Experiment B – fold split + initialisation.** Split seeds 31–33 (torch seed equal to split seed) → three ensembles. Captures how sensitive the result is to the cross-validation partition.

Each run is evaluated with `src/eval_ensemble.py`, and `src/aggregate.py` reports mean ± std, min–max, a 95% t-interval of the mean, and whether the paper's value falls inside that interval (allowing ±0.005 for the paper's rounding), plus single-model vs ensemble gains. The study is executed by `notebooks/02_multiseed_robustness_kaggle.ipynb` on Kaggle (≈ 33 min per ensemble; 8 ensembles ≈ 4.5 h).

Caveats that remain by design: (i) GPU runs are not bit-for-bit reproducible even with fixed seeds, because GPU scatter operations used by graph convolutions can be non-deterministic (not measured here), so seeds make runs comparable rather than identical; (ii) the decision threshold 0.1481 is the default of the authors' `evaluate.py`; the paper states that a threshold maximising MCC on the held-out fold was chosen in each fold and averaged, which presumably corresponds to this value. It is reused unchanged for the retrained models, so F1/BACC/MCC of retrained models also depend on how well that threshold transfers, whereas AUC-ROC and AUC-PR do not.

### 6.1 Results (8 retrained ensembles = 80 models, 32 min per ensemble on Kaggle GPU)

Ensemble metrics on the 45-antigen test set (raw output: `results/02_multiseed/summary.md`, per-run files in `results/02_multiseed/runs/`):

| Run | Seeds (split / init) | AUC-ROC | AUC-PR | F1 | BACC | MCC |
|---|---|---|---|---|---|---|
| init_s1 | 30 / 1 | 0.729 | 0.215 | 0.271 | 0.615 | 0.171 |
| init_s2 | 30 / 2 | 0.721 | 0.211 | 0.270 | 0.614 | 0.170 |
| init_s3 | 30 / 3 | 0.722 | 0.214 | 0.288 | 0.626 | 0.193 |
| init_s4 | 30 / 4 | 0.732 | 0.216 | 0.291 | 0.627 | 0.197 |
| init_s5 | 30 / 5 | 0.729 | 0.220 | 0.284 | 0.624 | 0.188 |
| **Experiment A, mean ± std** | | **0.726 ± 0.005** | **0.215 ± 0.003** | **0.281 ± 0.010** | **0.621 ± 0.006** | **0.184 ± 0.012** |
| split_s31 | 31 / 31 | 0.731 | 0.230 | 0.280 | 0.614 | 0.184 |
| split_s32 | 32 / 32 | 0.723 | 0.207 | 0.272 | 0.612 | 0.174 |
| split_s33 | 33 / 33 | 0.728 | 0.220 | 0.283 | 0.624 | 0.187 |
| **Experiment B, mean ± std** | | **0.727 ± 0.004** | **0.219 ± 0.012** | **0.278 ± 0.005** | **0.617 ± 0.006** | **0.182 ± 0.007** |
| Authors' checkpoints | – | 0.725 | 0.232 | 0.287 | 0.624 | 0.193 |
| Paper | – | 0.73 | 0.23 | 0.29 | 0.62 | 0.19 |

Standard deviations are sample standard deviations over runs (n = 5 and n = 3).

**Findings**

1. **AUC-ROC is reproduced and stable.** Mean 0.726 (A) and 0.727 (B) against 0.725 for the released checkpoints (difference +0.001 and +0.002). The spread between runs is 0.721–0.732, far below the test-set bootstrap interval. Paired against the released checkpoints, the AUC-ROC difference is between −0.007 and +0.004 per run, and every paired 95% CI contains zero (`results/02_multiseed/paired_comparison.md`).
2. **Experiment A and B give the same picture.** Changing the fold split in addition to the initialisation does not shift the mean (AUC-ROC 0.727 vs 0.726) but widens the spread of AUC-PR (std 0.012 vs 0.003); with only three runs in B this comparison is indicative only.
3. **F1, BACC and MCC.** BACC matches the paper (0.621 / 0.617 vs 0.62). F1 and MCC are slightly lower on average (F1 0.281 / 0.278 vs 0.29; MCC 0.184 / 0.182 vs 0.19), by 0.006–0.012. For all three the paper value lies inside the 95% t-interval of the retrained mean (with ±0.005 allowed for rounding of the paper's values).
4. **AUC-PR is systematically lower in the retrained runs.** All 8 ensembles score 0.207–0.230, below the checkpoints' 0.232. In Experiment A the paper value (0.23) lies outside the 95% t-interval of the mean ([0.211, 0.219]); in Experiment B, whose spread is larger, it lies inside. Single models show the same direction (0.204 ± 0.012 over 50 retrained models vs 0.214 ± 0.013 over the 10 released ones). A paired bootstrap over test antigens puts the difference at +0.015 AUC-PR (95% CI +0.001 to +0.028) for the average of all retrained ensembles, which is borderline against test-set sampling alone and small in absolute terms (about one eighth of the width of the AUC-PR interval in Section 4).
5. **Ensembling gain is stable.** Single models average AUC-ROC 0.709 ± 0.016 (n = 50, Experiment A) and 0.708 ± 0.023 (n = 30, Experiment B); the ensemble adds +0.017 and +0.019.
6. **Validation behaviour matches Section 5.** Mean best validation AUC per run is 0.675 ± 0.004 (A) and 0.672 ± 0.009 (B); the selected epoch has a median of 14 (A) and 15 (B), with a range of 2–33 over 80 folds.

**Possible reasons for the AUC-PR shortfall (not tested)**

- The released checkpoints are one draw of 10 models. Their own training variance is unknown; a draw at the favourable end, or checkpoints picked from several training attempts, would produce exactly this pattern. The data cannot distinguish this from the other options.
- The software environment differs from the one in the authors' `conda_env.yaml` (Python 3.8, PyTorch 1.10.1 with CUDA 11.3, PyG 2.0.4); the Kaggle runs used Python 3.13 and a current `torch_geometric`. In addition, the `set()` patch changes the order in which training antigens are batched.
- The released training pickle lacks one of the 200 antigens (Section 5, item 5).
- AUC-PR is the metric most sensitive to a handful of highly ranked residues in a test set with 10% positives, so small ranking differences move it more than AUC-ROC.

**Limits of this study.** Eight ensembles bound the retraining variance only roughly (a t-interval with 4 or 2 degrees of freedom is wide). GPU non-determinism means the seeds make runs comparable rather than identical. The decision threshold 0.1481 (the authors' default) is reused for retrained models.

## 7. Limitations

- Only one test set of 45 antigens; CIs are wide (Section 4). The DiscoTope-3 benchmark in the paper (Table 2) was not reproduced here.
- Competitor numbers are taken from Table 1 of the paper and not re-run; only aggregate metrics are available, so no paired statistics are possible.
- The pooled-residue metrics depend on how comparable scores are across antigens; the per-antigen AUC is substantially lower (Section 4).
- Training-set labels come from antibody–antigen complexes and are incomplete (unlabelled residues may be true epitopes), a limitation the paper acknowledges.

## 8. Conclusion

The published epitope3D results of EpiGraph are reproduced: exactly with the released checkpoints, and by retraining for AUC-ROC (0.726 ± 0.005 over five initialisation seeds, 0.727 ± 0.004 over three fold-split seeds, against 0.73 published), BACC, F1 and MCC. AUC-PR is the exception: retrained ensembles reach 0.215–0.219 instead of 0.23, a small but consistent shortfall whose cause (selection of the released checkpoints, software differences or chance) cannot be determined from these data. The model learns a useful signal (AUC-ROC ≈ 0.73, AUC-PR ≈ 2.2× chance), benefits clearly from ensembling (+0.017 AUC-ROC over single models), and its validation loss starts to rise early (epoch 8–21 in nine of ten folds). Retraining noise (std 0.005 AUC-ROC) is about ten times smaller than the test-set uncertainty: with 45 test antigens the 95% interval for AUC-ROC is about ±0.06, too wide to rank EpiGraph against BepiPred-3.0 or DiscoTope-3.0. Stronger claims need larger test sets or paired comparisons.

## Glossary

- **Seed**: the starting value of a pseudo-random number generator. The same seed gives the same sequence of "random" numbers, so a seed is what makes a random choice (shuffling the data, initialising weights) repeatable.
- **Ensemble**: here, the average of the probabilities of the 10 fold models.
- **AUC-ROC / AUC-PR**: threshold-free ranking metrics; AUC-PR is sensitive to the 10% positive rate.
- **BACC / MCC / F1**: threshold-dependent metrics, computed at 0.1481.
- **RSA**: relative solvent accessibility; residues with RSA ≥ 0.15 are treated as surface residues.
