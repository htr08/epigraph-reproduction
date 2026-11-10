# Authors' checkpoints minus retrained ensembles (paired bootstrap over test antigens)

Difference = metric(authors' 10 checkpoints) - metric(retrained ensemble); point estimate and 95% CI (2000 antigen-level bootstrap draws, same antigens for both models).

| Retrained ensemble | Diff. AUC-PR | Diff. AUC-ROC |
|---|---|---|
| init_s1 | +0.0178 [+0.0034, +0.0335] | -0.0036 [-0.0142, +0.0070] |
| init_s2 | +0.0211 [+0.0054, +0.0353] | +0.0039 [-0.0074, +0.0156] |
| init_s3 | +0.0182 [+0.0034, +0.0330] | +0.0032 [-0.0062, +0.0116] |
| init_s4 | +0.0166 [-0.0002, +0.0370] | -0.0068 [-0.0191, +0.0043] |
| init_s5 | +0.0123 [-0.0044, +0.0289] | -0.0037 [-0.0152, +0.0064] |
| split_s31 | +0.0020 [-0.0100, +0.0188] | -0.0062 [-0.0181, +0.0056] |
| split_s32 | +0.0254 [+0.0062, +0.0479] | +0.0025 [-0.0115, +0.0166] |
| split_s33 | +0.0120 [-0.0014, +0.0250] | -0.0026 [-0.0145, +0.0076] |
| mean of all 8 ensembles | +0.0146 [+0.0010, +0.0281] | -0.0033 [-0.0123, +0.0057] |
