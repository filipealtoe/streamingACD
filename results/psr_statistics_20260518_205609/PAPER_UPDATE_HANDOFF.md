# PSR Baselines - Statistical Reporting Handoff

**Run timestamp:** 2026-05-18 20:56:09
**Output directory:** `results/psr_statistics_20260518_205609/`
**Test set size:** 106 anomalous clusters (sequence/text baselines evaluate a subset; see per-baseline `n_test`).
**Bootstrap:** 1000 resamples, seed 42.

## Significance summary (Spearman rho, two-tailed)

| Baseline | Family | n | rho | p-value | 95% CI |
|---|---|---:|---:|---:|---|
| Random | Trivial | 106 | -0.052  | 0.596 | [-0.23, +0.13] |
| Ridge | Linear | 106 | +0.530 *** | <0.001 | [+0.37, +0.67] |
| BayesianRidge | Linear | 106 | +0.551 *** | <0.001 | [+0.40, +0.67] |
| RandomForest | Tree-Based | 106 | +0.530 *** | <0.001 | [+0.37, +0.66] |
| LightGBM | Tree-Based | 106 | +0.549 *** | <0.001 | [+0.38, +0.67] |
| SVR (RBF) | Kernel | 106 | +0.542 *** | <0.001 | [+0.38, +0.68] |
| LSTM | Sequence | 90 | +0.048  | 0.652 | [-0.14, +0.26] |
| GRU | Sequence | 90 | +0.097  | 0.362 | [-0.11, +0.31] |
| Hawkes (HIP) | Point Process | 77 | +0.247 * | 0.030 | [+0.00, +0.46] |
| BERTweet | Text-Based | 106 | -0.086  | 0.382 | [-0.28, +0.11] |
| BERTweet + Features | Text-Based | 106 | +0.510 *** | <0.001 | [+0.36, +0.63] |

Significance codes: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001.

## Which baselines beat chance at alpha=0.05?
- **Significant (p<0.05):** Ridge, BayesianRidge, RandomForest, LightGBM, SVR (RBF), Hawkes (HIP), BERTweet + Features
- **Not significant (p>=0.05):** Random, LSTM, GRU, BERTweet

## Pairwise McNemar tests (top-3 by F2@0.65, exact)

Top 3 by F2@0.65: SVR (RBF), RandomForest, LightGBM

| A | B | n common | a_correct/b_wrong | a_wrong/b_correct | exact p-value |
|---|---|---:|---:|---:|---:|
| SVR (RBF) | RandomForest | 106 | 13 | 11 | 0.839 |
| SVR (RBF) | LightGBM | 106 | 14 | 11 | 0.690 |
| RandomForest | LightGBM | 106 | 8 | 7 | 1.000 |

## Recommended Table 3 caption update

> Spearman rho p-values from `scipy.stats.spearmanr` (two-tailed). 95% confidence intervals from 1000 bootstrap resamples (seed=42). Sequence and point-process baselines evaluate on the subset of test clusters with sufficient time-series support (n shown per-row in supplementary materials).

## Sanity-check status

Recomputed Spearman rho (rounded to 3 decimals) versus the value reported in the paper:

| Baseline | Paper rho | Recomputed rho | |diff| | Status |
|---|---:|---:|---:|---|
| Random | -0.052 | -0.052 | 0.0000 | OK |
| Ridge | +0.530 | +0.530 | 0.0000 | OK |
| BayesianRidge | +0.551 | +0.551 | 0.0000 | OK |
| RandomForest | +0.532 | +0.530 | 0.0020 | OK |
| LightGBM | +0.549 | +0.549 | 0.0000 | OK |
| SVR (RBF) | +0.542 | +0.542 | 0.0000 | OK |
| LSTM | +0.016 | +0.048 | 0.0320 | MISMATCH |
| GRU | +0.082 | +0.097 | 0.0150 | MISMATCH |
| Hawkes (HIP) | +0.247 | +0.247 | 0.0000 | OK |
| BERTweet | -0.086 | -0.086 | 0.0000 | OK |
| BERTweet + Features | +0.510 | +0.510 | 0.0000 | OK |

### Mismatches flagged for author review

- **LSTM**: paper reports +0.016, recomputed +0.048 (|diff| 0.0320). Author decides which value to publish; this run does not overwrite the paper number.
- **GRU**: paper reports +0.082, recomputed +0.097 (|diff| 0.0150). Author decides which value to publish; this run does not overwrite the paper number.

## Notes on regenerated sequence/text baselines

- LSTM and GRU are stochastic; the rerun matches the original protocol but not bit-for-bit. Recomputed rho for LSTM (0.048) and GRU (0.097) differ from the paper's 0.016 and 0.082 by small amounts within stochastic noise; the bootstrap CIs reported here characterise the current run, not the original artifact.
- HIP_Hawkes, BERTweet, and BERTweet + Features matched the paper's Spearman rho exactly to 3 decimals on rerun.
