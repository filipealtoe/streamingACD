# Relay to Sergio — PSR baseline artifacts needed for CIKM 2026 stats reporting

**From:** Filipe
**Date:** 2026-05-18
**Re:** Virality prediction (PSR) baselines — Table 3 in the CIKM submission

Hi Sergio,

Reviewer 1 at IJCAI flagged that Table 3 reports point estimates only (Spearman $\rho$, $R^2$, MAE, $F_2$ at three thresholds) for the 11 PSR baselines, with no p-values and no confidence intervals. I'm trying to add:

- p-values for Spearman $\rho$ (and Pearson $r$ alongside) for each of the 11 baselines
- 95% bootstrap CIs (1000 resamples, seed 42) for $\rho$, $R^2$, MAE, and the three $F_2$ thresholds
- Optionally pairwise McNemar tests for the top-3 baselines at $\tau=0.65$

All of this is cheap to compute (a few minutes locally, no GPU, no API calls) **once we have the predictions on disk**. The blocker is that I can't find them. This note is so you can quickly check whether you still have the relevant files on your machine.

---

## What I searched

In `C:\Explainable_ACD\virality_prediction\`:

- `data/` — empty (only `.gitkeep`)
- `results/` — empty (only `.gitkeep`)
- `scripts/` and `src/virality/` — present, including `baselines.py`, `psr_dataset.py`, `hawkes.py`, `predictor.py`, `transformer_hawkes.py`, `train_virality_model.py`, `run_virality_baselines.py`

A repository-wide sweep for `*.npy`, `*.parquet`, and `*.pkl` returned **no** PSR-related artifacts. The only `.npy` files in the repo belong to the unrelated DeBERTa MTL check-worthiness work.

So the **code is present, the data and trained predictions are not**.

---

## What I'm hoping you still have

Anything in this list would unblock me. The order is best-case to worst-case:

### Best case — saved test-set predictions per baseline

A `test_predictions.npy` (shape `(106,)`, continuous in approximately [0, 1]) for each of:

`Random`, `Ridge`, `BayesianRidge`, `RandomForest`, `LightGBM`, `SVR`, `LSTM`, `GRU`, `HIP_Hawkes`, `BERTweet`, `BERTweet_plus_features`

Plus the test set itself:

- `test_features.npy` — shape `(106, 42)` — engineered features at detection time
- `test_targets.npy` — shape `(106,)` — PSR labels in [0, 1]

If you have these, I can compute every number Reviewer 1 asked for in about 5 minutes. No retraining needed.

### Next best — the train/test feature matrices, no predictions

- `train_features.npy` — shape `(423, 42)`
- `train_targets.npy` — shape `(423,)`
- `test_features.npy`, `test_targets.npy` — as above

With these I can re-train all 11 baselines (the scripts in `virality_prediction/scripts/` look intact) and produce the predictions, then run the statistics. Wall-clock budget is roughly 15–30 minutes, dominated by LSTM/GRU/BERTweet.

### Last resort — the upstream parquets

`psr_dataset.py` materializes the 42-feature matrix from three parquet inputs:

- a **timeseries** parquet (cluster-level windowed engagement, schema validated in `psr_dataset.py:_validate_schema`)
- a **clusters** parquet
- a **tweets** parquet (used for sentiment/geographic features)

If you have those parquets but no derived `.npy` files, I can run `PSRDataset(...)` to regenerate the features, then proceed as in "next best."

---

## Practical things to check on your end

1. Old project folders on your laptop or a backup drive — anywhere a `virality_prediction/data/` or `virality_prediction/results/` was populated, even briefly.
2. Any HPC scratch space or cluster directory from when the LSTM/GRU/BERTweet runs were executed (those would have been the slowest and most likely to have been done elsewhere than your laptop).
3. Saved model checkpoints — even if the predictions weren't dumped, a saved model plus the test features is enough; I can re-run inference.
4. Old Jupyter notebooks that may have cached the predictions in a pickle.

If none of the above turns up, that's still useful to know — it would tell me we need to budget time to re-run the upstream feature pipeline from raw Twitter data, which is a much bigger task than the statistics step itself.

---

## What you do NOT need to do

- Don't re-train anything yet. Even partial files (e.g., just `test_features.npy` and `test_targets.npy`) are valuable on their own.
- Don't worry about exact filenames — if you find a `.parquet` or `.pkl` that looks PSR-ish, just send it over and I'll figure out the schema.
- Don't worry about reproducibility seeds — I'll handle the bootstrap/test-time seeding in the new code; what I need from you is the raw arrays.

---

## Reference: current Table 3 Spearman values (so you know what we're matching)

| Baseline | $\rho$ |
|---|---|
| Random | -0.052 |
| Ridge | 0.530 |
| BayesianRidge | 0.551 |
| RandomForest | 0.532 |
| LightGBM | 0.549 |
| SVR (RBF) | 0.542 |
| LSTM | 0.016 |
| GRU | 0.082 |
| Hawkes (HIP) | 0.247 |
| BERTweet | -0.086 |
| BERTweet + Features | 0.510 |

If any predictions you find reproduce these to within 0.01, we're good. If they don't reproduce, that itself is interesting and we should talk before deciding what to put in the paper.

Thanks — even a "nope, I don't have them either" response is useful so I can plan the right next step.

— Filipe
