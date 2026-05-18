# PREFLIGHT FAILED — PSR Baselines Statistical Reporting

**Run timestamp:** 2026-05-18 13:33:51
**Output directory:** `C:\StreamingACD\results\psr_statistics_20260518_133351\`
**Status:** No computation performed. Required input artifacts could not be located.

---

## What the spec needed

The spec required, at minimum, one of two paths:

- **Path A** (preferred): saved test-set predictions for each of the 11 PSR baselines, plus the test-set features and targets, so that statistics could be computed directly.
- **Path B** (fallback): the train- and test-set features and targets, so that the 11 baselines could be re-trained and predictions re-generated.

Either path requires the engineered feature matrices and PSR targets at `C:\Explainable_ACD\data\psr\` (or equivalent). Path A additionally requires the per-baseline saved predictions at `C:\Explainable_ACD\results\psr_baselines\<baseline>\test_predictions.npy`.

## What was found

The path layout in the source repository differs from what the spec assumed. The actual PSR-relevant locations are:

| Spec expected | Actually present |
|---|---|
| `C:\Explainable_ACD\data\psr\` | does not exist |
| `C:\Explainable_ACD\results\psr_baselines\` | does not exist |
| `C:\Explainable_ACD\scripts\psr_baselines\` | does not exist |
| `C:\Explainable_ACD\virality_prediction\data\` | **empty** (only `.gitkeep`) |
| `C:\Explainable_ACD\virality_prediction\results\` | **empty** (only `.gitkeep`) |
| `C:\Explainable_ACD\virality_prediction\scripts\` | present (training/runner scripts) |
| `C:\Explainable_ACD\virality_prediction\src\virality\` | present (`baselines.py`, `psr_dataset.py`, `hawkes.py`, `predictor.py`, `transformer_hawkes.py`) |

A full repository sweep for `*.npy`, `*.parquet`, and `*.pkl` returned **no** PSR-related artifacts. The only `.npy` files in either repo belong to the unrelated DeBERTa MTL check-worthiness work.

## Why Path B is also blocked

`virality_prediction/src/virality/psr_dataset.py` constructs the feature matrices from raw input parquets via `PSRDataset(timeseries_path=..., clusters_path=..., tweets_path=...)`. Those upstream parquet files are also absent from the repository (the `virality_prediction/data/` directory contains no data). Re-training would therefore require first regenerating the raw timeseries/cluster/tweet inputs, which is a much larger pipeline step well outside the scope of this spec.

## Required artifacts (missing)

To proceed with **Path A**, the following must be supplied:

1. `test_features.npy` — shape (106, 42), engineered features at detection time for the 106 held-out anomalous clusters.
2. `test_targets.npy` — shape (106,), PSR labels in [0, 1].
3. For each baseline in {Random, Ridge, BayesianRidge, RandomForest, LightGBM, SVR, LSTM, GRU, HIP_Hawkes, BERTweet, BERTweet_plus_features}: a `test_predictions.npy` of shape (106,).

To proceed with **Path B**, additionally:

4. `train_features.npy` — shape (423, 42).
5. `train_targets.npy` — shape (423,).

Or, in the absence of (1)–(5), the upstream raw inputs that `PSRDataset` ingests:

6. A timeseries parquet (cluster-level windowed engagement).
7. A clusters parquet.
8. A tweets parquet (optional, used for sentiment/geographic features).

## No alternative scope was attempted

The spec's hard constraints (no modifications to source artifacts, graceful degradation, no silent overwriting of paper numbers) do not authorize regenerating the upstream PSR dataset from scratch. This run therefore stops at preflight and writes no statistics, no predictions, no LaTeX, and no handoff document beyond this report.

## Suggested next steps

Decide which of the following matches the actual state of the original work, and supply the corresponding inputs:

1. **Predictions exist on another machine or backup.** Copy the per-baseline `test_predictions.npy` files (and the test feature/target arrays) into `C:\Explainable_ACD\virality_prediction\data\psr\` and `C:\Explainable_ACD\virality_prediction\results\psr_baselines\<baseline>\`, then re-run.
2. **Raw PSR parquets exist on another machine.** Supply the timeseries/clusters parquets, run `psr_dataset.py` to materialize `train_features.npy` / `test_features.npy` / `train_targets.npy` / `test_targets.npy`, then re-run under Path B.
3. **Nothing has been persisted.** The full virality pipeline would need to be re-executed from raw Twitter data before any of the statistical reporting in this spec can be computed. That is a separate, larger task and should not be folded into this run.
