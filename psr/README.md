# PSR Virality Handoff For Filipe

Date: 2026-05-18

This package addresses the preflight failure in `PREFLIGHT_FAILED.md`.

## What Is Included

### Direct Path B Inputs

`derived_arrays/` contains the train/test arrays Filipe asked for:

- `train_features.npy`: `(423, 42)`
- `train_targets.npy`: `(423,)`
- `test_features.npy`: `(106, 42)`
- `test_targets.npy`: `(106,)`
- `train_cluster_ids.npy`, `test_cluster_ids.npy`
- `train_indices.npy`, `test_indices.npy`
- `feature_names.json`
- `derived_arrays_manifest.json`

These arrays were derived from:

`reproducibility/source_artifacts/virality/features_enhanced.parquet`

The split matches the existing manifest:

`train_test_split(indices, test_size=0.2, random_state=42, shuffle=True, stratify=None)`

The features were cleaned with:

`np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)`

This matches the behavior in the original virality baseline scripts.

### Partial Path A Predictions

`baseline_predictions/` contains regenerated test-set predictions for the feature-based baselines:

- Random
- Ridge
- BayesianRidge
- RandomForest
- LightGBM
- SVR_RBF

Each model folder includes:

- `test_predictions.npy`: clipped to `[0, 1]`, matching the original `evaluate()` function
- `test_predictions_raw.npy`: raw model output before clipping

`baseline_predictions/predictions_manifest.json` includes Spearman p-values, Pearson r/p-values, R2, MAE, and F2 values.

The originally missing baselines were regenerated under `reproduced_missing_predictions/`:

- `LSTM`
- `GRU`
- `HIP_Hawkes`
- `BERTweet`
- `BERTweet_plus_features`

Each folder includes:

- `test_predictions.npy`: length `106`; contains `NaN` where the original script did not evaluate a cluster
- `test_predictions_eval_subset.npy`: predictions used for the reported metric
- `valid_mask.npy`
- `valid_indices.npy`
- `test_cluster_ids.npy`
- `valid_cluster_ids.npy`
- `test_targets_eval_subset.npy`
- `metrics.json`

`reproduced_missing_predictions/manifest.json` summarizes the rerun.

Truth status for the five regenerated baselines:

| Baseline | Paper Spearman | Rerun Spearman | Status |
|---|---:|---:|---|
| `LSTM` | `0.016` | `0.048` | Same protocol rerun; stochastic model, not exact old run |
| `GRU` | `0.082` | `0.097` | Same protocol rerun; stochastic model, not exact old run |
| `HIP_Hawkes` | `0.247` | `0.247` | Matched |
| `BERTweet` | `-0.086` | `-0.086` | Matched |
| `BERTweet_plus_features` | `0.510` | `0.510` | Matched |

### Original Compact Artifacts

`reproducibility/source_artifacts/virality/` contains:

- `features_enhanced.parquet`
- `psr_labels.parquet`
- `tuned_baselines.json`
- `complete_baselines.json`
- baseline generation scripts
- LaTeX table

`explainableACD/experiments/results/` contains the local old-repo result folders:

- `virality/`
- `virality_complete/`
- `virality_tuned/`
- `virality_expanded/`
- `virality_feature_selection/`

`explainableACD/experiments/scripts/` contains the relevant old baseline scripts.

## Raw Upstream Folder

The raw upstream pipeline folder is included at:

`explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56`

Original local source path:

`/Users/sergiopinto/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56`

This folder contains:

- `cluster_timeseries.parquet`
- `tweets.parquet`
- `clusters.parquet`
- `claims.parquet`
- `users.parquet`
- registry files
- clustering state files

This is the folder used to regenerate `LSTM`, `GRU`, `HIP_Hawkes`, `BERTweet`, and `BERTweet_plus_features`.

## Truth Status

The direct train/test feature and target arrays are now available and match Filipe's requested shapes.

The full set of 11 original saved baseline prediction arrays was not found locally. I regenerated 6 feature-based prediction arrays from the packaged features and regenerated the 5 missing sequence/Hawkes/text baselines from the local raw virality folder.

The regenerated `test_predictions.npy` files are now present for all 11 requested baselines, but note that LSTM/GRU are stochastic reruns and the sequence/Hawkes scripts evaluate only the subset of test clusters with enough time-series/arrival data.
