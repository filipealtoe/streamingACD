# PSR virality reproduction arrays

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — documented the complete retained prediction workspace used by the camera-ready reproduction. -->

This directory contains the deterministic 423/106 split, feature and target
arrays, and retained predictions used by the CIKM 2026 virality experiments.

- `derived_arrays/`: train/test features, targets, cluster IDs, indices, and feature names;
- `baseline_predictions/`: Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR predictions;
- `reproduced_missing_predictions/`: LSTM, GRU, HIP Hawkes, BERTweet, and BERTweet-with-features predictions;
- `explainableACD/`: source snapshots and aggregate pipeline inputs used by the retained runs.

The numerical preprocessing replaces non-finite feature values with zero. The
split uses `test_size=0.2`, `random_state=42`, shuffle enabled, and no
stratification.

Recompute the paper's Spearman tests, bootstrap intervals, and exact McNemar
tests with:

```bash
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

All numerical arrays are bound by
`reproducibility/cikm2026/prediction_checksums.sha256`.
