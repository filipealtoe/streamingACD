# Virality PSR Split Manifest

Paper claim: `PSR prediction uses 529 anomalous clusters, 42 features, and a 423/106 split.`

Status: `reproduced_from_packaged_artifacts`

## Inputs

- `reproducibility/source_artifacts/virality/features_enhanced.parquet`
- `reproducibility/source_artifacts/virality/psr_labels.parquet`
- `reproducibility/source_artifacts/virality/tuned_baselines.json`
- `reproducibility/source_artifacts/virality/complete_baselines.json`

## Split Rule

- Function: `sklearn.model_selection.train_test_split`
- Input: row indices of `features_enhanced.parquet` in stored order
- `test_size=0.2`
- `random_state=42`
- `shuffle=True`
- `stratify=None`

## Counts

| Metric | Value |
|---|---:|
| Feature rows | `529` |
| Label rows | `529` |
| Feature columns | `42` |
| Train rows | `423` |
| Test rows | `106` |

The full cluster-id lists are stored in the JSON manifest.
