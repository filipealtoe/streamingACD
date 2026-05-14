# Anomaly EXPoSE Reproduction Attempt

Date: 2026-05-13

## Target Claim

The IJCAI draft reports EXPoSE at NAB `79.2`, detection `97.6%`, and median lead `+23h`.

## Method

I reran the EXPoSE NAB-style scoring calculation against the local
`/Users/sergiopinto/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/cluster_timeseries.parquet`
time-series data.

The existing grid-search script imports undeclared dependencies and is too slow interactively because it filters the full
time-series once per cluster per parameter combination. I therefore used an equivalent vectorized scoring pass for the known
best saved EXPoSE parameters:

- `decay=0.005`
- `gamma=1.0`
- `threshold=0.05`
- `use_arrival_rate=True`
- `n_components=20000`
- `fp_weight=0.1`

## Results

The documented/saved filter `min_rows=50` reproduces the packaged artifact exactly:

| min_rows | clusters | NAB | detection | median lead |
|---:|---:|---:|---:|---:|
| 50 | 1,543 | 72.2116 | 95.0097% | +23h |

A sensitivity pass over `min_rows` found candidate provenance for the paper's `79.2` NAB value:

| min_rows | clusters | NAB | detection | median lead |
|---:|---:|---:|---:|---:|
| 84 | 887 | 78.6208 | 97.6325% | +23h |
| 85 | 872 | 78.7834 | 97.7064% | +23h |
| 87 | 855 | 79.0185 | 97.7778% | +23h |
| 88 | 834 | 79.0697 | 97.8417% | +23h |
| 89 | 820 | 79.2236 | 97.9268% | +23h |
| 90 | 809 | 79.3562 | 98.0222% | +23h |

## Status

This does not fully reproduce the paper claim as written. It suggests the `79.2` NAB value is reachable under an undocumented
minimum-row filter near `min_rows=89`, while the `97.6%` detection value is closer to `min_rows=84`.

The claim should remain unresolved until we either recover the exact original configuration or rerun and rewrite the paper around
one declared configuration.
