# Anomaly Baseline Rerun at min_rows=89

Date: 2026-05-13

Paper claim: EXPoSE is compared with Random, Bayesian changepoint, Etsy Skyline, and Steuber Z-score using NAB-style score,
detection rate, and median lead time.

## Method

This rerun uses the same canonical local time-series artifact as the EXPoSE rerun:

- Input: `/Users/sergiopinto/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/cluster_timeseries.parquet`
- Filter: clusters with at least `89` time-series rows
- Evaluable clusters: `820`
- Signal: arrival rate with engagement boost, matching `evaluate_nab_style`
- Window: adaptive pre-peak window, matching `evaluate_nab_style`
- False-positive penalty: `0.1`

## Best Threshold Results

| Detector | Best threshold | NAB | Detection | Median lead | Paper value |
|---|---:|---:|---:|---:|---|
| Random | `0.3` | `72.2276` | `99.0244%` | `+22h` | `38.7 / 72.8% / +18h` |
| Bayesian Changept | `0.1` | `7.4289` | `40.9756%` | `+7.5h` | `14.1 / 39.3% / +13h` |
| Etsy Skyline | `0.3` | `8.2378` | `47.4390%` | `+6h` | `9.9 / 33.2% / +13h` |
| Steuber Z-score | `1.0` | `6.8232` | `41.8293%` | `+7h` | `5.8 / 21.3% / +11h` |

## Status

This does not reproduce the PDF baseline table as written.

It does produce a coherent replacement candidate under the same `min_rows=89` filter selected for the EXPoSE rerun:

- Random is much stronger than the PDF row.
- Steuber exceeds the PDF NAB and detection values, but not the median lead.
- Bayesian and Etsy do not match or surpass the PDF NAB/lead claims.

The paper should not keep the PDF baseline row unchanged unless the exact original configuration is recovered. If the anomaly
section is rewritten around the `min_rows=89` EXPoSE result, the baseline comparison should be rewritten around this rerun or a
new declared sensitivity rerun.

Machine-readable output: `results/anomaly_baseline_rerun_min_rows89_2026-05-13.json`.
