# CIKM anomaly-table reproduction

<!-- Sérgio Pinto, 2026-08-21 20:44 PDT — documented the retained table protocol and both detection-rate boundaries. -->

This fresh run reproduces the accepted paper's `min_rows=89` table from the released aggregate time series.

| Detector | Threshold | NAB-style | Historical detection | Strictly pre-peak | Median lead | Flagged points |
|---|---:|---:|---:|---:|---:|---:|
| EXPoSE | 0.05 | 79.2236 | 97.9268% | 97.9268% | +23.0h | 99.43% |
| Bayesian Changept | 0.10 | 7.4289 | 40.9756% | 31.2195% | +7.5h | 2.82% |
| Etsy Skyline | 0.30 | 8.2378 | 47.4390% | 35.6098% | +6.0h | 2.93% |
| Steuber Z-score | 1.00 | 6.8232 | 41.8293% | 31.0976% | +7.0h | 4.04% |

Additional sanity baseline (not a paper-table row):

| Detector | Threshold | NAB-style | Historical detection | Strictly pre-peak | Median lead | Flagged points |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.30 | 72.2276 | 99.0244% | 95.7317% | +22.0h | 67.03% |

Protocol notes:

- the retained cohort contains clusters with at least 89 hourly observations;
- detector thresholds are selected by the retained table sweep on these 820 clusters;
- the reported score is the paper's project-specific NAB-style early-detection score;
- the output records both strictly pre-peak and inclusive-at-peak detection rates;
- `summary.json` records both the method-paragraph and retained table-run parameters.

For exact reuse of these table values, use the retained table-run parameters in
`summary.json`: decay `0.005`, RBF gamma `1.0`, score threshold `0.05`, and no
warm-up. The method paragraph's EXPoSE values (`0.01`, `0.5`, `0.1`, and 30
observations) describe a different configuration.

The historical table preserved the strict pre-peak rate for EXPoSE and the
inclusive-at-peak rate for the three classical baselines. Both versions are
included in `summary.json` and `cluster_outcomes.parquet`, so a downstream user
can apply one definition consistently.
