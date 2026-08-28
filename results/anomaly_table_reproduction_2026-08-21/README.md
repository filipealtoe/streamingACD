# CIKM anomaly-table reproduction

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — documented the exact four-row table reproduction. -->

This fresh run reproduces the accepted paper's `min_rows=89` table from the released aggregate time series.

| Detector | Threshold | NAB-style | Detection | Median lead | Flagged points |
|---|---:|---:|---:|---:|---:|
| EXPoSE | 0.05 | 79.2236 | 97.9268% | +23.0h | 99.43% |
| Bayesian Changept | 0.10 | 7.4289 | 40.9756% | +7.5h | 2.82% |
| Etsy Skyline | 0.30 | 8.2378 | 47.4390% | +6.0h | 2.93% |
| Steuber Z-score | 1.00 | 6.8232 | 41.8293% | +7.0h | 4.04% |

Reproduction parameters:

- the retained cohort contains clusters with at least 89 hourly observations;
- detector thresholds are selected by the retained table sweep on these 820 clusters;
- the reported score is the paper's project-specific NAB-style early-detection score;
- EXPoSE uses decay `0.005`, RBF gamma `1.0`, score threshold `0.05`, and no warm-up.
