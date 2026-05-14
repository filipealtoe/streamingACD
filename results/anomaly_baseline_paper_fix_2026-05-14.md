# Anomaly Baseline Manuscript Fix

Date: 2026-05-14

## Scope

This note defines the paper-side correction for the anomaly-detection baseline comparison. It does not modify the manuscript
itself because no editable `.tex`, `.docx`, or source archive for the paper was found in the local repo search.

## Finding

The PDF baseline paragraph is not reproduced as written. A coherent replacement table exists under the same declared
`min_rows=89` filter used for the EXPoSE rerun.

Shared rerun setting:

- input: `data/pipeline_output/streaming_full/2026-01-17_03-56/cluster_timeseries.parquet`;
- filter: clusters with at least `89` time-series rows;
- evaluable clusters: `820`;
- signal: arrival rate with engagement boost;
- window: adaptive pre-peak window;
- false-positive penalty: `0.1`.

## Required Manuscript Edit

Replace the anomaly result paragraph/table with the following values:

| Detector | NAB | Detection | Median lead |
|---|---:|---:|---:|
| EXPoSE | `79.2` | `97.9%` | `+23h` |
| Random | `72.2` | `99.0%` | `+22h` |
| Bayesian Changept | `7.4` | `41.0%` | `+7.5h` |
| Etsy Skyline | `8.2` | `47.4%` | `+6h` |
| Steuber Z-score | `6.8` | `41.8%` | `+7h` |

Suggested nearby prose:

> Under the declared `min_rows=89` rerun, EXPoSE reaches NAB `79.2`, detection `97.9%`, and median lead time `+23h`. It obtains
> the highest NAB score among the evaluated detectors, while the random trigger baseline detects more clusters (`99.0%`) at lower
> NAB (`72.2`) and slightly lower median lead (`+22h`).

Do not claim that EXPoSE has the highest detection rate in this replacement table. The random trigger baseline has the highest
detection rate under this setting.

## Values To Remove

Remove or replace the previous baseline values:

- Random `38.7 / 72.8% / +18h`;
- Bayesian `14.1 / 39.3% / +13h`;
- Etsy `9.9 / 33.2% / +13h`;
- Steuber `5.8 / 21.3% / +11h`.

Those values are not reproduced by the packaged rerun.

## Evidence Used

- `/Users/sergiopinto/streamingACD/results/anomaly_expose_reproduction_attempt_2026-05-13.md`
- `/Users/sergiopinto/streamingACD/results/anomaly_baseline_rerun_min_rows89_2026-05-13.md`
- `/Users/sergiopinto/streamingACD/results/anomaly_baseline_rerun_min_rows89_2026-05-13.json`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/anomaly/expose_grid_search.csv`

## Remaining Blocker

No blocker remains for the declared `min_rows=89` replacement table. If the paper must keep the old baseline values, the original
configuration that produced them must be recovered and packaged.
