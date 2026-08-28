# Dataset card: streaming narrative virality

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — presented the released matrix, feature availability, provenance, and privacy boundary as factual dataset metadata. -->

## Summary

We release the first dataset for narrative-level streaming virality prediction
on Twitter/X, comprising 529 cluster instances with detection-time engineered
features and virality labels.

The public files are:

- `reproducibility/source_artifacts/virality/features_enhanced.parquet`;
- `reproducibility/source_artifacts/virality/psr_labels.parquet`.

Both files use `cluster_id` as their unique one-to-one key. The feature file has
529 rows and 44 columns: 42 engineered features, `cluster_id`, and `psr`.

## Provenance

The instances originate from the paper's Twitter/X US Election 2020 streaming
pipeline. The source corpus contains 1,522,909 posts across 600 hourly windows.
The canonical run produced 100,000 clusters and 535 normalized claims. The
released matrix contains 529 cluster instances linked to 447 canonical claims.

The dataset represents cluster state at the recorded detection boundary. The
features cover growth, engagement, temporal, user, geographic, propagation,
content, and network signals. Field-level definitions are in
[`SCHEMA.md`](SCHEMA.md).

## Target

The target is the Peak-to-Steady Ratio used by the released artifact:

```text
PSR = (final_engagement - engagement_at_detection) / final_engagement
```

The ratio is stored as `psr` in the feature matrix and as `psr_label` in the
label table. The verifier recomputes the ratio and confirms exact cluster-ID
alignment across both files.

## Feature availability

Eight columns are unobserved for all 529 released rows:

- `geographic_entropy_predetect`;
- `unique_countries_predetect`;
- `max_retweets_per_tweet`;
- `avg_likes_per_tweet`;
- `rt_like_ratio`;
- `engagement_skewness`;
- `engagement_gini`;
- `account_age_avg_days`.

`user_gini` and `top_user_ratio` each have 50 unobserved values. The numerical
baseline pipeline replaces non-finite inputs with zero, as recorded in
[`MODEL_PARAMETERS.json`](MODEL_PARAMETERS.json). `amplification_factor` is
present and equals zero in every released row.

## Split

The reproducibility split is deterministic:

- shuffle: true;
- random state: 42;
- test fraction: 0.2;
- stratification: none;
- training rows: 423;
- test rows: 106.

The exact row membership and hashes are stored in
[`results/virality_split_manifest_2026-05-12.json`](../../results/virality_split_manifest_2026-05-12.json).

## Privacy and access

The release contains no raw post text, usernames, user IDs, post IDs, profile
fields, or user table. `cluster_id` is an internal derived identifier. The
released feature and label tables are intended for numerical reproduction and
research on pre-peak narrative virality prediction.

## Integrity

The package verifier checks schema, row counts, unique keys, label formula,
feature availability, deterministic split membership, restricted-column
absence, and checksum identity:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```
