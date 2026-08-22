# Data Card: CIKM 2026 Narrative Virality Dataset

## Summary

This release contains cluster-level labels and numeric features used to evaluate narrative-level streaming virality prediction in
the CIKM 2026 paper “Predicting Narrative Virality on Social Media Streams for Pre-Peak Misinformation Triage.”

- **Unit of analysis:** one anomaly-flagged cluster instance.
- **Rows:** 529 unique `cluster_id` values.
- **Label:** `psr`, the fraction of final engagement occurring after detection, computed as
  `(e_final - e_detect) / e_final`.
- **Inputs:** 42 numeric features measured or derived at the retrospective detection-window boundary.
- **Stored split:** 423 training and 106 test instances.
<!-- Sérgio Pinto, 2026-08-21 21:46 PDT — linked the consolidated manuscript-to-public-package release-coverage audit. -->
<!-- Sérgio Pinto, 2026-08-21 20:44 PDT — clarified the actual pre-detection feature scope and retained split protocol. -->
<!-- Sérgio Pinto, 2026-08-21 20:29 PDT — replaced the ambiguous day count with the exact replay range and window count. -->
- **Domain:** a US political-event Twitter/X stream drawn from the US Election 2020 corpus, replayed in 600 hourly windows from
  2020-10-15 00:00 through 2020-11-08 23:00 (25 calendar dates).

## Files

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `reproducibility/source_artifacts/virality/psr_labels.parquet` | 529 | 5 | Cluster ID, detection time, engagement at detection, final engagement, and `psr` |
| `reproducibility/source_artifacts/virality/features_enhanced.parquet` | 529 | 44 | Cluster ID, 42 numeric features, and `psr` |
| `results/virality_split_manifest_2026-05-12.json` | — | — | Exact stored-order train/test membership and split rule |

The two Parquet files have the same 529 unique cluster IDs and exactly matching `psr` values after joining by `cluster_id`.

The public repository also contains a 535-row normalized-claim registry,
aggregate cluster and time-series tables, and a separate 535-row four-head score
artifact. The [release-coverage audit](../../results/artifact_release_coverage_audit_2026-08-21.json)
distinguishes these present scientific artifacts from the broader post-level,
rationale, embedding, and raw-response bundle described in the manuscript.
The canonical cluster-embedding matrix has since been recovered as a 153.6 MB
external-release candidate; its verified 100,000 × 384 shape and checksum are
documented in [`CLUSTER_EMBEDDINGS.md`](CLUSTER_EMBEDDINGS.md), but the matrix is
not yet uploaded.

## Source and derivation

The source stream was derived from the Kaggle
[US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets) corpus. The canonical pipeline
processed 1,522,909 posts and produced a
[separate registry of 535 normalized claims](../../psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/claims.parquet).
The released virality dataset instead uses cluster instances as rows; these counts are not expected to match.

<!-- Sérgio Pinto, 2026-08-21 17:52 PDT — Corrected the generator path so the data card points to the tracked source snapshot. -->
The feature-generation snapshot is retained at
`psr/reproducibility/source_artifacts/virality/generate_enhanced_features.py`. It depends on the raw post-level table, which is not
redistributed, so it documents the derivation but is not a public end-to-end regeneration path.
The source filters every aggregate and post-level input to timestamps at or before
detection; it uses all retained pre-detection history rather than a fixed six-hour
slice.

## Split

The stored order of `features_enhanced.parquet` was split with the Scikit-learn-compatible rule:

```text
train_test_split(row_indices, test_size=0.2, random_state=42, shuffle=True, stratify=None)
```

This yields 423 training and 106 test instances. The focused verifier independently rebuilds the equivalent
`RandomState(42)` permutation and checks every stored cluster ID. The split is
shuffled but is not stratified.

## Missing values

Eight feature columns contain IEEE NaN for all 529 rows:

- `geographic_entropy_predetect`
- `unique_countries_predetect`
- `max_retweets_per_tweet`
- `avg_likes_per_tweet`
- `rt_like_ratio`
- `engagement_skewness`
- `engagement_gini`
- `account_age_avg_days`

Two additional columns contain NaN in 50 of 529 rows:

- `user_gini`
- `top_user_ratio`

The packaged baseline source replaces NaN and positive/negative infinity with `0.0` before fitting. Consumers should not treat the
eight all-NaN columns as observed signals.

## Intended uses

- verifying the dataset shape, labels, stored split, and paper-facing result artifacts;
- freshly fitting Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR with the released feature matrix and stored
  preprocessing/split;
- comparing cluster-level virality predictors under the same split and feature boundary;
- studying retrospective pre-peak cluster-level intervention signals within the documented political-event setting.

## Out-of-scope uses

- identifying, profiling, or targeting individuals;
- reconstructing deleted or private posts;
- claiming performance on non-political, non-US, cross-platform, or live streams without new evaluation;
- treating the 529 cluster instances as 529 independent normalized claims;
- claiming that this package reproduces the complete streaming pipeline or every model fit from raw inputs.

## Privacy and platform boundary

The two CIKM virality Parquet files contain derived cluster-level numeric fields plus the cluster detection timestamp. They exclude
raw post text, post IDs, user IDs, usernames, profile descriptions, and individual locations. The current repository tree also
excludes the canonical run's user-level lookup table. Users remain responsible for applicable law, research-ethics requirements,
and current Twitter/X platform terms.

## Known limitations

- one political-event stream covering 600 hourly windows across 25 calendar dates;
- retrospective replay rather than a live no-lookahead deployment proof: likes and retweets are single collection-time snapshots,
  and the stored `detection_time` is the start of the triggering 60-minute window;
- temporal and platform-specific engagement features;
- eight features are entirely unavailable and two are partially unavailable in the released matrix;
- Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR can be freshly fit from the approved release files; local candidate
  anomaly, temporal, and embedding artifacts support heavier reruns but are not yet approved for publication;
- the separate 535-claim registry is public, but it is not a row-aligned version of this 529-instance virality dataset;
- the raw post table and some prediction arrays are not redistributed; the four-head
  model checkpoint is distributed separately through the documented GitHub release.

## License

No repository or dataset license was present when this package was prepared. The authors must add an explicit license before
describing the dataset as licensed for reuse.
