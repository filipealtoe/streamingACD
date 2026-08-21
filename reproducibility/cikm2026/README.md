# CIKM 2026 Reproducibility Artifact

This package supports the dataset and baseline evidence for the accepted paper:

> Filipe Altoe, Sérgio Miguel Gonçalves Pinto, and H. Sofia Pinto. 2026. “Predicting Narrative Virality on Social Media
> Streams for Pre-Peak Misinformation Triage.” 35th ACM International Conference on Information and Knowledge Management
> (CIKM 2026). Reserved ACM DOI:
> [10.1145/3799682.3840688](https://doi.org/10.1145/3799682.3840688) (expected to resolve after publication).

CIKM 2026 takes place in Rome, Italy, on November 7–11, 2026. See the
[official conference site](https://cikm2026.diag.uniroma1.it/).

## Quick verification

From the repository root, run:

```bash
uv run scripts/verify_cikm2026_artifacts.py
```

The script carries exact Python dependency versions in its inline metadata. It uses only files in the public clone and exits
non-zero if an integrity or consistency invariant fails.

The expected final line is:

```text
VERDICT: PASS
```

Freshly fit the six paper-facing numeric baselines that require only the public feature matrix:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

This command fits Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR under the stored split and preprocessing rule,
recomputes every reported metric, and compares the full-precision values with `tuned_baselines.json`. Its expected final line is
also `VERDICT: PASS`.

## What is public

| Artifact | Location | Publicly verifiable claim |
|---|---|---|
| Virality labels | [`psr_labels.parquet`](../source_artifacts/virality/psr_labels.parquet) | 529 unique anomaly-flagged cluster instances with detection time, engagement at detection/final engagement, and a virality label (`psr`) |
| Feature matrix | [`features_enhanced.parquet`](../source_artifacts/virality/features_enhanced.parquet) | The same 529 cluster IDs, the same `psr` values, and 42 numeric input features |
| Deterministic split | [`virality_split_manifest_2026-05-12.json`](../../results/virality_split_manifest_2026-05-12.json) | Stored-order split with 423 training and 106 test instances using the NumPy/Scikit-learn-compatible `RandomState(42)` permutation |
| Tuned baseline results | [`tuned_baselines.json`](../source_artifacts/virality/tuned_baselines.json) | Paper-facing results for Random, Ridge, BayesianRidge, RandomForest, LightGBM, SVR, and the text baselines |
| Additional baseline results | [`complete_baselines.json`](../source_artifacts/virality/complete_baselines.json) | Paper-facing LSTM, GRU, and Hawkes result artifacts |
| Table rendering | [`latex_table.tex`](../source_artifacts/virality/latex_table.tex) | Rounded values used for the paper table |
| Historical source scripts | [`source_artifacts/virality/`](../source_artifacts/virality/) | Source snapshots used to generate features and baselines in the original workspace |

Checksums for the release-facing files are in [`checksums.sha256`](checksums.sha256). Column-level details and missingness are
documented in [`SCHEMA.md`](SCHEMA.md); provenance, intended use, and limitations are in [`DATA_CARD.md`](DATA_CARD.md).

## 529 cluster instances versus 535 normalized claims

The two counts describe different artifacts and different units of analysis:

- the public virality dataset has **529 rows**, one per anomaly-flagged cluster instance;
- the canonical pipeline release contains **535 normalized claims** in a
  [separate claim registry](../../psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/claims.parquet);
- the 535-row claim registry is not a 535-row version of the public virality matrix.

This package verifies the public 529-row cluster-level virality dataset. The 535-row registry provides the corresponding
pipeline-level normalized-claim release as a separate artifact with a different unit of analysis.

## Reproducibility boundary

The focused verifier establishes that the committed dataset, split, checksums, documented missingness, and paper-facing result
files are internally consistent. The dedicated numeric reproducer additionally performs fresh fits for Random, Ridge,
BayesianRidge, RandomForest, LightGBM, and SVR.

Extended local audits cover sequence, Hawkes, BERTweet, anomaly detection, clustering, and threshold selection. Their candidate
inputs and generated outputs are deliberately excluded from the public release boundary until the authors complete the relevant
privacy and source-rights decisions. Dense embeddings must not be described as anonymized.

Fresh full-pipeline reproduction is outside this public package because the raw Twitter/X corpus, raw post table, large model
checkpoint, and some prediction arrays are not redistributed. Selected canonical run tables are retained under
`psr/explainableACD/data/pipeline_output/`. The historical scripts are retained as provenance snapshots; they should not be
described as a one-command end-to-end reproduction from the public clone. Historical Table 4 rows also used incompatible splits,
and the new reruns must not be presented as reproductions of those mixed-split numbers.

## Privacy and redistribution

The CIKM virality release-facing Parquet files contain derived cluster-level numeric fields plus the cluster detection timestamp.
They do not contain raw post text, post IDs, user IDs, usernames, profile descriptions, or individual locations. The repository
also retains normalized claims and aggregate cluster/time-series outputs from the canonical run. Raw Twitter/X content and the
user-level lookup table are excluded from the current public tree.

## License status

No repository or dataset license was present when this camera-ready artifact package was prepared. The authors must select and
add an explicit license before describing the files as licensed for reuse. Source-corpus licensing and Twitter/X platform terms do
not automatically grant a license for this derived release.
