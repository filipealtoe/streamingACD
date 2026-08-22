# CIKM 2026 Reproducibility Artifact

<!-- Sérgio Pinto, 2026-08-21 19:46 PDT — added the three-cell four-head reproducer and checksum-bound benchmark predictions. -->
<!-- Sérgio Pinto, 2026-08-21 19:21 PDT — added the text-free numerical bundle and fresh-fit reproduction for the three LLM-feature cells. -->
<!-- Sérgio Pinto, 2026-08-21 19:27 PDT — added a fixed CT24-development-threshold diagnostic alongside the faithful paper-value reproduction. -->
<!-- Sérgio Pinto, 2026-08-21 19:45 PDT — added the exact retained-component reproduction for the CT24 Fusion value. -->
<!-- Sérgio Pinto, 2026-08-21 19:58 PDT — added the recovered Encoder Only vectors, source snapshots, and aggregate ClaimBuster proof. -->
<!-- Sérgio Pinto, 2026-08-21 20:10 PDT — corrected the Encoder Only ClaimBuster boundary after checking the 1,032-row public benchmark. -->
<!-- Sérgio Pinto, 2026-08-21 20:13 PDT — added a cell-by-cell Table 2 audit and an exact reproducer for the retained claim-normalization cell. -->
<!-- Sérgio Pinto, 2026-08-21 20:18 PDT — added the Frontier-LLM aggregate audit and explicit missing-response boundary. -->
<!-- Sérgio Pinto, 2026-08-21 20:23 PDT — added one-run verification for the Table 3 metrics and paired statistical tests. -->
<!-- Sérgio Pinto, 2026-08-21 20:58 PDT — separated Table 3 numeric-cell reproduction from the RandomForest mixed-run consistency boundary. -->
<!-- Sérgio Pinto, 2026-08-21 21:40 PDT — added direct reproduction of every Table 3 significance mark and bootstrap interval. -->
<!-- Sérgio Pinto, 2026-08-21 20:29 PDT — added exact verification for the corpus, canonical pipeline, unit mapping, and clustering sample. -->
<!-- Sérgio Pinto, 2026-08-21 20:34 PDT — added direct reproduction of the cluster-72106 worked example. -->
<!-- Sérgio Pinto, 2026-08-21 20:40 PDT — corrected the Model A claim-normalization protocol boundary and added the shared-error audit. -->
<!-- Sérgio Pinto, 2026-08-21 20:44 PDT — added method-to-artifact protocol consistency checks for all paper sections. -->
<!-- Sérgio Pinto, 2026-08-21 20:51 PDT — extended prose-value checks to the clustering endpoints and claim-normalization side results. -->
<!-- Sérgio Pinto, 2026-08-21 21:11 PDT — added a direct single-item benchmark for the PSR latency claim. -->
<!-- Sérgio Pinto, 2026-08-21 21:14 PDT — separated the reproduced 88-post registry value from its temporal interpretation. -->
<!-- Sérgio Pinto, 2026-08-21 21:20 PDT — verified the CT24 dev-test row count against the official archive. -->
<!-- Sérgio Pinto, 2026-08-21 21:26 PDT — recovered CT24 sentence IDs and directly aligned the shared-error vectors. -->
<!-- Sérgio Pinto, 2026-08-21 21:31 PDT — added executable CT24 calibration and retained CUDA latency audits with their exact evidence boundaries. -->
<!-- Sérgio Pinto, 2026-08-21 21:36 PDT — added primary-source provenance for the three-system published-baseline row and exposed the LLM-feature threshold protocol. -->
<!-- Sérgio Pinto, 2026-08-21 21:46 PDT — added one consolidated audit of the manuscript's artifact-release promises and exact MTL/Fusion protocol scope. -->
<!-- Sérgio Pinto, 2026-08-21 21:55 PDT — added the recovered canonical cluster-embedding asset manifest and optional identity verifier. -->
<!-- Sérgio Pinto, 2026-08-21 22:08 PDT — bound the paper-value inventory to the exact camera-ready PDF. -->

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
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The script carries exact Python dependency versions in its inline metadata. It uses only files in the public clone and exits
non-zero if an integrity or consistency invariant fails.

The command prints every checked invariant and exits non-zero while any
paper-facing discrepancy remains unresolved.

The paper-value manifest records the exact 12-page camera-ready PDF by byte size
and SHA-256. Verify a local copy without adding it to the repository:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --paper-pdf /path/to/CIKM2026.pdf
```

It also verifies the prose-level dataset values: 1,522,909 posts over 600
hourly windows, 535 canonical normalized claims, 529 released virality cluster
instances mapping to 447 canonical claims, and the 5,000-post scope of the
reported clustering-threshold ablation.
The ablation check also reproduces the paper's endpoint summaries: similarity
`0.82` to `0.95` and cluster yield `97.2%` to `27.7%` across thresholds `0.55`
to `0.80` on the retained 5,000-post sample.

The same command also compares the executable protocols with the method prose.
It reports separately when a table value reproduces but its stated duration,
unit, parameter set, metric boundary, feature window, split rule, or training
seed does not match the retained run.

Reproduce the paper's worked example for cluster 72106:

```bash
uv run scripts/reproduce_cikm2026_case_study.py
```

The public aggregate tables and four-head claim scores reproduce every reported
numeric value, including the 88-post value stored by the claim registry. The
separate temporal audit remains inconsistent with the wording around that value:
the cluster time series contains 80 posts through the stated 07:00 detection
timestamp.

Audit the nine internal claim-normalization cells and recompute the retained
Approach 2 / Model A numerical recovery value:

```bash
uv run --no-project scripts/reproduce_cikm2026_claim_normalization_table.py
```

The command recomputes `0.5569184651538435` (`0.5569` in the paper), but marks
it as a numerical match under a different protocol. The paper states
`(10, 0.85, 0.65, 4, 2)`; the recovery run used
`(10, 0.85, 0.5, 5, 0)`. Exact paper-protocol outputs remain unresolved for all
nine internal cells.

Audit the paper's reported 86% shared CT24 misclassification rate:

```bash
uv run scripts/audit_cikm2026_shared_errors.py
```

The three retained vectors are joined exactly through the checksum-bound
official CT24 sentence-ID order. The command exits with `MISMATCH`: the paper
does not define the denominator behind `86%`, and the explicit all-three,
at-least-two, error-assignment, and pairwise definitions yield 27.69%, 46.15%,
69.03%, or at most 80%.

Audit the reported Frontier-LLM aggregate metrics, McNemar test, dimensional
agreement, latency, and cost arithmetic:

```bash
uv run scripts/audit_cikm2026_frontier_llm_ablation.py
```

The aggregate arithmetic is uniquely consistent with the reported values. The
command still exits non-zero with `UNRESOLVED` because the original 1,023 API
responses and per-example Opus predictions were not recovered.

Freshly fit the six paper-facing numeric baselines that require only the public feature matrix:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

This command fits Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR under the stored split and preprocessing rule,
recomputes every reported metric, and verifies the paper's exact McNemar tests from the checksum-bound retained predictions.
Every published tabular numeric cell is recoverable: the RandomForest Spearman `0.530` and the paired tests come from the retained
prediction vector, while the other five RandomForest cells come from the fresh deterministic fit. The command deliberately returns
`VERDICT: FAIL` because those values do not form one internally consistent RandomForest run: the fresh fit gives Spearman `0.532`,
and two paired p-values change when it is used. The complete cell-by-cell result is retained in
[`virality_tabular_reproduction_2026-08-21.json`](../../results/virality_tabular_reproduction_2026-08-21.json).

Recompute all Table 3 Spearman p-values, significance marks, 95% bootstrap
confidence intervals, and exact paired tests from the retained per-example
arrays:

```bash
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

This reproduces all 11 statistical rows with 1,000 bootstrap resamples using
seed 42 and all three exact McNemar p-values. The full-precision output is
[`virality_statistics_reproduction_2026-08-21.json`](../../results/virality_statistics_reproduction_2026-08-21.json).

Benchmark the paper's PSR prediction latency claim on the same 42-feature input:

```bash
uv run scripts/benchmark_cikm2026_psr_latency.py
```

The benchmark times one item per call for the paper-leading BayesianRidge and SVR
models. Its boundary includes feature scaling and model prediction, and excludes
feature generation, fitting, file I/O, and batching. The exact environment,
prediction hashes, and mean/p50/p95/p99 timings are recorded in
[`psr_latency_benchmark_2026-08-21.json`](../../results/psr_latency_benchmark_2026-08-21.json).

To verify the downloaded four-head model checkpoint at the same time:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py --checkpoint /path/to/best_model.pt
```

The canonical 100,000 × 384 cluster-embedding matrix is 153.6 MB and therefore
must be distributed as a release asset rather than a normal Git blob. Its exact
size, SHA-256, dtype, shape, and index identity are recorded in
[`CLUSTER_EMBEDDINGS.md`](CLUSTER_EMBEDDINGS.md). Once uploaded or otherwise
available, verify it with:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --embeddings /path/to/cluster_embeddings.npy
```

Recompute the three four-head F1 cells from the retained predictions:

```bash
uv run scripts/reproduce_cikm2026_mtl_table.py
```

Add `--checkpoint /path/to/best_model.pt` to repeat inference on ClaimBuster and CT23.

Audit the CT24 threshold sensitivity and the retained CUDA throughput trace:

```bash
uv run --no-project scripts/audit_cikm2026_mtl_calibration.py
uv run --no-project scripts/audit_cikm2026_mtl_latency.py
```

The calibration audit directly recovers a `-0.0287` test-F1 shift when the
development-selected threshold is used. It also records that the paper-facing
`0.833` is the test-sweep optimum rather than the value at that development
threshold. The latency audit verifies `6.09` and `4.56` ms per claim from the
retained CUDA trace as batch-amortized inference with batch size 32; it does not
represent single-request latency.

Recompute the CT24 and CT23 Encoder Only cells from retained probabilities and
audit the unresolved ClaimBuster aggregate reconstruction:

```bash
uv run scripts/reproduce_cikm2026_encoder_only.py
```

Freshly fit the PCA-64 + LLM + text-feature classifier and reproduce its three
paper-facing F1 cells:

```bash
uv run scripts/reproduce_cikm2026_llm_features.py
```

The result reproduces the retained per-benchmark threshold sweep and also reports
the three benchmarks with one threshold selected on CT24 development data.

Recompute the CT24 Fusion probability vector from its retained encoder and
XGBoost components and reproduce F1 `0.836`:

```bash
uv run scripts/reproduce_cikm2026_fusion_ct24.py
```

## What is public

| Artifact | Location | Publicly verifiable claim |
|---|---|---|
| Virality labels | [`psr_labels.parquet`](../source_artifacts/virality/psr_labels.parquet) | 529 unique anomaly-flagged cluster instances with detection time, engagement at detection/final engagement, and a virality label (`psr`) |
| Feature matrix | [`features_enhanced.parquet`](../source_artifacts/virality/features_enhanced.parquet) | The same 529 cluster IDs, the same `psr` values, and 42 numeric input features |
| Deterministic split | [`virality_split_manifest_2026-05-12.json`](../../results/virality_split_manifest_2026-05-12.json) | Stored-order split with 423 training and 106 test instances using the NumPy/Scikit-learn-compatible `RandomState(42)` permutation |
| Tuned baseline results | [`tuned_baselines.json`](../source_artifacts/virality/tuned_baselines.json) | Paper-facing results for Random, Ridge, BayesianRidge, RandomForest, LightGBM, SVR, and the text baselines |
| Additional baseline results | [`complete_baselines.json`](../source_artifacts/virality/complete_baselines.json) | Paper-facing LSTM, GRU, and Hawkes result artifacts |
| Table rendering | [`latex_table.tex`](../source_artifacts/virality/latex_table.tex) | Rounded values used for the paper table |
| Virality table audit | [`virality_tabular_reproduction_2026-08-21.json`](../../results/virality_tabular_reproduction_2026-08-21.json) | Fresh six-model metrics, exact retained-prediction McNemar tests, prediction hashes, and the one-run consistency result |
| Virality statistical audit | [`virality_statistics_reproduction_2026-08-21.json`](../../results/virality_statistics_reproduction_2026-08-21.json) | All 11 Spearman p-values and bootstrap intervals plus all three exact McNemar tests recomputed from checksum-bound prediction vectors |
| Artifact-release coverage audit | [`artifact_release_coverage_audit_2026-08-21.json`](../../results/artifact_release_coverage_audit_2026-08-21.json) | Compares the public scientific core with the broader artifact set promised in the manuscript, while preserving the no-post/user-identifiers boundary |
| PSR latency benchmark | [`psr_latency_benchmark_2026-08-21.json`](../../results/psr_latency_benchmark_2026-08-21.json) | Single-item scaling plus prediction timings for the checksum-matched BayesianRidge and SVR fits over 42 features |
| Historical source scripts | [`source_artifacts/virality/`](../source_artifacts/virality/) | Source snapshots used to generate features and baselines in the original workspace |
| Claim-normalization source | [`source_code/`](../source_artifacts/claim_normalization/source_code/) | Retrieval-augmented few-shot implementation used by the retained CT25 run |
| Claim-normalization result | [`ct25_claim_normalization_lambda_2026-05-15/`](../../results/ct25_claim_normalization_lambda_2026-05-15/) | Exact command, environment, aggregate score, and text-free per-sample metrics for 300 CT25 English test rows |
| Claim-normalization table audit | [`TABLE2_RECOVERY_AUDIT.json`](../source_artifacts/claim_normalization/TABLE2_RECOVERY_AUDIT.json) | Cell-by-cell distinction between the Model A numerical recovery match and the unrecovered exact paper protocols for all nine internal cells |
| Check-worthiness soft-label source | [`source_code/`](../source_artifacts/checkworthiness/source_code/) | Exact v4 feature-extraction runner and prompt configuration used by the four-head training path |
| Check-worthiness prompts | [`prompts/`](../../prompts/) | Original v4 soft-label configuration, ablation rendering, and claim-normalization prompt |
| Model parameters | [`MODEL_PARAMETERS.json`](MODEL_PARAMETERS.json) | Machine-readable parameters transcribed from retained executable sources |
| Paper-value and protocol manifest | [`PAPER_VALUE_MANIFEST.json`](PAPER_VALUE_MANIFEST.json) | Table cells, prose values, exact retained evidence boundaries, and method-to-run consistency statuses |
| CT24 dev-test count audit | [`ct24_devtest_metadata.json`](../source_artifacts/checkworthiness/benchmarks/ct24_devtest_metadata.json) | Official archive and inner-file hashes, 318 observed data rows versus the paper's 317, and the separate 1,031-row retained training-selection split |
<!-- Sérgio Pinto, 2026-08-21 22:29 PDT — added a live metadata-only check for both external release assets. -->
| External release assets | [`RELEASE_ASSETS.json`](RELEASE_ASSETS.json) | Expected names, byte sizes, and SHA-256 digests for the checkpoint and cluster embeddings, verified live by `scripts/verify_cikm2026_release_assets.py` |
| Four-head checkpoint | [`CHECKPOINT.md`](CHECKPOINT.md) | Release asset name, URL, byte size, and SHA-256 |
| Cluster-embedding asset | [`CLUSTER_EMBEDDINGS.md`](CLUSTER_EMBEDDINGS.md) | Recovered 100,000 × 384 float32 matrix identity and external-release verification instructions |
| Four-head benchmark predictions | [`benchmark_predictions/`](../runs/deberta_mtl_cikm_20260512_134553/benchmark_predictions/) | Exact ClaimBuster and CT23 probability/label arrays supporting F1 `0.973` and `0.933` |
| Four-head calibration audit | [`mtl_threshold_calibration_audit_2026-08-21.json`](../../results/mtl_threshold_calibration_audit_2026-08-21.json) | Recomputes the retained development/test threshold sweeps, the `-0.0287` F1 shift, and the test-optimized `0.833` selection boundary |
| Four-head CUDA latency audit | [`mtl_gpu_latency_audit_2026-08-21.json`](../../results/mtl_gpu_latency_audit_2026-08-21.json) | Verifies `6.09` and `4.56` ms/claim from the retained batch-size-32 CUDA trace under a batch-amortized inference boundary |
| Published-baseline provenance | [`PUBLISHED_BASELINE_AUDIT.json`](../source_artifacts/checkworthiness/published_baseline/PUBLISHED_BASELINE_AUDIT.json) | Traces CT24 `0.802` to FactFinders Llama2-7b, ClaimBuster `0.920` to G2CW, and CT23 `0.898` to OpenFact GPT-3; the values are valid references but not one Llama2 row |
| Encoder-only bundle | [`encoder_only/`](../source_artifacts/checkworthiness/encoder_only/) | Direct CT24 `0.821` and CT23 `0.928` probability reproduction; ClaimBuster `0.970` remains unresolved without the historical probability vector |
| LLM-feature classifier bundle | [`llm_features_classifier/`](../source_artifacts/checkworthiness/llm_features_classifier/) | Text-free numeric matrices, retained probabilities, and a fresh fit reproducing F1 `0.694`, `0.894`, and `0.846` |
| Fusion CT24 bundle | [`fusion_classifier/`](../source_artifacts/checkworthiness/fusion_classifier/) | Retained component probabilities and an executable reconstruction of CT24 F1 `0.836` |
| Frontier-LLM aggregate audit | [`frontier_llm_ablation/`](../source_artifacts/checkworthiness/frontier_llm_ablation/) | Unique aggregate reconstruction of the reported Opus metrics and paired test, with the missing 1,023-response log kept explicit |
| Shared-error audit | [`checkworthiness_shared_error_audit_2026-08-21.json`](../../results/checkworthiness_shared_error_audit_2026-08-21.json) | Joins all three 341-row vectors by official sentence ID and shows that explicit overlap definitions yield 27.69%, 46.15%, 69.03%, or at most 80%, not 86% |

Checksums for the release-facing files are in [`checksums.sha256`](checksums.sha256). Column-level details and missingness are
documented in [`SCHEMA.md`](SCHEMA.md); provenance, intended use, and limitations are in [`DATA_CARD.md`](DATA_CARD.md).
The [paper-to-artifact index](PAPER_ARTIFACT_INDEX.md) links every major method and
evaluation component to its implementation, inputs, and evidence.

## 529 cluster instances versus 535 normalized claims

The two counts describe different artifacts and different units of analysis:

- the public virality dataset has **529 rows**, one per anomaly-flagged cluster instance;
- the canonical pipeline release contains **535 normalized claims** in a
  [separate claim registry](../../psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/claims.parquet);
- the 535-row claim registry is not a 535-row version of the public virality matrix.

This package verifies the public 529-row cluster-level virality dataset. The 535-row registry provides the corresponding
pipeline-level normalized-claim release as a separate artifact with a different unit of analysis.

## Reproducibility levels

The focused verifier establishes that the committed dataset, split, checksums,
documented missingness, source snapshots, prompts, parameters, and paper-facing result
files are internally consistent. The dedicated numeric reproducer performs fresh fits
for Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR. The optional
`--checkpoint` check binds a downloaded release asset to the packaged four-head run.
The same numeric-baseline command verifies the retained exact McNemar results and
keeps the fresh single-run consistency discrepancy machine-detectable.

The canonical normalized-claim and aggregate cluster tables are retained under
`psr/explainableACD/data/pipeline_output/`. Raw Twitter/X posts and user-level data are
not needed for the public dataset checks or the fresh numeric-baseline reproduction.

The consolidated [release-coverage audit](../../results/artifact_release_coverage_audit_2026-08-21.json)
records the exact boundary between that reproducible aggregate scientific core
and the broader release description in the manuscript. It does not reintroduce
post-level or user-level identifiers.

## Privacy and redistribution

The CIKM virality release-facing Parquet files contain derived cluster-level numeric fields plus the cluster detection timestamp.
They do not contain raw post text, post IDs, user IDs, usernames, profile descriptions, or individual locations. The repository
also retains normalized claims and aggregate cluster/time-series outputs from the canonical run. Raw Twitter/X content and the
user-level lookup table are excluded from the current public tree.
