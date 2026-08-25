# Paper-to-artifact index

<!-- Sérgio Pinto, 2026-08-21 19:18 PDT — linked the executable anomaly-table evidence so every reported row has a public entry point. -->
<!-- Sérgio Pinto, 2026-08-21 19:21 PDT — linked the fresh LLM-feature fit, text-free numerical input, and full-precision result. -->
<!-- Sérgio Pinto, 2026-08-21 19:27 PDT — distinguished the retained paper threshold sweep from the fixed development-threshold transfer diagnostic. -->
<!-- Sérgio Pinto, 2026-08-21 19:45 PDT — linked the exact CT24 Fusion reconstruction and recorded the unavailable cross-benchmark per-sample evidence. -->
<!-- Sérgio Pinto, 2026-08-21 19:58 PDT — linked the recovered Encoder Only sources, numerical vectors, aggregate audit, and executable metric check. -->
<!-- Sérgio Pinto, 2026-08-21 20:10 PDT — corrected the Encoder Only ClaimBuster boundary after checking the 1,032-row public input. -->
<!-- Sérgio Pinto, 2026-08-21 20:13 PDT — linked the cell-by-cell claim-normalization table audit and exact retained result. -->
<!-- Sérgio Pinto, 2026-08-21 20:18 PDT — linked the Frontier-LLM aggregate audit and missing-response boundary. -->
<!-- Sérgio Pinto, 2026-08-21 20:23 PDT — linked the Table 3 fresh-fit and paired-test consistency evidence. -->
<!-- Sérgio Pinto, 2026-08-21 20:58 PDT — distinguished complete numeric-cell coverage from the RandomForest mixed-run consistency failure. -->
<!-- Sérgio Pinto, 2026-08-21 21:40 PDT — linked the fresh Table 3 significance, confidence-interval, and exact paired-test reproduction. -->
<!-- Sérgio Pinto, 2026-08-21 21:46 PDT — indexed the paper-to-public-package release-coverage audit and the exact MTL/Fusion protocol scope. -->
<!-- Sérgio Pinto, 2026-08-21 21:55 PDT — linked the recovered canonical cluster-embedding matrix metadata and external-release verification path. -->
<!-- Sérgio Pinto, 2026-08-21 20:29 PDT — linked exact corpus, pipeline, mapping, and scoped clustering-ablation counts. -->
<!-- Sérgio Pinto, 2026-08-21 20:34 PDT — linked the executable cluster-72106 worked-example evidence. -->
<!-- Sérgio Pinto, 2026-08-21 20:40 PDT — corrected the Model A protocol boundary and linked the CT24 shared-error audit. -->
<!-- Sérgio Pinto, 2026-08-21 20:44 PDT — indexed the cross-section method-to-run protocol audit. -->
<!-- Sérgio Pinto, 2026-08-21 21:11 PDT — linked the direct single-item PSR latency benchmark and its exact timing boundary. -->
<!-- Sérgio Pinto, 2026-08-21 21:14 PDT — separated the reproduced 88-post registry value from its temporal interpretation. -->
<!-- Sérgio Pinto, 2026-08-21 21:20 PDT — linked the official CT24 dev-test archive and verified its 318-row count. -->
<!-- Sérgio Pinto, 2026-08-21 21:26 PDT — linked the recovered CT24 sentence-ID order and direct three-model error join. -->
<!-- Sérgio Pinto, 2026-08-21 21:31 PDT — linked the executable CT24 threshold-calibration and retained CUDA latency audits. -->
<!-- Sérgio Pinto, 2026-08-21 21:36 PDT — linked primary-source provenance for the published baseline and the evaluation-set threshold mismatch. -->
<!-- Sérgio Pinto, 2026-08-21 22:08 PDT — linked the paper-value inventory to the exact camera-ready PDF identity. -->
<!-- Sérgio Pinto, 2026-08-21 22:29 PDT — linked the expected external assets to a live GitHub release-metadata verifier. -->
<!-- Sérgio Pinto, 2026-08-21 22:32 PDT — documented the portable JSON verification report and its explicit failure categories. -->
<!-- Sérgio Pinto, 2026-08-22 00:53 PDT — rebound the index to the exact final PDF and removed superseded release-scope and embedding-dimension claims. -->
<!-- Sérgio Pinto, 2026-08-25 14:18 WEST — linked fresh inference from the three public Encoder checkpoints and separated historical positional results from corrected sentence-ID-aligned LLM/Fusion replications. -->

This index connects the CIKM 2026 paper components to their public implementation,
parameters, prompts, and evidence. Paths are relative to the repository root.

| Paper component | Implementation and parameters | Prompt or data input | Result or verification evidence |
|---|---|---|---|
| Camera-ready paper identity | [paper-value manifest](PAPER_VALUE_MANIFEST.json) and [main verifier](../../scripts/verify_cikm2026_artifacts.py) | Exact 12-page `CIKM2026-camera-ready-2026-08-22.pdf`; the manuscript itself is not redistributed here | Optional `--paper-pdf` verification checks byte size and SHA-256 before treating the manifest as the paper's numerical inventory |
| Claim normalization | [source snapshot](../source_artifacts/claim_normalization/source_code/), [parameters](MODEL_PARAMETERS.json), and [table audit command](../../scripts/reproduce_cikm2026_claim_normalization_table.py) | [v1 recovery prompt](../../prompts/claim_normalization_cikm2026.md); official CheckThat! 2025 Task 2 English files identified by checksum in the run record | [retained CT25 run](../../results/ct25_claim_normalization_lambda_2026-05-15/) numerically matches Approach 2 / Model A `0.5569` under a different protocol; the [cell-by-cell recovery audit](../source_artifacts/claim_normalization/TABLE2_RECOVERY_AUDIT.json) keeps all nine exact paper protocols unresolved |
| Check-worthiness soft-label features | [exact v4 extraction runner](../source_artifacts/checkworthiness/source_code/scripts/runners/run_llm_checkworthiness_v4.py), [original v4 configuration](../../prompts/checkworthiness_prompts_zeroshot_v4.yaml), and [parameters](MODEL_PARAMETERS.json) | The three v4 checkability, verifiability, and harm prompts | [four-head training results and saved predictions](../runs/deberta_mtl_cikm_20260512_134553/) |
| Encoder-only DeBERTa | [training/evaluation sources](../source_artifacts/checkworthiness/source_code/scripts/), [historical table reproducer](../../scripts/reproduce_cikm2026_encoder_only.py), and [public-checkpoint runner](../../scripts/reproduce_cikm2026_public_encoder_fusion.py) | [retained CT24/CT23 vectors](../source_artifacts/checkworthiness/encoder_only/) plus three immutable public model revisions | The retained vectors reproduce CT24 `0.821` and CT23 `0.928`; [fresh per-example inference](../../results/public_encoder_fusion_replication_2026-08-25/) also reproduces CT23 `0.928`, while its best recorded ClaimBuster diagnostic is `0.966805`, so the historical `0.970` remains unmatched rather than inferred from an aggregate |
| PCA-64 + LLM + text-feature classifier | [feature-generation source](../source_artifacts/checkworthiness/source_code/scripts/feature_engineering/), [benchmark source](../source_artifacts/checkworthiness/source_code/scripts/benchmark_pca_llm_text.py), and [fresh-fit reproducer](../../scripts/reproduce_cikm2026_llm_features.py) | [text-free 153-feature matrices](../source_artifacts/checkworthiness/llm_features_classifier/), recovered [LLM features](../source_artifacts/checkworthiness/ct24_llm_features_v4/), and [text-free CT24 labels](../source_artifacts/checkworthiness/ct24_labels/) | The [full-precision result](../../results/llm_features_table_reproduction_2026-08-21.json) preserves the three paper values under the retained positional/test-sweep pipeline and separately reports the corrected one-to-one sentence-ID-aligned replication and fixed source-development threshold |
| Four-head multi-task DeBERTa | [training source](../runs/deberta_mtl_cikm_20260512_134553/scripts/finetune_deberta_mtl.py), [exact command](../runs/deberta_mtl_cikm_20260512_134553/command.txt), [table reproducer](../../scripts/reproduce_cikm2026_mtl_table.py), [calibration audit](../../scripts/audit_cikm2026_mtl_calibration.py), and [latency audit](../../scripts/audit_cikm2026_mtl_latency.py) | CT24 text plus the packaged [ClaimBuster and CT23 inputs](../source_artifacts/checkworthiness/benchmarks/) | [CT24 and cross-benchmark predictions](../runs/deberta_mtl_cikm_20260512_134553/), [fresh metric summary](../../results/mtl_table_reproduction_2026-08-21.json), [threshold audit](../../results/mtl_threshold_calibration_audit_2026-08-21.json), [retained CUDA trace audit](../../results/mtl_gpu_latency_audit_2026-08-21.json), and [checkpoint manifest](CHECKPOINT.md) |
| Fusion classifier | [CT24 reproducer](../../scripts/reproduce_cikm2026_fusion_ct24.py) and [public cross-benchmark runner](../../scripts/reproduce_cikm2026_public_encoder_fusion.py) | [retained CT24 components](../source_artifacts/checkworthiness/fusion_classifier/), public Encoder checkpoints, recovered LLM features, and text-free label/ID tables | The retained CT24 result reproduces `0.836`. The [fresh cross-benchmark result](../../results/public_encoder_fusion_replication_2026-08-25/) preserves the historical positional diagnostic and adds the corrected one-to-one ID-aligned three-seed path; neither fresh path is relabelled as an exact reproduction of historical ClaimBuster `0.961` or CT23 `0.915` |
| Frontier-LLM check-worthiness ablation | [inference](../../scripts/run_llm_ablation.py), [prompt loader](../../scripts/prompts.py), [analysis](../../scripts/analyze_llm_ablation.py), and [aggregate audit](../../scripts/audit_cikm2026_frontier_llm_ablation.py) | [human-readable v4 prompts](../../prompts/checkworthiness_v4_prompts.md) and retained CT24 label order | [aggregate recovery result](../../results/frontier_llm_ablation_audit_2026-08-21.json) uniquely checks the reported confusion, McNemar, agreement, cost, and latency arithmetic; direct reproduction remains unresolved because the historical 1,023 API responses are absent |
| CT24 shared-error analysis | [alignment audit](../../scripts/audit_cikm2026_shared_errors.py) | Three retained 341-row prediction vectors plus the [official CT24 test sentence-ID order](../source_artifacts/checkworthiness/llm_features_classifier/ct24_sentence_ids.npy) | [audit result](../../results/checkworthiness_shared_error_audit_2026-08-21.json) joins every row exactly and records that the explicit shared-error definitions yield 27.69%, 46.15%, 69.03%, or at most 80%, not 86% |
| CT24 evaluation split count | [official CheckThat! 2024 archive](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/raw/main/task1/data/CT24_checkworthy_english.zip) | Checksum-verified English dev-test file; no dataset text is copied into this artifact | [row-count metadata](../source_artifacts/checkworthiness/benchmarks/ct24_devtest_metadata.json) records 318 data rows versus the paper's `N=317`, and distinguishes the retained 1,031-row development split |
| Published check-worthiness baseline | [FactFinders Llama2-7b source](https://ceur-ws.org/Vol-3740/paper-47.pdf), [G2CW source](https://pmc.ncbi.nlm.nih.gov/articles/PMC9916500/), and [OpenFact source](https://ceur-ws.org/Vol-3497/paper-040.pdf) | Three external benchmark values | [provenance audit](../source_artifacts/checkworthiness/published_baseline/PUBLISHED_BASELINE_AUDIT.json) records that `0.802`, `0.920`, and `0.898` are valid reference values from three different systems, not one Llama2 cross-dataset row |
| Streaming claim and cluster registry | [canonical output tables](../../psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/) | Normalized claims, aggregate clusters, time series, and 600 hourly window results | [pipeline summary](../source_artifacts/pipeline/streaming_full_2026-01-17_03-56_summary.json) and the main verifier reproduce 1,522,909 posts, 535 claims, and the 529-cluster-to-447-claim mapping |
| Worked example: cluster 72106 | [executable reproducer](../../scripts/reproduce_cikm2026_case_study.py) | Canonical cluster/claim/time-series tables, released PSR row, and retained 535-row four-head claim scores | [Full-precision result](../../results/case_study_reproduction_2026-08-21.json) reproduces every reported numeric value, including 88 from the registry snapshot; the separate temporal audit records 80 posts in the time series through 07:00 |
| Online clustering threshold evidence | [threshold investigation](../../docs/investigations/clustering_threshold_ablation_20260519.md) | Retained 5,000-post sample | [cluster statistics](../source_artifacts/clustering/cluster_statistics.json) reproduce 86.9% yield and 0.8686 mean similarity at `tau=0.65`; these are scoped to the sample rather than the full stream |
| Anomaly-detection evaluation | [executable reproducer](../../scripts/reproduce_cikm2026_anomaly_table.py) | [aggregate numeric time series](../source_artifacts/anomaly/timeseries.parquet) and [retained full-precision references](../source_artifacts/anomaly/historical/) | [fresh five-detector run](../../results/anomaly_table_reproduction_2026-08-21/) reproducing the four paper rows exactly, plus a Random sanity baseline |
| Narrative-level virality dataset | [feature matrix](../source_artifacts/virality/features_enhanced.parquet) and [virality labels](../source_artifacts/virality/psr_labels.parquet) | 529 cluster instances, 42 detection-time engineered features, and virality labels | [data card](DATA_CARD.md), [schema](SCHEMA.md), [checksums](checksums.sha256), and the focused verifier |
| Numeric virality baselines | [fresh-fit reproducer](../../scripts/reproduce_cikm2026_tabular_baselines.py), [statistical reproducer](../../scripts/reproduce_cikm2026_virality_statistics.py), and [parameters](MODEL_PARAMETERS.json) | Public feature matrix, deterministic `RandomState(42)` split, checksum-bound retained predictions, and the [original statistical run](../../results/psr_statistics_20260518_205609/) | The [cell-by-cell result](../../results/virality_tabular_reproduction_2026-08-21.json) reproduces every tabular numeric cell; the [statistical result](../../results/virality_statistics_reproduction_2026-08-21.json) reproduces all 11 Spearman p-values and bootstrap intervals plus all three exact McNemar tests. The separate consistency audit exposes that the RandomForest row combines a retained Spearman cell with five fresh-fit cells |
| PSR prediction latency | [single-item benchmark](../../scripts/benchmark_cikm2026_psr_latency.py) | Released 42-feature matrix and exact BayesianRidge/SVR fits verified by prediction hashes | [Fresh benchmark result](../../results/psr_latency_benchmark_2026-08-21.json) measures scaling plus prediction for one item, excluding feature generation, fitting, file I/O, and batching |
| Sequence, Hawkes, and BERTweet virality rows | [historical experiment scripts](../../psr/explainableACD/experiments/scripts/) | Derived cluster-level inputs | [complete baselines](../source_artifacts/virality/complete_baselines.json), [tuned baselines](../source_artifacts/virality/tuned_baselines.json), and [paper table rendering](../source_artifacts/virality/latex_table.tex) |
| Cross-section protocol consistency | [main verifier](../../scripts/verify_cikm2026_artifacts.py) | Retained stream, pipeline, anomaly, feature-generation, split, and MTL-run sources | [paper-value manifest](PAPER_VALUE_MANIFEST.json) separates reproduced numbers from duration, unit, parameter, metric-definition, feature-window, split, seed, and unsupported side-claim gaps |
| Artifact-release coverage | [coverage audit](../../scripts/audit_cikm2026_release_coverage.py) | 529-row virality dataset, source snapshots, model parameters and checkpoint, prompts, schema, and repository evidence inventory | [Machine-readable result](../../results/artifact_release_coverage_audit_2026-08-22.json) verifies all six availability statements in the exact final PDF and the privacy-safe aggregate scientific core |
| External release assets | [expected asset manifest](RELEASE_ASSETS.json) and [live metadata verifier](../../scripts/verify_cikm2026_release_assets.py) | Public `cikm-2026-artifact-v1` GitHub release metadata | Checks asset presence, uploaded state, byte size, and GitHub SHA-256 digest without downloading either large binary |
| Canonical cluster embeddings | [asset manifest](CLUSTER_EMBEDDINGS.md), [metadata](cluster_embeddings.metadata.json), and [checksum](cluster_embeddings.sha256) | Recovered 100,000 × 384 float32 matrix and the committed 100,000-entry cluster-index mapping | Optional `--embeddings` verification checks byte size, SHA-256, shape, dtype, finiteness, and index identity; the live release verifier checks publication. The exact final PDF states no embedding dimension |

## Verification entry points

Run the complete public-package consistency check:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

Write the same result as a portable machine-readable report:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --json-output /path/to/verification.json
```

Verify the live external release assets from GitHub metadata only:

```bash
uv run --no-project scripts/verify_cikm2026_release_assets.py
```

Bind the inventory to a local copy of the audited camera-ready PDF:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --paper-pdf /path/to/CIKM2026-camera-ready-2026-08-22.pdf
```

Audit every claim-normalization table cell and recompute the retained
Approach 2 / Model A numerical recovery result:

```bash
uv run --no-project scripts/reproduce_cikm2026_claim_normalization_table.py
```

Audit the CT24 shared-error claim:

```bash
uv run scripts/audit_cikm2026_shared_errors.py
```

Audit the recoverable Frontier-LLM aggregates and the missing-response boundary:

```bash
uv run scripts/audit_cikm2026_frontier_llm_ablation.py
```

Reproduce the cluster-72106 worked example:

```bash
uv run scripts/reproduce_cikm2026_case_study.py
```

Freshly fit the six numeric virality baselines and verify the paired tests:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

Benchmark single-item PSR prediction over the released 42 features:

```bash
uv run scripts/benchmark_cikm2026_psr_latency.py
```

Reproduce every row of the paper's anomaly-detection table:

```bash
uv run scripts/reproduce_cikm2026_anomaly_table.py
```

Recompute the CT24, ClaimBuster, and CT23 four-head F1 cells:

```bash
uv run scripts/reproduce_cikm2026_mtl_table.py
```

Audit the retained CT24 threshold sweep and CUDA throughput trace:

```bash
uv run --no-project scripts/audit_cikm2026_mtl_calibration.py
uv run --no-project scripts/audit_cikm2026_mtl_latency.py
```

Recompute the two directly supported Encoder Only F1 cells and audit the unresolved ClaimBuster aggregate:

```bash
uv run scripts/reproduce_cikm2026_encoder_only.py
```

Freshly fit and evaluate the PCA-64 + LLM + text-feature classifier:

```bash
uv run scripts/reproduce_cikm2026_llm_features.py
```

Recompute the CT24 Fusion probability vector and F1:

```bash
uv run scripts/reproduce_cikm2026_fusion_ct24.py
```

Run the three public Encoder checkpoints and recompute the historical and
sentence-ID-aligned cross-benchmark Fusion paths:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py --inference
```

After downloading the four-head checkpoint release asset, verify its identity together
with the public package:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py --checkpoint /path/to/best_model.pt
```
