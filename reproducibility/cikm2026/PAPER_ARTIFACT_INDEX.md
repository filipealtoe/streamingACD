# CIKM 2026 paper-to-artifact index

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — limited the public index to checksum-bound inputs, executable reproductions, and verified outputs. -->
<!-- Sérgio Pinto, 2026-08-25 19:35 WEST — linked the retained, numerically verified performance measurements. -->
<!-- Sérgio Pinto, 2026-08-25 20:17 WEST — linked the two version-pinned RandomForest fits and all eleven complete virality rows. -->
<!-- Sérgio Pinto, 2026-08-25 21:56 WEST — describes the RandomForest result as a neutral version-pinned paper-cell reconstruction. -->
<!-- Sérgio Pinto, 2026-08-28 01:11 WEST — linked the checksum-bound CT23 Fusion reconstruction and full-precision result. -->
<!-- Sérgio Pinto, 2026-08-28 01:26 WEST — linked hardware-specific latency reruns to an explicit caller-selected output. -->

| Paper component | Inputs and source | Reproduction | Verified output |
|---|---|---|---|
| Released virality dataset | [`source_artifacts/virality/`](../source_artifacts/virality/) and [`SCHEMA.md`](SCHEMA.md) | `uv run --no-project scripts/verify_cikm2026_artifacts.py` | 529 instances, 42 features, PSR labels, deterministic split, privacy checks |
| Streaming pipeline aggregates | [`source_artifacts/pipeline/`](../source_artifacts/pipeline/) and [`source_artifacts/clustering/`](../source_artifacts/clustering/) | `uv run --no-project scripts/verify_cikm2026_artifacts.py` | corpus, cluster, claim, mapping, and threshold-ablation aggregates |
| Anomaly detection | [`source_artifacts/anomaly/`](../source_artifacts/anomaly/) and [`reproduce_cikm2026_anomaly_table.py`](../../scripts/reproduce_cikm2026_anomaly_table.py) | `uv run scripts/reproduce_cikm2026_anomaly_table.py` | [`anomaly_table_reproduction_2026-08-21/`](../../results/anomaly_table_reproduction_2026-08-21/) |
| Four-head check-worthiness | [run bundle](../runs/deberta_mtl_cikm_20260512_134553/), [benchmark inputs](../source_artifacts/checkworthiness/benchmarks/), and [`reproduce_cikm2026_mtl_table.py`](../../scripts/reproduce_cikm2026_mtl_table.py) | `uv run scripts/reproduce_cikm2026_mtl_table.py` | CT24 `0.833`, ClaimBuster `0.973`, CT23 `0.933` |
| LLM-feature check-worthiness | [numerical bundle](../source_artifacts/checkworthiness/llm_features_classifier/) and [`reproduce_cikm2026_llm_features.py`](../../scripts/reproduce_cikm2026_llm_features.py) | `uv run scripts/reproduce_cikm2026_llm_features.py` | CT24 `0.694`, ClaimBuster `0.894`, CT23 `0.846` |
| Encoder-only check-worthiness | [retained vectors](../source_artifacts/checkworthiness/encoder_only/) | `uv run --no-project scripts/verify_cikm2026_artifacts.py` | CT24 `0.821`, CT23 `0.928` |
| Fusion check-worthiness | [retained CT24 components](../source_artifacts/checkworthiness/fusion_classifier/), [retained CT23 encoder vectors](../source_artifacts/checkworthiness/encoder_only/), and [nine-feature matrices](../source_artifacts/checkworthiness/llm_features_classifier/) | `uv run scripts/reproduce_cikm2026_fusion_ct24.py`; `uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23` | CT24 `0.836`; CT23 `0.915` |
| Four-head GPU throughput | [retained CUDA trace and evaluation source](../runs/deberta_mtl_cikm_20260512_134553/) | `uv run --no-project scripts/verify_cikm2026_artifacts.py` | ClaimBuster `6.091 ms` and CT23 `4.564 ms` batch-amortized per claim |
| Numeric virality latency | [benchmark result](../../results/psr_latency_benchmark_2026-08-21.json) and [`benchmark_cikm2026_psr_latency.py`](../../scripts/benchmark_cikm2026_psr_latency.py) | `uv run --no-project scripts/benchmark_cikm2026_psr_latency.py --output /tmp/cikm2026_psr_latency.json` | Bayesian Ridge and SVR mean and p95 below `1 ms` per item |
| Virality prediction | [public matrix](../source_artifacts/virality/), [retained predictions](../../psr/), [`reproduce_cikm2026_tabular_baselines.py`](../../scripts/reproduce_cikm2026_tabular_baselines.py), [`reproduce_cikm2026_random_forest.py`](../../scripts/reproduce_cikm2026_random_forest.py), and [`reproduce_cikm2026_virality_statistics.py`](../../scripts/reproduce_cikm2026_virality_statistics.py) | `uv run scripts/reproduce_cikm2026_tabular_baselines.py`, `uv run --no-project scripts/reproduce_cikm2026_random_forest.py`, and `uv run scripts/reproduce_cikm2026_virality_statistics.py` | six deterministic fresh fits; version-pinned RandomForest paper-cell reconstruction; eleven Spearman tests and intervals; three exact McNemar tests; eleven complete six-metric rows |
| Four-head checkpoint | [`CHECKPOINT.md`](CHECKPOINT.md) and [`checkpoint.sha256`](checkpoint.sha256) | verifier option `--checkpoint` | byte size and SHA-256 identity |
| Cluster embeddings | [`CLUSTER_EMBEDDINGS.md`](CLUSTER_EMBEDDINGS.md) and [`cluster_embeddings.sha256`](cluster_embeddings.sha256) | verifier option `--embeddings` | 100,000 × 384 float32 matrix identity |
| Camera-ready PDF | [`PAPER_VALUE_MANIFEST.json`](PAPER_VALUE_MANIFEST.json) | verifier option `--paper-pdf` | 12 pages, byte size, and SHA-256 identity |

The machine-readable numerical inventory is
[`PAPER_VALUE_MANIFEST.json`](PAPER_VALUE_MANIFEST.json). Every entry in that
manifest is checked by the main verifier.
