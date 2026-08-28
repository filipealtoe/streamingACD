# CIKM 2026 reproducibility artifact

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — reduced the public guide to verified reproductions and direct reader commands. -->
<!-- Sérgio Pinto, 2026-08-25 19:35 WEST — added the performance measurements covered by the main verifier. -->
<!-- Sérgio Pinto, 2026-08-25 20:17 WEST — added the version-pinned RandomForest paper-row command and result. -->
<!-- Sérgio Pinto, 2026-08-25 21:56 WEST — labels the RandomForest result as a neutral version-pinned paper-cell reconstruction. -->
<!-- Sérgio Pinto, 2026-08-28 01:26 WEST — made hardware-specific latency reruns write only to an explicit caller-selected output. -->

This package accompanies the accepted CIKM 2026 paper **“Predicting Narrative
Virality on Social Media Streams for Pre-Peak Misinformation Triage.”** It
contains the released 529-instance virality dataset, text-free numerical
artifacts, source and prompt snapshots, retained prediction arrays, and
cryptographic identities for the large model and embedding assets.

## One-command verification

From the repository root:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The command verifies the committed files without network access. Optional local
copies of the paper and release assets can be checked with:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --paper-pdf /path/to/CIKM2026-camera-ready-2026-08-22.pdf \
  --checkpoint /path/to/best_model.pt \
  --embeddings /path/to/cluster_embeddings.npy
```

## Released dataset

The dataset is stored in
[`reproducibility/source_artifacts/virality/`](../source_artifacts/virality/):

- `features_enhanced.parquet`: 529 cluster instances, 42 engineered features, cluster ID, and PSR target;
- `psr_labels.parquet`: the same 529 cluster IDs with the engagement components and virality label;
- [`SCHEMA.md`](SCHEMA.md): field definitions;
- [`DATA_CARD.md`](DATA_CARD.md): provenance, intended use, feature availability, and privacy boundary.

The deterministic split uses `test_size=0.2`, `random_state=42`, and no
stratification, giving 423 training rows and 106 test rows.

## Numerical reproductions

| Paper component | Command | Result |
|---|---|---|
| Anomaly table | `uv run scripts/reproduce_cikm2026_anomaly_table.py` | [`results/anomaly_table_reproduction_2026-08-21/`](../../results/anomaly_table_reproduction_2026-08-21/) |
| Four-head check-worthiness row | `uv run scripts/reproduce_cikm2026_mtl_table.py` | [`mtl_table_reproduction_2026-08-21.json`](../../results/mtl_table_reproduction_2026-08-21.json) |
| LLM-feature check-worthiness row | `uv run scripts/reproduce_cikm2026_llm_features.py` | [`llm_features_table_reproduction_2026-08-21.json`](../../results/llm_features_table_reproduction_2026-08-21.json) |
| CT24 Fusion cell | `uv run scripts/reproduce_cikm2026_fusion_ct24.py` | [`fusion_ct24_reproduction_2026-08-21.json`](../../results/fusion_ct24_reproduction_2026-08-21.json) |
| CT23 Fusion cell | `uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23` | [`fusion_ct23_reproduction_2026-08-28.json`](../../results/fusion_ct23_reproduction_2026-08-28.json) |
| Six tabular fresh fits | `uv run scripts/reproduce_cikm2026_tabular_baselines.py` | [`virality_tabular_fresh_fit_2026-08-25.json`](../../results/virality_tabular_fresh_fit_2026-08-25.json) |
| RandomForest paper-cell reconstruction | `uv run --no-project scripts/reproduce_cikm2026_random_forest.py` | [`virality_random_forest_reproduction_2026-08-25/`](../../results/virality_random_forest_reproduction_2026-08-25/) |
| Virality statistics | `uv run scripts/reproduce_cikm2026_virality_statistics.py` | [`virality_statistics_reproduction_2026-08-21.json`](../../results/virality_statistics_reproduction_2026-08-21.json) |
| Four-head GPU throughput | `uv run --no-project scripts/verify_cikm2026_artifacts.py` | [`mtl_gpu_latency_audit_2026-08-21.json`](../../results/mtl_gpu_latency_audit_2026-08-21.json) |
| Numeric virality latency | `uv run --no-project scripts/benchmark_cikm2026_psr_latency.py --output /tmp/cikm2026_psr_latency.json` | [`psr_latency_benchmark_2026-08-21.json`](../../results/psr_latency_benchmark_2026-08-21.json) |

The [paper-to-artifact index](PAPER_ARTIFACT_INDEX.md) identifies every source,
input, command, and output included in the verified public package. The
[verified-value manifest](PAPER_VALUE_MANIFEST.json) records the exact rounded
paper values covered by these reproductions.

## Large release assets

The `cikm-2026-artifact-v1` GitHub release contains:

- `best_model.pt`: four-head DeBERTa checkpoint, 1,739,380,133 bytes;
- `cluster_embeddings.npy`: 100,000 × 384 float32 matrix, 153,600,128 bytes.

Their exact SHA-256 identities are in [`RELEASE_ASSETS.json`](RELEASE_ASSETS.json),
[`CHECKPOINT.md`](CHECKPOINT.md), and
[`CLUSTER_EMBEDDINGS.md`](CLUSTER_EMBEDDINGS.md).

## Artifact structure

- [`PAPER_ARTIFACT_INDEX.md`](PAPER_ARTIFACT_INDEX.md): paper component to public artifact mapping;
- [`PAPER_VALUE_MANIFEST.json`](PAPER_VALUE_MANIFEST.json): values included in the deterministic verifier;
- [`MODEL_PARAMETERS.json`](MODEL_PARAMETERS.json): executable parameter records;
- [`checksums.sha256`](checksums.sha256): data and result identities;
- [`code_checksums.sha256`](code_checksums.sha256): source, prompt, and manifest identities;
- [`prediction_checksums.sha256`](prediction_checksums.sha256): retained prediction identities;
- [`checkworthiness_checksums.sha256`](checkworthiness_checksums.sha256): check-worthiness input identities.
