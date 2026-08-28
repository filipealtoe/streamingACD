# Streaming ACD

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — focused the public camera-ready package on checksum-bound datasets, source, model assets, and verified numerical reproductions. -->
<!-- Sérgio Pinto, 2026-08-25 20:17 WEST — added the version-pinned RandomForest paper-row reproduction to the public commands. -->
<!-- Sérgio Pinto, 2026-08-25 21:56 WEST — clarified that the RandomForest command reconstructs published cells in neutral version-pinned environments. -->

This repository accompanies the accepted CIKM 2026 paper **“Predicting
Narrative Virality on Social Media Streams for Pre-Peak Misinformation
Triage”** by Filipe Altoe, Sérgio Miguel Gonçalves Pinto, and H. Sofia Pinto.

- Paper: [ACM DOI 10.1145/3799682.3840688](https://doi.org/10.1145/3799682.3840688)
- Conference: [35th ACM International Conference on Information and Knowledge Management (CIKM 2026)](https://cikm2026.diag.uniroma1.it/)
- Artifact guide: [`reproducibility/cikm2026/README.md`](reproducibility/cikm2026/README.md)
- Paper-to-artifact index: [`reproducibility/cikm2026/PAPER_ARTIFACT_INDEX.md`](reproducibility/cikm2026/PAPER_ARTIFACT_INDEX.md)
- Verified-value manifest: [`reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json`](reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json)
- Released virality dataset: [`reproducibility/source_artifacts/virality/`](reproducibility/source_artifacts/virality/)
- Four-head checkpoint: [`reproducibility/cikm2026/CHECKPOINT.md`](reproducibility/cikm2026/CHECKPOINT.md)
- Cluster embeddings: [`reproducibility/cikm2026/CLUSTER_EMBEDDINGS.md`](reproducibility/cikm2026/CLUSTER_EMBEDDINGS.md)

## Verify the artifact

From a fresh clone:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The verifier checks checksums, dataset structure, privacy boundaries, source and
prompt snapshots, parameter manifests, and every numerical reproduction listed
in the verified-value manifest.

The two large release assets can be checked by identity when downloaded:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --checkpoint /path/to/best_model.pt \
  --embeddings /path/to/cluster_embeddings.npy
```

The exact camera-ready PDF can also be checked without adding it to the
repository:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --paper-pdf /path/to/CIKM2026-camera-ready-2026-08-22.pdf
```

## Main reproductions

```bash
uv run scripts/reproduce_cikm2026_anomaly_table.py
uv run scripts/reproduce_cikm2026_mtl_table.py
uv run scripts/reproduce_cikm2026_llm_features.py
uv run scripts/reproduce_cikm2026_fusion_ct24.py
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
uv run scripts/reproduce_cikm2026_tabular_baselines.py
uv run --no-project scripts/reproduce_cikm2026_random_forest.py
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

The complete command and artifact mapping is maintained in the
[CIKM 2026 artifact guide](reproducibility/cikm2026/README.md).
