# Streaming ACD

<!-- Sérgio Pinto, 2026-08-21 21:53 PDT — replaced machine-local commands with the public camera-ready verification entry points. -->
<!-- Sérgio Pinto, 2026-08-21 22:08 PDT — documented optional identity verification for the exact audited camera-ready PDF. -->

Reproducibility workspace for the streaming automated claim detection experiments.

## CIKM 2026 Artifact

This repository accompanies the accepted CIKM 2026 paper
**“Predicting Narrative Virality on Social Media Streams for Pre-Peak Misinformation Triage”** by Filipe Altoe,
Sérgio Miguel Gonçalves Pinto, and H. Sofia Pinto.

- Paper: reserved [ACM DOI 10.1145/3799682.3840688](https://doi.org/10.1145/3799682.3840688) (expected to resolve after ACM publication)
- Conference: [35th ACM International Conference on Information and Knowledge Management (CIKM 2026)](https://cikm2026.diag.uniroma1.it/)
- Artifact guide: [`reproducibility/cikm2026/README.md`](reproducibility/cikm2026/README.md)
- Paper-to-artifact index: [`reproducibility/cikm2026/PAPER_ARTIFACT_INDEX.md`](reproducibility/cikm2026/PAPER_ARTIFACT_INDEX.md)
- Model parameters: [`reproducibility/cikm2026/MODEL_PARAMETERS.json`](reproducibility/cikm2026/MODEL_PARAMETERS.json)
- Paper-value manifest: [`reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json`](reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json)
- Dataset: [`reproducibility/source_artifacts/virality/`](reproducibility/source_artifacts/virality/)
- Four-head checkpoint: [`reproducibility/cikm2026/CHECKPOINT.md`](reproducibility/cikm2026/CHECKPOINT.md)
- Cluster-embedding asset manifest: [`reproducibility/cikm2026/CLUSTER_EMBEDDINGS.md`](reproducibility/cikm2026/CLUSTER_EMBEDDINGS.md)

Verify the public CIKM artifact from a fresh clone with:

```bash
uv run scripts/verify_cikm2026_artifacts.py
```

The global verifier checks dataset integrity, the deterministic split, documented
missingness, paper-facing results, source and prompt provenance, exact parameter
values, and checkpoint release metadata. Pass `--checkpoint` with the downloaded
release asset to verify its size and SHA-256 as well. It reports every supported
and unresolved claim separately and exits non-zero while any audited paper claim
or release promise remains unsupported.

To confirm that the inventory targets the same camera-ready file, pass the PDF:

```bash
uv run scripts/verify_cikm2026_artifacts.py --paper-pdf /path/to/CIKM2026.pdf
```

Reproduce all 66 Table 3 numeric cells and the complete statistical annotations:

```bash
uv run scripts/verify_cikm2026_virality_table.py
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

The first command checks all 11 rows and 66 cells. The second recomputes all
Spearman p-values, 95% bootstrap confidence intervals, significance marks, and
the three exact McNemar tests from retained per-example arrays.

## Current Reproduced Run

The four-head multi-task DeBERTa run is packaged at:

`reproducibility/runs/deberta_mtl_cikm_20260512_134553/`

The reproduced CT24 best test F1 is `0.8333` at threshold `0.50`. Cross-dataset evaluation from the same checkpoint gives
ClaimBuster F1 `0.9730` and CT23 F1 `0.9327`.

The 1.7 GB model checkpoint is distributed through the
`cikm-2026-artifact-v1` GitHub release. Its expected byte size and SHA-256 are recorded
in [`CHECKPOINT.md`](reproducibility/cikm2026/CHECKPOINT.md).

## Reproducibility Package

The `reproducibility/` directory contains the public manifests, source snapshots,
retained run artifacts, and checksums. Large model files are distributed as release
assets and bound to the repository by byte size and SHA-256.

Audit the exact boundary between the public scientific core and the broader
artifact set described in the manuscript with:

```bash
uv run scripts/audit_cikm2026_release_coverage.py
```

The detailed public commands, evidence boundaries, and paper-to-artifact links
are maintained in the [artifact guide](reproducibility/cikm2026/README.md).
