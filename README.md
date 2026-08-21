# Streaming ACD

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
- Dataset: [`reproducibility/source_artifacts/virality/`](reproducibility/source_artifacts/virality/)
- Four-head checkpoint: [`reproducibility/cikm2026/CHECKPOINT.md`](reproducibility/cikm2026/CHECKPOINT.md)

Verify the public CIKM artifact from a fresh clone with:

```bash
uv run scripts/verify_cikm2026_artifacts.py
```

The focused verifier checks dataset integrity, the deterministic split, documented
missingness, paper-facing results, source and prompt provenance, exact parameter
values, and checkpoint release metadata. Pass `--checkpoint` with the downloaded
release asset to verify its size and SHA-256 as well.

Freshly fit the six numeric baselines that depend only on the released matrix with:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

This reproduces Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR metrics
and compares them with the committed full-precision result artifact.

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

Run the current local verifier with:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/verify_local_artifacts.py
```

Recompute CT24 metrics from saved probability artifacts with:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/recompute_ct24_metrics.py
```

The paper-claim status matrix is tracked in `docs/paper_claim_reproduction_matrix.md`.
