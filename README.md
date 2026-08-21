# Streaming ACD

Reproducibility workspace for the streaming automated claim detection experiments.

## CIKM 2026 Artifact

This repository accompanies the accepted CIKM 2026 paper
**“Predicting Narrative Virality on Social Media Streams for Pre-Peak Misinformation Triage”** by Filipe Altoe,
Sérgio Miguel Gonçalves Pinto, and H. Sofia Pinto.

- Paper: reserved [ACM DOI 10.1145/3799682.3840688](https://doi.org/10.1145/3799682.3840688) (expected to resolve after ACM publication)
- Conference: [35th ACM International Conference on Information and Knowledge Management (CIKM 2026)](https://cikm2026.diag.uniroma1.it/)
- Artifact guide: [`reproducibility/cikm2026/README.md`](reproducibility/cikm2026/README.md)
- Dataset: [`reproducibility/source_artifacts/virality/`](reproducibility/source_artifacts/virality/)

Verify the public CIKM artifact from a fresh clone with:

```bash
uv run scripts/verify_cikm2026_artifacts.py
```

The focused verifier checks dataset integrity, the deterministic split, documented missingness, and the paper-facing result
artifacts. It does not claim to rerun the full streaming pipeline or models whose raw inputs or checkpoints are not public.

Freshly fit the six numeric baselines that depend only on the released matrix with:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

This reproduces Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR metrics and compares them with the committed result
artifact. Extended audit inputs and outputs remain local-only until the authors complete the relevant privacy and source-rights
decisions.

Current focus:

- rebuild the CT24 check-worthiness baselines from source data;
- save per-sample prediction artifacts for paired comparisons;
- document exact commands, thresholds, seeds, package versions, and dataset checksums;
- keep paper-facing metrics tied to regenerable code and committed manifests.

Exact result values should only be used in a submission after they are regenerated or verified from saved artifacts in this repository.

## Current Reproduced Run

The four-head multi-task DeBERTa run is packaged at:

`reproducibility/runs/deberta_mtl_cikm_20260512_134553/`

The reproduced CT24 best test F1 is `0.8333` at threshold `0.50`. Cross-dataset evaluation from the same checkpoint gives
ClaimBuster F1 `0.9730` and CT23 F1 `0.9327`.

The model checkpoint is not committed because it is 1.7 GB. Its SHA-256 is recorded in the run bundle.

## Reproducibility Package

The `reproducibility/` directory contains the current local artifact manifest and checksums. Large datasets, embeddings, model files, and prediction arrays are not committed by default; they are referenced by path, size, and SHA-256 until they are regenerated or placed in external storage.

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
