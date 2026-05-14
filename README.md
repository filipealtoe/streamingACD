# Streaming ACD

Reproducibility workspace for the streaming automated claim detection experiments.

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
  uv run --with pyarrow python scripts/verify_local_artifacts.py
```

The current verifier status is summarized in `results/current_reproducibility_status.md`.

Recompute CT24 metrics from saved probability artifacts with:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/recompute_ct24_metrics.py
```

The paper-claim status matrix is tracked in `docs/paper_claim_reproduction_matrix.md`.
