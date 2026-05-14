# Current Reproducibility Status

Date: 2026-05-14

Command run:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  uv run --with pyarrow python scripts/verify_local_artifacts.py
```

Current result:

```text
Summary: 0 failures, 2 warnings, 4 info
```

## What Is Now Checked From Packaged Repository Artifacts

The verifier now prefers `reproducibility/source_artifacts/` before falling back to the old local `explainableACD` checkout.

Packaged checks that pass:

| Area | Current verifier evidence |
|---|---|
| Source artifact integrity | 56 files match `reproducibility/source_artifacts/sha256sums.txt` |
| CT24 splits | train `22402`, dev `1031`, test `341` rows |
| CT23 and ClaimBuster inputs | ClaimBuster `1032`; CT23 input/gold `318` each |
| LLM feature inputs | CT24 train/dev/test and CT23/ClaimBuster feature Parquets match expected row counts |
| Canonical pipeline summary | `1522909` tweets processed, `535` claims, `100000` clusters |
| Canonical pipeline external bundle | parquet row counts plus `cluster_embeddings.npy`, `cluster_id_to_idx.json`, and `cluster_ids_order.json` present in the local external cache |
| Clustering threshold result | threshold `0.65`: yield `86.9`, mean intra-similarity `0.8685999195826681`, `5000` tweets |
| DeBERTa Table 3 reproduced rows | ensemble `0.834`, four-head MTL `0.833`, fusion `0.836`; packaged fusion probabilities have `341` rows |
| Four-head MTL run bundle | CT24 test F1 `0.8333333333333333`; saved dev/test predictions and labels checksum clean |
| Virality/PSR artifacts | `529` feature rows, `529` label rows, `42` features, `423/106` split, packaged baseline metrics |

## Current Warnings

These are warnings, not verifier failures:

| Warning | Meaning |
|---|---|
| Claim normalization exact Table 1 pending | Filipe needs to check/recover the exact `N=300` run artifacts. |
| Formative evaluation quantitative stats pending | Filipe needs to provide anonymized participant responses and the analysis script if the paper keeps exact human-study numbers. |

## Informational Notes

These are not current paper blockers:

| Note | Meaning |
|---|---|
| CT24 LLM feature-generation checkpoints not packaged | The feature Parquets are present, complete, and checksummed. The missing checkpoint JSON files only affect run-resume provenance. |
| Raw corpus language column missing | The local raw corpus supports row count/date span, but not the old full-corpus English-share wording. Use `69.9%` for the full corpus detector estimate, or `86.6%` only for the three-day clustering-analysis subset documented in `results/clustering_subset_language_share_2026-05-14.*`. |
| Single DeBERTa F1 corrected value | The saved probability artifact supports F1 `0.8214285714285715`; use `0.821` if the baseline row is retained. |
| Four-head checkpoint omitted | `best_model.pt` is not in Git; checksum is recorded and saved predictions are packaged. |

## Immediate Interpretation

The repo now has a passing reproducibility verifier for the artifacts currently under our control. The remaining hard paper risks are
not hidden in the verifier: claim normalization and formative evaluation still depend on Filipe-owned artifacts, and the paper should
not restore those exact numbers unless those artifacts are recovered.
