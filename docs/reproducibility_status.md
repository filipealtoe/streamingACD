# Reproducibility Status

Date: 2026-05-12

## Current Position

The four-head multi-task check-worthiness result has been rerun and packaged. The CIKM draft can now cite a reproduced CT24 result
for the four-head architecture, provided the paper states the threshold protocol clearly.

The rest of the paper should still be treated as partially reproduced. Several exact claims have local artifacts in the previous
checkout, but not all of them are packaged into this repository with a clean command and checksum trail. Table 3 is materially
stronger now: Fusion Classifier has been rerun to the paper-facing value, while the PCA/LLM/LogReg row has been separated into the
parts that match and the CT24 cell that does not.

## Completed

### Four-head check-worthiness

Packaged run:

`reproducibility/runs/deberta_mtl_cikm_20260512_134553/`

Main results:

- CT24 best test F1: `0.8333333333333333` at threshold `0.50`
- CT24 test accuracy: `0.9120234604105572`
- CT24 test precision: `0.8152173913043478`
- CT24 test recall: `0.8522727272727273`
- ClaimBuster F1: `0.9730`
- CT23 F1: `0.9327`

Included artifacts:

- training/evaluation scripts used for the run;
- command and environment files;
- training and benchmark logs;
- CT24 dev/test probability arrays;
- CT24 dev/test labels and `Sentence_id` arrays in the exact row order used by the MTL join;
- CT24 results JSON;
- cross-dataset benchmark summary;
- SHA-256 manifests.

The 1.7 GB `best_model.pt` checkpoint is not committed. Its checksum is recorded in the run bundle.

Important detail: the four-head MTL probabilities must be evaluated against the packaged `test_labels.npy`, not the older
`CT24_clean` row order. The MTL script evaluates after joining CT24 feature rows with LLM feature rows, and that join order differs
from the older baseline artifacts.

## Still Open

See `docs/paper_claim_reproduction_matrix.md` for the full claim matrix.

Highest-risk open items:

1. claim normalization Table 1 exact METEOR values;
2. anomaly detection NAB/detection/lead-time values;
3. formative evaluation aggregate data and analysis;
4. the CT24 `0.761` LLM-feature cell in Table 3 if the paper keeps that exact value.

## Full Paper Claim Audit

A paper-level audit script is now packaged:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/audit_full_paper_claims.py
```

The audit writes `results/full_paper_claim_audit_2026-05-12.*` and intentionally exits non-zero while exact paper claims remain
unreproduced. The current blocking categories are not hidden:

- claim normalization Table 1 does not match the available local comparison artifact;
- anomaly detection paper values do not match the packaged EXPoSE grid-search artifact;
- the `87% English` corpus claim conflicts with the canonical run summary and no raw language column exists;
- Table 3 still contains exact mismatches unless the paper uses the reproduced four-head and LLM-row values;
- the formative evaluation aggregate data and exact sample-report example are not packaged.

## Table 3 Audit

The Table 3 audit is packaged at:

`results/table3_reproduction_2026-05-12.*`

Current result:

- single DeBERTa: F1 `0.8214`, near the paper-facing `0.8242` value;
- three-seed DeBERTa ensemble: F1 `0.8343`, reproduced;
- four-head MTL retrain: F1 `0.8333`, reproduced;
- PCA-64 + LLM + text LogReg CT24 rerun: F1 `0.6936`, not the paper-facing `0.761` value;
- Fusion Classifier rerun: F1 `0.8362`, reproduced from the saved T=`0.3` ensemble probabilities and a fresh XGBoost v4 LLM-feature component.

The Fusion Classifier row now has a reproducible CT24 rerun. The PCA/LLM/LogReg paper row is still not clean as written: the literal
benchmark rerun matches ClaimBuster `0.8939` and CT23 `0.8458`, but CT24 recomputes at `0.6936`.

There is also a separate replacement candidate, not a reproduction of the paper row as labeled:
`results/deberta_cls_llm_logreg_candidate_2026-05-12.md` records a targeted held-out CT24 rerun of DeBERTa CLS embeddings plus
seven LLM features with tuned Logistic Regression at F1 `0.7929`.

## Next Slice

Attack one of the still-unpackaged or still-mismatched paper claims:

- the CT24 `0.761` LLM-feature cell if the row remains in the paper;
- claim normalization Table 1 exact METEOR values;
- anomaly detection NAB/detection/lead-time values;
- formative evaluation aggregate data and analysis.
