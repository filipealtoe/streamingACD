# Reproducibility Status

Date: 2026-05-12

## Current Position

The four-head multi-task check-worthiness result has been rerun and packaged. The CIKM draft can now cite a reproduced CT24 result
for the four-head architecture, provided the paper states the threshold protocol clearly.

The rest of the paper should still be treated as partially reproduced. Several exact claims have local artifacts in the previous
checkout, but not all of them are packaged into this repository with a clean command and checksum trail.

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
4. Fusion Classifier and LLM-feature rows in Table 3 if the paper keeps those rows.

## Table 3 Audit

The Table 3 audit is packaged at:

`results/table3_reproduction_2026-05-12.*`

Current result:

- single DeBERTa: F1 `0.8214`, near the paper-facing `0.8242` value;
- three-seed DeBERTa ensemble: F1 `0.8343`, reproduced;
- four-head MTL retrain: F1 `0.8333`, reproduced;
- saved local LLM-feature artifacts: F1 `0.7135` and `0.7200`, not the paper-facing `0.761` value;
- saved local fusion artifacts: best found F1 `0.8295`, not the paper-facing `0.8362` value.

The fusion and LLM-feature rows should not be used as exact final paper claims unless the documented XGBoost v4 protocol is rerun and
packaged.

## Next Slice

Attack one of the still-unpackaged paper claims:

- claim normalization Table 1 exact METEOR values;
- anomaly detection NAB/detection/lead-time values;
- formative evaluation aggregate data and analysis.
