# Single DeBERTa Baseline Paper Fix

Date: 2026-05-14

## Issue

The old table value for the Single DeBERTa CT24 baseline was `0.8242`. The saved local probability artifact recomputes to
F1 `0.8214285714285715` at threshold `0.50`.

## Evidence

- Probability artifact:
  `/Users/sergiopinto/explainableACD/experiments/results/deberta_checkworthy/deberta-v3-large/test_probs.npy`
- Labels:
  `/Users/sergiopinto/explainableACD/data/processed/CT24_clean/CT24_test_clean.parquet`
- Reproduction report:
  `results/table3_reproduction_2026-05-12.json`
- Probability SHA-256:
  `f27e63143bd4fb0aa448c1bc034a4b13533544819721c75f29429a254e7d3c84`

Computed metrics:

| Metric | Value |
|---|---:|
| Threshold | `0.50` |
| Accuracy | `0.9120234604105572` |
| Precision | `0.8625` |
| Recall | `0.7840909090909091` |
| F1 | `0.8214285714285715` |

## Manuscript Replacement

If the Single DeBERTa baseline row is retained, report CT24 F1 as `0.821` instead of `0.824`.

Do not use the old `0.8242` value unless a separate exact probability artifact or run log is recovered.
