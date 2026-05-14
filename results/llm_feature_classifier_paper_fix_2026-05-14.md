# LLM-Feature Classifier Manuscript Fix

Date: 2026-05-14

## Scope

This note defines the paper-side correction for the Table 3 row labeled `LLM_features PCA 64 + Logreg`. It does not modify the
manuscript itself because no editable `.tex`, `.docx`, or source archive for the paper was found in the local repo search.

## Finding

The runnable PCA-64 + LLM + text Logistic Regression path does not reproduce the paper's CT24 `0.761` value.

Supported rerun values for the literal PCA-64 + LLM + text Logistic Regression path:

| Dataset | Supported F1 | Paper-facing value |
|---|---:|---:|
| CT24 | `0.694` | `0.761` |
| ClaimBuster | `0.894` | `0.894` |
| CT23 | `0.846` | `0.846` |

The CT24 `0.761` number should not be carried forward under the old `LLM_features PCA 64 + Logreg` wording.

## Required Manuscript Edit

If the paper keeps this row as the literal rerun-supported baseline, replace the row with:

| Row label | CT24 | ClaimBuster | CT23 |
|---|---:|---:|---:|
| PCA-64 + LLM + Text (LogReg) | `0.694` | `0.894` | `0.846` |

Suggested nearby prose:

> The reproducible PCA-64 + LLM + text Logistic Regression baseline reaches F1 `0.694` on CT24, while retaining the previously
> documented rounded ClaimBuster and CT23 values of `0.894` and `0.846`; we therefore do not use the unsupported CT24 `0.761`
> value from the earlier manuscript.

## Replacement Candidate Boundary

A separate DeBERTa CLS + LLM Logistic Regression diagnostic reached CT24 F1 `0.7929`, but it is not a reproduction of the
`LLM_features PCA 64 + Logreg` row.

Use that value only if the method and row are rewritten around the replacement method:

| Row label | CT24 |
|---|---:|
| DeBERTa CLS + LLM Features (LogReg) | `0.793` |

Do not silently replace `0.761` with `0.793` under the old PCA-64 row label.

## Evidence Used

- `/Users/sergiopinto/streamingACD/results/pca64_llm_text_benchmark_2026-05-12.md`
- `/Users/sergiopinto/streamingACD/results/table3_reproduction_2026-05-12.md`
- `/Users/sergiopinto/streamingACD/results/table3_reproduction_2026-05-12.json`
- `/Users/sergiopinto/streamingACD/results/deberta_cls_llm_logreg_candidate_2026-05-12.md`
- `/Users/sergiopinto/sergio-os/tasks/cikm2026/table3_manuscript_revision_2026-05-12.md`

## Remaining Blocker

No blocker remains for the conservative `0.694/0.894/0.846` correction. If the paper needs a stronger CT24 baseline, the
replacement candidate can be used only after the table row and method text are rewritten to match that classifier.
