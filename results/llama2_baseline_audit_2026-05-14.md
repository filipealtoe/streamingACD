# Llama2 Baseline Row Audit

Date: 2026-05-14

## Finding

The Table 3 row labeled `Baseline Llama2-7b Fine-tuned` with values `0.802 / 0.920 / 0.898` is not one reproduced model row.
It combines external reference values from different sources.

## Evidence

| Value | Current paper label | Evidence found | Interpretation |
|---|---|---|---|
| CT2024 `0.802` | `Llama2-7b Fine-tuned` | CheckThat! 2024 official results list FactFinders at `0.802`; the FactFinders paper reports fine-tuned open LLM experiments including Llama2-7b. Local `run_together_baselines.py` stores Llama2 under `PAPER_BASELINES`, not under runnable model outputs. | External prior-work baseline, not our rerun |
| ClaimBuster `0.920` | `Llama2-7b Fine-tuned` | Local benchmark scripts store `0.920` as `G2CW (GloVe+GRU)` / ClaimBuster SOTA. | External SOTA reference, not Llama2 |
| CT2023 `0.898` | `Llama2-7b Fine-tuned` | Local benchmark scripts store `0.898` as `OpenFact (CEUR-WS Vol-3497)` / CT23 SOTA. | External SOTA reference, not Llama2 |

The paired Together prediction parquet for the 2025-12-29 run contains 600 rows: `llama-3.1-8b`, `llama-3.1-70b`,
`llama-3.3-70b`, `llama-3.2-3b`, `mistral-7b`, and `mixtral-8x7b`, each with 100 rows. It contains no Llama2 rows.

## Sources

- FactFinders / CheckThat! 2024 prior: https://arxiv.org/abs/2406.18297
- CEUR paper PDF: https://ceur-ws.org/Vol-3740/paper-47.pdf
- CheckThat! 2024 Task 1 results page: https://checkthat.gitlab.io/clef2024/task1/
- Packaged local metadata: `reproducibility/source_artifacts/checkworthiness/together_baselines_metrics_20251229_194021.json`
- Packaged runner: `reproducibility/source_artifacts/checkworthiness/together_baselines/run_together_baselines.py`

## Paper Action

Do not keep `0.802 / 0.920 / 0.898` as one `Llama2-7b Fine-tuned` row. Either split the row into separate external references
per dataset, or relabel it as a prior-work/SOTA comparison row with explicit citations.
