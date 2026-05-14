# Formative Evaluation Artifact Audit

Date: 2026-05-14

## Finding

The local repos contain formative-evaluation stimulus artifacts, but not the participant responses or aggregate analysis needed to
reproduce the paper's human-study statistics.

## Evidence Found

| Artifact | Rows/files | What it supports |
|---|---:|---|
| `reproducibility/source_artifacts/formative_evaluation/select_claims_for_study.py` | 1 script | The intended selection logic for high-confidence, negative, and borderline report candidates |
| `reproducibility/source_artifacts/formative_evaluation/expose_fast_study_claims_selection.csv` | 59 rows | Candidate claims: 20 high-confidence positive, 19 negative recommendation, 20 borderline |
| `reproducibility/source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv` | 58 rows | Candidate claims: 20 high-confidence positive, 18 negative recommendation, 20 borderline |
| `/Users/sergiopinto/explainableACD/data/pipeline_output/.../study_reports/` | HTML report files | Generated report stimuli/candidates |

## Evidence Not Found

No anonymized participant-level response table, survey export, aggregate rating file, or analysis script was found for:

- `22/27` agree or partially agree;
- `16/27` full agreement;
- usefulness means `4.04`, `3.81`, `3.78`, `3.11`;
- `8/9` report-structure alignment;
- `7/9` triage-effort reduction;
- `5/9` trust-calibration count.

The PDF also has a participant-count conflict: abstract/limitations say `N=7`, while the body says `N=9`.

## Paper Action

Use `results/formative_evaluation_paper_fix_2026-05-14.md` to remove the unsupported quantitative claims from the manuscript, or
add an anonymized participant-response artifact plus analysis script and reopen this audit.
