# Sample Explainability Report Manuscript Fix

Date: 2026-05-14

## Scope

This note defines the paper-side correction for the sample explainability-report claim. It does not modify the manuscript itself
because no editable `.tex`, `.docx`, or source archive for the paper was found in the local repo search.

## Finding

The exact report artifact was recovered, but it does not support the paper's `567 engagements` value.

Recovered artifact:

- `reproducibility/source_artifacts/explainability_report/03_9dea2d0d_eng564.html`

Recovered claim row:

- `reproducibility/source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv`

Supported values:

- claim ID: `9dea2d0d-0604-4753-8bde-f9498ee5a2b0`;
- check-worthiness confidence: `71.7%`;
- detection lead time: `8.0h`;
- peak engagement: `564`;
- total engagement shown in report metadata: `1,172`;
- cluster size: `73 tweets`.

Unsupported value:

- `567 engagements`.

## Required Manuscript Edit

Replace any sentence saying the report had `567 engagements` with `564 peak engagements`.

If the sentence currently says:

> The report predicted virality eight hours before peak and reached 567 engagements, with 71.7% check-worthiness confidence.

Use:

> The recovered report for claim `9dea2d0d-0604-4753-8bde-f9498ee5a2b0` predicted virality `8.0` hours before its peak,
> reached `564` peak engagements, and assigned `71.7%` check-worthiness confidence.

If the surrounding paper text does not need a concrete anecdotal example, delete the sentence instead. Do not keep `567`.

## Evidence Used

- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/explainability_report/03_9dea2d0d_eng564.html`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv`

## Remaining Blocker

No blocker remains for the corrected `564` version. If the paper must keep `567`, then a different source report or calculation
must be recovered and packaged.
