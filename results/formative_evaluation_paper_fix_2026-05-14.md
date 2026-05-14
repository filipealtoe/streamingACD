# Formative Evaluation Manuscript Fix

Date: 2026-05-14

## Scope

This note defines the paper-side correction for the formative-evaluation claims in the IJCAI draft. It does not modify the
manuscript itself because no editable `.tex`, `.docx`, or source archive for the paper was found in the local repo search.

## Decision

Do not carry forward the quantitative formative-evaluation results unless an anonymized participant-response artifact and
analysis script are recovered.

The local reproducibility package supports only the study-stimulus preparation step:

- candidate report-selection script;
- selected candidate reports for high-confidence positive, negative recommendation, and borderline cases;
- generated report candidates in the prior local `explainableACD` checkout.

It does not support the paper's exact participant statistics:

- `N=7` or `N=9`;
- `27` evaluated report pairs;
- `22/27` agreement or partial agreement;
- `16/27` full agreement;
- component means `4.04`, `3.81`, `3.78`, and `3.11`;
- `8/9` report-structure alignment;
- `7/9` triage-effort reduction;
- `5/9` trust-calibration count.

## Required Manuscript Edits

### Abstract

Remove the current formative-evaluation sentence that claims professional fact-checker agreement, report-structure ratings, or
triage-effort reduction.

Preferred fix: delete the sentence entirely. If the abstract needs a replacement sentence, use:

> We release study-stimulus selection artifacts for future expert evaluation of generated claim-checking reports.

Do not describe this as a completed quantitative user study.

### Contributions

Replace the current contribution claiming a completed professional fact-checker evaluation with:

> We release the stimulus-selection procedure used to prepare a formative expert-review protocol; no quantitative human-study
> result is claimed in this version.

### Evaluation Section

Replace the quantitative formative-evaluation paragraph with:

> To support expert review of the generated reports, we prepared candidate report stimuli spanning high-confidence positive,
> negative recommendation, and borderline cases. We release the selection script and selected candidate-report tables. This
> version reports the protocol and stimuli only; it does not claim agreement percentages, usefulness means, or triage-effort
> statistics as empirical results.

### Limitations

Replace any sentence using `N=7`, `N=9`, or exact formative-evaluation percentages with:

> A quantitative expert evaluation of report usefulness remains future work; the current release provides the protocol and
> stimuli needed to conduct that evaluation.

## Evidence Used

- `/Users/sergiopinto/streamingACD/results/formative_evaluation_artifact_audit_2026-05-14.md`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/formative_evaluation/select_claims_for_study.py`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/formative_evaluation/expose_fast_study_claims_selection.csv`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv`

## Remaining Blocker

If Filipe or Sergio recovers the survey export, anonymized participant table, or aggregate analysis file, this decision can be
reopened. The required repair would be to package the response artifact, add a verifier script, resolve the `N=7` versus `N=9`
conflict, and then restore only the exact statistics that the verifier recomputes.
