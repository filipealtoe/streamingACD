# Claim Normalization Table 1 Handoff

Date: 2026-05-14

## Scope

This note defines the status and handoff for the claim-normalization Table 1 claim. It does not modify the manuscript itself
because no editable `.tex`, `.docx`, or source archive for the paper was found in the local repo search.

## Finding

The paper's Table 1 claim is not reproduced from the artifacts currently packaged in this repo.

Paper-facing claim:

- CT2025 test set size: `N=300`;
- Approach 2 METEOR values around `0.5583`, `0.5463`, and `0.5691`.

Packaged artifact:

- `reproducibility/source_artifacts/claim_normalization/comparison_test_20260113_123010.json`

What the packaged artifact supports:

- sample size: `n=1285`, not `N=300`;
- best available average METEOR: `0.3449` from `deepseek-v3`;
- no saved Table 1 prediction file or summary matching the paper values.

## Decision

Treat this as Filipe-owned until the exact Table 1 protocol and artifacts are recovered or rerun.

Do not keep the Table 1 values `0.5583`, `0.5463`, or `0.5691` as reproduced unless the exact `N=300` prediction artifacts and
summary files are provided.

## Required Filipe Artifact Package

To restore the Table 1 claim, Filipe needs to provide or rerun:

1. the exact CT2025 `N=300` test split used for Table 1;
2. the command/script used to generate Approach 1/2/3 outputs;
3. the exact prompt/config/model identifiers;
4. per-sample predictions for each reported approach;
5. a summary JSON or CSV recomputing METEOR from the per-sample predictions;
6. checksums for the split, predictions, and summary outputs.

Minimum acceptance rule:

```text
script + input split + per-sample predictions -> recomputed METEOR values matching Table 1 after rounding
```

## Manuscript Fallback

If the exact Table 1 artifacts cannot be recovered before submission, remove the claim-normalization Table 1 result or rewrite it
around the packaged result boundary:

> In a broader `n=1285` diagnostic comparison, the best packaged claim-normalization run reaches average METEOR `0.345`; this
> diagnostic is not the `N=300` Table 1 result from the earlier draft.

Use that fallback only if the paper wants to report the diagnostic. Otherwise, delete Table 1.

## Message To Filipe

> Filipe, the claim-normalization Table 1 values are still not reproducible in the repo. The only packaged artifact I found is
> `comparison_test_20260113_123010.json`, which has `n=1285` and best METEOR `0.3449`, not the paper's `N=300` values
> `0.5583/0.5463/0.5691`. Can you either send the exact Table 1 split, per-sample predictions, summary file, and command, or rerun
> the exact Table 1 protocol and save those artifacts? Without that, we should remove or rewrite Table 1.

## Evidence Used

- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/claim_normalization/comparison_test_20260113_123010.json`
- `/Users/sergiopinto/sergio-os/tasks/cikm2026/evidence_manifest_2026-05-11.md`
- `/Users/sergiopinto/explainableACD/experiments/results/claim_normalization/`
- `/Users/sergiopinto/explainableACD/experiments/scripts/run_claim_normalization_ct25.py`

## Remaining Blocker

The exact `N=300` Table 1 run artifact is missing.
