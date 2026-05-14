# Claim Normalization Source Package

This directory packages the local `claim_norm` runner, prompts, and CheckThat! 2025 CSV inputs recovered from
`/Users/sergiopinto/explainableACD`.

Important: this is runnable source material, not evidence that the paper's Table 1 values are reproduced. The exact paper table
still needs the `N=300` split, per-sample predictions, summary metrics, command, and checksums.

Packaged contents:

- `source_code/`: recovered `claim_norm` package, excluding cache and local data directories.
- `check_that_25/`: recovered CheckThat! 2025 train/dev/test CSV files used by the package.
- `comparison_test_20260113_123010.json`: local comparison artifact with `n=1285`; this does not match the paper Table 1 protocol.

Use `results/claim_normalization_filipe_request_2026-05-14.md` for the missing-artifact request.
