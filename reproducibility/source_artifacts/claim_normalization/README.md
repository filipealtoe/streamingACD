# Claim-normalization source snapshot

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — retained the exact source and prompt material for a fresh claim-normalization run. -->

This directory contains the source snapshot and prompt configurations used by
the claim-normalization experiments:

- `source_code/scripts/run_claim_normalization_ct25.py`;
- `source_code/scripts/finetune_claim_normalizer.py`;
- `source_code/scripts/tune_fewshot_hyperparams.py`;
- `source_code/prompts/claim_normalization.yaml`;
- `source_code/prompts/claim_normalization_deepseek.yaml`.

The source identities are bound by
`reproducibility/cikm2026/code_checksums.sha256`. Generated outputs should record
the model revision, prompt hash, input hash, environment, per-example
predictions, and full-precision METEOR values.
