# CIKM 2026 ClaimBuster historical input recovery

Author: Sérgio Pinto
Timestamp: 2026-08-28 03:38 WEST (+0100)
Reason: restore the exact four-confidence benchmark inputs and bind the
ClaimBuster Encoder Only and Fusion cells to the fresh A10 replication.

## Change

- Restored the exact ClaimBuster and CT23 four-confidence parquet files by
  their original SHA-256 identities.
- Bound the fresh A10 `seed_0` run (effective RNG seed 42) to the ClaimBuster
  benchmark order and model SHA-256 identity.
- Reproduced Encoder Only F1 `0.970085` at threshold `0.65`.
- Reproduced Fusion F1 `0.961373` from the seed-42 encoder vector and recovered
  four-confidence XGBoost component at threshold `0.60`.
- Preserved the separately reproduced nine-feature CT23 Fusion path.
- Added a regression test for the retained ClaimBuster LLM-only F1 `0.633333`.

## Verification

From the repository root:

```bash
uv run pytest -q tests/test_cikm2026_checkworthiness_lambda.py
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-claimbuster
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
uv run --no-project scripts/verify_cikm2026_artifacts.py
```
