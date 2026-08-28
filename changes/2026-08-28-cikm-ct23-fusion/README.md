# CIKM 2026 CT23 Fusion reproduction

Author: Sérgio Pinto
Timestamp: 2026-08-28 01:11 WEST (+0100)
Reason: connect the paper's CT23 Fusion value to a complete, checksum-bound
public reconstruction and its full-precision result.

## Change

- Added CT23 Fusion `0.915094`, reported as `0.915`, to the paper-value
  manifest and paper-to-artifact index.
- Bound the reconstruction to the two retained encoder probability vectors,
  nine ordered LLM features, XGBoost 2.1.1, temperature `0.7`, Fusion weights
  `0.6/0.4`, and threshold `0.5`.
- Added a deterministic full-precision result file and main-verifier checks.
- Added XGBoost 2.1.1 to the public project dependencies so the documented
  command runs from the repository root.

## Verification

From the repository root:

```bash
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
uv run pytest -q tests/test_cikm2026_checkworthiness_lambda.py \
  tests/test_cikm2026_public_artifact.py
uv run scripts/verify_cikm2026_artifacts.py
```
