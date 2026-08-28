# ClaimBuster Encoder Only and Fusion reproduction

<!-- Sérgio Pinto, 2026-08-28 03:38 WEST — published the checksum-bound output of the fresh A10 ClaimBuster reproduction. -->

This bundle reproduces the accepted paper's ClaimBuster check-worthiness cells
from the fresh A10 `seed_0` run (effective RNG seed 42) and the recovered
four-confidence XGBoost inputs.

Run from the repository root:

```bash
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-claimbuster
```

The command verifies the benchmark order and input hashes, then writes:

- `summary.json`: full-precision metrics, thresholds, model identity and method;
- `per_example_predictions.npz`: sentence IDs, labels, and the Encoder Only,
  XGBoost and Fusion probability vectors;
- `checksums.sha256`: SHA-256 identities for both outputs.

The reproduced F1 values are `0.970085` for Encoder Only and `0.961373` for
Fusion, which round to the paper values `0.970` and `0.961`.
