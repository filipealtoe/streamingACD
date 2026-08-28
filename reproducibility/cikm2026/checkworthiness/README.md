# CIKM 2026 claim check-worthiness reproduction

<!-- Sérgio Pinto, 2026-08-28 03:38 WEST — linked the fresh A10 ClaimBuster Encoder Only and Fusion reproduction to the public artifact entrypoint. -->

The accepted paper's task and reported values are recorded in
[`PAPER_PROTOCOL.json`](PAPER_PROTOCOL.json). The public package includes:

- numerical soft-label files for CT24, ClaimBuster, and CT23;
- text-free CT24 sentence IDs and labels;
- immutable source snapshots and prompts;
- retained Encoder Only, LLM-feature, four-head, and Fusion predictions;
- the four-head checkpoint identity and exact run bundle.

Run the verified numerical paths with:

```bash
uv run scripts/reproduce_cikm2026_mtl_table.py
uv run scripts/reproduce_cikm2026_llm_features.py
uv run scripts/reproduce_cikm2026_fusion_ct24.py
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-claimbuster
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The ClaimBuster command verifies the fresh A10 seed-42 vector against the
public benchmark order, reconstructs the four-confidence XGBoost component,
and reproduces Encoder Only F1 `0.970085` and Fusion F1 `0.961373`. It writes
the full-precision metrics and text-free per-example predictions to
`results/claimbuster_encoder_fusion_reproduction_2026-08-28/`.

The CT23 command reconstructs F1 `0.915094` from the two retained encoder
probability vectors and the nine-feature XGBoost input, and writes the
full-precision result to `results/fusion_ct23_reproduction_2026-08-28.json`.

The historical GPU path uses Python 3.10, PyTorch 2.7.0 with CUDA 12.8,
Transformers 4.44.0, and an NVIDIA A10. Install the pinned environment before
running `preflight`, `train --track historical`, and
`evaluate --track historical`; the runner verifies that machine contract
before training starts. The complete operator sequence is in
[`LAMBDA_RUN.md`](LAMBDA_RUN.md), and its machine-readable configuration is in
[`LAMBDA_RUN_SPEC.json`](LAMBDA_RUN_SPEC.json).

The released numerical tables contain benchmark sentence IDs, labels,
probabilities, and engineered features. They contain no Twitter/X post or user
identifiers, usernames, profiles, or locations.
