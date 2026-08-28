# Encoder-only numerical bundle

<!-- Sérgio Pinto, 2026-08-28 03:38 WEST — added the fresh A10 ClaimBuster seed-42 probabilities and linked their exact paper-cell reconstruction. -->

This directory contains Encoder Only numerical predictions for:

- CT24: 341 examples, F1 `0.821` at threshold `0.50`;
- ClaimBuster: 1,032 examples, F1 `0.970085` at threshold `0.65`;
- CT23: 318 examples, F1 `0.928` at threshold `0.50`.

`claimbuster_seed_42_predictions.npz` contains the benchmark sentence IDs,
labels and fresh A10 probabilities from the historical `seed_0` configuration
(effective RNG seed 42). The corresponding model SHA-256 is
`3765638fb1f60a87741fdd6c576faeece8be1cb520d074d08e7e2abe8c3feb0f`.
Together with the recovered four-confidence inputs, it reproduces the paper's
ClaimBuster Encoder Only `0.970` and Fusion `0.961` cells:

```bash
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-claimbuster
```

The CT23 probabilities in `ct23_probs.npy` are the retained `seed_0` run
(effective RNG seed 42). `ct23_seed_456_probs.json` preserves the second
retained run. Together with the released nine-feature matrices, they
recompute the CT23 Fusion value `0.915094`, reported as `0.915` in the paper:

```bash
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
```

`retained_ct24_results.json` preserves the CT24 threshold sweep. Matching
text-free labels are distributed in the check-worthiness artifact bundle. The
main artifact verifier recomputes both F1 values directly from labels and
probabilities:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The files contain numerical probabilities and metrics only. They contain no
post text, Twitter/X post or user identifiers, usernames, profiles, or
locations.
