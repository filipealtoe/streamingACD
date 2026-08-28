# Encoder-only numerical bundle

<!-- Sérgio Pinto, 2026-08-28 01:11 WEST — added the retained second-run probabilities needed for the CT23 Fusion paper-cell reconstruction. -->

This directory contains retained Encoder Only probabilities for:

- CT24: 341 examples, F1 `0.821` at threshold `0.50`;
- CT23: 318 examples, F1 `0.928` at threshold `0.50`.

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
