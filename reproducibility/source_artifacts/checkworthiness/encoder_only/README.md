# Encoder-only numerical bundle

<!-- Sérgio Pinto, 2026-08-21 21:48 PDT — added the checksum-bound non-sensitive historical aggregate extract. -->
<!-- Sérgio Pinto, 2026-08-21 22:05 PDT — recorded the failed later-checkpoint recovery without retaining benchmark text. -->

This directory supports the `Encoder Only — DeBERTa v3 Large` row in the
CIKM 2026 paper:

- `ct24_probs.npy` is the retained 341-example CT24 probability vector;
- `ct23_probs.npy` is the retained 318-example CT23 probability vector from
  the January 12 run;
- `retained_ct24_results.json` preserves the original CT24 threshold sweep;
- `ENCODER_RECOVERY_AUDIT.json` records the provenance and the ClaimBuster
  recovery boundary;
- `HISTORICAL_RUN_EXTRACT.md` preserves the non-sensitive author-recorded
  ClaimBuster aggregate and command.

The matching labels are already present in the text-free numerical matrix at
`../llm_features_classifier/matrices.npz`. Recompute the three paper-facing
F1 values with:

```bash
uv run scripts/reproduce_cikm2026_encoder_only.py
```

CT24 `0.821` and CT23 `0.928` are recomputed directly from per-example labels
and probabilities. For ClaimBuster, the command checks that the author-recorded
F1 `0.9702` and accuracy `0.9864` are arithmetically consistent with the
checksum-bound 1,032-row public benchmark. The per-example probability vector
was not retained, so ClaimBuster remains unresolved as direct reproduction even
though its historical aggregate extract is now public and checksum-bound.
Fresh inference with the later locally available seed-0 checkpoint gives F1
`0.9657` at the historical `0.65` threshold, rather than `0.9702`; the recovery
audit therefore excludes that checkpoint from the paper-value evidence.

The bundle contains only numeric probabilities, numeric metrics, and
provenance. It contains no benchmark text, post or user identifiers, profiles,
or locations. The original training and evaluation source snapshots are under
`../source_code/scripts/finetuning/` and `../source_code/scripts/evaluation/`.
