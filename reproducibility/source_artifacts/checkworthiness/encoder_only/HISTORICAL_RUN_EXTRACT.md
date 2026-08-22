<!-- Sérgio Pinto, 2026-08-21 21:48 PDT — preserved the non-sensitive ClaimBuster aggregate record without benchmark text or per-example identifiers. -->

# Encoder-only ClaimBuster historical aggregate

This is a non-sensitive extract from Sérgio Pinto's author design record
`design_approaches_2026-01-17_Sergio.md` (SHA-256
`aaa0e65f80f063cbe53cc089a5eab77240ce1ec6de02628c2f4fddda67b6e8ec`,
filesystem modification time `2026-01-17T10:48:07-0800`). It makes the
historical aggregate record public and checksum-bound; it does not replace the
missing per-example probability vector.

- Record date: 2026-01-17
- Benchmark: ClaimBuster groundtruth
- Rows: 1,032
- Positive labels: 238
- Model: DeBERTa seed 0
- Threshold: 0.65
- F1: 0.9702
- Confusion counts: TP 228, FP 4, FN 10, TN 790

The recorded command was:

```bash
python experiments/scripts/evaluate_deberta_claimbuster.py \
  --model-path lambda_backup/ubuntu/ensemble_results/seed_0/deberta-v3-large/best_model \
  --auto-threshold
```

This extract contains no benchmark text, prediction vector, post/user
identifier, profile, or location.
