# Four-head benchmark predictions

<!-- Sérgio Pinto, 2026-08-21 19:32 PDT — added the exact cross-benchmark arrays so all three F1 cells can be recomputed without relying on a log summary. -->

These arrays preserve the exact label order and probabilities used for the
ClaimBuster and CT23 cells in the paper's four-head row. The CT24 arrays remain
one directory above because they are part of the original retained training run.

Run the metric-only verification from the repository root:

```bash
uv run scripts/reproduce_cikm2026_mtl_table.py
```

After downloading the checkpoint release asset, repeat the benchmark inference:

```bash
uv run scripts/reproduce_cikm2026_mtl_table.py --checkpoint /path/to/best_model.pt
```

The expected F1 values are `0.833` on CT24, `0.973` on ClaimBuster, and `0.933`
on CT23 after the paper's three-decimal rounding.
