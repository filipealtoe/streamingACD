# CIKM 2026 virality statistics

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — documented the retained statistical inputs and portable reproduction command. -->

This run stores the Table 3 prediction vectors, their hashes, full-precision
metrics, and the statistical source script. It supports all eleven Spearman
tests and confidence intervals plus the three exact pairwise McNemar tests.

Run from the repository root:

```bash
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

The portable output is
`results/virality_statistics_reproduction_2026-08-21.json`.
