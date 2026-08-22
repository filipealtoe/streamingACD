<!-- Sérgio Pinto, 2026-08-21 21:40 PDT — clarified the retained statistics package's authoritative camera-ready scope. -->

# CIKM 2026 virality statistics

This retained run is the authoritative source for the Table 3 Spearman
significance marks, 95% bootstrap confidence intervals, and exact pairwise
McNemar tests. It contains the per-example prediction vectors, their hashes,
the full-precision metrics, and the original statistical source script.

Run the portable repository-level reproducer from the repository root:

```bash
uv run scripts/reproduce_cikm2026_virality_statistics.py
```

The paper-facing metric cells are governed by
`reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json`. The retained
`TABLE3_UPDATED.tex` is a historical statistical-reporting handoff and is not
the camera-ready table source; in particular, the final RandomForest row uses
the cell-level evidence recorded in the paper-value manifest.
