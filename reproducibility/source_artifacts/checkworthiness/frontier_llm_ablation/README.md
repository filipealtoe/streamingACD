# Frontier-LLM ablation evidence

<!-- Sérgio Pinto, 2026-08-21 20:18 PDT — added an explicit recovery boundary and aggregate arithmetic audit for the Opus ablation. -->

The paper reports Claude Opus 4.7 on the 341-row CT24 test-gold split with
three calls per row. The reported aggregate values determine a unique confusion
matrix against the retained 88-positive label vector, and the reported McNemar
statistic and p-value determine a unique aggregate paired-correctness table
against the retained four-head result.

Audit that arithmetic with:

```bash
uv run scripts/audit_cikm2026_frontier_llm_ablation.py
```

This is an aggregate consistency audit, not a fresh Opus inference run. The
original 1,023 API responses and per-example Opus predictions were not found in
the recovered workspace. The command therefore reports `UNRESOLVED` even when
the aggregate arithmetic passes. Repeating those paid API calls requires an
explicit author decision and would produce a new run rather than recover the
historical responses.
