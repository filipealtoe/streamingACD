# PCA-64 + LLM + Text LogReg Benchmark Rerun

Date: 2026-05-12

Command run from `/Users/sergiopinto/explainableACD`:

```bash
uv run python experiments/scripts/benchmark_pca_llm_text.py
```

The script trains Logistic Regression on CT24 train+dev features, then evaluates CT24 test, ClaimBuster, and CT23 with threshold
search as implemented in the source script.

| Dataset | Rerun F1 | Precision | Recall | Threshold | Paper-facing rounded cell |
|---|---:|---:|---:|---:|---:|
| CT24 test | `0.6936` | `0.7059` | `0.6818` | `0.55` | `0.761` |
| ClaimBuster | `0.8939` | `0.8690` | `0.9202` | `0.70` | `0.894` |
| CT23 | `0.8458` | `0.8067` | `0.8889` | `0.45` | `0.846` |

Conclusion: this runnable PCA-64 + LLM + text LogReg path supports the ClaimBuster and CT23 cells after rounding, but it does not
support the paper-facing CT24 `0.761` value.
