# LLM-feature classifier numerical bundle

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — documented the exact numerical reproduction inputs and outputs. -->

`matrices.npz` contains the CT24 train, development, and test feature matrices
and the ClaimBuster and CT23 evaluation matrices used by the paper's PCA-64 +
LLM + text-feature Logistic Regression row. Each row has 153 numeric features:

- 35 deterministic text features;
- 64 BGE embedding principal components;
- 54 LLM-derived features.

`reference_predictions.npz` preserves the corresponding probabilities. The
reproducer fits the scaler and classifier from the numerical matrices, confirms
the retained classification decisions, and recomputes all three F1 values:

```bash
uv run scripts/reproduce_cikm2026_llm_features.py
```

The paper values are CT24 `0.694`, ClaimBuster `0.894`, and CT23 `0.846`. The
bundle contains labels and numerical features only, with no post text, post or
user identifiers, usernames, or profile fields.
