# LLM-feature classifier numerical bundle

<!-- Sérgio Pinto, 2026-08-21 19:21 PDT — packaged the exact numerical inputs needed to reproduce the three paper-facing F1 values without redistributing source text. -->
<!-- Sérgio Pinto, 2026-08-21 19:27 PDT — documented the historical per-benchmark threshold sweep and added a fixed development-selected threshold diagnostic. -->

`matrices.npz` contains the CT24 train, development, and test feature matrices
and the ClaimBuster and CT23 evaluation matrices used by the paper's
PCA-64 + LLM + text-feature Logistic Regression row. Each row has 153 numeric
features:

- 35 deterministic text features;
- 64 BGE embedding principal components;
- 54 LLM-derived features.

The bundle contains labels and numeric features only. It contains no post text,
post or user identifiers, usernames, or profile fields.

`reference_predictions.npz` preserves the probabilities from the retained
full-precision execution. The public reproducer freshly fits the scaler and
classifier from `matrices.npz`, checks classification decisions against those
reference probabilities, and recomputes all three F1 values:

```bash
uv run scripts/reproduce_cikm2026_llm_features.py
```

The paper-facing values follow the retained source and select the maximum-F1
threshold separately for each evaluation benchmark. For a stricter transfer
diagnostic, the output also selects one threshold on the CT24 development split
using a training-only model, then holds that threshold fixed when evaluating the
final train-plus-development model on CT24, ClaimBuster, and CT23.

The matrices were reconstructed with the retained feature-extraction,
embedding/PCA, and benchmark scripts under `source_code/`. PCA was fitted on
the CT24 training split only; the scaler and classifier are fitted on CT24
train plus development data.
