# Fusion classifier numerical bundle

<!-- Sérgio Pinto, 2026-08-28 01:11 WEST — connected the verified CT24 and CT23 Fusion reproductions. -->

The numerical vectors in this directory reproduce the CT24 Fusion result:

- `ct24_encoder_t03_probs.npy`: temperature-0.3 DeBERTa ensemble probabilities;
- `ct24_llm_xgboost_test_probs.npy`: XGBoost probabilities;
- `ct24_fusion_test_probs.npy`: equal-weight Fusion probabilities.

Recompute the equal-weight probabilities and F1 `0.836` with:

```bash
uv run scripts/reproduce_cikm2026_fusion_ct24.py
```

The bundle contains numerical probabilities only and no source text or user
data.

The CT23 Fusion cell uses the two retained encoder vectors in
`../encoder_only/` and the nine-feature matrices in
`../llm_features_classifier/`. Recompute its full-precision F1 `0.915094` with:

```bash
uv run scripts/run_cikm2026_checkworthiness_lambda.py verify-retained-ct23
```
