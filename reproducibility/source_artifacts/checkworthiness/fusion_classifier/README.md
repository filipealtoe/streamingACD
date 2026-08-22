# Fusion classifier numerical bundle

<!-- Sérgio Pinto, 2026-08-21 21:48 PDT — added the checksum-bound non-sensitive historical cross-benchmark aggregate extract. -->
<!-- Sérgio Pinto, 2026-08-21 22:01 PDT — distinguished public aggregate evidence from local checkpoint observations. -->

The three numeric vectors in this directory reproduce the CT24 Fusion result:

- `ct24_encoder_t03_probs.npy`: retained temperature-0.3 DeBERTa ensemble
  probabilities;
- `ct24_llm_xgboost_test_probs.npy`: retained XGBoost probabilities;
- `ct24_fusion_test_probs.npy`: retained equal-weight Fusion probabilities.

Recompute the equal-weight probabilities and CT24 F1 `0.836`:

```bash
uv run scripts/reproduce_cikm2026_fusion_ct24.py
```

The checksum-bound [`FUSION_RECOVERY_AUDIT.json`](FUSION_RECOVERY_AUDIT.json)
preserves the historical ClaimBuster `0.961` and CT23 `0.915` aggregate metrics,
records the candidate-checkpoint hashes and filesystem times as local author
observations, and includes the fresh diagnostic showing that those candidates do
not reproduce the paper values.
[`HISTORICAL_RUN_EXTRACT.md`](HISTORICAL_RUN_EXTRACT.md) makes the corresponding
non-sensitive author aggregate record public and documents that the historical
cross-benchmark evaluation loaded two seeds.
Their per-sample prediction arrays were not retained, so exact cross-benchmark
metric reproduction requires the historical checkpoint bytes or corresponding
prediction arrays.
