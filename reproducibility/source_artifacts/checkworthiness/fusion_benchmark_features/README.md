<!-- Sérgio Pinto, 2026-08-28 02:46 WEST — restored the exact four-confidence benchmark inputs used by the historical Fusion evaluation. -->

# Retained benchmark inputs for Fusion

These are the exact ClaimBuster and CheckThat! 2023 four-confidence inputs
used by the historical Fusion evaluator.

| Benchmark | Rows | Positive labels | SHA-256 |
|---|---:|---:|---|
| ClaimBuster | 1,032 | 238 | `60bdfcf5a89002b8eddfcfdf220b2d35665ce0121cc228f5ea0cd30e70858326` |
| CheckThat! 2023 | 318 | 108 | `07591b77a281939c852aebd5627c6e1cf36731bc84457839f6aa170ff4536327` |

Each row stores the public benchmark text and label together with
`checkability_conf`, `verifiability_conf`, `harm_conf`, and their row mean in
`avg_confidence`. The `idx` field fixes the benchmark order. ClaimBuster and
CT23 reproduce the recorded LLM-only F1 values `0.633333` and `0.684615`
respectively when evaluated by the historical XGBoost path.
