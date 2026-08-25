<!-- Sérgio Pinto, 2026-08-25 14:18 WEST — recorded direct inference from the three immutable public checkpoints and the corrected sentence-ID-aligned Fusion replication. -->

# Public Encoder and Fusion replication

This result contains fresh per-example inference from Sérgio Pinto's three
public DeBERTa checkpoints. Each checkpoint is fixed to a Hugging Face revision,
model byte size, and SHA-256 in `summary.json`. The prediction bundle contains
only public benchmark sentence IDs, labels, probabilities, and derived numeric
scores; it contains no benchmark text or Twitter/X user/post identifiers.

The paper values remain the historical targets. The fresh run gives:

| Path | ClaimBuster F1 | CT23 F1 |
|---|---:|---:|
| Public seed-0 Encoder, threshold 0.50 | 0.963265 | 0.927536 |
| Public seed-0 Encoder, evaluation diagnostic | 0.966805 | 0.927536 |
| Historical two-seed positional Fusion | 0.951168 | 0.900000 |
| Corrected three-seed ID-aligned Fusion | 0.941909 | 0.861386 |
| Paper Encoder | 0.970 | 0.928 |
| Paper Fusion | 0.961 | 0.915 |

The Encoder run directly reproduces the CT23 paper cell after rounding. The
public checkpoints do not reproduce the historical ClaimBuster Encoder value or
the two cross-benchmark Fusion values. No threshold, seed, feature order, or
model parameter was changed to force a paper-value match.

The corrected Fusion path validates complete one-to-one `sentence_id` joins for
CT24 train/dev, ClaimBuster, and CT23 before fitting the LLM XGBoost component.
The historical positional path is retained only to explain the original
pipeline behavior; it is not presented as the scientifically corrected result.

Run or resume checkpoint inference:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py --inference
```

Regenerate all model probabilities:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py \
  --inference --force-inference
```

Recompute every metric offline from the retained prediction bundle:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py
```
