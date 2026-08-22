# Synchronize the CIKM 2026 artifact with the final PDF

- Author: Sérgio Pinto
- Timestamp: 2026-08-22 00:53 PDT (-0700)
- Reason: bind the public evidence package to the exact final camera-ready PDF
  and remove conclusions that belonged to earlier manuscript revisions.

## What changed

- Replaced the superseded August 13 PDF identity with the exact 12-page final
  build: 620,920 bytes and SHA-256
  `4d82abd01d66de5e04d7107e8c4bb21d3b3d1a7148aa235e95b78b803df78b9c`.
- Realigned the release-coverage audit with the final paper's availability
  statements. The 529-row dataset, 42 engineered features, virality labels,
  source snapshots, model parameters and checkpoint, prompts, schema, and
  reproducibility artifacts are all present.
- Removed the stale 768-versus-384 manuscript mismatch. The exact final PDF
  states no embedding dimension; the published matrix remains verified at
  100,000 by 384 float32 values.
- Retained the existing evidence boundary for the historical Encoder Only
  ClaimBuster and Fusion ClaimBuster/CT23 results after fresh inference showed
  that the later local checkpoints do not reproduce the paper-run vectors.

## Verification

```bash
uv run --no-project scripts/audit_cikm2026_release_coverage.py
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --paper-pdf /path/to/CIKM2026-camera-ready-2026-08-22.pdf
```

The PDF is used only for optional byte-identity verification and is not added to
the repository. This change does not modify the Overleaf project and does not
publish raw posts, post identifiers, or user-level data.
