# CIKM check-worthiness reproduction

<!-- Sérgio Pinto, 2026-08-25 13:41 WEST — recorded the purpose and verification boundary of the check-worthiness artifact recovery. -->

## Why

The public artifact already reproduces most Table 5 cells, but the Encoder Only
ClaimBuster and Fusion ClaimBuster/CT23 cells lacked direct per-example evidence.
Some retained runs also selected thresholds from final evaluation labels or
combined feature rows by position.

## What

- Freeze the accepted-paper check-worthiness values and methods in a
  machine-readable protocol contract.
- Restore checksum-identified, text-free soft-label feature artifacts from
  Sérgio Pinto's earlier public artifact commit.
- Add text-free CT24 sentence-ID/label tables and enforce complete one-to-one
  joins before fitting the corrected Fusion diagnostic.
- Bind the three public Encoder Only models to immutable Hugging Face revisions
  and model-file hashes.
- Add one resumable inference and evaluation path that reports historical and
  corrected ID-aligned Fusion results separately.
- Add deterministic guards for one-to-one sentence-ID alignment, evaluation
  policy labels, model revisions, input checksums, and privacy-safe schemas.

## Risk and rollback

The change adds reproducibility evidence and scripts; it does not modify the
accepted paper or production pipeline. Model downloads are cached outside the
repository. Reverting the eventual commit removes the new public evidence path
without changing historical results.
