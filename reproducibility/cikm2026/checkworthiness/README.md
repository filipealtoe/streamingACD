<!-- Sérgio Pinto, 2026-08-25 13:41 WEST — added a paper-frozen reproduction path for the claim check-worthiness work, with explicit safeguards against row-order joins and test-label threshold selection. -->

# CIKM 2026 claim check-worthiness reproduction

The accepted paper is the immutable numerical and methodological target. This
directory separates three kinds of evidence:

1. **Historical paper evidence** preserves the values and retained outputs from
   the experiments used for the paper.
2. **Fresh direct reproduction** reruns a checksum-identified public checkpoint
   or deterministic numerical pipeline.
3. **Protocol-conformant replication** selects models and thresholds without
   consulting the final evaluation labels and reports the resulting values even
   when they differ from the historical table.

The machine-readable contract is [`PAPER_PROTOCOL.json`](PAPER_PROTOCOL.json).

## Recovered author artifacts

The numerical soft-label files under
`reproducibility/source_artifacts/checkworthiness/ct24_llm_features_v4/` and
`benchmark_llm_features/` were recovered byte-for-byte from Sérgio Pinto's
May 14, 2026 artifact commit `093bb9accfa62b0864ce6a65f797ba3a23e81418`.
They contain benchmark sentence IDs and numerical LLM-derived features, but no
source text, Twitter/X post or user identifiers, usernames, profiles, or
locations.

The text-free CT24 label tables under `ct24_labels/` preserve only public
benchmark sentence IDs and binary labels. They make the corrected path
independent of row order: every recovered soft-label row is joined by
`sentence_id` with a one-to-one and complete-set check before model fitting.

The three Encoder Only checkpoints are public, immutable Hugging Face revisions:

| Seed | Model revision | Model SHA-256 |
|---:|---|---|
| 0 | `a7e8f07153c72c853ce39bb4c03b4bc817c1e39c` | `9c4c555ce0e839b109d7ca2d9df69b8b5b86fe63be2a493fe1ecb1cfa467440b` |
| 42 | `972bbd2a25acbe26c46358f11c55c42cfafa0090` | `74ef9ea94b3345ecf87b84e65b07ff390f8cf9d20f6defb67d7ec9a1f278b7c0` |
| 456 | `eda7cc7a98577ab52dab370642ca9a9ecd33acfc` | `6c76e81fd86773e516fd3d26e0d243c49c68bd88ae331bd060b41cde26fbd517` |

## Commands

Run or resume direct inference from the public checkpoints:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py --inference
```

Add `--force-inference` to regenerate an already complete prediction bundle.
The runner uses the fast tokenizer selected by the recovered historical source
and pins its conversion/runtime dependencies.

Recompute the metrics from retained local probabilities without downloading
models:

```bash
uv run scripts/reproduce_cikm2026_public_encoder_fusion.py
```

The output records individual Encoder Only results, two- and three-seed
ensembles, the historical positional Fusion diagnostic, and the corrected
sentence-ID-aligned Fusion replication. Evaluation-set threshold optima are
always marked as diagnostics; fixed-threshold results are reported separately.

The CT24 training text is not duplicated here. It is available from the
official CheckThat! 2024 Task 1 repository. The recovered numerical soft labels
join to the official rows by `Sentence_id` with one-to-one validation.
