# Reproducibility Package

This directory tracks evidence for paper-facing claims without committing large datasets, model checkpoints, or raw prediction dumps by default.

## Contents

- `artifact_checksums_2026-05-11.md`: human-readable artifact manifest.
- `artifact_checksums_2026-05-11.json`: machine-readable artifact manifest.
- `runs/deberta_mtl_cikm_20260512_134553/`: reproduced four-head MTL run bundle.
- `../scripts/verify_local_artifacts.py`: read-only verifier for the local artifact set.

## Current Scope

The current manifest covers local evidence for:

- CT24 split sizes and source files;
- CT23 and ClaimBuster source files;
- the US Election 2020 raw corpus count and date range;
- the candidate canonical streaming run `2026-01-17_03-56`;
- the clustering threshold-ablation statistics;
- DeBERTa CT24 single-model and ensemble prediction artifacts;
- virality tabular baseline results.

The current run bundle additionally covers:

- four-head MTL CT24 test F1 `0.8333`;
- CT24 dev/test probability artifacts for paired comparisons;
- CT24 dev/test labels and `Sentence_id` arrays in the exact MTL join order;
- ClaimBuster F1 `0.9730` and CT23 F1 `0.9327` from the reproduced checkpoint;
- exact training and benchmark scripts used for this run.

## Rules

- Small manifests, summary JSON files, and verifier scripts can live in Git.
- Raw datasets, embeddings, model checkpoints, and large Parquet/NPY files should be regenerated or stored externally with SHA-256 checksums.
- A paper metric is accepted only when a verifier or evaluation script can recompute it from a declared artifact.
- Unsupported or conflicting claims stay out of the paper until rerun or rewritten.

## Local Verification

The verifier reads from the previous local checkout by default:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/verify_local_artifacts.py
```

Expected current result:

```text
Summary: 0 failures, 2 warnings
```

The warnings are expected until resolved:

- the raw election corpus file does not contain a language column for the `87% English` claim;
- the old `0.814` value appears in single-head DeBERTa artifacts, but the new four-head run is now separately packaged.

## CT24 Metric Recalculation

Recompute CT24 metrics from saved probability artifacts with:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/recompute_ct24_metrics.py
```

The output is written to `results/ct24_metric_reproduction_2026-05-12.json`.
