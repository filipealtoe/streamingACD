# Reviewer Reproducibility Artifact

Date: 2026-05-15

This package consolidates the reproducibility material for the CIKM 2026 submission draft.

## How To Read This Package

- `docs/` contains the paper-claim audit and current reproduction status.
- `reproducibility/` contains source artifacts, packaged runs, scripts, and checksums.
- `results/` contains rerun summaries, manuscript-fix notes, and claim-specific evidence notes.
- `scripts/` contains audit/recompute helpers used by the package.
- `private_raw_artifacts/` contains local-only raw artifacts that were intentionally not committed to the public GitHub repo.

## Main Covered Claims

| Claim area | Status | Primary artifact |
|---|---|---|
| Four-head CT24 check-worthiness | Reproduced | `reproducibility/runs/deberta_mtl_cikm_20260512_134553/`, including `best_model.pt` |
| Four-head ClaimBuster/CT23 cross-dataset checks | Reproduced | `reproducibility/runs/deberta_mtl_cikm_20260512_134553/benchmark_summary.json` |
| Three-seed DeBERTa ensemble | Reproduced | `results/table3_reproduction_2026-05-12.*` |
| Fusion classifier | Reproduced | `results/table3_reproduction_2026-05-12.*`; `reproducibility/source_artifacts/checkworthiness/fusion_classifier/` |
| CT25 claim normalization first-300 result | Reproduced with boundary | `results/ct25_claim_normalization_lambda_2026-05-15/`; raw predictions in `private_raw_artifacts/ct25_claim_normalization_lambda/` |
| Virality split and PSR artifacts | Artifact-backed | `reproducibility/source_artifacts/virality/`; `results/virality_split_manifest_2026-05-12.*` |
| Anomaly detection replacement values | Rerun/rewrite packet | `results/anomaly_expose_reproduction_attempt_2026-05-13.md`; `results/anomaly_baseline_paper_fix_2026-05-14.md` |
| Corpus language composition | Narrowed denominator required | `results/corpus_language_share_2026-05-12.*`; `results/clustering_subset_language_share_2026-05-14.*` |
| Formative evaluation quantitative stats | Not reproduced | `results/formative_evaluation_paper_fix_2026-05-14.md`; `results/formative_evaluation_artifact_audit_2026-05-14.md` |

## CT25 Claim Normalization Boundary

The GitHub-safe artifact commits only per-sample metrics and provenance. This reviewer package additionally includes the full
unredacted Lambda prediction JSONL:

```text
private_raw_artifacts/ct25_claim_normalization_lambda/local_Mistral-7B-Instruct-v0.3_test.jsonl
```

That file contains `post`, `gold_claim`, and `predicted_claim` fields. It is included here so the first-300 METEOR value can be
audited from raw text if redistribution is permitted by the dataset terms. It should be removed from any public artifact if the
dataset license does not allow redistribution of those fields.

## Included Large Checkpoint

This reviewer package includes the four-head checkpoint:

```text
reproducibility/runs/deberta_mtl_cikm_20260512_134553/best_model.pt
```

Checkpoint SHA-256:

```text
5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081
```

## Not Included

The package intentionally excludes:

- Hugging Face cache files and unrelated model weights;
- Python virtual environments;
- Lambda package caches;
- API keys and secrets.

## Current Manuscript Risks

The repository does not support every old IJCAI paper value as originally written. The current safe path is to use the reproduced
values and rewrite unsupported exact claims according to the notes in:

```text
docs/paper_claim_reproduction_matrix.md
results/manuscript_rewrite_patch_2026-05-14.md
```

The main remaining unsupported empirical area is formative-evaluation quantitative participant statistics, unless the paper removes
or qualitatively rewords those numbers.
