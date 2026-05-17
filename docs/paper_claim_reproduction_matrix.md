# Paper Claim Reproduction Matrix

Date: 2026-05-12

This matrix tracks paper-facing empirical claims against local evidence. A claim is treated as reproducible only when the repository
contains or references the exact input artifacts, command or script, output artifacts, and checksums needed to verify the number.

Machine-readable full-paper audit:

- `scripts/audit_full_paper_claims.py`
- `results/full_paper_claim_audit_2026-05-12.json`
- `results/full_paper_claim_audit_2026-05-12.md`

The full-paper audit intentionally exits non-zero while exact IJCAI claims remain mismatched or missing.

| Claim area | Paper-facing claim | Status | Evidence now in this repo | Next action |
|---|---|---|---|---|
| Four-head check-worthiness on CT24 | Four-head MTL model reaches CT24 F1 in the low 0.8s; reproduced run gives F1 `0.8333` | Reproduced | `reproducibility/runs/deberta_mtl_cikm_20260512_134553/` | Use the reproduced number and cite the run bundle |
| Four-head cross-dataset check-worthiness | ClaimBuster F1 `0.973`, CT23 F1 `0.933` | Reproduced | `benchmark_eval.log`, `benchmark_summary.json` in the run bundle | Use reproduced ClaimBuster `0.9730` and CT23 `0.9327` |
| Single DeBERTa CT24 baseline | Encoder-only DeBERTa F1 around `0.824` | Near paper value | `results/table3_reproduction_2026-05-12.*` recomputes F1 `0.8214` | Use as an approximate baseline or cite the recomputed value |
| Three-seed DeBERTa ensemble | Ensemble F1 around `0.834` | Reproduced | `results/table3_reproduction_2026-05-12.*` recomputes F1 `0.8343` | Keep as a reproduced baseline |
| Fusion classifier | Ensemble + LLM features F1 around `0.836` | Reproduced | `results/table3_reproduction_2026-05-12.*` reruns the XGBoost v4 LLM component and recomputes Fusion Classifier F1 `0.8362` | Keep the row, but cite the exact rerun configuration |
| LLM-feature classifier | Paper row says `LLM_features PCA 64 + Logreg`: CT24 `0.761`, ClaimBuster `0.894`, CT23 `0.846` | Partially reproduced | `results/pca64_llm_text_benchmark_2026-05-12.md` matches ClaimBuster `0.8939` and CT23 `0.8458`; `results/table3_reproduction_2026-05-12.*` reruns CT24 at `0.6936` | Do not keep CT24 `0.761` as-is unless a matching held-out test run is recovered; `results/deberta_cls_llm_logreg_candidate_2026-05-12.md` records a stronger replacement candidate at CT24 F1 `0.7929` |
| Clustering threshold | `tau=0.65` gives mean intra-cluster similarity about `0.87` | Reproduced local | `reproducibility/source_artifacts/clustering/cluster_statistics.json` and the full-paper audit | Keep the copied result artifact plus checksum |
| Pipeline dataset release | Pipeline has 535 normalized claims and linked output artifacts | Reproduced local | `reproducibility/source_artifacts/pipeline/streaming_full_2026-01-17_03-56_summary.json` and the full-paper audit | Add a schema/count verifier for the larger canonical run artifacts if the paper cites file-level schema details |
| Corpus language composition | Corpus is approximately `87%` English | Contradicted by packaged summary | `reproducibility/source_artifacts/pipeline/streaming_full_2026-01-17_03-56_summary.json` implies English share `0.699`; raw parquet has no language column | Recompute from a clear language field/detection log or remove the exact `87%` claim |
| Virality prediction | Table 2 PSR results on 529 anomalous clusters | Artifact-backed, not fully rerunnable | `reproducibility/source_artifacts/virality/*`; local parquet verifies 529 rows and 42 feature columns | Add the exact 423/106 split manifest and rerun command |
| Claim normalization | Table 1 METEOR values, including Approach 2 up to `0.5691` | Not reproduced | `reproducibility/source_artifacts/claim_normalization/comparison_test_20260113_123010.json` has `n=1285`, best METEOR `0.3449`, not the paper table | Rerun or rewrite the claim around the values that have artifacts |
| Anomaly detection | EXPoSE NAB `79.2`, detection `97.6%`, median lead `+23h` | Not reproduced | `reproducibility/source_artifacts/anomaly/expose_grid_search.csv` best row is NAB `72.2116`, detection `95.0097`, lead `+23h` | Rerun until match/surpass or rewrite to supported values |
| Four-head training details | Focal Loss, layer-wise LR decay, R-Drop, and FGM adversarial training | Not reproduced as written | Packaged `finetune_deberta_mtl.py` is the reproduced 3-phase BCE/MSE auxiliary-loss trainer and does not contain those regularizers | Package exact regularized run or revise method text |
| Llama2 baseline row | Llama2-7b fine-tuned baseline has F1 `0.802/0.920/0.898` | Not reproduced | No local result/prediction bundle found | Package exact baseline artifact or cite as external prior work |
| Formative evaluation | 9 participants, 27 report pairs, agreement and usefulness ratings | Not reproduced | No anonymized aggregate survey file packaged yet | Add anonymized aggregate data and analysis script, or soften/remove the exact claims |
| Sample explainability report | High-confidence report example and 8-hour pre-peak claim | Missing exact artifact | No exact local artifact was found for the combined `567 engagements`, `71.7%`, and `8-hour` example | Recover the exact report and source claim ID, or remove the exact example |

## Current Priority

The four-head, ensemble, and fusion CT24 check-worthiness results are no longer blocking. The highest-risk remaining exact claims
are claim normalization, anomaly detection, corpus language composition, the formative evaluation, and the exact sample report
because those are not tied to clean, packaged artifacts. The LLM-feature classifier row still needs a CT24 rewrite or a recovered
held-out test run that actually matches `0.761`.
