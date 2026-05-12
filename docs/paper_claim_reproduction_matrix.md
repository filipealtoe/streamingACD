# Paper Claim Reproduction Matrix

Date: 2026-05-12

This matrix tracks paper-facing empirical claims against local evidence. A claim is treated as reproducible only when the repository
contains or references the exact input artifacts, command or script, output artifacts, and checksums needed to verify the number.

| Claim area | Paper-facing claim | Status | Evidence now in this repo | Next action |
|---|---|---|---|---|
| Four-head check-worthiness on CT24 | Four-head MTL model reaches CT24 F1 in the low 0.8s; reproduced run gives F1 `0.8333` | Reproduced | `reproducibility/runs/deberta_mtl_cikm_20260512_134553/` | Use the reproduced number and cite the run bundle |
| Four-head cross-dataset check-worthiness | ClaimBuster F1 `0.973`, CT23 F1 `0.933` | Reproduced | `benchmark_eval.log`, `benchmark_summary.json` in the run bundle | Use reproduced ClaimBuster `0.9730` and CT23 `0.9327` |
| Single DeBERTa CT24 baseline | Encoder-only DeBERTa F1 around `0.824` | Near paper value | `results/table3_reproduction_2026-05-12.*` recomputes F1 `0.8214` | Use as an approximate baseline or cite the recomputed value |
| Three-seed DeBERTa ensemble | Ensemble F1 around `0.834` | Reproduced | `results/table3_reproduction_2026-05-12.*` recomputes F1 `0.8343` | Keep as a reproduced baseline |
| Fusion classifier | Ensemble + LLM features F1 around `0.836` | Reproduced | `results/table3_reproduction_2026-05-12.*` reruns the XGBoost v4 LLM component and recomputes Fusion Classifier F1 `0.8362` | Keep the row, but cite the exact rerun configuration |
| LLM-feature classifier | Paper row says `LLM_features PCA 64 + Logreg`: CT24 `0.761`, ClaimBuster `0.894`, CT23 `0.846` | Partially reproduced | `results/pca64_llm_text_benchmark_2026-05-12.md` matches ClaimBuster `0.8939` and CT23 `0.8458`; `results/table3_reproduction_2026-05-12.*` reruns CT24 at `0.6936` | Do not keep CT24 `0.761` as-is unless a matching held-out test run is recovered; `results/deberta_cls_llm_logreg_candidate_2026-05-12.md` records a stronger replacement candidate at CT24 F1 `0.7929` |
| Clustering threshold | `tau=0.65` gives mean intra-cluster similarity about `0.87` | Local artifact found | `artifact_checksums_2026-05-11.*`; previous `threshold_ablation/cluster_statistics.json` | Package the ablation JSON or copy a small derived summary |
| Pipeline dataset release | Pipeline has 535 normalized claims and linked output artifacts | Local artifact found | Artifact manifest references canonical streaming run files | Add a schema/count verifier for the canonical run |
| Virality prediction | Table 2 PSR results on 529 anomalous clusters | Local artifact found | Previous `virality_tuned/tuned_baselines.json` referenced in manifest | Verify split, input data, and all paper table rows |
| Claim normalization | Table 1 METEOR values, including Approach 2 up to `0.5691` | Not reproduced | Exact table artifacts not found in the current package | Rerun or rewrite the claim around the values that have artifacts |
| Anomaly detection | EXPoSE NAB `79.2`, detection `97.6%`, median lead `+23h` | Not reproduced | Related anomaly files exist locally, but no exact matching result bundle identified | Rerun or rewrite the anomaly section with supported metrics |
| Formative evaluation | 9 participants, 27 report pairs, agreement and usefulness ratings | Not reproduced | No anonymized aggregate survey file packaged yet | Add anonymized aggregate data and analysis script, or soften/remove the exact claims |
| Sample explainability report | High-confidence report example and 8-hour pre-peak claim | Local artifact likely found | Previous pipeline output includes report HTML candidates | Copy the exact report and record the source claim ID, or remove the URL claim |

## Current Priority

The four-head, ensemble, and fusion check-worthiness results are no longer blocking. The highest-risk remaining exact claims are
claim normalization, anomaly detection, and the formative evaluation because those are not yet tied to clean, packaged artifacts.
The LLM-feature classifier row still needs a CT24 rewrite or a recovered held-out test run that actually matches `0.761`.
