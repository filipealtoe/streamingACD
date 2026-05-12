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
| Fusion classifier | Ensemble + LLM features F1 around `0.836` | Not cleanly reproduced | `results/table3_reproduction_2026-05-12.*` finds best saved local fusion F1 `0.8295`, not the documented XGBoost v4 run | Rerun the documented XGBoost v4 fusion protocol or remove/soften the row |
| LLM-feature classifier | LLM-feature classifier F1 around `0.761` | Not reproduced | `results/table3_reproduction_2026-05-12.*` finds saved local LLM artifacts at F1 `0.7135` and `0.7200` | Rerun the documented XGBoost v4 protocol or remove/soften the row |
| Clustering threshold | `tau=0.65` gives mean intra-cluster similarity about `0.87` | Local artifact found | `artifact_checksums_2026-05-11.*`; previous `threshold_ablation/cluster_statistics.json` | Package the ablation JSON or copy a small derived summary |
| Pipeline dataset release | Pipeline has 535 normalized claims and linked output artifacts | Local artifact found | Artifact manifest references canonical streaming run files | Add a schema/count verifier for the canonical run |
| Virality prediction | Table 2 PSR results on 529 anomalous clusters | Local artifact found | Previous `virality_tuned/tuned_baselines.json` referenced in manifest | Verify split, input data, and all paper table rows |
| Claim normalization | Table 1 METEOR values, including Approach 2 up to `0.5691` | Not reproduced | Exact table artifacts not found in the current package | Rerun or rewrite the claim around the values that have artifacts |
| Anomaly detection | EXPoSE NAB `79.2`, detection `97.6%`, median lead `+23h` | Not reproduced | Related anomaly files exist locally, but no exact matching result bundle identified | Rerun or rewrite the anomaly section with supported metrics |
| Formative evaluation | 9 participants, 27 report pairs, agreement and usefulness ratings | Not reproduced | No anonymized aggregate survey file packaged yet | Add anonymized aggregate data and analysis script, or soften/remove the exact claims |
| Sample explainability report | High-confidence report example and 8-hour pre-peak claim | Local artifact likely found | Previous pipeline output includes report HTML candidates | Copy the exact report and record the source claim ID, or remove the URL claim |

## Current Priority

The four-head and ensemble check-worthiness results are no longer blocking. The highest-risk remaining exact claims are claim
normalization, anomaly detection, and the formative evaluation because those are not yet tied to clean, packaged artifacts. The
fusion and LLM-feature rows also need either a documented XGBoost v4 rerun or paper wording changes.
