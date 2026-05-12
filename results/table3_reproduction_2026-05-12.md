# Table 3 Reproduction Audit

Dataset: CT24 check-worthiness English test (341 samples, 88 positives)

| Row | Paper F1 | Recomputed F1 | Threshold | Status | Evidence |
|---|---:|---:|---:|---|---|
| Single DeBERTa | 0.8242 | 0.8214 | 0.50 | near paper value | /Users/sergiopinto/explainableACD/experiments/results/deberta_checkworthy/deberta-v3-large/test_probs.npy |
| 3-seed DeBERTa ensemble | 0.8343 | 0.8343 | 0.60 | reproduced | /Users/sergiopinto/explainableACD/lambda_backup/ubuntu/ensemble_results/test_temp_0.3_probs.npy |
| 4-head MTL retrain | 0.8333 | 0.8333 | 0.50 | reproduced | /Users/sergiopinto/streamingACD/reproducibility/runs/deberta_mtl_cikm_20260512_134553/test_probs.npy |
| PCA-64 + LLM + text LogReg CT24 rerun | 0.7610 | 0.6936 | 0.55 | not reproduced | experiments/scripts/benchmark_pca_llm_text.py CT24 training path, mirrored in this audit |
| Fusion classifier rerun | 0.8362 | 0.8362 | 0.50 | reproduced | saved T=0.3 ensemble probabilities + rerun 24-feature XGBoost v4, weight=0.5 |

## Conclusion

The three-seed ensemble, the four-head MTL retrain, and the Fusion Classifier now have reproducible CT24 evidence. The runnable PCA-64 + LLM + text LogReg path matches the paper's cross-dataset values after rounding, but its CT24 test F1 does not reproduce the paper-facing 0.761 value.
