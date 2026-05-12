# Table 3 Reproduction Audit

Dataset: CT24 check-worthiness English test (341 samples, 88 positives)

| Row | Paper F1 | Recomputed F1 | Threshold | Status | Evidence |
|---|---:|---:|---:|---|---|
| Single DeBERTa | 0.8242 | 0.8214 | 0.50 | near paper value | /Users/sergiopinto/explainableACD/experiments/results/deberta_checkworthy/deberta-v3-large/test_probs.npy |
| 3-seed DeBERTa ensemble | 0.8343 | 0.8343 | 0.60 | reproduced | /Users/sergiopinto/explainableACD/lambda_backup/ubuntu/ensemble_results/test_temp_0.3_probs.npy |
| 4-head MTL retrain | 0.8333 | 0.8333 | 0.50 | reproduced | /Users/sergiopinto/streamingACD/reproducibility/runs/deberta_mtl_cikm_20260512_134553/test_probs.npy |
| Saved LLM probability artifact: logreg_llm_embed | 0.7610 | 0.7135 | 0.65 | not reproduced | /Users/sergiopinto/explainableACD/experiments/results/deberta_checkworthy/logreg_llm_embed_probs.npy |
| Saved LLM probability artifact: logreg_test | 0.7610 | 0.7200 | 0.50 | not reproduced | /Users/sergiopinto/explainableACD/experiments/results/deberta_checkworthy/logreg_test_probs.npy |
| Fusion classifier from saved local probability artifacts | 0.8362 | 0.8295 | 0.55 | not cleanly reproduced | lambda_temp_0_3 + logreg_llm_embed, DeBERTa weight=0.9 |

## Conclusion

Single DeBERTa, the three-seed ensemble, and the four-head MTL retrain have usable local evidence. The saved local LLM/fusion probability artifacts do not reproduce the paper's LLM-feature or fusion values.
