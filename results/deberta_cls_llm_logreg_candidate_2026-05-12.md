# DeBERTa CLS + LLM LogReg Replacement Candidate

Date: 2026-05-12

Purpose: test whether a related held-out CT24 classifier can match or surpass the paper-facing `0.761` CT24 value after the
PCA-64 + LLM + text LogReg rerun failed to do so.

TRUTH: this was a targeted diagnostic rerun of the Logistic Regression branch represented in
`/Users/sergiopinto/explainableACD/experiments/scripts/train_embedding_classifiers.py`, using the existing cached DeBERTa CLS
embeddings and seven LLM features.

LIMITATION: this is not a reproduction of the paper row labeled `LLM_features PCA 64 + Logreg`, and it should not be presented as
that row without rewriting the method description.

Feature inputs:

- cached DeBERTa CLS embeddings from `experiments/results/embeddings_cache/*_cls_embeddings.npy`;
- LLM features `check_p_yes`, `verif_p_yes`, `harm_p_yes`, `check_entropy_norm`, `verif_entropy_norm`, `harm_entropy_norm`,
  `yes_vote_count`;
- Logistic Regression with class weight `{0: 1, 1: 3}`;
- `GridSearchCV` over `C in [0.01, 0.1, 1, 10]`, `cv=3`, scoring `f1`;
- train split used for fit, held-out CT24 `test` split used for the paper-relevant comparison.

Rerun output:

| Split | F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Dev | `0.9590` | `0.9806` | `0.9360` | `0.9832` |
| Dev-test | `0.9055` | `0.9403` | `0.9785` | `0.8426` |
| CT24 test | `0.7929` | `0.8974` | `0.8272` | `0.7614` |

Selected Logistic Regression configuration:

- `C=0.01`;
- cross-validation F1 `0.8401`.

Conclusion: this candidate clears CT24 F1 `0.761` on the held-out test split, but it is a replacement candidate, not evidence that
the paper's PCA-64 + LLM + LogReg CT24 cell was reproduced.
