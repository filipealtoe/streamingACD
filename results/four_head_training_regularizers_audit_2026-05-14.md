# Four-Head Training Regularizer Attribution

Date: 2026-05-14

## Finding

The Focal Loss, layer-wise learning-rate decay, R-Drop, and FGM adversarial-training sentence is not supported for the reproduced
four-head MTL run.

## Evidence

| Code path | Evidence | Supported training recipe |
|---|---|---|
| Reproduced four-head MTL | `reproducibility/runs/deberta_mtl_cikm_20260512_134553/scripts/finetune_deberta_mtl.py` | Shared DeBERTa encoder, one main check-worthiness head, three auxiliary regression heads, 3-phase training, weighted cross-entropy main loss, MSE auxiliary losses, entropy weighting |
| Regularized single-head DeBERTa | `reproducibility/source_artifacts/checkworthiness/regularized_single_head/finetune_deberta_multimodel.py` | Single sequence-classification head with Focal Loss, layer-wise learning-rate decay, R-Drop, and FGM support |
| Seed ensemble launcher | `reproducibility/source_artifacts/checkworthiness/regularized_single_head/ensemble_deberta_seeds.py` | Enables `focal_loss`, `llrd`, `rdrop`, and `fgm` for each single-head seed model |

## Paper Action

Use `results/four_head_training_regularizers_paper_fix_2026-05-14.md` to revise the method text so the regularized recipe
describes the single-head DeBERTa seed ensemble. The four-head MTL method should be described as the reproduced weighted
cross-entropy/MSE auxiliary-training run unless a new four-head model is retrained with those regularizers and its predictions,
command, script, and metrics are packaged.
