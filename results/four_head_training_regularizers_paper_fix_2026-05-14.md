# Four-Head Training Regularizers Manuscript Fix

Date: 2026-05-14

## Scope

This note defines the paper-side correction for the claim that the four-head model used Focal Loss, layer-wise learning-rate
decay, R-Drop, and FGM adversarial training. It does not modify the manuscript itself because no editable `.tex`, `.docx`, or
source archive for the paper was found in the local repo search.

## Finding

The regularized training recipe exists in the repo, but it is attached to the single-head DeBERTa seed-ensemble path, not the
reproduced four-head MTL run.

Supported four-head MTL recipe:

- shared DeBERTa encoder;
- one main check-worthiness classification head;
- three auxiliary regression heads;
- 3-phase training;
- weighted cross-entropy for the main classification task;
- MSE auxiliary losses;
- entropy weighting.

Supported single-head/ensemble recipe:

- single sequence-classification head;
- Focal Loss;
- layer-wise learning-rate decay;
- R-Drop;
- FGM adversarial training;
- enabled by the seed-ensemble launcher for each single-head seed model.

## Required Manuscript Edit

Do not say that the reproduced four-head MTL model used Focal Loss, layer-wise learning-rate decay, R-Drop, or FGM.

Replace any sentence with this meaning:

> The four-head model was trained with Focal Loss, layer-wise learning-rate decay, R-Drop, and FGM adversarial training.

with:

> The four-head MTL model uses a shared DeBERTa encoder with one check-worthiness classification head and three auxiliary
> regression heads. The reproduced run uses 3-phase training, weighted cross-entropy for the main task, MSE auxiliary losses,
> and entropy weighting.

If the paper also describes the single-head seed ensemble, attach the regularizer sentence there:

> The single-head DeBERTa seed-ensemble trainer supports Focal Loss, layer-wise learning-rate decay, R-Drop consistency
> regularization, and FGM adversarial training, and the ensemble launcher enables those options for each seed model.

## Evidence Used

- `/Users/sergiopinto/streamingACD/reproducibility/runs/deberta_mtl_cikm_20260512_134553/scripts/finetune_deberta_mtl.py`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/checkworthiness/regularized_single_head/finetune_deberta_multimodel.py`
- `/Users/sergiopinto/streamingACD/reproducibility/source_artifacts/checkworthiness/regularized_single_head/ensemble_deberta_seeds.py`
- `/Users/sergiopinto/streamingACD/results/four_head_training_regularizers_audit_2026-05-14.md`

## Remaining Blocker

No blocker remains for the corrected wording. If the paper must claim those regularizers for a four-head model, a new four-head
run using those regularizers must be trained and packaged with script, command, predictions, and metrics.
