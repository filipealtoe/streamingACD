# DeBERTa MTL Four-Head Run

Run ID: `deberta_mtl_cikm_20260512_134553`

This run reproduces the four-head multi-task check-worthiness result for the CIKM draft.

## Result

| Evaluation | F1 | Precision | Recall | Accuracy | Threshold |
|---|---:|---:|---:|---:|---:|
| CT24 test | 0.8333 | 0.8152 | 0.8523 | 0.9120 | 0.50 |
| ClaimBuster | 0.9730 | 0.9630 | 0.9832 | - | 0.60 |
| CT23 | 0.9327 | 0.9700 | 0.8981 | - | 0.30 |

## Included

- `results.json`: CT24 dev/test threshold sweep and best result.
- `dev_probs.npy`: CT24 dev probabilities, 1,031 rows.
- `test_probs.npy`: CT24 test probabilities, 341 rows.
- `dev_labels.npy`, `test_labels.npy`: labels in the exact post-join row order used by the MTL run.
- `dev_sentence_ids.npy`, `test_sentence_ids.npy`: sentence IDs in the exact post-join row order used by the MTL run.
- `saved_label_order_summary.json`: metric recomputation from the saved probabilities and saved label order.
- `benchmark_eval.log`: ClaimBuster and CT23 evaluation log.
- `benchmark_summary.json`: cross-dataset summary.
- `training.log`: training and final-evaluation log.
- `command.txt`: exact training command.
- `environment.txt`: remote training environment summary.
- `sha256sums.txt`: checksums for run artifacts and external checkpoint.
- `label_order_sha256sums.txt`: checksums for saved label-order artifacts.
- `script_sha256sums.txt`: checksums for copied scripts and run summary.
- `scripts/finetune_deberta_mtl.py`: training script used for the run.
- `scripts/evaluate_mtl_benchmarks.py`: cross-dataset evaluation script used for the run.

## Not Included

`best_model.pt` is not committed because it is 1.7 GB. Its SHA-256 is recorded in `sha256sums.txt`:

```text
5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081  best_model.pt
```

Use Git LFS or external artifact storage if the checkpoint needs to move with the repository.

## Label Order Note

The MTL training script evaluates after joining `CT24_features` with `CT24_llm_features_v4`. That join order is not the same as the
older `CT24_clean` label order used by some single-model and ensemble artifacts. Use the packaged `test_labels.npy` when recomputing
the four-head MTL row.

## Training Recipe Note

This run supports the four-head architecture claim, not the regularized single-head DeBERTa recipe. The packaged training script uses
a shared DeBERTa encoder, one main check-worthiness head, three auxiliary regression heads, 3-phase training, weighted
cross-entropy for the main task, MSE auxiliary losses, and entropy weighting.

This run does not use Focal Loss, layer-wise learning-rate decay, R-Drop, or FGM adversarial training. Those regularizers are
supported by the single-head DeBERTa/seed-ensemble source artifacts copied under
`reproducibility/source_artifacts/checkworthiness/regularized_single_head/`.
