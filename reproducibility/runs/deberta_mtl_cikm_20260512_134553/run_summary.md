# CIKM 2026 MTL Retrain Run Summary

Run ID: `deberta_mtl_cikm_20260512_134553`
Remote host: `ubuntu@129.213.210.158`
Hardware: NVIDIA A10 24 GB

## CT24 Result

- Best test F1: `0.8333333333333333`
- Best test threshold: `0.50`
- Test accuracy: `0.9120234604105572`
- Test precision: `0.8152173913043478`
- Test recall: `0.8522727272727273`
- Dev-selected best threshold: `0.60`
- Saved prediction arrays:
  - `dev_probs.npy`: 1031 rows
  - `test_probs.npy`: 341 rows

## Cross-Dataset Evaluation

- ClaimBuster: F1 `0.9730`, precision `0.9630`, recall `0.9832`, threshold `0.60`
- CT23: F1 `0.9327`, precision `0.9700`, recall `0.8981`, threshold `0.30`

## Local Artifacts

- `best_model.pt`
- `results.json`
- `dev_probs.npy`
- `test_probs.npy`
- `training.log`
- `benchmark_eval.log`
- `benchmark_summary.json`
- `command.txt`
- `environment.txt`
- `sha256sums.txt`
- `script_sha256sums.txt`
- `scripts/finetune_deberta_mtl.py`
- `scripts/evaluate_mtl_benchmarks.py`

Intermediate remote checkpoints `checkpoint_phase1.pt`, `checkpoint_phase2.pt`, and `checkpoint_phase3.pt` were not copied locally.
They remain on the Lambda instance unless the instance is terminated.
