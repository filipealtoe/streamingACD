# Four-head DeBERTa checkpoint

The trained checkpoint for the packaged CIKM 2026 four-head run is published as a
GitHub release asset so the repository remains lightweight.

| Field | Value |
|---|---|
| Run | `deberta_mtl_cikm_20260512_134553` |
| File | `best_model.pt` |
| Size | `1,739,380,133` bytes |
| SHA-256 | `5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081` |
| Release tag | `cikm-2026-artifact-v1` |
| Asset URL | [download `best_model.pt`](https://github.com/filipealtoe/streamingACD/releases/download/cikm-2026-artifact-v1/best_model.pt) |

After downloading the asset, verify its byte identity:

```bash
uv run scripts/verify_cikm2026_artifacts.py --checkpoint /path/to/best_model.pt
```

The training command, source snapshot, predictions, saved label order, evaluation
results, environment, and their checksums are in
`reproducibility/runs/deberta_mtl_cikm_20260512_134553/`.
