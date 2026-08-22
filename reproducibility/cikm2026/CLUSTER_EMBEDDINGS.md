# Canonical cluster embeddings

<!-- Sérgio Pinto, 2026-08-21 21:55 PDT — documented the recovered canonical matrix and its external-release boundary. -->
<!-- Sérgio Pinto, 2026-08-21 22:29 PDT — linked the expected public asset identity to a live, metadata-only release check. -->
<!-- Sérgio Pinto, 2026-08-21 23:43 PDT — recorded the completed public upload and stable download URL after verifying the remote digest. -->
<!-- Sérgio Pinto, 2026-08-22 00:53 PDT — removed the stale manuscript-dimension mismatch after checking the exact final PDF. -->

The canonical `streaming_full/2026-01-17_03-56` cluster-embedding matrix has
been recovered and authenticated:

- file: `cluster_embeddings.npy`;
- shape: `100000 × 384`;
- dtype: `float32`;
- byte size: `153600128`;
- SHA-256: `514e9f44c095b1de5175cc6c76db548e9762c3e1bcdda93b8cb8f16f09bd1619`;
- all values are finite;
- its 100,000 row indices exactly match the committed
  `cluster_id_to_idx.json` mapping.

At 153.6 MB, the matrix is above GitHub's normal 100 MB file limit and must not
be committed as a normal Git blob. Its public delivery path is an external asset
on the existing [`cikm-2026-artifact-v1`](https://github.com/filipealtoe/streamingACD/releases/tag/cikm-2026-artifact-v1)
release. The published file is available as
[`cluster_embeddings.npy`](https://github.com/filipealtoe/streamingACD/releases/download/cikm-2026-artifact-v1/cluster_embeddings.npy).
`RELEASE_ASSETS.json` records the expected remote identity, and the metadata-only
command below verifies the live release without downloading the matrix:

```bash
uv run --no-project scripts/verify_cikm2026_release_assets.py
```

The exact final camera-ready PDF does not state an embedding dimension. The
metadata therefore records the observed and checksum-verified 384-dimensional
shape without asserting a manuscript mismatch.

After downloading the published file, verify its contents with:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py \
  --embeddings /path/to/cluster_embeddings.npy
```
