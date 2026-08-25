# RandomForest paper-cell reconstruction

<!-- Sérgio Pinto, 2026-08-25 20:17 WEST — documents the exact command and version-to-cell mapping for the published RandomForest row. -->
<!-- Sérgio Pinto, 2026-08-25 20:30 WEST — identifies the pinned versions as exact reproduction environments and fixes the checksum-bound output location. -->
<!-- Sérgio Pinto, 2026-08-25 21:56 WEST — replaces chronological run labels with neutral version identities and describes the result as a cell-level reconstruction. -->

Run from the repository root:

```bash
uv run --no-project scripts/reproduce_cikm2026_random_forest.py
```

The command fits the same RandomForest parameters on the released 423/106
split in two isolated, version-pinned package environments. The published
cells are reconstructed and verified as follows:

| Paper cell | Reproduced with |
|---|---|
| Spearman rho | scikit-learn 1.3.2 fit |
| R2, MAE, F2@0.65, F2@0.75, F2@0.85 | scikit-learn 1.5.2 fit |

The two text-free prediction vectors, package identities, full-precision
metrics, and paper-cell verification are stored in this directory.
