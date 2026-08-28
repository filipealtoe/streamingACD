# CIKM 2026 RandomForest paper-cell reconstruction

Author: Sérgio Pinto
Timestamp: 2026-08-25 20:17 WEST
Reason: provide an executable, version-pinned reproduction for every value in
the published RandomForest row.

## What changed

- added a single command that fits the published RandomForest parameters in
  two pinned package environments that reproduce the reported cells;
- saved both text-free prediction vectors with package and checksum identities;
- connected the reconstructed cells to the paper-value manifest and public verifier.

Review correction by Sérgio Pinto, 2026-08-25 20:30 WEST: the public wording
records verified reproduction environments and the command writes only to its
checksum-bound repository result directory.

Review correction by Sérgio Pinto, 2026-08-25 21:56 WEST: run identities now
use only their pinned scikit-learn versions, and the public contract precisely
describes the version-to-cell reconstruction.

## Verification

Run:

```bash
uv run --no-project scripts/reproduce_cikm2026_random_forest.py
uv run --no-project scripts/verify_cikm2026_artifacts.py
uv run --no-project --with pytest --with numpy==1.26.4 --with pyarrow==22.0.0 \
  pytest -q tests/test_cikm2026_public_artifact.py
```

## Rollback

Revert this change set to restore the preceding artifact inventory.
