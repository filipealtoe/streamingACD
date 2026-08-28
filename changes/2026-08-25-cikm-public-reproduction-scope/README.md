# CIKM 2026 public reproduction scope

Author: Sérgio Pinto
Timestamp: 2026-08-25 18:18 WEST (+0100)
Reason: make the camera-ready artifact direct, coherent, and verifiable from a
fresh public clone.

## Change

- Focused the public guide and paper-to-artifact index on checksum-bound
  reproductions.
- Restricted the machine-readable value inventory to results that the public
  verifier can recompute from released inputs.
- Kept dataset feature availability and privacy facts in neutral, testable form.
- Consolidated executable model and evaluation parameters used by the verified
  package.

## Verification

From the repository root:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

The command must report `VERDICT: PASS` from a fresh clone. The optional paper,
checkpoint, and embedding paths add byte-size and SHA-256 identity checks.
