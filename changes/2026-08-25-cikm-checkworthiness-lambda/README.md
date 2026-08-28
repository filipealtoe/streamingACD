# CIKM 2026 two-track check-worthiness run

Author: Sérgio Pinto
Timestamp: 2026-08-27 19:24 WEST (+0100)
Reason: reproduce the remaining Encoder Only and Fusion paper values with the
recovered historical configuration and exact NVIDIA A10 environment.

## Change

- Added a machine-readable two-track run specification.
- Added a checksum-bound preflight for the CT24, ClaimBuster, and CT23 inputs.
- Added deterministic benchmark-text deduplication and grouped internal
  validation for the strict track.
- Added GPU training, per-seed model receipts, cross-benchmark inference,
  Fusion evaluation, and text-free prediction output.
- Bound each seed receipt to the exact preflight, staged splits, runner,
  specification, requirements, and training configuration.
- Pinned the Python dependency layer and recorded the CUDA/PyTorch environment
  used by each run.
- Bound the retained CT23 Fusion reconstruction to model scores, `p_yes`
  probabilities, then entropy values in their verified column order.
- Added a direct retained-output check for the CT23 Fusion value `0.915094`,
  reported as `0.915` in the paper.
- Required Python 3.10, Transformers 4.44.0, PyTorch 2.7.0 with CUDA 12.8,
  and an NVIDIA A10 before historical training can begin.

## Verification

The local preflight must report `PREFLIGHT: PASS`, the retained CT23 check must
report `RETAINED CT23 FUSION: PASS`, and the unit suite must pass. On the GPU
host, each executed track must produce `summary.json`,
`per_example_predictions.npz`, and `checksums.sha256` after its bound seed
receipts pass.
