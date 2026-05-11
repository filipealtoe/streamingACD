# Reproducibility Status

Date: 2026-05-11

## Current Decision Point

The resubmission should not carry exact headline metrics from the prior paper unless those values can be regenerated from code or verified from saved artifacts.

The first reproducibility target is the CT24 check-worthiness baseline described as the four-head multi-task model. The prior repository contains useful scripts and local artifacts, but the specific four-head test predictions and checkpoint for the reported value have not been found in committed files.

## Working Findings

- The previous repository documents CT24 single-model, ensemble, and fusion results.
- The four-head design-log result is documented, but the matching saved checkpoint and test predictions have not been found.
- The entropy-weighted multi-task training script loads train and dev data, but does not perform a test-set evaluation or save test predictions.
- Other multi-task variants do include test evaluation and saved probability outputs, making them better candidates for a fresh reproducibility run.

## Next Slice

Create the minimal CT24 reproducibility path:

1. copy only the required data-loading, training, and evaluation code;
2. define a command-line run contract with explicit seed, threshold, and output directory;
3. save `dev_probs.npy`, `test_probs.npy`, `results.json`, and a run manifest;
4. rerun CT24 first, then rerun cross-dataset checks against CT23 and ClaimBuster.
