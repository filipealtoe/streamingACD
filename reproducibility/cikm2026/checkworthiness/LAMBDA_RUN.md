# Two-track check-worthiness GPU reproduction

<!-- Sérgio Pinto, 2026-08-28 01:11 WEST — bound the retained CT23 Fusion reconstruction and the GPU rerun to their verified inputs and environment. -->

The run contract is frozen in [`LAMBDA_RUN_SPEC.json`](LAMBDA_RUN_SPEC.json).
It produces two separately named result tracks:

- `historical`: the recorded split, ensemble, temperature, fusion, and
  threshold configuration;
- `strict`: model and threshold selection on an internal CT24 training split,
  followed by ClaimBuster and CT23 evaluation.

The historical track trains the retained `seed_0` configuration with effective
RNG seed 42, gradient accumulation 2, and FGM epsilon 0.5, followed by the
`seed_456` configuration with gradient accumulation 4 and FGM epsilon 1.0.
Both runs use FP16, Python 3.10, Transformers 4.44.0, PyTorch 2.7.0 with
CUDA 12.8, and an NVIDIA A10 (compute capability 8.6). The runner checks this
environment before training.

For CT23, the Encoder Only cell uses the retained `seed_0` vector, while the
Fusion cell combines the retained `seed_0` and `seed_456` vectors before adding
the XGBoost probability component.

The strict preflight groups duplicate text before splitting and checks that its
training and internal-validation rows do not overlap the external benchmarks.

From the repository root, prepare the run directory outside the repository:

```bash
python scripts/run_cikm2026_checkworthiness_lambda.py preflight \
  --data-root /path/to/data \
  --run-root /path/to/cikm-checkworthiness-run
```

On the NVIDIA A10 host with the packages in `requirements-lambda.txt`
installed, reproduce the historical paper cells with:

```bash
python scripts/run_cikm2026_checkworthiness_lambda.py train \
  --track historical \
  --run-root /path/to/cikm-checkworthiness-run

python scripts/run_cikm2026_checkworthiness_lambda.py evaluate \
  --track historical \
  --run-root /path/to/cikm-checkworthiness-run
```

The independent strict track can then be run with:

```bash
python scripts/run_cikm2026_checkworthiness_lambda.py train \
  --track strict \
  --run-root /path/to/cikm-checkworthiness-run

python scripts/run_cikm2026_checkworthiness_lambda.py evaluate \
  --track strict \
  --run-root /path/to/cikm-checkworthiness-run
```

Each track retains the environment, model identities, full-precision metrics,
text-free per-example prediction arrays, and SHA-256 manifest.
