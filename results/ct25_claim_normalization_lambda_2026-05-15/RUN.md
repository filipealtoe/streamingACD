# CT25 Claim Normalization Lambda Reproduction

Date: 2026-05-15

## Purpose

This folder records the Lambda rerun used to verify the paper claim that the CT25 English claim-normalization component reaches roughly 0.558 METEOR on the first 300 English test examples.

## Result

| Metric | Value |
|---|---:|
| Split | CT25 English test |
| Rows | first 300 rows, idx 0-299 |
| Model | Mistral-7B-Instruct-v0.3 |
| Inference mode | local Hugging Face model on Lambda A10 |
| METEOR | 0.5569184651538435 |
| Prior SOTA used by script | 0.4569 |
| Delta vs prior SOTA | +0.1000184651538435 |
| Retrieval hits | 227 |
| Local generations | 73 |
| API cost | 0.0 USD |

## Source

- Source repository: `https://github.com/filipealtoe/explainableACD`
- Source commit: `aa4864c299bbcd0b6dd60c13ace7742a2d0644a1`
- Remote run directory: `/home/ubuntu/claimnorm-repro/claim_norm`
- Local private archive: `/Users/sergiopinto/repro/claimnorm-lambda-20260515`

## Dataset Inputs

The script used local copies of the official CheckThat! 2025 Task 2 English files:

| File | Rows | SHA256 |
|---|---:|---|
| `train-eng.csv` | 11374 | `e95e1a7f9881ec7ab76b0e32a67eafae47bbda45582da80993ca756c7f8dc1fc` |
| `dev-eng.csv` | 1171 | `63c0f28e09cd7bc3fa1c4ff6afe647ed556e4f555563d58e6ab1eded3c64c3a5` |
| `test-eng-gold.csv` | 1285 | `977a09196c4779d5c23fbd8a5a784b5c1e92b27aede5843dbcfb0b883846b9c5` |

The dataset files themselves are not committed here.

## Command

```bash
cd /home/ubuntu/claimnorm-repro/claim_norm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/.local/bin/uv run python scripts/run_claim_normalization_ct25.py \
  --local \
  --local-model mistralai/Mistral-7B-Instruct-v0.3 \
  --split test \
  --limit 300 \
  --num-examples 5 \
  --topic-clusters 10 \
  --retrieval-threshold 0.85 \
  --claim-verify-threshold 0.5 \
  --batch-size 8 \
  --no-resume
```

## Artifacts

- `summary.json`: aggregate metrics emitted by the script.
- `per_sample_metrics.jsonl`: per-row metric/provenance records with benchmark text and labels removed.
- `lambda_environment.txt`: Lambda hardware/runtime inventory and remote result-file hashes.
- `checksums.sha256`: checksums for the committed artifacts in this folder.

## Truth Boundary

This reproduces the approximate first-300 CT25 claim-normalization result using local Lambda inference, not the Together API. It is not a full 1285-row CT25 test run. The score is close to the previously cited 0.558 value, but not byte-identical because the original exact prediction artifact was not present in the source repository.
