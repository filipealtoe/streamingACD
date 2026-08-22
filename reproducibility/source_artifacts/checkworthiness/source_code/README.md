# Check-worthiness soft-label source snapshot

<!-- Sérgio Pinto, 2026-08-21 19:21 PDT — added the retained PCA/text feature extraction, embedding/PCA, and cross-benchmark evaluation sources. -->
<!-- Sérgio Pinto, 2026-08-21 19:58 PDT — added the original encoder-only training and benchmark-evaluation sources used by the recovered paper row. -->

This directory preserves the v4 feature-extraction runner and prompt configuration
used to create the checkability, verifiability, and harm soft labels consumed by the
four-head DeBERTa run.

- Source repository: `https://github.com/filipealtoe/explainableACD`
- Source commit: `4ef8fb77c130f291efc236f7b3e05de7992b251e`
- Runner: `scripts/runners/run_llm_checkworthiness_v4.py`
- Runner SHA-256:
  `adef3e287b19b08ff4770cd6e8b37223c87bdab966c3f3422b7d52a1b0f4d1f9`
- Prompt: `prompts/checkworthiness_prompts_zeroshot_v4.yaml`
- Prompt SHA-256:
  `ac167033361da470b3ce3f811f2eaa38a0b23e678e69c3a2c17a22df0e867f66`

The runner reads CT24 feature splits from `data/processed/CT24_features/` and writes
the three-dimensional v4 features to `data/processed/CT24_llm_features_v4/`. It calls
`mistralai/Mistral-Small-24B-Instruct-2501` through the Together OpenAI-compatible
endpoint with temperature `0`, JSON output, token log-probabilities, and the prompt
limits stored in the YAML file.

No source posts, labels, generated responses, or credentials are included in this
snapshot.

## PCA-64 + LLM + text-feature classifier

The same snapshot also retains the three scripts that define the LLM-feature
classifier row:

- `scripts/feature_engineering/extract_text_features.py` extracts the 35
  deterministic text features;
- `scripts/feature_engineering/generate_embeddings.py` generates normalized
  BGE-small embeddings and fits PCA on the CT24 training split;
- `scripts/benchmark_pca_llm_text.py` fits the scaled Logistic Regression on
  CT24 train plus development data and evaluates CT24, ClaimBuster, and CT23.

The public numerical bundle removes all source text while preserving the 153
features and labels consumed by the final scaler and classifier.

## Encoder-only DeBERTa

The snapshot also includes the original single-encoder implementation and its
benchmark evaluators:

- `scripts/finetuning/finetune_deberta_multimodel.py` defines the
  DeBERTa-v3-large fine-tuning and threshold sweep;
- `scripts/evaluation/evaluate_deberta_claimbuster.py` defines the
  ClaimBuster evaluation;
- `scripts/evaluation/evaluate_deberta_ct23.py` defines the CT23 evaluation.

These files are preserved unchanged from the historical workspace and are
checksum-bound by the CIKM code manifest.
