# Check-worthiness soft-label source snapshot

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
