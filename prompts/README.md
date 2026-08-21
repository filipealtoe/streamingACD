# Prompt index

The prompt files correspond to two distinct experimental paths:

| File | Experimental path |
|---|---|
| [`checkworthiness_prompts_zeroshot_v4.yaml`](checkworthiness_prompts_zeroshot_v4.yaml) | Original v4 prompts and generation limits used by the [soft-label runner](../reproducibility/source_artifacts/checkworthiness/source_code/scripts/runners/run_llm_checkworthiness_v4.py) to create the checkability, verifiability, and harm features |
| [`checkworthiness_v4_prompts.md`](checkworthiness_v4_prompts.md) | Human-readable rendering used by [`scripts/run_llm_ablation.py`](../scripts/run_llm_ablation.py) for the frontier-LLM zero-shot ablation |
| [`claim_normalization_cikm2026.md`](claim_normalization_cikm2026.md) | Exact v1 system and few-shot templates used by the retained CT25 claim-normalization run |

The soft-label generation path used the YAML assistant prefix. The Opus ablation uses
the same system/user content but omits assistant prefill because that model endpoint
does not accept it; this behavior is explicit in
[`scripts/run_llm_ablation.py`](../scripts/run_llm_ablation.py).
