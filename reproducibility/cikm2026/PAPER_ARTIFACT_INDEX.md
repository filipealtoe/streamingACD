# Paper-to-artifact index

This index connects the CIKM 2026 paper components to their public implementation,
parameters, prompts, and evidence. Paths are relative to the repository root.

| Paper component | Implementation and parameters | Prompt or data input | Result or verification evidence |
|---|---|---|---|
| Claim normalization | [source snapshot](../source_artifacts/claim_normalization/source_code/) and [parameters](MODEL_PARAMETERS.json) | [exact v1 prompt](../../prompts/claim_normalization_cikm2026.md); official CheckThat! 2025 Task 2 English files identified by checksum in the run record | [retained CT25 run](../../results/ct25_claim_normalization_lambda_2026-05-15/) |
| Check-worthiness soft-label features | [exact v4 extraction runner](../source_artifacts/checkworthiness/source_code/scripts/runners/run_llm_checkworthiness_v4.py), [original v4 configuration](../../prompts/checkworthiness_prompts_zeroshot_v4.yaml), and [parameters](MODEL_PARAMETERS.json) | The three v4 checkability, verifiability, and harm prompts | [four-head training results and saved predictions](../runs/deberta_mtl_cikm_20260512_134553/) |
| Four-head multi-task DeBERTa | [training source](../runs/deberta_mtl_cikm_20260512_134553/scripts/finetune_deberta_mtl.py) and [exact command](../runs/deberta_mtl_cikm_20260512_134553/command.txt) | CT24 text joined with the three LLM soft-label dimensions | [results and saved label order](../runs/deberta_mtl_cikm_20260512_134553/) plus the [checkpoint manifest](CHECKPOINT.md) |
| Frontier-LLM check-worthiness ablation | [inference](../../scripts/run_llm_ablation.py) and [prompt loader](../../scripts/prompts.py) | [human-readable v4 prompts](../../prompts/checkworthiness_v4_prompts.md) | [analysis script](../../scripts/analyze_llm_ablation.py) and the retained CT24 label order |
| Streaming claim and cluster registry | [canonical output tables](../../psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/) | Normalized claims, aggregate clusters, time series, and window results | [pipeline summary](../source_artifacts/pipeline/streaming_full_2026-01-17_03-56_summary.json) |
| Online clustering threshold evidence | [threshold investigation](../../docs/investigations/clustering_threshold_ablation_20260519.md) | Canonical streaming run | [cluster statistics](../source_artifacts/clustering/cluster_statistics.json) |
| Anomaly-detection evaluation | [retained evaluation artifacts](../source_artifacts/anomaly/) | Aggregate benchmark and streaming detection outputs | Baselines, detector comparison, F2/lead-time evaluation, and EXPoSE grid search in the linked directory |
| Narrative-level virality dataset | [feature matrix](../source_artifacts/virality/features_enhanced.parquet) and [virality labels](../source_artifacts/virality/psr_labels.parquet) | 529 cluster instances, 42 detection-time engineered features, and virality labels | [data card](DATA_CARD.md), [schema](SCHEMA.md), [checksums](checksums.sha256), and the focused verifier |
| Numeric virality baselines | [fresh-fit reproducer](../../scripts/reproduce_cikm2026_tabular_baselines.py) and [parameters](MODEL_PARAMETERS.json) | Public feature matrix and deterministic `RandomState(42)` split | Fresh-fit comparison with [full-precision results](../source_artifacts/virality/tuned_baselines.json) |
| Sequence, Hawkes, and BERTweet virality rows | [historical experiment scripts](../../psr/explainableACD/experiments/scripts/) | Derived cluster-level inputs | [complete baselines](../source_artifacts/virality/complete_baselines.json), [tuned baselines](../source_artifacts/virality/tuned_baselines.json), and [paper table rendering](../source_artifacts/virality/latex_table.tex) |

## Verification entry points

Run the complete public-package consistency check:

```bash
uv run scripts/verify_cikm2026_artifacts.py
```

Freshly fit the six numeric virality baselines:

```bash
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

After downloading the four-head checkpoint release asset, verify its identity together
with the public package:

```bash
uv run scripts/verify_cikm2026_artifacts.py --checkpoint /path/to/best_model.pt
```
