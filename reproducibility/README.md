# Reproducibility Package

This directory tracks evidence for paper-facing claims without committing large datasets, model checkpoints, or raw prediction dumps by default.

## Contents

- `artifact_checksums_2026-05-11.md`: human-readable artifact manifest.
- `artifact_checksums_2026-05-11.json`: machine-readable artifact manifest.
- `runs/deberta_mtl_cikm_20260512_134553/`: reproduced four-head MTL run bundle.
- `source_artifacts/`: small source result artifacts copied from the prior local `explainableACD` checkout for paper-claim
  auditing. Large raw datasets, parquet tables, prediction dumps, and model checkpoints are still referenced by path and checksum
  rather than copied into Git.
- `source_artifacts/sha256sums.txt`: checksum manifest for the copied source result artifacts.
- `../scripts/verify_local_artifacts.py`: read-only verifier for the local artifact set.
- `../scripts/audit_full_paper_claims.py`: paper-level audit that checks all IJCAI result-bearing claims against packaged and
  local evidence.

## Current Scope

The current manifest covers local evidence for:

- CT24 split sizes and source files;
- CT23 and ClaimBuster source files;
- packaged CT24 clean splits, CT24 LLM feature Parquets, CT23/ClaimBuster benchmark inputs, and benchmark LLM feature Parquets for
  local check-worthiness reruns;
- the US Election 2020 raw corpus count and date range;
- corpus language-share recomputation from the recovered Kaggle-derived parquet;
- the claim-normalization runner/prompts, CheckThat! 2025 CSV inputs, diagnostic artifact, and Filipe handoff boundary for the
  unsupported Table 1 values;
- the candidate canonical streaming run `2026-01-17_03-56`;
- the anomaly EXPoSE and baseline replacement table under the declared `min_rows=89` rerun;
- the clustering threshold-ablation statistics;
- DeBERTa CT24 single-model and ensemble prediction artifacts;
- the PCA-64 + LLM + text Logistic Regression rerun and replacement-candidate boundary for the unsupported CT24 `0.761` row;
- the Together baseline comparison metadata, including the unsupported Llama2 constants;
- the Together baseline runner used to compare available Llama 3.x/Mistral/Mixtral models against paper baseline constants;
- regularized single-head DeBERTa/seed-ensemble scripts that contain the Focal/LLRD/R-Drop/FGM recipe;
- formative-evaluation stimulus selection artifacts, but not participant responses;
- the recovered sample explainability report artifact for claim `9dea2d0d-0604-4753-8bde-f9498ee5a2b0`;
- virality PSR labels, enhanced 42-feature matrix, split manifest, scripts, and tabular baseline results.

The current run bundle additionally covers:

- four-head MTL CT24 test F1 `0.8333`;
- CT24 dev/test probability artifacts for paired comparisons;
- CT24 dev/test labels and `Sentence_id` arrays in the exact MTL join order;
- ClaimBuster F1 `0.9730` and CT23 F1 `0.9327` from the reproduced checkpoint;
- exact training and benchmark scripts used for this run.
- explicit evidence that this four-head MTL run did not use the Focal/LLRD/R-Drop/FGM recipe from the single-head ensemble.

## Rules

- Small manifests, summary JSON files, and verifier scripts can live in Git.
- Raw datasets, embeddings, model checkpoints, and large Parquet/NPY files should be regenerated or stored externally with SHA-256 checksums.
- A paper metric is accepted only when a verifier or evaluation script can recompute it from a declared artifact.
- Unsupported or conflicting claims stay out of the paper until rerun or rewritten.

## Local Verification

The verifier prefers packaged artifacts in this repository. It reads from the previous local checkout only for large external
artifacts that are intentionally not committed.

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  uv run --with pyarrow python scripts/verify_local_artifacts.py
```

Expected current result:

```text
Summary: 0 failures, 7 warnings
```

The warnings are expected until resolved:

- CT24 LLM feature checkpoint JSON files were not packaged, although the feature Parquets are present;
- the raw election corpus file does not contain a language column for the removed `87% English` claim;
- single DeBERTa recomputes near, but not exactly at, the old paper value;
- the fusion probability file is not available as a standalone external artifact, although the reproduced summary is packaged;
- the four-head checkpoint is omitted from Git and represented by checksum plus saved predictions;
- claim normalization Table 1 and formative-evaluation quantitative claims are waiting on Filipe-owned artifacts.

See `../results/current_reproducibility_status.md` for the latest readable status summary.

## Four-Head Training Details

The reproduced four-head MTL run is not the same training recipe as the regularized single-head DeBERTa seed ensemble.
The four-head run uses a shared DeBERTa encoder, a main check-worthiness classification head, three auxiliary regression heads,
3-phase training, weighted cross-entropy for the main task, MSE auxiliary losses, and entropy weighting.

The Focal Loss, layer-wise learning-rate decay, R-Drop, and FGM recipe is supported by:

- `source_artifacts/checkworthiness/regularized_single_head/finetune_deberta_multimodel.py`
- `source_artifacts/checkworthiness/regularized_single_head/ensemble_deberta_seeds.py`

Do not describe those regularizers as part of the reproduced four-head MTL run unless a new four-head retrain actually includes
them and packages its script, command, predictions, and metrics.

Use `../results/four_head_training_regularizers_paper_fix_2026-05-14.md` for manuscript replacement text.

## Llama2 Baseline Row

The Table 3 `Baseline Llama2-7b Fine-tuned` row is not reproduced by this repo. The packaged Together run stores Llama2 only as
an external CT24 prior-work constant and the paired local predictions are for available Llama 3.x/Mistral/Mixtral models, not
Llama2. The ClaimBuster `0.920` and CT23 `0.898` values are separate SOTA reference constants in the benchmark scripts, not
Llama2 outputs.

Keep these values only as clearly cited external references, or split the row by source/dataset in the paper.

## LLM-Feature Classifier Table Row

The Table 3 row labeled `LLM_features PCA 64 + Logreg` is only partially reproduced. The literal PCA-64 + LLM + text Logistic
Regression rerun supports:

- CT24 F1: `0.694`, not the paper's `0.761`;
- ClaimBuster F1: `0.894`;
- CT23 F1: `0.846`.

Use `../results/llm_feature_classifier_paper_fix_2026-05-14.md` to rewrite the row to
`PCA-64 + LLM + Text (LogReg)` with CT24 `0.694`, or rewrite the method and row around the separate DeBERTa CLS + LLM LogReg
replacement candidate before using CT24 `0.793`.

## Anomaly Detection Baseline Comparison

The original anomaly baseline paragraph is not reproduced as written. The repo now has a coherent replacement table under the
same declared `min_rows=89` filter used for the EXPoSE rerun:

- EXPoSE: NAB `79.2`, detection `97.9%`, median lead `+23h`;
- Random: NAB `72.2`, detection `99.0%`, median lead `+22h`;
- Bayesian Changept: NAB `7.4`, detection `41.0%`, median lead `+7.5h`;
- Etsy Skyline: NAB `8.2`, detection `47.4%`, median lead `+6h`;
- Steuber Z-score: NAB `6.8`, detection `41.8%`, median lead `+7h`.

Use `../results/anomaly_baseline_paper_fix_2026-05-14.md` to rewrite the anomaly paragraph/table. Do not claim that EXPoSE has the
highest detection rate in this replacement table; the random trigger baseline has higher detection under this setting.

## Claim Normalization Table 1

The claim-normalization Table 1 values are not reproduced by the packaged artifact. The repo currently packages:

- `source_artifacts/claim_normalization/comparison_test_20260113_123010.json`
- `source_artifacts/claim_normalization/source_code/`
- `source_artifacts/claim_normalization/check_that_25/`

This artifact has `n=1285` and best average METEOR `0.3449`, not the paper's `N=300` Table 1 values around `0.5583`, `0.5463`,
and `0.5691`.

Use `../results/claim_normalization_handoff_2026-05-14.md` for the exact artifact request to Filipe. Do not restore the Table 1
values unless the `N=300` split, per-sample predictions, summary, command, and checksums are provided.

The copied source package includes its historical README and command examples. Treat those as recovery notes until an exact rerun
produces and packages the Table 1 output artifacts.

## Formative Evaluation

The local repo contains study-stimulus artifacts, not the participant-response data needed to reproduce the human-study statistics.
The packaged stimulus artifacts are:

- `source_artifacts/formative_evaluation/select_claims_for_study.py`
- `source_artifacts/formative_evaluation/expose_fast_study_claims_selection.csv`
- `source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv`

These support that candidate reports were selected/generated for a study-style evaluation, but they cannot recompute agreement,
full agreement, component-usefulness means, report-structure alignment, triage-effort reduction, or trust-calibration counts.
The paper also has an internal sample-size conflict: the abstract/limitations say `N=7`, while the body says `N=9`.

Use `../results/formative_evaluation_paper_fix_2026-05-14.md` to remove the unsupported quantitative formative-evaluation claims
from the manuscript unless the missing participant-response artifact and analysis script are recovered.

## Sample Explainability Report

The local repo now packages the recovered sample report artifact:

- `source_artifacts/explainability_report/03_9dea2d0d_eng564.html`

This artifact supports the check-worthiness confidence and lead-time parts of the paper example, but corrects the engagement
value. Supported values are:

- claim ID: `9dea2d0d-0604-4753-8bde-f9498ee5a2b0`;
- check-worthiness confidence: `71.7%`;
- detection lead time: `8.0h`;
- peak engagement: `564`;
- total engagement in the report metadata: `1,172`;
- cluster size: `73 tweets`.

Use `../results/sample_report_paper_fix_2026-05-14.md` to rewrite the paper example from `567 engagements` to `564 peak
engagements`, or remove the anecdotal example if exact examples are omitted from the revision.

## CT24 Metric Recalculation

Recompute CT24 metrics from saved probability artifacts with:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/recompute_ct24_metrics.py
```

The output is written to `results/ct24_metric_reproduction_2026-05-12.json`.

## Full Paper Claim Audit

Run:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/effort/.venv/bin/python scripts/audit_full_paper_claims.py
```

The script writes:

- `results/full_paper_claim_audit_2026-05-12.json`
- `results/full_paper_claim_audit_2026-05-12.md`

Expected current result: non-zero exit. That is intentional because several IJCAI paper claims are still not reproducible from the
repo package as written. The audit separates reproduced claims from mismatches and missing exact artifacts.

## Virality PSR Table

The Table 2 virality/PSR row is backed by packaged artifacts:

- `source_artifacts/virality/psr_labels.parquet`
- `source_artifacts/virality/features_enhanced.parquet`
- `source_artifacts/virality/tuned_baselines.json`
- `source_artifacts/virality/complete_baselines.json`
- `source_artifacts/virality/latex_table.tex`
- `../results/virality_split_manifest_2026-05-12.json`

The split manifest records the exact deterministic split used by the scripts:

```text
train_test_split(row_indices, test_size=0.2, random_state=42)
```

Current packaged counts: `529` PSR rows, `42` feature columns, `423` train rows, and `106` test rows. The copied result artifacts
match the paper's reported Table 2 values. A future fresh rerun should execute the packaged `run_tuned_baselines.py` and
`run_complete_baselines.py` in a matching Python environment.

## Corpus Language Share

The paper claim that the corpus is approximately `87%` English is not reproduced. The source dataset was recovered from
[Kaggle](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets) through KaggleHub, but the CSV files do not include a
language column. The canonical local parquet was therefore recomputed with the same `langdetect` rule used by the streaming pipeline:

```bash
EXPLAINABLE_ACD_ROOT=/Users/sergiopinto/explainableACD \
  /Users/sergiopinto/explainableACD/.venv/bin/python scripts/recompute_corpus_language_share.py
```

The output is written to:

- `results/corpus_language_share_2026-05-12.json`
- `results/corpus_language_share_2026-05-12.md`

Current result: `1,064,253 / 1,522,909 = 0.6988` English. The script intentionally exits non-zero while the paper claim remains
unrevised.
