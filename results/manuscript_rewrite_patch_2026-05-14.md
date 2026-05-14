# Manuscript Rewrite Patch

Date: 2026-05-14

Scope: paper-side edits needed to align the current IJCAI-derived manuscript with the reproducibility package. This file is not a
paper source file; it is a direct patch guide for Overleaf or whichever editable manuscript source is used next.

## 0. Abstract Replacement

Use this abstract if the paper source still contains the old claim-normalization, full-corpus `87% English`, or formative-evaluation
claims:

```latex
Misinformation narratives that reach virality on social media typically do so
within a median of 1.5 days, while professional fact-checkers require a median
of 4.5 days to publish corrections. When a false claim circulates unaddressed,
polarization sets in around it, eroding the corrective impact of fact-checks
that arrive after peak spread. Existing virality prediction operates either on
individual posts or on coarse events, missing the narrative, a cluster of
semantically equivalent claims, as the natural unit of analysis for fact-checking
workflows.

We introduce narrative-level streaming virality prediction as a task formulation
for fact-checking triage. Our pipeline clusters semantically equivalent posts
into evolving narratives, detects anomalous activity over cluster time series,
normalizes clustered posts into claim-level representations, and predicts
post-detection spread through a Post-Detection Spread Ratio (PSR) metric that
quantifies the remaining intervention window at detection time. This formulation
surfaces triage decisions before peak virality while preserving intermediate
artifacts needed for reproducibility.

On CheckThat! 2024 check-worthiness, our reproduced 4-head DeBERTa-v3-large
multi-task model reaches F1=0.833 on CT24 with dimension-wise confidence
decomposition, and generalizes zero-shot to ClaimBuster (F1=0.973) and
CheckThat! 2023 (F1=0.933). A fusion classifier combining the DeBERTa ensemble
with LLM-derived features reaches F1=0.836 on CT24. For streaming virality
prediction, we package the feature matrix, PSR labels, train/test split, and
baseline result artifacts over 529 anomalous clusters.

We release dataset and reproducibility artifacts for narrative-level streaming
virality prediction, including virality labels, check-worthiness annotations, and
intermediate features over 535 narratives derived from 1.52M tweets.
```

Do not include the claim-normalization `10%` SOTA sentence unless Filipe restores the exact `N=300` Table 1 artifacts. Do not include
formative-evaluation percentages unless the anonymized response export and analysis script are added.

## 1. Dataset Language Claim

Do not write that the full `1.52M` corpus is `87%` English.

Use this narrowed wording:

```latex
In the three-day clustering-analysis subset, a seeded langdetect sample
(n=2,000) estimated the language distribution at 86.6\% English and
13.4\% non-English, motivating the use of multilingual embeddings.
```

Evidence:

- `results/corpus_language_share_2026-05-12.*`: full corpus detector estimate is `69.88%` English.
- `results/clustering_subset_language_share_2026-05-14.*`: three-day subset estimate is `86.6%` English.

## 2. Check-Worthiness Table

Replace the CT24 values in the check-worthiness table with reproduced values:

| Method | CT24 F1 | ClaimBuster F1 | CT23 F1 | Status |
|---|---:|---:|---:|---|
| Single DeBERTa | `0.821` | remove or leave blank | remove or leave blank | CT24 only reproduced |
| 3-seed DeBERTa ensemble | `0.834` | remove or leave blank | remove or leave blank | CT24 reproduced |
| 4-head MTL DeBERTa | `0.833` | `0.973` | `0.933` | reproduced |
| PCA-64 + LLM + Text (LogReg) | `0.694` | `0.894` | `0.846` | reproduced literal row |
| Fusion classifier | `0.836` | remove or leave blank | remove or leave blank | CT24 reproduced |

Do not keep:

- Single DeBERTa CT24 `0.824`;
- LLM-feature CT24 `0.761`;
- Fusion cross-dataset values unless separate artifacts are added;
- Single DeBERTa cross-dataset values unless separate artifacts are added;
- one row labeled `Llama2-7b Fine-tuned` with `0.802 / 0.920 / 0.898`.

If the paper needs the `0.802 / 0.920 / 0.898` values, split them into explicit external prior-work rows with citations rather
than presenting them as one reproduced Llama2 model.

## 3. Four-Head Training Description

Replace any sentence that says the four-head MTL model used Focal Loss, layer-wise learning-rate decay, R-Drop, or FGM with:

```latex
The four-head MTL model uses a shared DeBERTa encoder with one check-worthiness
classification head and three auxiliary regression heads. The reproduced run uses
three-phase training, weighted cross-entropy for the main task, MSE auxiliary losses,
and entropy weighting.
```

If the single-head seed ensemble is described, attach the regularizers there:

```latex
The single-head DeBERTa seed-ensemble trainer supports Focal Loss, layer-wise
learning-rate decay, R-Drop consistency regularization, and FGM adversarial
training, and the ensemble launcher enables those options for each seed model.
```

## 4. Anomaly Detection Results

Replace the anomaly result table/paragraph with the declared `min_rows=89` rerun:

| Detector | NAB | Detection | Median lead |
|---|---:|---:|---:|
| EXPoSE | `79.2` | `97.9%` | `+23h` |
| Random | `72.2` | `99.0%` | `+22h` |
| Bayesian Changept | `7.4` | `41.0%` | `+7.5h` |
| Etsy Skyline | `8.2` | `47.4%` | `+6h` |
| Steuber Z-score | `6.8` | `41.8%` | `+7h` |

Use this prose:

```latex
Under the declared min\_rows=89 rerun, EXPoSE reaches NAB 79.2,
detection 97.9\%, and median lead time +23h. It obtains the highest NAB score
among the evaluated detectors, while the random trigger baseline detects more
clusters (99.0\%) at lower NAB (72.2) and slightly lower median lead (+22h).
```

Do not claim that EXPoSE has the highest detection rate under this replacement table.

## 5. Sample Report Anecdote

Replace `567 engagements` with `564 peak engagements`.

Use:

```latex
The recovered report for claim 9dea2d0d-0604-4753-8bde-f9498ee5a2b0 predicted
virality 8.0 hours before its peak, reached 564 peak engagements, and assigned
71.7\% check-worthiness confidence.
```

Delete the anecdote instead if the exact claim ID is too specific for the manuscript.

## 6. Claim Normalization

Do not keep the CheckThat! 2025 Table 1 values `0.5583`, `0.5463`, or `0.5691` as reproduced unless Filipe provides the exact
`N=300` split, per-sample predictions, command, and summary.

If the table is not recovered, delete it or replace it with this explicit diagnostic boundary:

```latex
In a broader n=1,285 diagnostic comparison, the best packaged claim-normalization
run reaches average METEOR 0.345; this diagnostic is not the N=300 Table 1 result
from the earlier draft.
```

## 7. Formative Evaluation

Remove exact formative-evaluation statistics unless Filipe provides the anonymized response export and analysis script.

Delete or replace claims involving:

- `N=7` or `N=9`;
- `27` report pairs;
- `22/27` agreement;
- `16/27` full agreement;
- component means `4.04`, `3.81`, `3.78`, `3.11`;
- `8/9`, `7/9`, or `5/9` participant-count claims.

Use this fallback:

```latex
To support expert review of the generated reports, we prepared candidate report
stimuli spanning high-confidence positive, negative recommendation, and borderline
cases. We release the selection script and selected candidate-report tables. This
version reports the protocol and stimuli only; it does not claim agreement
percentages, usefulness means, or triage-effort statistics as empirical results.
```

## 8. Abstract-Safe Numbers

The abstract can safely keep:

- `1.52M` tweets;
- `535` narratives or normalized claims, depending on final terminology;
- CT24 four-head F1 `0.833`;
- ClaimBuster four-head F1 `0.973`;
- CT23 four-head F1 `0.933`.

The abstract should not keep:

- claim-normalization `10%` SOTA claim unless Filipe restores Table 1 artifacts;
- formative-evaluation quantitative claims;
- full-corpus `87% English`.

## Final Manuscript Pass Order

1. Apply the language-denominator correction.
2. Replace the check-worthiness table.
3. Replace the anomaly table and paragraph.
4. Fix the four-head training description.
5. Fix or delete the sample-report anecdote.
6. Remove claim-normalization and formative-evaluation exact claims unless Filipe provides artifacts.
7. Rerun `scripts/audit_full_paper_claims.py` after the manuscript text is updated.
