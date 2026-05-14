# Clustering Subset Language Share Reproduction

Paper-safe claim: `The three-day clustering-analysis subset is approximately 87% English.`

Status: `reproduced_with_narrowed_denominator`

## Result

| Metric | Value |
|---|---:|
| Source rows | `87,911` |
| Sample size | `2,000` |
| English rows in sample | `1,732` |
| English share | `0.8660` (`86.6%`) |
| Non-English / other / short / unknown rows | `268` |
| Non-English / other / short / unknown share | `0.1340` (`13.4%`) |

## Top Detected Languages

| Language | Count | Share |
|---|---:|---:|
| `en` | `1,732` | `86.6%` |
| `es` | `61` | `3.0%` |
| `de` | `52` | `2.6%` |
| `nl` | `40` | `2.0%` |
| `fr` | `40` | `2.0%` |
| `it` | `17` | `0.8%` |

## Method

- Source: `/Users/sergiopinto/explainableACD/data/pipeline_output/expose_3day_full/2026-01-19_10-57/tweets.parquet`.
- Source rows: `87,911`.
- Language columns in source: none.
- Detector: `langdetect.detect` with `DetectorFactory.seed = 0`.
- Sampling: Polars `sample(n=2000, seed=42)`.
- Input column: `text`.
- Short text rule: `short` when stripped text has fewer than 20 characters.
- Invalid text rule: `unknown` when detection raises.

## Interpretation

This reproduces the old `~87% English` number only for the three-day clustering-analysis subset. It does not reproduce the claim that the full `1,522,909`-tweet corpus is `87%` English. The full-corpus detector-derived estimate remains `69.88%` English in `results/corpus_language_share_2026-05-12.*`.

## Paper-Safe Wording

```latex
In the three-day clustering-analysis subset, a seeded langdetect sample
(n=2,000) estimated the language distribution at 86.6\% English and
13.4\% non-English, motivating the use of multilingual embeddings.
```
