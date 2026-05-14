# Corpus Language Share Reproduction

Paper claim: `Corpus is approximately 87% English.`

Status: `not_reproduced`

## Result

| Metric | Value |
|---|---:|
| Total rows | `1,522,909` |
| English rows | `1,064,253` |
| English share | `0.6988` (`69.88%`) |
| Non-English or unknown rows | `458,656` |
| Non-English or unknown share | `0.3012` (`30.12%`) |

## Top Languages

| Language | Count | Share |
|---|---:|---:|
| `en` | `1,064,253` | `0.6988` |
| `es` | `95,260` | `0.0626` |
| `de` | `68,682` | `0.0451` |
| `fr` | `62,757` | `0.0412` |
| `nl` | `50,637` | `0.0333` |
| `it` | `40,737` | `0.0267` |
| `tr` | `22,044` | `0.0145` |
| `ro` | `15,472` | `0.0102` |
| `pt` | `15,000` | `0.0098` |
| `id` | `10,887` | `0.0071` |
| `vi` | `8,992` | `0.0059` |
| `da` | `7,400` | `0.0049` |
| `ca` | `7,351` | `0.0048` |
| `ar` | `5,109` | `0.0034` |
| `hi` | `4,938` | `0.0032` |
| `no` | `4,365` | `0.0029` |
| `af` | `3,529` | `0.0023` |
| `pl` | `3,403` | `0.0022` |
| `fi` | `2,991` | `0.0020` |
| `fa` | `2,990` | `0.0020` |

## Method

- Detector: `langdetect.detect` with `DetectorFactory.seed = 0`.
- Input: original `tweet` text from the canonical local parquet.
- Short or invalid text is counted as `unknown`, matching the pipeline helper behavior.
- The downloaded Kaggle CSV headers do not include `lang` or `language`, so this is detector-derived.

## Conclusion

The recomputed canonical corpus language share is about 70% English, not 87%. The online Kaggle CSV headers do not expose a language column, so this result is detector-derived.

The old `~87%` value was recovered for a narrower denominator: the three-day clustering-analysis subset in
`/Users/sergiopinto/explainableACD/data/pipeline_output/expose_3day_full/2026-01-19_10-57/tweets.parquet`. See
`results/clustering_subset_language_share_2026-05-14.*` for the seeded `n=2,000` sample that gives `86.6%` English.
