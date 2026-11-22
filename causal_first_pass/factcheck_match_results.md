# Fact-check match results

Counts only: no tweet text, no usernames. 3,955 narratives (`sample_k10_h48.parquet`), linked to tweets by `cluster_id`.

## Answer

**About 8% of narratives (329 of 3,955) make a main claim that an existing fact-check already settles** (95% interval 6.5% to 11.1%). The judge directly flagged 316 (8.0%); the estimate corrects that for its measured error rates.

## How it was done

1. **Fact-checks.** PrimeFacts is gated (access requested, not granted). Fact-checks came from the Google Fact Check Tools API, all publishers, via 8,694 keyword searches. Window 2020-09-01 to 2020-11-30 by claim date (review date when missing): **3,224 fact-checks**, 381 from PolitiFact. A lower bound, since the API only answers searches.
2. **Candidates.** Each narrative's first 10 tweets were embedded one by one (paraphrase-multilingual-mpnet-base-v2) and averaged; its 5 most similar fact-checks became candidates.
3. **Judge.** Claude Haiku saw the 10 tweets and the 5 candidates and named the one, if any, that settles the narrative's main claim, so a fact-checker could point to it instead of writing a new check. One call per narrative, all 3,955 narratives, no missing answers.
4. **Checks.** Precision: 50 random narratives Haiku called covered were read by hand. Recall: Haiku's answers on the 28 matches found in earlier hand labelling.

Similarity alone could not do this: no similarity threshold reached 90% correct matches (see `factcheck_match_results_similarity_only.md`).

## Judge accuracy

- Precision: 39 of 50 covered calls correct (78%). The wrong ones are mostly the same subject with a different claim.
- Recall: 21 of 28 earlier hand-found matches recovered (75%).
- 142 of the 316 matches were not the single most similar fact-check, so looking at 5 candidates mattered.
- Labels are one rater's (Claude's). The recall set was labelled partly under a stricter definition, and both checks are small, hence the wide interval.

## Matches by verdict of the fact-check

| verdict group | matched narratives |
|---|---|
| false | 130 |
| misleading or partly false | 86 |
| true | 65 |
| other or unrated | 21 |
| satire | 14 |

## Matches by publisher (top 10)

| publisher | matched narratives |
|---|---|
| snopes.com | 79 |
| politifact.com | 60 |
| misbar.com | 39 |
| usatoday.com | 32 |
| factcheck.org | 25 |
| washingtonpost.com | 20 |
| factcheck.afp.com | 17 |
| leadstories.com | 13 |
| apnews.com | 11 |
| rappler.com | 8 |

## Match rate: flagged versus unflagged

- ever_flagged true: 8.8% (49 of 559)
- ever_flagged false: 7.9% (267 of 3,396)

## Match rate by tercile of tweets_after

- low (0-3 tweets after): 6.4% (84 of 1,318)
- middle (3-12 tweets after): 6.9% (91 of 1,318)
- high (12-1638 tweets after): 10.7% (141 of 1,319)

Rates in these two sections are the judge's direct counts, not corrected; the correction is assumed to apply equally across groups.

## Timing

- Matches whose fact-check is dated on or before the narrative's t0: 212 of 316 (67%). The rest were fact-checked after the narrative started.

## Files

In `C:\StreamingACD\causal_first_pass\`: `haiku_top5_all\` (candidates, judge answers, call logs, the 50-call precision sample), `factchecks\` (API results and window table), hand labels `label_sample_150.csv`, `label_sample_100_new.csv`, `label_sample_100_v3.csv`, and the earlier similarity-only report.
