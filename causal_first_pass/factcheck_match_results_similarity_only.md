# Fact-check match results

Counts only: no tweet text, no usernames. Sample: 3,955 narratives (`sample_k10_h48.parquet`), linked to tweets by `cluster_id` (all 3,955 found).

## Deviations from the plan

- **Source.** PrimeFacts (Hugging Face `premsa/prime-facts`) is gated and access was not yet granted. Fact-checks came instead from the Google Fact Check Tools API (ClaimReview), all publishers, English-language search. The API only answers keyword searches, so 8,694 search terms (the most distinctive terms of each narrative plus 47 election-period seed terms) were run, up to 300 results each. The count below is therefore a lower bound on fact-checks in the window.
- **Publishers.** Every publisher the API returns was kept, as agreed (all are fact-checking organisations). Verdict wordings differ by publisher, so they were mapped to five groups.
- **Date.** Claim date, or the review date when the claim date is missing.
- **Narrative embedding.** The model reads at most 128 tokens, about two or three tweets. Each of the first 10 tweets (URLs removed) was embedded separately and the 10 vectors were averaged.
- **Labels.** One rater (Claude) labelled the 150 pairs, in shuffled order without seeing similarity or band. "Same claim" means the tweets assert or discuss the specific claim checked; shared general election talk counts as unrelated.
- **Sample per band.** Only 14 narratives fall below 0.4, so all 14 were taken and 34 from each other band (150 in total).

## Fact-checks in the window (2020-09-01 to 2020-11-30)

- Unique fact-checks retrieved (any date): 90,839
- In the window, with claim text: **3,224**
- PolitiFact among them: 381

| publisher (top 10) | fact-checks |
|---|---|
| snopes.com | 413 |
| politifact.com | 381 |
| factcheck.afp.com | 375 |
| leadstories.com | 225 |
| misbar.com | 220 |
| boomlive.in | 208 |
| usatoday.com | 184 |
| factcheck.org | 160 |
| apnews.com | 139 |
| factly.in | 108 |

| verdict group | fact-checks |
|---|---|
| false | 1,999 |
| misleading or partly false | 691 |
| other or unrated | 269 |
| true | 237 |
| satire | 28 |

## The 150 labelled pairs, by similarity band

| band | narratives in band | same claim | same topic, different claim | unrelated | pairs |
|---|---|---|---|---|---|
| <0.4 | 14 | 0 | 0 | 14 | 14 |
| 0.4-0.5 | 86 | 0 | 2 | 32 | 34 |
| 0.5-0.6 | 1,068 | 0 | 8 | 26 | 34 |
| 0.6-0.7 | 2,252 | 3 | 9 | 22 | 34 |
| >0.7 | 535 | 7 | 9 | 18 | 34 |

## Threshold

**No threshold reaches 90% "same claim".** Precision among labelled pairs at or above each threshold:

| threshold | narratives at or above | labelled pairs | same claim | precision |
|---|---|---|---|---|
| 0.60 | 2,787 | 68 | 10 | 15% |
| 0.65 | 1,517 | 50 | 8 | 16% |
| 0.70 | 535 | 34 | 7 | 21% |
| 0.72 | 338 | 23 | 5 | 22% |
| 0.74 | 208 | 15 | 4 | 27% |
| 0.76 | 125 | 10 | 2 | 20% |
| 0.78 | 72 | 6 | 2 | 33% |
| 0.80 | 43 | 4 | 2 | 50% |

The 10 same-claim pairs have similarities from 0.63 to 0.82, while unrelated pairs reach 0.80 and above, so similarity alone does not separate them. Step 5 could not be applied.

## Share of narratives matched

With no usable threshold, the match rate is estimated from the labels instead: each band's same-claim rate, times the number of narratives in that band.

- Estimated narratives whose best fact-check is the same claim: **309 of 3,955 (7.8%)**, 95% bootstrap interval 110 to 552 (2.8% to 13.9%).
- All of it comes from bands 0.6-0.7 (3 of 34 labelled) and above 0.7 (7 of 34).
- This counts only fact-checks the keyword search found, and only each narrative's single best match, so it is likely an underestimate.

## Verdicts of matches; flagged versus unflagged; terciles

Not computed: with no threshold there is no set of matched narratives, and the 10 same-claim pairs are too few to break down. Verdicts of those 10: misleading or partly false 3, true 2, false 2, other or unrated 2, satire 1.

For reference only (this is **not** a match rate: about one pair in five above 0.7 is the same claim), the share of narratives whose best similarity exceeds 0.7:

- ever_flagged true: 22.9% of 559; false: 12.0% of 3396
- tweets_after tercile low (0-3): 10.4%
- tweets_after tercile middle (3-12): 11.7%
- tweets_after tercile high (12-1638): 18.5%

## Files

In `C:\StreamingACD\causal_first_pass\`: `factchecks\` (raw API results, deduplicated table, window table, search terms, claim embeddings), `narrative_texts_first10.parquet`, `narrative_emb_meantweets.npy`, `best_match.parquet`, `label_sample_150.csv` (pair ids, cluster ids, similarity, fact-check URL, label).
