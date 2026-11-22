# causal_first_pass

One row per narrative for a first causal analysis (`build_sample.py`, `discover.py`, `effects.py`, `results_k10_h48.txt`), and a measurement of how many of those narratives had already been fact-checked (everything else in this folder, produced 2026-11-22).

## Fact-check matching: result

About **8% of the 3,955 narratives (329, 95% interval 6.5% to 11.1%)** make a main claim that an existing fact-check, published 2020-09-01 to 2020-11-30, already settles. Full counts and breakdowns (verdict, publisher, `ever_flagged`, tercile of `tweets_after`, timing): `factcheck_match_results.md`.

"Settles" is the triage question asked before fact-checking: could a fact-checker point to the existing check instead of writing a new one? Matches on the same subject with a different claim do not count.

## Inputs

| Input | Source |
|---|---|
| Narratives | `sample_k10_h48.parquet`: 3,955 narratives (clusters), K = 10, H = 48 h, from `psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/` |
| Tweets | that run's `tweets.parquet` (not in the repository). The first 10 tweets of each narrative, ordered by `created_at` then `tweet_id`, are listed by ID in `narrative_tweet_ids.csv` (39,550 IDs) |
| Fact-checks | Google Fact Check Tools API (ClaimReview), `claims:search`, `languageCode=en`, all publishers, queried 2026-11-22. 8,694 search terms (`factchecks/google_query_terms.json`), up to 300 results each; 90,839 unique reviews, 3,224 in the window |

PrimeFacts (Hugging Face `premsa/prime-facts`, CC BY-NC 4.0) was the intended source. It is gated, and access was requested but not granted at the time; only its README and LICENSE are in `factchecks/`.

## Method

| Step | Script | Output |
|---|---|---|
| Search terms (TF-IDF over narrative text, plus 47 seed terms) | `scripts/terms.py` | `factchecks/google_query_terms.json` |
| Fact-check retrieval | `scripts/sweep.py` | `factchecks/google_raw.jsonl.gz` |
| Deduplicate, window by claim date (review date if missing) | `scripts/dedupe.py`, `scripts/match.py` | `factchecks/google_factchecks_all.parquet`, `factchecks/factchecks_window.parquet` |
| Narrative text: first 10 tweets | `scripts/build_texts.py` | `narrative_texts_first10.parquet` (local only) |
| Embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, each tweet embedded separately (URLs removed), mean of the 10, L2-normalized; the model truncates at 128 tokens | `scripts/embed_tweets.py`, `scripts/match.py` | `narrative_emb_meantweets.npy` (rows ordered as `narrative_emb_ids.parquet`), `factchecks/claim_emb.npy` |
| Best match by cosine similarity | `scripts/match.py` | `best_match.parquet` |
| Similarity-only attempt (150 hand-labelled pairs; no threshold reached 90% precision) | `scripts/sample.py`, `scripts/labels.py`, `scripts/report.py` | `label_sample_150.csv`, `factcheck_match_results_similarity_only.md` |
| Judge tests: Haiku and Sonnet against hand labels, three prompt versions | `scripts/haiku150.py`, `scripts/haiku_v2.py`, `scripts/sonnet_v2.py`, `scripts/judge_v3.py`, `scripts/compare.py` | `haiku_check_150/`, `haiku_check_v2/`, `sonnet_check_v2/`, `haiku_check_v3/`, `sonnet_check_v3/`, `label_sample_100_new.csv`, `label_sample_100_v3.csv` |
| Final judge: top 5 candidates per narrative, all 3,955 narratives | `scripts/judge_top5.py` | `haiku_top5_all/candidates.parquet`, `haiku_top5_all/labels.jsonl` |
| Precision check (50 random positives, hand-labelled) and recall check (28 earlier hand-found matches) | `scripts/sample_pos.py`, `scripts/final_report.py` | `haiku_top5_all/precision_sample_50.csv`, `factcheck_match_results.md` |
| Verdict wording to five groups | `scripts/verdict.py` | used by the reports |
| Public, text-free copies | `scripts/prepare_release.py` | `narrative_tweet_ids.csv`, `*/calls_redacted.jsonl.gz` |

**Judge model.** `claude-haiku-5-5`, called through the Claude Code CLI 2.1.330 in headless mode with no tools (wrapper: `verifier/model_call.py` in `filipealtoe/factchecked-causal-priors`). One call per narrative; 3,958 calls including 3 parse retries; no missing answers. The exact prompt is in `scripts/judge_top5.py`.

**Prompt history.** The first two prompts asked whether the tweets made the *same claim*. That definition proved stricter than the triage purpose: on 100 new pairs, Haiku agreed on 10 of 13 same-claim pairs, and Sonnet's same-claim calls agreed with the hand labels in 12 of 18. The third prompt asks whether the fact-check *settles the narrative's main claim*. On 100 fresh pairs, Haiku found 5 of 5 covered pairs with 1 false positive; Sonnet found 5 of 5 with 3. Haiku was used for the full run.

**Correction.** Estimate = judge positives × precision ÷ recall = 316 × 39/50 ÷ 21/28. The interval is a percentile bootstrap (10,000 draws) over both check samples.

## Limitations

- All hand labels are one rater's (Claude, model `claude-opus-5-5`), read blind to similarity and model output, except the precision sample, which was drawn from model positives.
- The recall set (28 matches) was labelled partly under the stricter same-claim definition.
- The fact-check pool is a lower bound: the API answers keyword searches only.
- Only each narrative's 5 most similar fact-checks were judged.
- Group breakdowns use the judge's direct counts, assuming the same error rates in every group.

## Tweet text

This repository is public, so no tweet text is committed. Rehydrate `narrative_tweet_ids.csv` through the X API to rebuild the inputs. Tweets deleted since collection cannot be recovered. With the local `tweets.parquet`, `scripts/build_texts.py` and `scripts/prepare_release.py` rebuild the local-only files:

- `narrative_texts_first10.parquet`
- `tweet_id_text_map.parquet`
- `*/calls.jsonl`: the full call logs, with prompts

The redacted logs keep each prompt's SHA-256, so a rebuilt prompt can be checked against the one that was sent.

Scripts were written for this machine: paths are absolute, and `sweep.py` reads `GOOGLE_API_KEY` from the environment. The scratch scripts `embed_narr.py` (the first, truncated embedding, `narrative_emb.npy`, superseded) and `wait.sh` are kept for completeness.
