# Narrative Case Study Candidates

Date: 2026-05-19
Source run: `psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56`
Source for PSR labels: `psr/explainableACD/experiments/results/virality/psr_labels.parquet`

## Filter audit

| Stage | Description | Passing |
|---|---|---:|
| Initial | All claims | 535 |
| F1 | Procedural keyword present AND no direct candidate-family attack | 27 |
| F2 | `viral_confidence` > 0.7 (used as proxy for the 4-head "overall" score) | 7 |
| F3 | Computed lead time (peak engagement timestamp minus anomaly trigger timestamp) >= 4 h | 3 |
| F4 | `total_tweets` >= 10 | 3 |
| F5 | Per-dimension variation >= 0.1 | **SKIPPED** |
| Final | Top 5 by `viral_confidence` then PSR | 3 |

### Data availability gap (load-bearing)

Per-claim 4-Parallel-Heads dimensional scores (checkability, verifiability, harm
potential, overall) are **not persisted** in any released artifact for the 535
normalized claims. The `aux_checkability/aux_verifiability/aux_harm/main_probs`
files at `results/llm_ablation_20260517_110633/mtl_predictions/` are predictions
on the CT24 test split, not on the streaming claims.

Consequences:

- Filter 5 (dimensional variation) cannot be applied; it has been skipped.
- The "Dimensional scores" row of each candidate below is marked
  **unavailable**. Producing them requires running the four-head checkpoint
  (`data/4-heads_model/best_model.pt`) over the 535 normalized claim texts,
  which the spec forbids ("No model loading, no inference").
- Filter 2 uses `viral_confidence` (the pipeline's `is_viral` probability
  output) as a proxy for the missing "overall check-worthiness" score.
  `viral_confidence` is a virality signal, not a check-worthiness signal;
  it correlates but is not identical. This substitution is flagged on every
  candidate.

### Filter 1 keyword list (for audit)

- Procedural include: `ballot`, `voting machine`, `mail-in`, `mail in`,
  `absentee`, `vote count`, `votes count`, `recount`, `certif`, `audit`,
  `poll worker`, `polling place`, `polling station`, `voter registration`,
  `voter id`, `voter roll`, `election official`, `election worker`,
  `signature verification`, `precinct`, `ballot box`, `drop box`,
  `chain of custody`, `tabulat`, `vote tally`, `tally`, `dominion`,
  `voter fraud`.
- Candidate-attack exclude: `hunter biden`, `biden family`, `trump family`,
  `don jr`, `donald trump jr`, `hunter's laptop`, `biden corruption`,
  `trump pee`, `trump tax`.

The lead-time and peak-engagement values used by F3 were computed from
`cluster_timeseries.parquet` (the `claims.parquet` columns `peak_time` and
`lead_time_hours` are entirely null in the released bundle).

Three candidates passed all applicable filters, so no relaxation of F1 was
required.

---

## Candidate 1: Mail-in ballot counting in battleground states

**Topic:** mail-in ballot counting timeline
**Cluster ID:** 23761
**Cluster size:** 24 posts

**Pipeline trace:**

| Stage | Output |
|---|---|
| Cluster formation | 2020-10-19 20:00 UTC |
| Anomaly detection trigger | 2020-11-04 07:00 UTC, z-score 3.37 |
| Lead time before peak | 5.0 hours |
| Engagement at detection | 30 |
| Engagement at peak (12:00 UTC) | 3,479 |
| Detection-time engagement (PSR `e_detect`) | 38 |
| Final engagement (PSR `e_final`) | 5,416 |
| Predicted virality | viral, viral_confidence 0.85 |
| Actual PSR | 0.993 |

**Representative post phrasings (paraphrased, derived from cluster member tweets
ordered by distance from cluster centroid):**

1. A get-out-the-vote appeal urging Pennsylvania voters to back one ticket,
   framed around the certainty that the count would eventually be completed.
2. A user expressing confusion about why all ballots are not simply counted
   before a winner is declared, and asking who decides what gets counted first.
3. A jocular post invoking a historical figure to dramatize ongoing
   vote-counting in Washington State.
4. A defense of slow state-by-state vote counting as the expected behavior of
   the United States electoral system, contrasted with framing it as
   irregular.
5. A summary of asymmetric counting demands from one campaign: calls to halt
   counting where the lead was eroding and to continue counting where the
   lead might be regained.

**Normalized claim (verbatim from the pipeline output):**
"Counting mail-in ballots in key battleground states will skew the election
results in favor of Trump."

**Dimensional scores:**

| Dimension | Confidence | Binary decision |
|---|---|---|
| Checkability | unavailable | unavailable |
| Verifiability | unavailable | unavailable |
| Harm potential | unavailable | unavailable |
| Overall | 0.85 (proxy from `viral_confidence`) | check-worthy |

**Suitability notes:** Procedurally focused (mail-in counting timeline);
underlying narrative was extensively fact-checked across major
fact-checking organizations as part of the "blue shift" and "Stop the Count"
coverage. Five-hour lead before peak engagement on November 4 is a clean
illustrative example. The normalized claim summary is partisan in framing
even though the cluster contents are predominantly procedural commentary,
which is itself an interesting failure mode the case study could highlight if
the paper wishes to discuss claim normalization fidelity. PSR 0.99 places this
near the top of the corpus by actual virality.

---

## Candidate 2: Kentucky early vote counts

**Topic:** early vote count, Kentucky
**Cluster ID:** 25588
**Cluster size:** 91 posts

**Pipeline trace:**

| Stage | Output |
|---|---|
| Cluster formation | 2020-10-20 10:00 UTC |
| Anomaly detection trigger | 2020-11-03 23:00 UTC, z-score 4.29 |
| Lead time before peak | 92.0 hours |
| Engagement at detection | 46 |
| Engagement at peak (2020-11-07 19:00 UTC) | 322 |
| Detection-time engagement (PSR `e_detect`) | 77 |
| Final engagement (PSR `e_final`) | 676 |
| Predicted virality | viral, viral_confidence 0.85 |
| Actual PSR | 0.886 |

**Representative post phrasings (paraphrased):**

1. A pre-election poll reference showing one candidate ahead by roughly 17
   points in Kentucky, with comparison to the prior cycle's wider margin.
2. An election-night observation that one candidate appeared to be winning
   Kentucky after roughly two-thirds of votes had been tallied.
3. A breakdown of poll results by Kentucky region and demographic group,
   noting where the other candidate held a narrower lead.
4. A commentary speculating whether the tighter-than-expected Kentucky margin
   signaled broader weakness for the incumbent.
5. An observation that the early-called states matched expectations, with
   neither indicating a landslide for either side.

**Normalized claim (verbatim from the pipeline output):**
"Joe Biden is unexpectedly leading in early vote counts in Kentucky,
potentially challenging Donald Trump's expected win."

**Dimensional scores:**

| Dimension | Confidence | Binary decision |
|---|---|---|
| Checkability | unavailable | unavailable |
| Verifiability | unavailable | unavailable |
| Harm potential | unavailable | unavailable |
| Overall | 0.85 (proxy from `viral_confidence`) | check-worthy |

**Suitability notes:** Borderline procedural. The cluster nominally concerns
vote counting in Kentucky, but the content is predominantly partisan poll
commentary; only the act of counting is procedural. The 92-hour lead time
is unusually long, which is rhetorically attractive but raises questions
about whether peak engagement is meaningfully tied to the detection event
or is simply the post-election general engagement curve. Total engagement
(322) is small. Weaker overall than Candidate 1 or 3 for paper purposes;
include only if a "long lead, low engagement" example is desired.

---

## Candidate 3: Pennsylvania uncounted mail-in ballots

**Topic:** Pennsylvania mail-in ballots, post-election dispute
**Cluster ID:** 72106
**Cluster size:** 88 posts

**Pipeline trace:**

| Stage | Output |
|---|---|
| Cluster formation | 2020-11-02 13:00 UTC |
| Anomaly detection trigger | 2020-11-04 07:00 UTC, z-score 3.37 |
| Lead time before peak | 14.0 hours |
| Engagement at detection | 301 |
| Engagement at peak (2020-11-04 21:00 UTC) | 2,678 |
| Detection-time engagement (PSR `e_detect`) | 1,031 |
| Final engagement (PSR `e_final`) | 6,851 |
| Predicted virality | viral, viral_confidence 0.85 |
| Actual PSR | 0.850 |

**Representative post phrasings (paraphrased):**

1. A characterization of remarks attributed to a state attorney general
   predicting one candidate would lose Pennsylvania if all ballots were
   counted.
2. A campaign-source claim cataloguing specific Pennsylvania counties with
   tens of thousands of uncounted ballots said to favor that campaign.
3. A betting-style post forecasting a particular Pennsylvania margin.
4. A comparison invoking Taiwan's same-day vote-counting timeline to question
   why the Pennsylvania count was taking longer.
5. A profanity-laced denunciation of the state governor, asserting a
   particular outcome with allusions to ballot harvesting and fraud.

**Normalized claim (verbatim from the pipeline output):**
"The legitimacy of Donald Trump's Pennsylvania win in the 2020 US
presidential election is disputed due to uncounted mail-in ballots."

**Dimensional scores:**

| Dimension | Confidence | Binary decision |
|---|---|---|
| Checkability | unavailable | unavailable |
| Verifiability | unavailable | unavailable |
| Harm potential | unavailable | unavailable |
| Overall | 0.85 (proxy from `viral_confidence`) | check-worthy |

**Suitability notes:** Heavily procedural (county-level ballot counts,
counting timelines, comparisons to other jurisdictions). Underlying narrative
was extensively fact-checked. The normalized claim references "Trump's
Pennsylvania win," which is factually inverted (Pennsylvania was called for
Biden); this is a claim-normalization artifact worth flagging if used. Cluster
contents include a profane post (#5 above), which the paper draft should not
quote even in paraphrase if maximum decorum is required; consider showing the
first three or four phrasings only. Politically charged context (Pennsylvania
was the decisive state). Lead time 14 hours, PSR 0.85.

---

## Recommendation

**Top pick: Candidate 1 (mail-in ballot counting in battleground states,
cluster 23761).**

Rationale:
- Highest PSR (0.99) among the three, providing a clean "predicted-viral,
  actually-viral" story for §4.
- Short, clean five-hour lead time on November 4 is intuitive for readers.
- Cluster contents are recognizably procedural (counting timeline,
  state-by-state mechanics) and the underlying narrative is among the
  most extensively fact-checked of the cycle.
- Small cluster (24 posts) keeps the case-study figure compact.
- The mismatch between the partisan-framed normalized claim and the
  more procedural cluster contents is a useful talking point about
  claim-normalization fidelity, if the paper wants to acknowledge a
  limitation.

Candidate 3 is the strongest backup; choose it if the paper benefits from a
larger cluster and a higher-engagement detection point.

Candidate 2 is included for completeness but is not recommended on its own:
the procedural angle is weak, total engagement is low, and the 92-hour lead
time is hard to claim as causally meaningful.

## Outstanding action before any of these are paper-ready

Run the four-head checkpoint at `data/4-heads_model/best_model.pt` over the
535 normalized claim texts and persist per-claim
`{checkability, verifiability, harm, overall}` probabilities. Without that,
§4 cannot honestly describe the dimensional decomposition for the selected
case.
