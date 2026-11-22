import sys, os, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from verdict import group
O = r"C:\StreamingACD\causal_first_pass"; FC = os.path.join(O, "factchecks")
P = "NYYYNYYYYN" "NYNYYNYYYN" "YYYNYYYYYY" "YYYYYYYNYY" "YYNYYNYYYY"; assert len(P) == 50
ps = pd.read_csv(os.path.join(O, "haiku_top5_all", "precision_sample_50.csv")); ps["correct"] = [c == "Y" for c in P]
ps.to_csv(os.path.join(O, "haiku_top5_all", "precision_sample_50.csv"), index=False)
w = pd.read_parquet(os.path.join(FC, "factchecks_window.parquet")); w["verdict"] = w.rating.map(group)
r = pd.DataFrame([json.loads(l) for l in open(os.path.join(O, "haiku_top5_all", "labels.jsonl"), encoding="utf-8")])
s = pd.read_parquet(os.path.join(O, "sample_k10_h48.parquet"), columns=["cluster_id", "t0", "ever_flagged", "tweets_after"])
d = s.merge(r, on="cluster_id"); d["matched"] = d.match.fillna(0) > 0
d["tercile"] = pd.qcut(d.tweets_after.rank(method="first"), 3, labels=["low", "middle", "high"])
rng_t = d.groupby("tercile", observed=True).tweets_after.agg(["min", "max"])
m = d[d.matched].copy(); m = m.join(w[["claim", "site", "verdict", "date"]], on="fc_index")
m["fc_before_t0"] = pd.to_datetime(m.date) <= pd.to_datetime(m.t0).dt.normalize()
# recall from earlier labelled positives
lab = pd.concat([pd.read_csv(os.path.join(O, f)) for f in ["label_sample_150.csv", "label_sample_100_new.csv", "label_sample_100_v3.csv"]])
pos = lab[lab.label.isin(["same claim", "covered"])].merge(r, on="cluster_id")
rec_hits = (pos.match.fillna(0) > 0).values
prec_hits = ps.correct.values; n_yes = int(d.matched.sum()); N = len(d)
prec, rec = prec_hits.mean(), rec_hits.mean(); est = n_yes * prec / rec
g = np.random.default_rng(20261125)
boot = [n_yes * g.choice(prec_hits, len(prec_hits)).mean() / max(g.choice(rec_hits, len(rec_hits)).mean(), 1e-9) for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
def md(df):
    df = df.reset_index(); cols = list(map(str, df.columns))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row) + " |" for row in df.itertuples(index=False)])
def rate(x): return f"{x.matched.mean():.1%} ({int(x.matched.sum())} of {len(x):,})"
L = ["# Fact-check match results", "",
     "Counts only: no tweet text, no usernames. 3,955 narratives (`sample_k10_h48.parquet`), linked to tweets by `cluster_id`.", "",
     "## Answer", "",
     f"**About {est/N:.0%} of narratives ({est:.0f} of {N:,}) make a main claim that an existing fact-check already settles** (95% interval {lo/N:.1%} to {hi/N:.1%}). The judge directly flagged {n_yes} ({n_yes/N:.1%}); the estimate corrects that for its measured error rates.", "",
     "## How it was done", "",
     "1. **Fact-checks.** PrimeFacts is gated (access requested, not granted). Fact-checks came from the Google Fact Check Tools API, all publishers, via 8,694 keyword searches. Window 2020-09-01 to 2020-11-30 by claim date (review date when missing): **3,224 fact-checks**, 381 from PolitiFact. A lower bound, since the API only answers searches.",
     "2. **Candidates.** Each narrative's first 10 tweets were embedded one by one (paraphrase-multilingual-mpnet-base-v2) and averaged; its 5 most similar fact-checks became candidates.",
     "3. **Judge.** Claude Haiku saw the 10 tweets and the 5 candidates and named the one, if any, that settles the narrative's main claim, so a fact-checker could point to it instead of writing a new check. One call per narrative, all 3,955 narratives, no missing answers.",
     "4. **Checks.** Precision: 50 random narratives Haiku called covered were read by hand. Recall: Haiku's answers on the 28 matches found in earlier hand labelling.", "",
     "Similarity alone could not do this: no similarity threshold reached 90% correct matches (see `factcheck_match_results_similarity_only.md`).", "",
     "## Judge accuracy", "",
     f"- Precision: {int(prec_hits.sum())} of 50 covered calls correct ({prec:.0%}). The wrong ones are mostly the same subject with a different claim.",
     f"- Recall: {int(rec_hits.sum())} of {len(rec_hits)} earlier hand-found matches recovered ({rec:.0%}).",
     f"- {int((m.match > 1).sum())} of the {n_yes} matches were not the single most similar fact-check, so looking at 5 candidates mattered.",
     "- Labels are one rater's (Claude's). The recall set was labelled partly under a stricter definition, and both checks are small, hence the wide interval.", "",
     "## Matches by verdict of the fact-check", "",
     md(m.verdict.value_counts().rename("matched narratives").to_frame().rename_axis("verdict group")), "",
     "## Matches by publisher (top 10)", "",
     md(m.site.value_counts().head(10).rename("matched narratives").to_frame().rename_axis("publisher")), "",
     "## Match rate: flagged versus unflagged", "",
     f"- ever_flagged true: {rate(d[d.ever_flagged])}", f"- ever_flagged false: {rate(d[~d.ever_flagged])}", "",
     "## Match rate by tercile of tweets_after", ""] + \
    [f"- {t} ({int(x['min'])}-{int(x['max'])} tweets after): {rate(d[d.tercile == t])}" for t, x in rng_t.iterrows()] + ["",
     "Rates in these two sections are the judge's direct counts, not corrected; the correction is assumed to apply equally across groups.", "",
     "## Timing", "",
     f"- Matches whose fact-check is dated on or before the narrative's t0: {int(m.fc_before_t0.sum())} of {n_yes} ({m.fc_before_t0.mean():.0%}). The rest were fact-checked after the narrative started.", "",
     "## Files", "",
     "In `C:\StreamingACD\causal_first_pass\`: `haiku_top5_all\` (candidates, judge answers, call logs, the 50-call precision sample), `factchecks\` (API results and window table), hand labels `label_sample_150.csv`, `label_sample_100_new.csv`, `label_sample_100_v3.csv`, and the earlier similarity-only report."]
open(os.path.join(O, "factcheck_match_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
