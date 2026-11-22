import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from verdict import group
OUT = r"C:\StreamingACD\causal_first_pass"; FC = OUT + r"\factchecks"
order = ["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"]
w = pd.read_parquet(FC + r"\factchecks_window.parquet"); w["verdict"] = w.rating.map(group)
res = pd.read_parquet(OUT + r"\best_match.parquet")
res["band"] = pd.cut(res.best_sim, [-1, .4, .5, .6, .7, 1], labels=order).astype(str)
s = pd.read_parquet(OUT + r"\sample_k10_h48.parquet", columns=["cluster_id", "ever_flagged", "tweets_after"])
res = res.merge(s, on="cluster_id")
res["tercile"] = pd.qcut(res.tweets_after.rank(method="first"), 3, labels=["low", "middle", "high"])
terc_rng = res.groupby("tercile", observed=True).tweets_after.agg(["min", "max"])
smp = pd.read_csv(OUT + r"\label_sample_150.csv"); smp["same"] = smp.label.eq("same claim")
pop = res.band.value_counts()
rate = smp.groupby("band").same.mean().reindex(order)
est = float((pop.reindex(order) * rate).sum())
rng = np.random.default_rng(20261122); boots = []
for _ in range(5000):
    tot = 0
    for b in order:
        x = smp[smp.band == b].same.values; tot += pop[b] * rng.choice(x, len(x)).mean()
    boots.append(tot)
lo, hi = np.percentile(boots, [2.5, 97.5]); N = len(res)
tab = pd.crosstab(smp.band, smp.label).reindex(order).fillna(0).astype(int); tab["pairs"] = tab.sum(1)
tab.insert(0, "narratives in band", pop.reindex(order).values)
scan = []
for t in [0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]:
    x = smp[smp.best_sim >= t]
    scan.append([f"{t:.2f}", int((res.best_sim >= t).sum()), len(x), int(x.same.sum()), f"{x.same.mean():.0%}"])
def md(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = [[f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r] for r in d.itertuples(index=False)]
    return chr(10).join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])
hi7 = lambda g: f"{(g.best_sim > 0.7).mean():.1%}"
L = []
L += ["# Fact-check match results", "",
      "Counts only: no tweet text, no usernames. Sample: 3,955 narratives (`sample_k10_h48.parquet`), linked to tweets by `cluster_id` (all 3,955 found).", "",
      "## Deviations from the plan", "",
      "- **Source.** PrimeFacts (Hugging Face `premsa/prime-facts`) is gated and access was not yet granted. Fact-checks came instead from the Google Fact Check Tools API (ClaimReview), all publishers, English-language search. The API only answers keyword searches, so 8,694 search terms (the most distinctive terms of each narrative plus 47 election-period seed terms) were run, up to 300 results each. The count below is therefore a lower bound on fact-checks in the window.",
      "- **Publishers.** Every publisher the API returns was kept, as agreed (all are fact-checking organisations). Verdict wordings differ by publisher, so they were mapped to five groups.",
      "- **Date.** Claim date, or the review date when the claim date is missing.",
      "- **Narrative embedding.** The model reads at most 128 tokens, about two or three tweets. Each of the first 10 tweets (URLs removed) was embedded separately and the 10 vectors were averaged.",
      "- **Labels.** One rater (Claude) labelled the 150 pairs, in shuffled order without seeing similarity or band. \"Same claim\" means the tweets assert or discuss the specific claim checked; shared general election talk counts as unrelated.",
      "- **Sample per band.** Only 14 narratives fall below 0.4, so all 14 were taken and 34 from each other band (150 in total).", "",
      "## Fact-checks in the window (2020-09-01 to 2020-11-30)", "",
      f"- Unique fact-checks retrieved (any date): {len(pd.read_parquet(FC + r'/google_factchecks_all.parquet')):,}",
      f"- In the window, with claim text: **{len(w):,}**", f"- PolitiFact among them: {int((w.site == 'politifact.com').sum())}", "",
      md(w.site.value_counts().head(10).rename("fact-checks").to_frame().rename_axis("publisher (top 10)")), "",
      md(w.verdict.value_counts().rename("fact-checks").to_frame().rename_axis("verdict group")), "",
      "## The 150 labelled pairs, by similarity band", "", md(tab), "",
      "## Threshold", "",
      "**No threshold reaches 90% \"same claim\".** Precision among labelled pairs at or above each threshold:", "",
      md(pd.DataFrame(scan, columns=["threshold", "narratives at or above", "labelled pairs", "same claim", "precision"]), index=False), "",
      "The 10 same-claim pairs have similarities from 0.63 to 0.82, while unrelated pairs reach 0.80 and above, so similarity alone does not separate them. Step 5 could not be applied.", "",
      "## Share of narratives matched", "",
      "With no usable threshold, the match rate is estimated from the labels instead: each band's same-claim rate, times the number of narratives in that band.", "",
      f"- Estimated narratives whose best fact-check is the same claim: **{est:.0f} of {N:,} ({est/N:.1%})**, 95% bootstrap interval {lo:.0f} to {hi:.0f} ({lo/N:.1%} to {hi/N:.1%}).",
      "- All of it comes from bands 0.6-0.7 (3 of 34 labelled) and above 0.7 (7 of 34).",
      "- This counts only fact-checks the keyword search found, and only each narrative's single best match, so it is likely an underestimate.", "",
      "## Verdicts of matches; flagged versus unflagged; terciles", "",
      "Not computed: with no threshold there is no set of matched narratives, and the 10 same-claim pairs are too few to break down. Verdicts of those 10: "
      + ", ".join(f"{k} {v}" for k, v in smp[smp.same].merge(res[["cluster_id", "fc_rating"]], on="cluster_id").fc_rating.map(group).value_counts().items()) + ".", "",
      "For reference only (this is **not** a match rate: about one pair in five above 0.7 is the same claim), the share of narratives whose best similarity exceeds 0.7:", "",
      f"- ever_flagged true: {hi7(res[res.ever_flagged])} of {int(res.ever_flagged.sum())}; false: {hi7(res[~res.ever_flagged])} of {int((~res.ever_flagged).sum())}",
      ] + [f"- tweets_after tercile {t} ({int(r['min'])}-{int(r['max'])}): {hi7(res[res.tercile == t])}" for t, r in terc_rng.iterrows()] + ["",
      "## Files", "",
      "In `C:\StreamingACD\causal_first_pass\`: `factchecks\` (raw API results, deduplicated table, window table, search terms, claim embeddings), `narrative_texts_first10.parquet`, `narrative_emb_meantweets.npy`, `best_match.parquet`, `label_sample_150.csv` (pair ids, cluster ids, similarity, fact-check URL, label)."]
open(OUT + r"\factcheck_match_results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
