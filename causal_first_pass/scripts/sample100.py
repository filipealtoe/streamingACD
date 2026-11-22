import pandas as pd, re
OUT = r"C:\StreamingACD\causal_first_pass"
res = pd.read_parquet(OUT + r"\best_match.parquet")
order = ["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"]
res["band"] = pd.cut(res.best_sim, [-1, .4, .5, .6, .7, 1], labels=order).astype(str)
old = pd.read_csv(OUT + r"\label_sample_150.csv")
pool = res[~res.cluster_id.isin(old.cluster_id)]
quota = {"0.5-0.6": 20, "0.6-0.7": 40, ">0.7": 40}
smp = pd.concat([pool[pool.band == b].sample(n, random_state=20261123) for b, n in quota.items()]).sample(frac=1, random_state=2).reset_index(drop=True)
smp["pair_id"] = range(201, 301)
smp[["pair_id", "cluster_id", "band", "best_sim", "fc_url"]].to_csv(OUT + r"\label_sample_100_new.csv", index=False)
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet").set_index("cluster_id").narrative_text
with open("pairs100.txt", "w", encoding="utf-8") as f:
    for r in smp.itertuples():
        tw = [re.sub(r"https?://\S+", "", x).strip()[:200] for x in texts[r.cluster_id].split("\n") if x.strip()][:6]
        f.write(f"=== PAIR {r.pair_id}\nFC: {r.fc_claim[:260]}\n- " + "\n- ".join(tw) + "\n\n")
print(smp.band.value_counts().to_dict())
