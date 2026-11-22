import pandas as pd, re
OUT = r"C:\StreamingACD\causal_first_pass"; SP = r"C:\Users\filip\AppData\Local\Temp\claude\9f358d908271be3f\f603f27f-85c5-46a5-88df-6d2e9957acbc\scratchpad\fc"
res = pd.read_parquet(OUT + r"\best_match.parquet")
res["band"] = pd.cut(res.best_sim, [-1, .4, .5, .6, .7, 1], labels=["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"])
quota = {"<0.4": 14, "0.4-0.5": 34, "0.5-0.6": 34, "0.6-0.7": 34, ">0.7": 34}
smp = pd.concat([g.sample(min(quota[b], len(g)), random_state=20261122) for b, g in res.groupby("band", observed=True)])
smp = smp.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle so I label blind to band
smp["pair_id"] = range(1, len(smp) + 1)
smp[["pair_id", "cluster_id", "band", "best_sim", "fc_url"]].to_csv(OUT + r"\label_sample_150.csv", index=False)
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet").set_index("cluster_id").narrative_text
with open(SP + r"\pairs_to_read.txt", "w", encoding="utf-8") as f:
    for r in smp.itertuples():
        tw = [re.sub(r"https?://\S+", "", x).strip()[:220] for x in texts[r.cluster_id].split("\n")][:6]
        f.write(f"=== PAIR {r.pair_id}\nFACT-CHECK ({r.fc_site}, {r.fc_date}): {r.fc_claim[:300]}\nTWEETS:\n- " + "\n- ".join(tw) + "\n\n")
print(len(smp), smp.band.value_counts().to_dict())
