import json, os, re, pandas as pd
O = r"C:\StreamingACD\causal_first_pass"
r = pd.DataFrame([json.loads(l) for l in open(os.path.join(O, "haiku_top5_all", "labels.jsonl"), encoding="utf-8")])
w = pd.read_parquet(os.path.join(O, "factchecks", "factchecks_window.parquet"))
best = pd.read_parquet(os.path.join(O, "best_match.parquet"))[["cluster_id", "fc_index"]].rename(columns={"fc_index": "best_fc"})
# recall check against all earlier positives (strict "same claim" or "covered")
lab = pd.concat([pd.read_csv(os.path.join(O, f)) for f in ["label_sample_150.csv", "label_sample_100_new.csv", "label_sample_100_v3.csv"]])
pos = lab[lab.label.isin(["same claim", "covered"])].merge(r, on="cluster_id").merge(best, on="cluster_id")
pos["found_same_fc"] = pos.fc_index == pos.best_fc
print("earlier positives:", len(pos), "Haiku covered:", int((pos.match > 0).sum()), "with the same fact-check:", int(pos.found_same_fc.sum()))
print("missed:", sorted(pos[pos.match == 0].pair_id))
yes = r[r.match > 0].sample(50, random_state=20261125).reset_index(drop=True)
yes["check_id"] = range(1, 51)
yes.to_csv(os.path.join(O, "haiku_top5_all", "precision_sample_50.csv"), index=False)
texts = pd.read_parquet(os.path.join(O, "narrative_texts_first10.parquet")).set_index("cluster_id").narrative_text
with open("pos50.txt", "w", encoding="utf-8") as f:
    for x in yes.itertuples():
        tw = [re.sub(r"https?://\S+", "", t).strip()[:200] for t in texts[x.cluster_id].split("\n") if t.strip()][:6]
        f.write(f"=== CHECK {x.check_id}\nFC: {w.claim.iloc[int(x.fc_index)][:260]}\n- " + "\n- ".join(tw) + "\n\n")
