import pandas as pd, numpy as np
OUT = r"C:\StreamingACD\causal_first_pass"
L = ("TTUUUUUUUU"
     "UUTUUUTUUU"
     "UTTUUUUUUU"
     "TUUTUUUUTU"
     "UUSUUSUUUT"
     "UUTTTSSUUU"
     "UUTUUTUUUT"
     "USUUUTUUUU"
     "UUTTUUUUUU"
     "UUTSTUSUUU"
     "UUUUUUUUUS"
     "UUTUUUTUSS"
     "UUUUUUUUUT"
     "UUUUUUUUTU"
     "UUUUTTTUUU")
assert len(L) == 150, len(L)
name = {"S": "same claim", "T": "same topic, different claim", "U": "unrelated"}
smp = pd.read_csv(OUT + r"\label_sample_150.csv"); smp["label"] = [name[c] for c in L]
smp.to_csv(OUT + r"\label_sample_150.csv", index=False)
order = ["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">0.7"]
tab = pd.crosstab(smp.band, smp.label).reindex(order).fillna(0).astype(int); tab["pairs"] = tab.sum(1)
print(tab.to_string())
res = pd.read_parquet(OUT + r"\best_match.parquet")
res["band"] = pd.cut(res.best_sim, [-1, .4, .5, .6, .7, 1], labels=order).astype(str)
pop = res.band.value_counts(); smp["w"] = smp.band.map(pop) / smp.band.map(smp.band.value_counts())
smp["same"] = smp.label.eq("same claim")
print(smp[smp.same][["pair_id", "band", "best_sim"]].sort_values("best_sim").to_string())
for t in np.arange(0.60, 0.86, 0.01):
    s = smp[smp.best_sim >= t]
    if len(s) == 0: break
    print(f"t={t:.2f} pairs={len(s):3d} same={s.same.sum():2d} precision={s.same.mean():.2f} weighted={np.average(s.same, weights=s.w):.2f} narratives_above={(res.best_sim >= t).sum()}")
