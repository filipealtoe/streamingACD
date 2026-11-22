import pandas as pd, json, os
O = r"C:\StreamingACD\causal_first_pass"
s = pd.concat([pd.read_csv(os.path.join(O, "label_sample_150.csv")).assign(set="old150"),
               pd.read_csv(os.path.join(O, "label_sample_100_new.csv")).assign(set="new100")])
for m in ["haiku_check_v2", "sonnet_check_v2"]:
    h = pd.DataFrame([json.loads(l) for l in open(os.path.join(O, m, "labels.jsonl"), encoding="utf-8")]).rename(columns={"haiku": m})
    s = s.merge(h, on="pair_id")
x = s[s.set == "new100"]
print(pd.crosstab(x.label, x.sonnet_check_v2, margins=True).to_string())
for m in ["haiku_check_v2", "sonnet_check_v2"]:
    for st, g in s.groupby("set"):
        p = g[m].eq("same claim"); t = g.label.eq("same claim")
        print(m, st, "calls", p.sum(), "agree", (p & t).sum(), "mine", t.sum(), "extra", sorted(g[p & ~t].pair_id), "missed", sorted(g[~p & t].pair_id))
