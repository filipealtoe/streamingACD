import json, pandas as pd, collections
FC = r"C:\StreamingACD\causal_first_pass\factchecks"
rows = {}
for line in open(FC + r"\google_raw.jsonl", encoding="utf-8"):
    for c in json.loads(line)["claims"]:
        for r in c.get("claimReview", []):
            key = r.get("url")
            if not key or key in rows: continue
            rows[key] = {"claim": c.get("text", ""), "claimant": c.get("claimant"), "claim_date": (c.get("claimDate") or "")[:10],
                         "review_date": (r.get("reviewDate") or "")[:10], "publisher": r.get("publisher", {}).get("name"),
                         "site": r.get("publisher", {}).get("site"), "rating": r.get("textualRating"), "url": key,
                         "review_language": r.get("languageCode")}
df = pd.DataFrame(rows.values())
df["date"] = df.claim_date.where(df.claim_date != "", df.review_date)
print("unique reviews", len(df), "missing claim_date", (df.claim_date == "").sum())
w = df[(df.date >= "2020-09-01") & (df.date <= "2020-11-30")]
print("in window", len(w))
print(w.site.value_counts().head(40).to_string())
df.to_parquet(FC + r"\google_factchecks_all.parquet", index=False)
