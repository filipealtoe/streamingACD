import pandas as pd, re, collections
OUT = r"C:\StreamingACD\causal_first_pass"
s = pd.read_parquet(OUT + r"\sample_k10_h48.parquet")
t = pd.read_parquet(r"C:\StreamingACD\psr\explainableACD\data\pipeline_output\streaming_full\2026-01-17_03-56\tweets.parquet",
                    columns=["tweet_id", "cluster_id", "text", "created_at"])
t = t[t.cluster_id.isin(s.cluster_id)].sort_values(["cluster_id", "created_at", "tweet_id"])
first = t.groupby("cluster_id").head(10)
texts = first.groupby("cluster_id")["text"].apply(lambda x: "\n".join(x)).rename("narrative_text").reset_index()
texts["n_tweets"] = first.groupby("cluster_id").size().values
texts.to_parquet(OUT + r"\narrative_texts_first10.parquet", index=False)
print(len(texts), texts.n_tweets.describe().to_dict())
ascii_share = texts.narrative_text.map(lambda x: sum(c.isascii() for c in x) / max(len(x), 1))
print("share of narratives mostly ASCII:", (ascii_share > 0.9).mean())
print("example lengths (chars):", texts.narrative_text.str.len().describe().to_dict())
