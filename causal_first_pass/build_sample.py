"""Build one row per narrative for a first causal analysis.

Unit: a narrative (cluster) that reaches K tweets at time t0, with t0 at least
H hours before the moment the pipeline stopped creating clusters.
Early window: the first K tweets. Outcome window: (t0, t0 + H hours].
Output holds aggregates only: no text, no user names, no user ids.
"""
import os, sys
import numpy as np, pandas as pd

K, H = 10, 48
D = os.path.expanduser("~/mnt/StreamingACD/psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/")
OUT = os.path.dirname(os.path.abspath(__file__))

cols = ["tweet_id", "user_id", "cluster_id", "text", "created_at", "retweet_count_at_collection",
        "likes_at_collection", "cluster_similarity", "user_join_date", "user_followers_count",
        "country", "collected_at"]
tw = pd.read_parquet(D + "tweets.parquet", columns=cols).sort_values(["cluster_id", "created_at"]).reset_index(drop=True)
tw["rank"] = tw.groupby("cluster_id").cumcount() + 1
cap = tw.loc[tw["rank"] == 1, "created_at"].max()
lag_h = (tw.collected_at - tw.created_at).dt.total_seconds() / 3600
print("collection lag hours quantiles:", lag_h.quantile([.01, .25, .5, .75, .99]).round(2).tolist(), "null share", round(lag_h.isna().mean(), 3))
print("country null share:", round(tw.country.isna().mean(), 3), " join date null share:", round(tw.user_join_date.isna().mean(), 4))

t0 = tw.loc[tw["rank"] == K].set_index("cluster_id")["created_at"]
t0 = t0[t0 <= cap - pd.Timedelta(hours=H)]
tw = tw[tw.cluster_id.isin(t0.index)].copy()
tw["t0"] = tw.cluster_id.map(t0)
early = tw[tw["rank"] <= K].copy()
late = tw[(tw.created_at > tw.t0) & (tw.created_at <= tw.t0 + pd.Timedelta(hours=H))]

txt = early.text.fillna("")
early["has_url"] = txt.str.contains("http", regex=False)
early["has_mention"] = txt.str.contains("@", regex=False)
early["n_words"] = txt.str.split().str.len()
early["age_days"] = (early.created_at - early.user_join_date).dt.total_seconds() / 86400
early["eng"] = early.likes_at_collection + early.retweet_count_at_collection

g = early.groupby("cluster_id")
df = pd.DataFrame({
    "t0": t0,
    "max_followers": g.user_followers_count.max(),
    "median_followers": g.user_followers_count.median(),
    "unique_users": g.user_id.nunique(),
    "account_age_days": g.age_days.median(),
    "url_share": g.has_url.mean(),
    "mention_share": g.has_mention.mean(),
    "mean_words": g.n_words.mean(),
    "similarity": g.cluster_similarity.mean(),
    "hours_to_k": (g.created_at.max() - g.created_at.min()).dt.total_seconds() / 3600,
    "early_engagement": g.eng.sum(),
})
early_users = early.groupby("cluster_id").user_id.agg(set)
lg = late.groupby("cluster_id")
df["tweets_after"] = lg.size().reindex(df.index).fillna(0)
df["engagement_after"] = (late.likes_at_collection + late.retweet_count_at_collection).groupby(late.cluster_id).sum().reindex(df.index).fillna(0)
late_users = lg.user_id.agg(set).reindex(df.index)
df["new_users_after"] = [len(l - early_users[c]) if isinstance(l, set) else 0 for c, l in late_users.items()]

# context: how busy the whole stream was in the 24 hours before t0, and position in the corpus
all_t = pd.read_parquet(D + "tweets.parquet", columns=["created_at"]).created_at.sort_values().values
t0v = df.t0.values
df["stream_volume_24h"] = np.searchsorted(all_t, t0v) - np.searchsorted(all_t, t0v - np.timedelta64(24, "h"))
df["day_index"] = (df.t0 - all_t[0]).dt.total_seconds() / 86400
df["hour_utc"] = df.t0.dt.hour

ts = pd.read_parquet(D + "cluster_timeseries.parquet", columns=["cluster_id", "timestamp", "is_anomaly_trigger"])
first_flag = ts[ts.is_anomaly_trigger].groupby("cluster_id").timestamp.min()
df["ever_flagged"] = df.index.isin(first_flag.index)

df = df.reset_index()
df.to_parquet(os.path.join(OUT, "sample_k10_h48.parquet"), index=False)
print("rows", len(df), "cap", cap)
print(df.drop(columns=["cluster_id", "t0"]).describe().T[["min", "50%", "max"]].round(2).to_string())
print("zero outcome share:", round((df.tweets_after == 0).mean(), 3))
