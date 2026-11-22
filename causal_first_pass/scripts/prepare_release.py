"""Make the public, text-free copies of this folder's outputs.

- narrative_tweet_ids.csv: the first 10 tweet IDs of each narrative, in the order used
  (created_at, then tweet_id). Rehydrating these IDs rebuilds the judge's inputs.
- tweet_id_text_map.parquet: local only (gitignored), tweet ID -> text.
- <run>/calls_redacted.jsonl.gz: each model call without its prompt (which carries tweet
  text); keeps model, response, token usage, timing, and the prompt's SHA-256 and length.
- factchecks/google_raw.jsonl.gz: compressed raw Google Fact Check API responses.
"""
import glob, gzip, hashlib, json, os, shutil
import pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWEETS = r"C:\StreamingACD\psr\explainableACD\data\pipeline_output\streaming_full\2026-01-17_03-56\tweets.parquet"
s = pd.read_parquet(os.path.join(HERE, "sample_k10_h48.parquet"), columns=["cluster_id"])
t = pd.read_parquet(TWEETS, columns=["tweet_id", "cluster_id", "text", "created_at"])
t = t[t.cluster_id.isin(s.cluster_id)].sort_values(["cluster_id", "created_at", "tweet_id"]).groupby("cluster_id").head(10).copy()
t["position"] = t.groupby("cluster_id").cumcount() + 1
t["tweet_id"] = t.tweet_id.astype("int64").astype(str)
t[["cluster_id", "position", "tweet_id"]].to_csv(os.path.join(HERE, "narrative_tweet_ids.csv"), index=False)
t[["tweet_id", "cluster_id", "position", "text"]].to_parquet(os.path.join(HERE, "tweet_id_text_map.parquet"), index=False)
print("tweet ids:", len(t), "narratives:", t.cluster_id.nunique())
DROP = {"prompt", "cli_output", "stderr"}
for f in glob.glob(os.path.join(HERE, "*", "calls.jsonl")):
    out = f.replace("calls.jsonl", "calls_redacted.jsonl.gz"); n = 0
    with open(f, encoding="utf-8") as src, gzip.open(out, "wt", encoding="utf-8") as dst:
        for line in src:
            r = json.loads(line); p = r.get("prompt") or ""
            red = {k: v for k, v in r.items() if k not in DROP}
            red["prompt_sha256"] = hashlib.sha256(p.encode("utf-8")).hexdigest(); red["prompt_chars"] = len(p)
            dst.write(json.dumps(red) + "\n"); n += 1
    print(os.path.relpath(out, HERE), n)
raw = os.path.join(HERE, "factchecks", "google_raw.jsonl")
with open(raw, "rb") as a, gzip.open(raw + ".gz", "wb") as b: shutil.copyfileobj(a, b)
print("google_raw.jsonl.gz", os.path.getsize(raw + ".gz"))
