import pandas as pd, numpy as np, time, os, re, sys
from sentence_transformers import SentenceTransformer
OUT = r"C:\StreamingACD\causal_first_pass"; PART = OUT + r"\emb_parts"; os.makedirs(PART, exist_ok=True)
s = pd.read_parquet(OUT + r"\sample_k10_h48.parquet", columns=["cluster_id"])
t = pd.read_parquet(r"C:\StreamingACD\psr\explainableACD\data\pipeline_output\streaming_full\2026-01-17_03-56\tweets.parquet",
                    columns=["tweet_id", "cluster_id", "text", "created_at"])
t = t[t.cluster_id.isin(s.cluster_id)].sort_values(["cluster_id", "created_at", "tweet_id"]).groupby("cluster_id").head(10)
t["clean"] = t.text.map(lambda x: re.sub(r"\s+", " ", re.sub(r"https?://\S+", " ", x)).strip() or x)
t = t.reset_index(drop=True); CH = 4000
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
deadline = time.time() + float(sys.argv[1]) * 60
for i in range(0, len(t), CH):
    f = f"{PART}\part_{i:06d}.npy"
    if os.path.exists(f): continue
    if time.time() > deadline: break
    np.save(f, m.encode(t.clean.iloc[i:i+CH].tolist(), batch_size=64, normalize_embeddings=True).astype(np.float32))
    print("done", i, flush=True)
parts = sorted(os.listdir(PART))
if len(parts) == -(-len(t) // CH):
    E = np.concatenate([np.load(PART + "\\" + p) for p in parts])
    g = pd.Series(range(len(t))).groupby(t.cluster_id.values)
    ids = s.cluster_id.values
    N = np.stack([E[g.get_group(c).values].mean(0) for c in ids]); N /= np.linalg.norm(N, axis=1, keepdims=True)
    np.save(OUT + r"\narrative_emb_meantweets.npy", N); pd.Series(ids).to_frame("cluster_id").to_parquet(OUT + r"\narrative_emb_ids.parquet")
    print("ALL DONE", N.shape)
