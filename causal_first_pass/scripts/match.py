import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
OUT = r"C:\StreamingACD\causal_first_pass"; FC = OUT + r"\factchecks"
df = pd.read_parquet(FC + r"\google_factchecks_all.parquet")
w = df[(df.date >= "2020-09-01") & (df.date <= "2020-11-30") & (df.claim.str.strip() != "")].reset_index(drop=True)
w.to_parquet(FC + r"\factchecks_window.parquet", index=False)
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
C = m.encode(w.claim.tolist(), batch_size=64, normalize_embeddings=True).astype(np.float32)
np.save(FC + r"\claim_emb.npy", C)
N = np.load(OUT + r"\narrative_emb_meantweets.npy"); ids = pd.read_parquet(OUT + r"\narrative_emb_ids.parquet").cluster_id
S = N @ C.T; best = S.argmax(1)
res = pd.DataFrame({"cluster_id": ids, "best_sim": S.max(1), "fc_index": best})
res = res.join(w.add_prefix("fc_"), on="fc_index")
res.to_parquet(OUT + r"\best_match.parquet", index=False)
print(len(w), res.best_sim.describe().round(3).to_dict())
print(pd.cut(res.best_sim, [-1, .4, .5, .6, .7, 1]).value_counts().sort_index().to_string())
