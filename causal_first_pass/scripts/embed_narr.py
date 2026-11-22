import pandas as pd, numpy as np, time
from sentence_transformers import SentenceTransformer
OUT = r"C:\StreamingACD\causal_first_pass"
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet")
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("max_seq_length", m.max_seq_length, flush=True)
t = time.time()
E = m.encode(texts.narrative_text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
np.save(OUT + r"\narrative_emb.npy", E.astype(np.float32))
print(E.shape, round(time.time() - t), "s")
