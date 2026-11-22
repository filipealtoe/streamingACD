import sys, json, re, os, time
sys.path.insert(0, r"C:\Causalpriors")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from verifier.model_call import call_model
from verifier.parse import extract_json
OUT = r"C:\StreamingACD\causal_first_pass"; FC = os.path.join(OUT, "factchecks")
RUN = Path(OUT) / "haiku_top5_all"; RUN.mkdir(exist_ok=True)
deadline = time.time() + float(sys.argv[1]) * 60
w = pd.read_parquet(os.path.join(FC, "factchecks_window.parquet"))
cand_file = RUN / "candidates.parquet"
if not cand_file.exists():
    N = np.load(os.path.join(OUT, "narrative_emb_meantweets.npy")); C = np.load(os.path.join(FC, "claim_emb.npy"))
    ids = pd.read_parquet(os.path.join(OUT, "narrative_emb_ids.parquet")).cluster_id.values
    S = N @ C.T; top = np.argsort(-S, axis=1)[:, :5]
    pd.DataFrame({"cluster_id": ids, "cand": [list(map(int, r)) for r in top],
                  "sims": [[float(S[i, j]) for j in r] for i, r in enumerate(top)]}).to_parquet(cand_file)
cands = pd.read_parquet(cand_file)
texts = pd.read_parquet(os.path.join(OUT, "narrative_texts_first10.parquet")).set_index("cluster_id").narrative_text
def prompt(cid, cand):
    tw = [re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", x)).strip() for x in texts[cid].split("\n")]
    tw = "\n".join(f"- {x}" for x in tw if x)
    fcs = "\n".join(f"{k}. {w.claim.iloc[j]}" for k, j in enumerate(cand, 1))
    return f"""This is a triage step before fact-checking. Below are tweets from one online narrative and five claims that fact-checkers have already checked.

Question: does one of these existing fact-checks settle the main factual claim the tweets make, so that a fact-checker could point to it instead of writing a new check? It counts even if the wording, details or framing differ. It does not count if the tweets are only about the same subject but make a different claim, make no checkable claim, or only share general election or political talk.

Reply with JSON only: {{"match": <the number of the fact-check that settles the main claim, or 0 if none does>}}

Fact-checked claims:
{fcs}

Tweets:
{tw}
"""
log = RUN / "labels.jsonl"; done = set()
if log.exists(): done = {json.loads(l)["cluster_id"] for l in open(log, encoding="utf-8")}
def run(row):
    if row.cluster_id in done or time.time() > deadline: return
    cand = [int(x) for x in row.cand]; m = None
    for attempt in range(2):
        res = call_model(prompt(row.cluster_id, cand), "haiku", RUN, {"unit_id": f"n:{row.cluster_id}"})
        try:
            m = int(extract_json(res.text)["match"])
            if 0 <= m <= 5: break
        except Exception: m = None
    rec = {"cluster_id": int(row.cluster_id), "match": m, "fc_index": (cand[m - 1] if m else None)}
    with open(log, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
t0 = time.time()
with ThreadPoolExecutor(6) as ex: list(ex.map(run, cands.itertuples()))
n = sum(1 for _ in open(log, encoding="utf-8")) if log.exists() else 0
print(f"done {n} of {len(cands)}; this chunk {time.time() - t0:.0f}s", flush=True)
