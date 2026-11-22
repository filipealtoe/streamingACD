import sys, json, re, os
sys.path.insert(0, r"C:\Causalpriors")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from verifier.model_call import call_model
from verifier.parse import extract_json
MODEL = sys.argv[1]
OUT = r"C:\StreamingACD\causal_first_pass"; RUN = Path(OUT) / f"{MODEL}_check_v3"; RUN.mkdir(exist_ok=True)
smp = pd.read_csv(os.path.join(OUT, "label_sample_100_v3.csv"))
best = pd.read_parquet(os.path.join(OUT, "best_match.parquet")).set_index("cluster_id")
texts = pd.read_parquet(os.path.join(OUT, "narrative_texts_first10.parquet")).set_index("cluster_id").narrative_text
LABELS = {"covered": "covered", "related": "related, not covered", "unrelated": "unrelated"}
def prompt(cid):
    tw = [re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", x)).strip() for x in texts[cid].split("\n")]
    tw = "\n".join(f"- {x}" for x in tw if x)
    return f"""This is a triage step before fact-checking. Below are tweets from one online narrative and one claim that a fact-checker has already checked.

Question: would this existing fact-check settle the main factual claim the tweets make, so that a fact-checker could point to it instead of writing a new check?
- "covered": yes. The fact-check addresses the narrative's main claim, even if the wording, details or framing differ.
- "related": the tweets are about the same subject (event, person, policy), but their main claim would need a separate fact-check.
- "unrelated": anything else, including tweets that make no checkable claim or only share general election or political talk.

Reply with JSON only: {{"label": "covered" | "related" | "unrelated"}}

Fact-checked claim: {best.loc[cid, 'fc_claim']}

Tweets:
{tw}
"""
log = RUN / "labels.jsonl"; done = set()
if log.exists(): done = {json.loads(l)["pair_id"] for l in open(log, encoding="utf-8")}
def run(row):
    if row.pair_id in done: return
    lab = None
    for attempt in range(2):
        res = call_model(prompt(row.cluster_id), MODEL, RUN, {"unit_id": f"pair:{row.pair_id}"})
        try:
            lab = extract_json(res.text)["label"]
            if lab in LABELS: break
        except Exception: lab = None
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pair_id": int(row.pair_id), "model": LABELS.get(lab, "missing")}) + "\n")
with ThreadPoolExecutor(4) as ex: list(ex.map(run, smp.itertuples()))
m = smp.merge(pd.DataFrame([json.loads(l) for l in open(log, encoding="utf-8")]), on="pair_id")
print("==", MODEL); print(pd.crosstab(m.label, m.model, margins=True).to_string())
p = m.model.eq("covered"); t = m.label.eq("covered")
print("calls", p.sum(), "agree", (p & t).sum(), "mine", t.sum(), "extra", sorted(m[p & ~t].pair_id), "missed", sorted(m[~p & t].pair_id))
