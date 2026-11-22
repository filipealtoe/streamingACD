import sys, json, re
sys.path.insert(0, r"C:\Causalpriors")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from verifier.model_call import call_model
from verifier.parse import extract_json
OUT = r"C:\StreamingACD\causal_first_pass"; RUN = Path(OUT) / "haiku_check_150"; RUN.mkdir(exist_ok=True)
smp = pd.read_csv(OUT + r"\label_sample_150.csv")
best = pd.read_parquet(OUT + r"\best_match.parquet").set_index("cluster_id")
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet").set_index("cluster_id").narrative_text
LABELS = {"same_claim": "same claim", "same_topic": "same topic, different claim", "unrelated": "unrelated"}
def prompt(cid):
    tw = [re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", x)).strip() for x in texts[cid].split("\n")]
    tw = "\n".join(f"- {x}" for x in tw if x)
    return f"""Below are tweets from one online narrative and one claim that a fact-checker checked.

Decide how they relate:
- "same_claim": the tweets assert, repeat, discuss or dispute the specific claim that was fact-checked.
- "same_topic": the tweets are about the same specific subject (event, person, policy), but not that claim.
- "unrelated": anything else, including tweets that only share general election or political talk.

Reply with JSON only: {{"label": "same_claim" | "same_topic" | "unrelated"}}

Fact-checked claim: {best.loc[cid, 'fc_claim']}

Tweets:
{tw}
"""
done = {}
log = RUN / "labels.jsonl"
if log.exists():
    for line in open(log, encoding="utf-8"): r = json.loads(line); done[r["pair_id"]] = r
def run(row):
    if row.pair_id in done: return
    for attempt in range(2):
        res = call_model(prompt(row.cluster_id), "haiku", RUN, {"unit_id": f"pair:{row.pair_id}"})
        try:
            lab = extract_json(res.text)["label"]
            if lab in LABELS: break
        except Exception: lab = None
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pair_id": int(row.pair_id), "haiku": LABELS.get(lab, "missing")}) + "\n")
with ThreadPoolExecutor(4) as ex: list(ex.map(run, smp.itertuples()))
h = pd.DataFrame([json.loads(l) for l in open(log, encoding="utf-8")])
m = smp.merge(h, on="pair_id")
print(pd.crosstab(m.label, m.haiku, margins=True).to_string())
print("exact agreement", round((m.label == m.haiku).mean(), 3))
