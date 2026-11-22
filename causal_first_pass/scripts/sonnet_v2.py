import sys, json, re
sys.path.insert(0, r"C:\Causalpriors")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from verifier.model_call import call_model
from verifier.parse import extract_json
OUT = r"C:\StreamingACD\causal_first_pass"; RUN = Path(OUT) / "sonnet_check_v2"; RUN.mkdir(exist_ok=True)
smp = pd.concat([pd.read_csv(OUT + r"\label_sample_150.csv").assign(set="old150"), pd.read_csv(OUT + r"\label_sample_100_new.csv").assign(set="new100")])
best = pd.read_parquet(OUT + r"\best_match.parquet").set_index("cluster_id")
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet").set_index("cluster_id").narrative_text
LABELS = {"same_claim": "same claim", "same_topic": "same topic, different claim", "unrelated": "unrelated"}
def prompt(cid):
    tw = [re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", x)).strip() for x in texts[cid].split("\n")]
    tw = "\n".join(f"- {x}" for x in tw if x)
    return f"""Below are tweets from one online narrative and one claim that a fact-checker checked.

Decide how they relate. Be strict about "same_claim".
- "same_claim": the tweets make, repeat or dispute the same specific factual assertion that was checked: the same who did or said what. Saying the subject is true or false in general is not enough; the specific assertion must be there.
- "same_topic": the tweets are about the same specific event, person, policy or controversy, but make a different assertion. A different event of the same kind (another rally, another lawsuit, another state's count, another statement by the same person) is same_topic, not same_claim.
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
        res = call_model(prompt(row.cluster_id), "sonnet", RUN, {"unit_id": f"pair:{row.pair_id}"})
        try:
            lab = extract_json(res.text)["label"]
            if lab in LABELS: break
        except Exception: lab = None
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pair_id": int(row.pair_id), "haiku": LABELS.get(lab, "missing")}) + "\n")
with ThreadPoolExecutor(4) as ex: list(ex.map(run, smp.itertuples()))
h = pd.DataFrame([json.loads(l) for l in open(log, encoding="utf-8")])
m = smp.merge(h, on="pair_id")
for name, x in m.groupby("set"):
    print("==", name); print(pd.crosstab(x.label, x.haiku, margins=True).to_string())
    sc = x.haiku.eq("same claim"); tc = x.label.eq("same claim")
    print(f"exact agreement {(x.label == x.haiku).mean():.2f}; Haiku same_claim {sc.sum()}, of which mine same claim {(sc & tc).sum()}; my same claim {tc.sum()}, Haiku found {(sc & tc).sum()}")
    print("disagreements on same claim:", sorted(x[sc != tc].pair_id.tolist()))
