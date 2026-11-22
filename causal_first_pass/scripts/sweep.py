import os, json, time, requests, threading, sys
from concurrent.futures import ThreadPoolExecutor
FC = r"C:\StreamingACD\causal_first_pass\factchecks"
CACHE = FC + r"\google_raw.jsonl"
terms = json.load(open(FC + r"\google_query_terms.json"))
done = set()
if os.path.exists(CACHE):
    for line in open(CACHE, encoding="utf-8"):
        done.add(json.loads(line)["term"])
todo = [t for t in terms if t not in done]
print("todo", len(todo), flush=True)
lock = threading.Lock(); deadline = time.time() + float(sys.argv[1]) * 60
KEY = os.environ["GOOGLE_API_KEY"]; S = requests.Session()
def get(params):
    for attempt in range(6):
        r = S.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params=params, timeout=30)
        if r.status_code == 200: return r.json()
        if r.status_code in (429, 500, 503): time.sleep(5 * 2 ** attempt); continue
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    raise RuntimeError("retries exhausted")
def run(term):
    if time.time() > deadline: return
    claims, tok = [], None
    for _ in range(3):
        p = {"query": term, "languageCode": "en", "pageSize": 100, "key": KEY}
        if tok: p["pageToken"] = tok
        d = get(p); claims += d.get("claims", []); tok = d.get("nextPageToken")
        if not tok: break
    with lock, open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"term": term, "claims": claims}) + "\n")
with ThreadPoolExecutor(8) as ex: list(ex.map(run, todo))
n = sum(1 for _ in open(CACHE, encoding="utf-8")); print("terms done", n, "of", len(terms))
