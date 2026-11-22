"""First causal discovery pass on the narrative sample.

PC (stable) with Fisher-Z on normal scores of the ranks, and time-order
knowledge as tiers: context -> who/what posted first -> speed -> later spread.
Stability: bootstrap that resamples whole days, and a first-half / second-half split.
"""
import json, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
warnings.filterwarnings("ignore")

TIERS = {
    0: ["stream_volume_24h", "day_index"],
    1: ["max_followers", "median_followers", "unique_users", "account_age_days",
        "url_share", "mention_share", "mean_words", "similarity"],
    2: ["hours_to_k"],
    3: ["tweets_after"],
}
VARS = [v for t in sorted(TIERS) for v in TIERS[t]]
TIER_OF = {v: t for t, vs in TIERS.items() for v in vs}
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

def normal_scores(df):
    n = len(df)
    return np.column_stack([norm.ppf(rankdata(df[v].values) / (n + 1)) for v in VARS])

def run_pc(df, tiers=True):
    X = normal_scores(df)
    bk = None
    if tiers:
        bk = BackgroundKnowledge()
        for i, v in enumerate(VARS):
            bk.add_node_to_tier(GraphNode(f"X{i+1}"), TIER_OF[v])
    cg = pc(X, alpha=ALPHA, indep_test="fisherz", stable=True, background_knowledge=bk,
            show_progress=False, verbose=False)
    G = cg.G.graph  # G[i,j]=-1 and G[j,i]=1 means i -> j ; both -1 means undirected
    edges = {}
    p = len(VARS)
    for i in range(p):
        for j in range(i + 1, p):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1: kind = (VARS[i], VARS[j], "->")
            elif a == 1 and b == -1: kind = (VARS[j], VARS[i], "->")
            else: kind = (VARS[i], VARS[j], "--")
            edges[frozenset((VARS[i], VARS[j]))] = kind
    return edges

def fmt(e): return f"{e[0]} {e[2]} {e[1]}"

if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1])
    df["day"] = pd.to_datetime(df.t0).dt.date
    print("n =", len(df), "alpha =", ALPHA)
    full = run_pc(df)
    print("\nFull-sample graph with time-order knowledge:")
    for e in full.values(): print("  ", fmt(e))
    free = run_pc(df, tiers=False)
    agree = sum(1 for k, e in full.items() if k in free)
    same_dir = sum(1 for k, e in full.items() if k in free and free[k] == e)
    print(f"\nWithout time-order knowledge: {len(free)} edges; shares {agree} adjacencies with the graph above; "
          f"{same_dir} with the same orientation; undirected or differently oriented: {agree - same_dir}")

    # day-block bootstrap
    rng = np.random.default_rng(20261121)
    days = sorted(df.day.unique()); B = 200
    adj, dire = {}, {}
    groups = {d: g for d, g in df.groupby("day")}
    for b in range(B):
        pick = rng.choice(len(days), len(days), replace=True)
        bs = pd.concat([groups[days[i]] for i in pick], ignore_index=True)
        for k, e in run_pc(bs).items():
            adj[k] = adj.get(k, 0) + 1
            dire[e] = dire.get(e, 0) + 1
    print(f"\nEdge stability over {B} day-block bootstrap samples (share of samples with the edge):")
    rows = sorted(adj.items(), key=lambda x: -x[1])
    for k, c in rows:
        a, b2 = sorted(k, key=VARS.index)
        best = max([(v, e) for e, v in dire.items() if frozenset(e[:2]) == k])
        mark = "*" if k in full else " "
        print(f" {mark} {c/B:5.0%}  {a} ~ {b2}   most common form: {fmt(best[1])} ({best[0]/B:.0%})")
    # halves
    mid = days[len(days) // 2]
    h1, h2 = run_pc(df[df.day < mid]), run_pc(df[df.day >= mid])
    both = set(h1) & set(h2)
    print(f"\nSplit by time (before / from {mid}): n = {(df.day < mid).sum()} / {(df.day >= mid).sum()}; "
          f"edges {len(h1)} / {len(h2)}; shared {len(both)}")
    for k in both: print("   shared:", fmt(h1[k]), "|", fmt(h2[k]))
    out = {"alpha": ALPHA, "n": len(df), "full": [fmt(e) for e in full.values()],
           "bootstrap_adjacency": {" ~ ".join(sorted(k, key=VARS.index)): c / B for k, c in rows}}
    json.dump(out, open(f"discovery_alpha{ALPHA}.json", "w"), indent=1)
