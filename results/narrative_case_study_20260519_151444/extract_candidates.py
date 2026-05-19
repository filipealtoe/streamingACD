"""
Narrative case study extraction for CIKM paper §4.

Read-only join across claims.parquet, cluster_timeseries.parquet, psr_labels.parquet,
tweets.parquet. Produces a candidate list with full pipeline traces.

Per-claim 4-head dimensional check-worthiness scores are NOT persisted in any released
artifact; that field is flagged as unavailable in the report.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/StreamingACD/psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56")
PSR  = Path(r"C:/StreamingACD/psr/explainableACD/experiments/results/virality/psr_labels.parquet")
OUT  = Path(r"C:/StreamingACD/results/narrative_case_study_20260519_151444")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load ----------------------------------------------------------------
claims   = pd.read_parquet(ROOT / "claims.parquet")
ts       = pd.read_parquet(ROOT / "cluster_timeseries.parquet")
psr      = pd.read_parquet(PSR)

# ---- Compute detection & peak timestamps and lead time per cluster -------
trig = (ts[ts["is_anomaly_trigger"] == True]
        .sort_values("timestamp")
        .drop_duplicates("cluster_id", keep="first")
        [["cluster_id", "timestamp", "engagement"]]
        .rename(columns={"timestamp": "detect_ts", "engagement": "engagement_at_detect"}))

peak = (ts.sort_values("engagement", ascending=False)
        .drop_duplicates("cluster_id", keep="first")
        [["cluster_id", "timestamp", "engagement"]]
        .rename(columns={"timestamp": "peak_ts", "engagement": "engagement_at_peak"}))

formation = (ts.sort_values("timestamp")
             .drop_duplicates("cluster_id", keep="first")
             [["cluster_id", "timestamp"]]
             .rename(columns={"timestamp": "formation_ts"}))

# Join per-cluster timings into claims via trigger_cluster_id
claims = claims.merge(trig,      left_on="trigger_cluster_id", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
claims = claims.merge(peak,      left_on="trigger_cluster_id", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
claims = claims.merge(formation, left_on="trigger_cluster_id", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

claims["lead_time_hours_calc"] = (claims["peak_ts"] - claims["detect_ts"]).dt.total_seconds() / 3600.0

# PSR labels join (529/535 covered)
claims = claims.merge(psr.rename(columns={"detection_time": "psr_detection_time"}),
                       left_on="trigger_cluster_id", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

print(f"[load] claims: {len(claims)}")
print(f"[load] peak_ts populated: {claims['peak_ts'].notna().sum()}")
print(f"[load] lead_time computed (>0h): {(claims['lead_time_hours_calc'] > 0).sum()}")
print(f"[load] PSR labels matched: {claims['psr'].notna().sum()}")

# ---- Filter 1: procedural include, candidate-attack exclude --------------
PROCEDURAL = [
    "ballot", "voting machine", "mail-in", "mail in", "absentee", "vote count", "votes count",
    "recount", "certif", "audit", "poll worker", "polling place", "polling station",
    "voter registration", "voter id", "voter roll", "election official", "election worker",
    "signature verification", "precinct", "ballot box", "drop box", "chain of custody",
    "tabulat", "vote tally", "tally", "dominion", "voter fraud",
]
CANDIDATE_ATTACK_PRIMARY = [
    "hunter biden", "biden family", "trump family", "don jr", "donald trump jr",
    "hunter's laptop", "biden corruption", "trump pee", "trump tax",
]

def lc(s):
    return s.lower() if isinstance(s, str) else ""

claims["text_lc"] = claims["claim_text"].map(lc)
claims["is_procedural"] = claims["text_lc"].apply(lambda t: any(kw in t for kw in PROCEDURAL))
claims["is_candidate_attack"] = claims["text_lc"].apply(lambda t: any(kw in t for kw in CANDIDATE_ATTACK_PRIMARY))

f1 = claims[claims["is_procedural"] & ~claims["is_candidate_attack"]].copy()
print(f"[F1 procedural include & not candidate attack] passing: {len(f1)}")

# ---- Filter 2: viral_confidence (proxy for overall) > 0.7 ----------------
# Note: dimensional check-worthiness scores were not persisted; viral_confidence is the
# closest available overall signal from the pipeline and is used here as a proxy.
f2 = f1[f1["viral_confidence"] > 0.7].copy()
print(f"[F2 viral_confidence > 0.7 (proxy)] passing: {len(f2)}")

# ---- Filter 3: lead time >= 4 hours --------------------------------------
f3 = f2[f2["lead_time_hours_calc"] >= 4].copy()
print(f"[F3 lead_time_hours >= 4] passing: {len(f3)}")

# ---- Filter 4: cluster size (total_tweets) >= 10 -------------------------
f4 = f3[f3["total_tweets"] >= 10].copy()
print(f"[F4 total_tweets >= 10] passing: {len(f4)}")

# ---- Filter 5: dimensional variation -- SKIPPED (data unavailable) -------
print(f"[F5 dimensional variation] SKIPPED - no per-claim 4-head scores in released artifacts")

# Rank by viral_confidence descending then PSR desc, take top 5
f4 = f4.sort_values(["viral_confidence", "psr"], ascending=[False, False]).reset_index(drop=True)
final = f4.head(5)
print(f"[final] selected: {len(final)} candidates")

# ---- Save filter audit ---------------------------------------------------
audit = {
    "filter_counts": {
        "initial": int(len(claims)),
        "f1_procedural_include_and_not_candidate_attack": int(len(f1)),
        "f2_viral_confidence_proxy_gt_0_7": int(len(f2)),
        "f3_lead_time_hours_gte_4": int(len(f3)),
        "f4_total_tweets_gte_10": int(len(f4)),
        "f5_dimensional_variation_gte_0_1": "SKIPPED - per-claim 4-head scores not persisted",
        "final_selected": int(len(final)),
    },
    "filter_2_caveat": (
        "viral_confidence is used as a proxy for the 4-head 'overall check-worthiness' "
        "score because per-claim dimensional scores from the 4-Parallel-Heads classifier "
        "were not found in any released artifact. viral_confidence is the pipeline's own "
        "is_viral probability, not the check-worthiness model output."
    ),
    "filter_5_caveat": (
        "Per-claim checkability, verifiability, harm potential scores from the 4-head "
        "model are not persisted. Filter 5 cannot be applied without re-running model "
        "inference on the 535 normalized claims."
    ),
}
(OUT / "filter_audit.json").write_text(json.dumps(audit, indent=2))

# ---- Persist the candidate slice -----------------------------------------
keep = [
    "claim_id", "claim_text", "trigger_cluster_id", "cluster_ids", "total_tweets",
    "total_engagement", "detection_z_score", "is_viral", "viral_confidence",
    "formation_ts", "detect_ts", "peak_ts", "lead_time_hours_calc",
    "engagement_at_detect", "engagement_at_peak", "psr", "e_detect", "e_final",
]
final[keep].to_csv(OUT / "candidates.csv", index=False)
print(f"[save] candidates.csv with {len(final)} rows")

# ---- Per-candidate representative posts ----------------------------------
# Load tweets and cluster_embeddings lazily; sample 5 posts per candidate's trigger cluster
import sys
sys.stdout.flush()

tweets_df = pd.read_parquet(ROOT / "tweets.parquet")
print(f"[load] tweets: {len(tweets_df)}")

candidate_payload = []
for _, row in final.iterrows():
    cid = int(row["trigger_cluster_id"])
    cluster_tweets = tweets_df[tweets_df["cluster_id"] == cid].copy()
    # Sort by engagement-ish proxy: cluster_similarity desc then retweet desc
    cluster_tweets["eng_proxy"] = cluster_tweets["retweet_count_at_collection"].fillna(0) + cluster_tweets["likes_at_collection"].fillna(0)
    # Pick 1 centroid-most-similar + 4 with diverse engagement / similarity
    centroid_tw = cluster_tweets.sort_values("cluster_similarity", ascending=False).head(1)
    rest = cluster_tweets.drop(centroid_tw.index).sort_values("cluster_similarity", ascending=True).head(4)
    sample = pd.concat([centroid_tw, rest])
    payload = {
        "claim_id": row["claim_id"],
        "claim_text": row["claim_text"],
        "trigger_cluster_id": cid,
        "cluster_ids": list(row["cluster_ids"]) if row["cluster_ids"] is not None else [],
        "total_tweets": int(row["total_tweets"]),
        "total_engagement": int(row["total_engagement"]),
        "detection_z_score": float(row["detection_z_score"]),
        "is_viral": bool(row["is_viral"]),
        "viral_confidence": float(row["viral_confidence"]),
        "formation_ts": str(row["formation_ts"]),
        "detect_ts": str(row["detect_ts"]),
        "peak_ts": str(row["peak_ts"]),
        "lead_time_hours": float(row["lead_time_hours_calc"]),
        "engagement_at_detect": int(row["engagement_at_detect"]) if pd.notna(row["engagement_at_detect"]) else None,
        "engagement_at_peak": int(row["engagement_at_peak"]) if pd.notna(row["engagement_at_peak"]) else None,
        "psr": float(row["psr"]) if pd.notna(row["psr"]) else None,
        "e_detect": int(row["e_detect"]) if pd.notna(row["e_detect"]) else None,
        "e_final": int(row["e_final"]) if pd.notna(row["e_final"]) else None,
        "sample_post_texts": sample["text"].tolist(),
        "sample_post_engagements": sample["eng_proxy"].astype(int).tolist(),
        "sample_post_similarities": sample["cluster_similarity"].astype(float).tolist(),
    }
    candidate_payload.append(payload)

(OUT / "candidates_full.json").write_text(json.dumps(candidate_payload, indent=2, default=str))
print(f"[save] candidates_full.json with {len(candidate_payload)} candidates")
