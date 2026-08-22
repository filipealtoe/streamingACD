#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow==25.0.1",
# ]
# ///
"""Reproduce the numeric CIKM 2026 worked example for cluster 72106."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


CLUSTER_ID = 72106
CLAIM_ID = "72614ee1-b470-4d6f-90a3-da78f7ccb70e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the public CIKM artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path (default: results/case_study_reproduction_2026-08-21.json).",
    )
    return parser.parse_args()


def one_row(table: Any, column: str, value: Any) -> dict[str, Any]:
    rows = table.filter(pc.equal(table[column], value)).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one {column}={value!r} row, found {len(rows)}")
    return rows[0]


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    canonical_root = (
        repo_root
        / "psr"
        / "explainableACD"
        / "data"
        / "pipeline_output"
        / "streaming_full"
        / "2026-01-17_03-56"
    )

    timeseries = pq.read_table(canonical_root / "cluster_timeseries.parquet")
    timeseries = timeseries.filter(pc.equal(timeseries["cluster_id"], CLUSTER_ID))
    timeseries_rows = sorted(timeseries.to_pylist(), key=lambda row: row["timestamp"])
    trigger_rows = [row for row in timeseries_rows if row["is_anomaly_trigger"]]
    if not trigger_rows:
        raise ValueError(f"No anomaly trigger found for cluster {CLUSTER_ID}")
    trigger = trigger_rows[0]
    formation_time = timeseries_rows[0]["timestamp"]
    members_at_detection = sum(
        int(row["tweet_count"])
        for row in timeseries_rows
        if row["timestamp"] <= trigger["timestamp"]
    )
    peak = max(
        (row for row in timeseries_rows if row["timestamp"] >= trigger["timestamp"]),
        key=lambda row: row["engagement"],
    )
    final_engagement = sum(int(row["engagement"]) for row in timeseries_rows)

    label = one_row(
        pq.read_table(
            repo_root
            / "reproducibility"
            / "source_artifacts"
            / "virality"
            / "psr_labels.parquet"
        ),
        "cluster_id",
        CLUSTER_ID,
    )
    claim = one_row(
        pq.read_table(canonical_root / "claims.parquet"),
        "claim_id",
        CLAIM_ID,
    )

    scores_path = (
        repo_root
        / "results"
        / "four_heads_inference_20260519_152747"
        / "claim_scores.csv"
    )
    with scores_path.open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    score = next(row for row in score_rows if row["claim_id"] == CLAIM_ID)
    metadata = json.loads(
        (
            repo_root
            / "results"
            / "four_heads_inference_20260519_152747"
            / "inference_metadata.json"
        ).read_text(encoding="utf-8")
    )

    actual = {
        "cluster_id": CLUSTER_ID,
        "claim_id": CLAIM_ID,
        "formation_time": formation_time.isoformat(),
        "detection_time": trigger["timestamp"].isoformat(),
        "detection_z_score": float(trigger["z_score"]),
        "timeseries_members_through_detection": members_at_detection,
        "registry_total_tweets_at_normalization": int(claim["total_tweets"]),
        "engagement_at_detection_window": int(trigger["engagement"]),
        "psr_engagement_at_detection": float(label["e_detect"]),
        "peak_time": peak["timestamp"].isoformat(),
        "peak_window_engagement": int(peak["engagement"]),
        "hours_detection_to_peak": (
            peak["timestamp"] - trigger["timestamp"]
        ).total_seconds()
        / 3600.0,
        "final_engagement": final_engagement,
        "psr_final_engagement": float(label["e_final"]),
        "psr": float(label["psr"]),
        "checkability": float(score["checkability_score"]),
        "verifiability": float(score["verifiability_score"]),
        "harm": float(score["harm_score"]),
        "overall": float(score["overall_score"]),
        "harm_corpus_p75": float(metadata["per_head_summary"]["harm"]["p75"]),
        "score_rows": len(score_rows),
        "claim_text_sha256": hashlib.sha256(
            claim["claim_text"].encode("utf-8")
        ).hexdigest(),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    expected = {
        "formation_time": "2020-11-02T13:00:00",
        "detection_time": "2020-11-04T07:00:00",
        "detection_z_score_rounded_2": 3.37,
        "members_at_detection": 88,
        "psr_engagement_at_detection": 1031.0,
        "peak_time": "2020-11-04T21:00:00",
        "peak_window_engagement": 2678,
        "hours_detection_to_peak": 14.0,
        "final_engagement": 6851,
        "psr_rounded_2": 0.85,
        "checkability_rounded_3": 0.711,
        "verifiability_rounded_3": 0.982,
        "harm_rounded_3": 0.394,
        "overall_rounded_3": 0.989,
        "harm_corpus_p75_rounded_3": 0.293,
        "score_rows": 535,
    }
    comparisons = {
        "formation_time": actual["formation_time"] == expected["formation_time"],
        "detection_time": actual["detection_time"] == expected["detection_time"],
        "detection_z_score": round(actual["detection_z_score"], 2)
        == expected["detection_z_score_rounded_2"],
        "paper_member_value_from_registry": actual[
            "registry_total_tweets_at_normalization"
        ]
        == expected["members_at_detection"],
        "detection_timestamp_member_count": actual[
            "timeseries_members_through_detection"
        ]
        == expected["members_at_detection"],
        "psr_engagement_at_detection": math.isclose(
            actual["psr_engagement_at_detection"],
            expected["psr_engagement_at_detection"],
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "peak_time": actual["peak_time"] == expected["peak_time"],
        "peak_window_engagement": actual["peak_window_engagement"]
        == expected["peak_window_engagement"],
        "hours_detection_to_peak": actual["hours_detection_to_peak"]
        == expected["hours_detection_to_peak"],
        "final_engagement": actual["final_engagement"]
        == expected["final_engagement"]
        and actual["psr_final_engagement"] == expected["final_engagement"],
        "psr": round(actual["psr"], 2) == expected["psr_rounded_2"],
        "checkability": round(actual["checkability"], 3)
        == expected["checkability_rounded_3"],
        "verifiability": round(actual["verifiability"], 3)
        == expected["verifiability_rounded_3"],
        "harm": round(actual["harm"], 3) == expected["harm_rounded_3"],
        "overall": round(actual["overall"], 3)
        == expected["overall_rounded_3"],
        "harm_corpus_p75": round(actual["harm_corpus_p75"], 3)
        == expected["harm_corpus_p75_rounded_3"],
        "score_rows": actual["score_rows"] == expected["score_rows"],
        "claim_linkage": trigger["claim_id"] == CLAIM_ID
        and claim["trigger_cluster_id"] == CLUSTER_ID
        and CLUSTER_ID in claim["cluster_ids"],
        "checkpoint_identity": metadata["checkpoint_sha256"]
        == "5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081",
    }
    status = "PASS" if all(comparisons.values()) else "PARTIAL"
    output = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 21:14 PDT — Separated direct "
            "reproduction of the paper's 88-post registry value from the "
            "80-post temporal-boundary mismatch in the cluster time series."
        ),
        "actual": actual,
        "paper_values": expected,
        "comparisons": comparisons,
        "status": status,
    }
    output_path = args.output or (
        repo_root / "results" / "case_study_reproduction_2026-08-21.json"
    )
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("CIKM 2026 cluster-72106 worked-example reproduction")
    for name, passed in comparisons.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"Result: {output_path}")
    print(f"VERDICT: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
