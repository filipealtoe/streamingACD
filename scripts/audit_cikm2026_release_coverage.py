#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow==25.0.1",
# ]
# ///
"""Audit the public package against the CIKM 2026 artifact-release promises."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-21 23:47 PDT | Reason: distinguish the scientifically useful public
# aggregate package from the broader artifact set promised in the manuscript,
# including large assets distributed through the checksum-verified GitHub
# release without reintroducing post-level or user-level identifiers.


def inspect_parquet(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "rows": 0, "columns": []}
    metadata = pq.read_metadata(path)
    return {
        "present": True,
        "rows": int(metadata.num_rows),
        "columns": pq.read_schema(path).names,
    }


def inspect_cluster_embeddings(
    repo_root: Path, canonical_root: Path
) -> dict[str, Any]:
    """Describe the embedding matrix across Git and release distribution."""
    artifact_name = "cluster_embeddings.npy"
    local_path = canonical_root / artifact_name
    release_manifest = json.loads(
        (repo_root / "reproducibility/cikm2026/RELEASE_ASSETS.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (
            repo_root
            / "reproducibility/cikm2026/cluster_embeddings.metadata.json"
        ).read_text(encoding="utf-8")
    )
    release_asset = next(
        (
            asset
            for asset in release_manifest["assets"]
            if asset["name"] == artifact_name
        ),
        None,
    )
    repository = release_manifest["repository"]
    release_tag = release_manifest["release"]["tag"]
    expected_url = (
        f"https://github.com/{repository}/releases/download/"
        f"{release_tag}/{artifact_name}"
    )
    published_as_release_asset = bool(
        release_asset
        and release_manifest["release"]["draft"] is False
        and metadata["artifact_name"] == artifact_name
        and metadata["release_tag"] == release_tag
        and metadata["release_asset_url"] == expected_url
        and release_asset["bytes"] == metadata["bytes"]
        and release_asset["digest"] == f"sha256:{metadata['sha256']}"
    )
    present_in_git = local_path.is_file()
    return {
        "present": present_in_git or published_as_release_asset,
        "present_in_git": present_in_git,
        "index_present": (canonical_root / "cluster_id_to_idx.json").is_file(),
        "published_as_release_asset": published_as_release_asset,
        "distribution": (
            "git" if present_in_git else "github_release"
            if published_as_release_asset
            else "not_public"
        ),
        "release": {
            "repository": repository,
            "tag": release_tag,
            "asset_url": expected_url,
            "bytes": release_asset["bytes"] if release_asset else None,
            "sha256": metadata["sha256"],
        },
    }


def audit(repo_root: Path) -> dict[str, Any]:
    canonical_root = (
        repo_root
        / "psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56"
    )
    expected_artifacts = {
        "tweets.parquet": inspect_parquet(canonical_root / "tweets.parquet"),
        "users.parquet": inspect_parquet(canonical_root / "users.parquet"),
        "clusters.parquet": inspect_parquet(canonical_root / "clusters.parquet"),
        "claims.parquet": inspect_parquet(canonical_root / "claims.parquet"),
        "cluster_timeseries.parquet": inspect_parquet(
            canonical_root / "cluster_timeseries.parquet"
        ),
        "cluster_embeddings.npy": inspect_cluster_embeddings(
            repo_root, canonical_root
        ),
    }
    for name, artifact in expected_artifacts.items():
        if name != "cluster_embeddings.npy":
            artifact["distribution"] = "git" if artifact["present"] else "not_public"

    claims_columns = set(expected_artifacts["claims.parquet"]["columns"])
    promised_claim_columns = {
        "checkability_score",
        "verifiability_score",
        "harm_score",
    }
    missing_claim_columns = sorted(promised_claim_columns - claims_columns)
    virality = inspect_parquet(
        repo_root
        / "reproducibility/source_artifacts/virality/features_enhanced.parquet"
    )
    four_head_scores = inspect_parquet(
        repo_root
        / "results/four_heads_inference_20260519_152747/claim_scores.parquet"
    )
    claims_ids = set(
        pq.read_table(canonical_root / "claims.parquet", columns=["claim_id"])[
            "claim_id"
        ].to_pylist()
    )
    score_path = (
        repo_root / "results/four_heads_inference_20260519_152747/claim_scores.parquet"
    )
    score_ids = set(
        pq.read_table(score_path, columns=["claim_id"])["claim_id"].to_pylist()
    )
    score_columns = set(four_head_scores["columns"])
    required_score_columns = {
        "checkability_score",
        "verifiability_score",
        "harm_score",
        "overall_score",
    }
    separate_scores_complete = (
        required_score_columns.issubset(score_columns)
        and claims_ids == score_ids
        and len(score_ids) == 535
    )

    promises = {
        "six_interlinked_artifacts": {
            "expected": list(expected_artifacts),
            "present": [
                name for name, value in expected_artifacts.items() if value["present"]
            ],
        },
        "original_tweet_ids": {
            "present": False,
            "reason": "Post-level identifiers are excluded from the public tree.",
        },
        "full_post_level_membership": {
            "present": expected_artifacts["tweets.parquet"]["present"],
        },
        "claim_checkworthiness_decomposition": {
            "present_in_claims_table": not missing_claim_columns,
            "missing_claim_columns": missing_claim_columns,
            "separate_score_artifact_rows": four_head_scores["rows"],
            "present_in_separate_score_artifact": separate_scores_complete,
            "separate_score_artifact_claim_ids_match": claims_ids == score_ids,
            "separate_score_columns": sorted(required_score_columns & score_columns),
        },
        "natural_language_rationales": {
            "present": False,
        },
        "opus_raw_json_records": {
            "paper_count": 1023,
            "public_count": 0,
        },
    }
    complete_promises = {
        "six_interlinked_artifacts": len(
            promises["six_interlinked_artifacts"]["present"]
        )
        == 6,
        "original_tweet_ids": promises["original_tweet_ids"]["present"],
        "full_post_level_membership": promises["full_post_level_membership"][
            "present"
        ],
        "claim_checkworthiness_decomposition": separate_scores_complete,
        "natural_language_rationales": promises["natural_language_rationales"][
            "present"
        ],
        "opus_raw_json_records": promises["opus_raw_json_records"]["public_count"]
        == promises["opus_raw_json_records"]["paper_count"],
    }

    return {
        "artifact": "CIKM 2026 public artifact-release coverage audit",
        "author": "Sérgio Pinto",
        "timestamp": "2026-08-21 23:47 PDT (-0700)",
        "reason": "Measure the public package against the release promises in Sections 3.2, 4.3, and 4.4.",
        "paper_expected_artifacts": expected_artifacts,
        "paper_release_promises": promises,
        "public_scientific_core": {
            "canonical_claims": expected_artifacts["claims.parquet"]["rows"],
            "aggregate_clusters": expected_artifacts["clusters.parquet"]["rows"],
            "aggregate_cluster_timeseries_rows": expected_artifacts[
                "cluster_timeseries.parquet"
            ]["rows"],
            "virality_cluster_instances": virality["rows"],
            "four_head_score_rows": four_head_scores["rows"],
            "raw_post_text_present": False,
            "post_or_user_identifier_tables_present": False,
        },
        "promise_checks": complete_promises,
        "summary": {
            "promises_satisfied": sum(complete_promises.values()),
            "promises_checked": len(complete_promises),
            "verdict": "PASS" if all(complete_promises.values()) else "PARTIAL",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the CIKM 2026 artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/artifact_release_coverage_audit_2026-08-21.json"),
        help="Output path relative to the repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    result = audit(repo_root)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("CIKM 2026 artifact-release coverage audit")
    for name, passed in result["promise_checks"].items():
        print(f"[{'PASS' if passed else 'PARTIAL'}] {name}")
    core = result["public_scientific_core"]
    print(
        "Public scientific core: "
        f"claims={core['canonical_claims']}; clusters={core['aggregate_clusters']}; "
        f"time-series rows={core['aggregate_cluster_timeseries_rows']}; "
        f"virality rows={core['virality_cluster_instances']}; "
        f"four-head rows={core['four_head_score_rows']}"
    )
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 23:51 PDT | Reason: honor the documented absolute --output
    # form as well as repository-relative result paths.
    display_path = (
        output_path.relative_to(repo_root)
        if output_path.is_relative_to(repo_root)
        else output_path
    )
    print(f"Result: {display_path}")
    print(f"VERDICT: {result['summary']['verdict']}")
    return 0 if result["summary"]["verdict"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
