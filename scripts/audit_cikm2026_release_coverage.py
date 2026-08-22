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
# 2026-08-22 00:53 PDT | Reason: align this audit with the exact final PDF,
# whose release statement covers the 529-row virality dataset, source code,
# model parameters, prompts, schema, and accompanying reproducibility artifacts.


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
    pipeline_inventory = {
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
    for name, artifact in pipeline_inventory.items():
        if name != "cluster_embeddings.npy":
            artifact["distribution"] = "git" if artifact["present"] else "not_public"

    virality = inspect_parquet(
        repo_root
        / "reproducibility/source_artifacts/virality/features_enhanced.parquet"
    )
    virality_labels = inspect_parquet(
        repo_root
        / "reproducibility/source_artifacts/virality/psr_labels.parquet"
    )
    feature_path = (
        repo_root
        / "reproducibility/source_artifacts/virality/features_enhanced.parquet"
    )
    label_path = (
        repo_root / "reproducibility/source_artifacts/virality/psr_labels.parquet"
    )
    feature_ids = set(
        pq.read_table(feature_path, columns=["cluster_id"])["cluster_id"].to_pylist()
    )
    label_ids = set(
        pq.read_table(label_path, columns=["cluster_id"])["cluster_id"].to_pylist()
    )
    engineered_feature_columns = sorted(
        set(virality["columns"]) - {"cluster_id", "psr"}
    )
    dataset_complete = (
        virality["present"]
        and virality_labels["present"]
        and virality["rows"] == 529
        and virality_labels["rows"] == 529
        and len(engineered_feature_columns) == 42
        and "psr" in virality["columns"]
        and "psr" in virality_labels["columns"]
        and feature_ids == label_ids
        and len(feature_ids) == 529
    )

    release_manifest = json.loads(
        (repo_root / "reproducibility/cikm2026/RELEASE_ASSETS.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = next(
        (asset for asset in release_manifest["assets"] if asset["name"] == "best_model.pt"),
        None,
    )
    checkpoint_accounted = bool(
        checkpoint
        and checkpoint["required"] is True
        and checkpoint["bytes"] == 1_739_380_133
        and checkpoint["digest"]
        == "sha256:5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081"
        and release_manifest["release"]["draft"] is False
    )
    required_source_paths = [
        "scripts/reproduce_cikm2026_tabular_baselines.py",
        "scripts/verify_cikm2026_artifacts.py",
        "reproducibility/source_artifacts/checkworthiness/source_code/README.md",
        "reproducibility/source_artifacts/claim_normalization/source_code/README.md",
    ]
    required_prompt_paths = [
        "prompts/checkworthiness_prompts_zeroshot_v4.yaml",
        "prompts/claim_normalization_cikm2026.md",
    ]
    required_reproducibility_paths = [
        "reproducibility/cikm2026/PAPER_ARTIFACT_INDEX.md",
        "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json",
        "reproducibility/cikm2026/checksums.sha256",
    ]
    promise_checks = {
        "virality_dataset_529_rows_42_features_and_labels": dataset_complete,
        "source_code": all((repo_root / path).is_file() for path in required_source_paths),
        "model_parameters_and_checkpoint": (
            (repo_root / "reproducibility/cikm2026/MODEL_PARAMETERS.json").is_file()
            and checkpoint_accounted
        ),
        "llm_prompts": all((repo_root / path).is_file() for path in required_prompt_paths),
        "reproducibility_artifacts": all(
            (repo_root / path).is_file() for path in required_reproducibility_paths
        ),
        "pipeline_schema": (
            repo_root / "reproducibility/cikm2026/SCHEMA.md"
        ).is_file(),
    }

    return {
        "artifact": "CIKM 2026 public artifact-release coverage audit",
        "author": "Sérgio Pinto",
        "timestamp": "2026-08-22 00:53 PDT (-0700)",
        "paper_pdf_sha256": "4d82abd01d66de5e04d7107e8c4bb21d3b3d1a7148aa235e95b78b803df78b9c",
        "reason": "Measure the public package against the exact availability statements in Sections 3 and 4.4 of the final camera-ready PDF.",
        "paper_release_scope": {
            "dataset": "529 cluster instances with detection-time engineered features and virality labels",
            "method_artifacts": "source code, model parameters, LLM prompts, pipeline schema, and accompanying reproducibility artifacts",
        },
        "dataset": {
            "features": virality,
            "labels": virality_labels,
            "engineered_feature_columns": len(engineered_feature_columns),
            "cluster_ids_match": feature_ids == label_ids,
            "unique_cluster_ids": len(feature_ids),
        },
        "release_checkpoint": {
            "accounted_for": checkpoint_accounted,
            "asset": checkpoint,
            "release_tag": release_manifest["release"]["tag"],
        },
        "pipeline_output_inventory": pipeline_inventory,
        "public_scientific_core": {
            "canonical_claims": pipeline_inventory["claims.parquet"]["rows"],
            "aggregate_clusters": pipeline_inventory["clusters.parquet"]["rows"],
            "aggregate_cluster_timeseries_rows": pipeline_inventory[
                "cluster_timeseries.parquet"
            ]["rows"],
            "virality_cluster_instances": virality["rows"],
            "raw_post_text_present": False,
            "post_or_user_identifier_tables_present": False,
        },
        "promise_checks": promise_checks,
        "summary": {
            "promises_satisfied": sum(promise_checks.values()),
            "promises_checked": len(promise_checks),
            "verdict": "PASS" if all(promise_checks.values()) else "FAIL",
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
        default=Path("results/artifact_release_coverage_audit_2026-08-22.json"),
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
        f"engineered features={result['dataset']['engineered_feature_columns']}"
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
    return 0 if result["summary"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
