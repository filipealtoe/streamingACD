#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.2",
#   "pyarrow==25.0.1",
# ]
# ///
"""Verify the public CIKM 2026 dataset and result artifacts.

This verifier intentionally uses only files committed to the public repository.
It checks integrity and artifact consistency; it does not claim a fresh model fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_nan(column: pa.ChunkedArray) -> int:
    if not pa.types.is_floating(column.type):
        return 0
    return int(pc.sum(pc.cast(pc.is_nan(column), pa.int64())).as_py())


def verify_checksums(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = repo_root / relative.strip()
        entries += 1
        if not path.is_file():
            missing.append(relative.strip())
        elif sha256_file(path) != expected:
            mismatched.append(relative.strip())
    passed = not missing and not mismatched and entries == 6
    return Check(
        "Release checksums",
        passed,
        f"{entries} entries; missing={missing or 'none'}; mismatched={mismatched or 'none'}",
    )


def verify_dataset(repo_root: Path) -> list[Check]:
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"
    features = pq.read_table(artifact_root / "features_enhanced.parquet")
    labels = pq.read_table(artifact_root / "psr_labels.parquet")
    split = load_json(repo_root / "results" / "virality_split_manifest_2026-05-12.json")

    checks: list[Check] = []
    expected_feature_columns = split["feature_columns"]
    checks.append(
        Check(
            "Dataset shape",
            features.num_rows == labels.num_rows == 529
            and features.num_columns == 44
            and labels.num_columns == 5,
            f"features={features.num_rows}x{features.num_columns}; labels={labels.num_rows}x{labels.num_columns}",
        )
    )
    checks.append(
        Check(
            "Feature schema",
            features.column_names == ["cluster_id", *expected_feature_columns, "psr"]
            and labels.column_names
            == ["cluster_id", "detection_time", "e_detect", "e_final", "psr"],
            f"42 manifest features; feature schema has {features.num_columns - 2} inputs",
        )
    )

    feature_ids = features["cluster_id"].to_pylist()
    label_ids = labels["cluster_id"].to_pylist()
    checks.append(
        Check(
            "Cluster identifiers",
            len(set(feature_ids)) == len(set(label_ids)) == 529
            and set(feature_ids) == set(label_ids),
            f"unique features={len(set(feature_ids))}; unique labels={len(set(label_ids))}; same set={set(feature_ids) == set(label_ids)}",
        )
    )

    feature_psr = dict(zip(feature_ids, features["psr"].to_pylist(), strict=True))
    label_rows = {row["cluster_id"]: row for row in labels.to_pylist()}
    matching_psr = all(
        feature_psr[cluster_id] == row["psr"] for cluster_id, row in label_rows.items()
    )
    formula_matches = all(
        row["e_final"] > 0
        and 0 <= row["e_detect"] <= row["e_final"]
        and math.isclose(
            row["psr"],
            (row["e_final"] - row["e_detect"]) / row["e_final"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in label_rows.values()
    )
    psr_values = list(feature_psr.values())
    checks.append(
        Check(
            "Virality labels",
            matching_psr
            and formula_matches
            and min(psr_values) >= 0.0
            and max(psr_values) <= 1.0,
            f"keyed tables match={matching_psr}; formula matches={formula_matches}; range=[{min(psr_values):.6f}, {max(psr_values):.6f}]",
        )
    )

    expected_nan_counts = {
        "geographic_entropy_predetect": 529,
        "unique_countries_predetect": 529,
        "max_retweets_per_tweet": 529,
        "avg_likes_per_tweet": 529,
        "rt_like_ratio": 529,
        "engagement_skewness": 529,
        "engagement_gini": 529,
        "account_age_avg_days": 529,
        "user_gini": 50,
        "top_user_ratio": 50,
    }
    actual_nan_counts = {
        name: count
        for name in features.column_names
        if (count := count_nan(features[name])) > 0
    }
    checks.append(
        Check(
            "Documented missingness",
            actual_nan_counts == expected_nan_counts,
            f"NaN counts match the data card: {actual_nan_counts}",
        )
    )

    restricted_columns = {
        "text",
        "raw_text",
        "tweet_id",
        "post_id",
        "user_id",
        "username",
        "screen_name",
        "handle",
        "profile_description",
        "location",
    }
    present_restricted = restricted_columns.intersection(
        features.column_names + labels.column_names
    )
    checks.append(
        Check(
            "Public-data boundary",
            not present_restricted,
            f"raw text/user/post identifier columns present={sorted(present_restricted) or 'none'}",
        )
    )

    counts = split["counts"]
    test_size = counts["test_rows"]
    permutation = np.random.RandomState(42).permutation(len(feature_ids))
    derived_test = [feature_ids[index] for index in permutation[:test_size]]
    derived_train = [feature_ids[index] for index in permutation[test_size:]]
    stored_train = split["train_cluster_ids"]
    stored_test = split["test_cluster_ids"]
    checks.append(
        Check(
            "Deterministic split",
            counts
            == {
                "feature_columns": 42,
                "features_rows": 529,
                "labels_rows": 529,
                "test_rows": 106,
                "train_rows": 423,
            }
            and derived_train == stored_train
            and derived_test == stored_test
            and not set(stored_train).intersection(stored_test)
            and set(stored_train + stored_test) == set(feature_ids),
            f"train={len(stored_train)}; test={len(stored_test)}; RandomState(42) membership matches",
        )
    )
    return checks


def rounded_metrics(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        round(float(row[key]), 3)
        for key in ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")
    )


def verify_result_artifacts(repo_root: Path) -> list[Check]:
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"
    tuned = {
        row["name"]: row for row in load_json(artifact_root / "tuned_baselines.json")
    }
    complete = {
        row["name"]: row for row in load_json(artifact_root / "complete_baselines.json")
    }
    table_text = (artifact_root / "latex_table.tex").read_text(encoding="utf-8")

    expected = {
        "Random": (tuned, (-0.052, -1.141, 0.381, 0.281, 0.167, 0.075)),
        "Ridge": (tuned, (0.530, 0.293, 0.222, 0.410, 0.203, 0.080)),
        "BayesianRidge": (tuned, (0.551, 0.293, 0.226, 0.413, 0.131, 0.041)),
        "RandomForest": (tuned, (0.532, 0.319, 0.217, 0.547, 0.322, 0.041)),
        "LightGBM": (tuned, (0.549, 0.333, 0.211, 0.581, 0.322, 0.081)),
        "SVR (RBF)": (tuned, (0.542, 0.336, 0.207, 0.632, 0.343, 0.041)),
        "LSTM": (complete, (0.016, -0.021, 0.280, 0.263, 0.180, 0.041)),
        "GRU": (complete, (0.082, -0.023, 0.281, 0.335, 0.149, 0.041)),
        "Hawkes (HIP)": (complete, (0.247, -3.491, 0.587, 0.000, 0.000, 0.000)),
        "BERTweet + Features": (tuned, (0.510, 0.255, 0.222, 0.541, 0.256, 0.111)),
    }

    checks: list[Check] = []
    for name, (source, expected_metrics) in expected.items():
        actual = rounded_metrics(source[name]) if name in source else ()
        checks.append(
            Check(
                f"Result artifact: {name}",
                actual == expected_metrics and name in table_text,
                f"rounded metrics={actual}; expected={expected_metrics}; present in LaTeX={name in table_text}",
            )
        )
    return checks


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root to verify (defaults to the parent of this script's directory).",
    )
    return parser.parse_args()


def run_check_group(name: str, check_group: Any) -> list[Check]:
    try:
        result = check_group()
    except Exception as error:  # noqa: BLE001 - malformed public artifacts must produce a readable FAIL
        return [Check(f"{name} execution", False, f"{type(error).__name__}: {error}")]
    return [result] if isinstance(result, Check) else list(result)


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    checks = [
        *run_check_group("Checksum verification", lambda: verify_checksums(repo_root)),
        *run_check_group("Dataset verification", lambda: verify_dataset(repo_root)),
        *run_check_group(
            "Result-artifact verification", lambda: verify_result_artifacts(repo_root)
        ),
    ]

    print("CIKM 2026 public artifact verification")
    print(f"Repository root: {repo_root}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    failures = [check for check in checks if not check.passed]
    print(f"\nSummary: {len(checks) - len(failures)} passed, {len(failures)} failed")
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
