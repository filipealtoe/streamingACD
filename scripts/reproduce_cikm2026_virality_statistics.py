#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "scipy==1.16.3",
# ]
# ///
"""Reproduce the CIKM 2026 Table 3 significance and confidence intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-21 21:40 PDT | Reason: make the paper's Table 3 Spearman p-values,
# significance marks, bootstrap confidence intervals, and exact paired tests
# independently reproducible from the retained per-example arrays.

BASELINE_PATHS = {
    "Random": ("baseline_predictions", "Random"),
    "Ridge": ("baseline_predictions", "Ridge"),
    "BayesianRidge": ("baseline_predictions", "BayesianRidge"),
    "RandomForest": ("baseline_predictions", "RandomForest"),
    "LightGBM": ("baseline_predictions", "LightGBM"),
    "SVR_RBF": ("baseline_predictions", "SVR_RBF"),
    "LSTM": ("reproduced_missing_predictions", "LSTM"),
    "GRU": ("reproduced_missing_predictions", "GRU"),
    "HIP_Hawkes": ("reproduced_missing_predictions", "HIP_Hawkes"),
    "BERTweet": ("reproduced_missing_predictions", "BERTweet"),
    "BERTweet_plus_features": (
        "reproduced_missing_predictions",
        "BERTweet_plus_features",
    ),
}
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
PAIR_THRESHOLD = 0.65


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_arrays(repo_root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    source_group, folder = BASELINE_PATHS[name]
    source_root = repo_root / "psr" / source_group / folder
    if source_group == "baseline_predictions":
        labels_path = repo_root / "psr" / "derived_arrays" / "test_targets.npy"
        predictions_path = source_root / "test_predictions.npy"
    else:
        labels_path = source_root / "test_targets_eval_subset.npy"
        predictions_path = source_root / "test_predictions_eval_subset.npy"
    labels = np.asarray(np.load(labels_path), dtype=np.float64)
    predictions = np.asarray(np.load(predictions_path), dtype=np.float64)
    if labels.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError(f"{name}: incompatible arrays {labels.shape}/{predictions.shape}")
    if not np.all(np.isfinite(labels)) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"{name}: evaluation arrays contain non-finite values")
    return labels, predictions


def bootstrap_spearman_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    rng = np.random.default_rng(seed)
    sample_count = len(labels)
    values = np.full(n_resamples, np.nan, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, sample_count, size=sample_count)
        value = spearmanr(labels[sample], predictions[sample]).statistic
        if np.isfinite(value):
            values[index] = float(value)
    if int(np.isfinite(values).sum()) < n_resamples // 2:
        raise ValueError("Fewer than half of the bootstrap statistics are finite")
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def significance_code(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    smaller = min(discordant_a, discordant_b)
    one_tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (
        2**total
    )
    return min(1.0, 2.0 * one_tail)


def reproduce(repo_root: Path) -> dict[str, Any]:
    retained_root = repo_root / "results" / "psr_statistics_20260518_205609"
    retained = load_json(retained_root / "metrics.json")
    copied_manifest = load_json(retained_root / "predictions" / "manifest.json")
    copied_by_name = {row["name"]: row for row in copied_manifest["baselines"]}
    retained_by_name = {row["name"]: row for row in retained["baselines"]}

    rows: dict[str, Any] = {}
    row_failures: list[str] = []
    for name in BASELINE_PATHS:
        labels, predictions = load_evaluation_arrays(repo_root, name)
        correlation = spearmanr(labels, predictions)
        point = float(correlation.statistic)
        p_value = float(correlation.pvalue)
        ci95 = bootstrap_spearman_ci(labels, predictions)
        expected = retained_by_name[name]["spearman_rho"]

        source_group, folder = BASELINE_PATHS[name]
        canonical_full_path = (
            repo_root / "psr" / source_group / folder / "test_predictions.npy"
        )
        copied_path = retained_root / "predictions" / f"{name}.npy"
        copied_hash = sha256_file(copied_path)
        canonical_hash = sha256_file(canonical_full_path)
        manifest_hash = copied_by_name[name]["predictions_sha256"]

        checks = {
            "n_test": len(labels) == int(retained_by_name[name]["n_test"]),
            "spearman_point": math.isclose(
                point, float(expected["point"]), rel_tol=0.0, abs_tol=1e-15
            ),
            "spearman_p_value": math.isclose(
                p_value, float(expected["p_value"]), rel_tol=0.0, abs_tol=1e-15
            ),
            "bootstrap_ci95": all(
                math.isclose(actual, float(wanted), rel_tol=0.0, abs_tol=1e-15)
                for actual, wanted in zip(ci95, expected["ci95"], strict=True)
            ),
            "prediction_copy": copied_hash == canonical_hash == manifest_hash,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        if status == "FAIL":
            row_failures.append(name)
        rows[name] = {
            "n_test": len(labels),
            "spearman_rho": point,
            "two_tailed_p_value": p_value,
            "significance_code": significance_code(p_value),
            "bootstrap_ci95": ci95,
            "bootstrap_ci95_rounded_2": [round(value, 2) for value in ci95],
            "prediction_sha256": canonical_hash,
            "checks": checks,
            "status": status,
        }

    labels = np.asarray(
        np.load(repo_root / "psr" / "derived_arrays" / "test_targets.npy"),
        dtype=np.float64,
    )
    truth = labels > PAIR_THRESHOLD
    expected_tests = {
        (row["baseline_a"], row["baseline_b"]): row
        for row in retained["pairwise_mcnemar"]["tests"]
    }
    pair_name_map = {
        "SVR_RBF": "SVR_RBF",
        "RandomForest": "RandomForest",
        "LightGBM": "LightGBM",
    }
    binary_predictions = {
        name: np.asarray(
            np.load(
                repo_root
                / "psr"
                / "baseline_predictions"
                / folder
                / "test_predictions.npy"
            ),
            dtype=np.float64,
        )
        > PAIR_THRESHOLD
        for name, folder in pair_name_map.items()
    }
    paired_tests: list[dict[str, Any]] = []
    pair_failures: list[str] = []
    for names, expected in expected_tests.items():
        name_a, name_b = names
        correct_a = binary_predictions[name_a] == truth
        correct_b = binary_predictions[name_b] == truth
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        p_value = exact_mcnemar_p(a_only, b_only)
        passed = (
            a_only == int(expected["table_a_correct_b_wrong"])
            and b_only == int(expected["table_a_wrong_b_correct"])
            and math.isclose(
                p_value, float(expected["p_value"]), rel_tol=0.0, abs_tol=1e-15
            )
        )
        if not passed:
            pair_failures.append(f"{name_a}/{name_b}")
        paired_tests.append(
            {
                "baseline_a": name_a,
                "baseline_b": name_b,
                "discordant_a_only": a_only,
                "discordant_b_only": b_only,
                "exact_p_value": p_value,
                "status": "PASS" if passed else "FAIL",
            }
        )

    return {
        "artifact": "CIKM 2026 Table 3 significance and confidence-interval reproduction",
        "author": "Sérgio Pinto",
        "timestamp": "2026-08-21 21:40 PDT (-0700)",
        "reason": "Recompute the paper's statistical annotations from retained per-example arrays.",
        "protocol": {
            "spearman_p_value": "scipy.stats.spearmanr two-tailed",
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_interval": "2.5th and 97.5th percentiles",
            "paired_test": "exact two-sided McNemar",
            "paired_threshold": PAIR_THRESHOLD,
        },
        "rows": rows,
        "paired_tests": paired_tests,
        "summary": {
            "row_count": len(rows),
            "row_failures": row_failures,
            "paired_test_failures": pair_failures,
            "verdict": "PASS" if not row_failures and not pair_failures else "FAIL",
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
        default=Path("results/virality_statistics_reproduction_2026-08-21.json"),
        help="Output path relative to the repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    result = reproduce(repo_root)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("CIKM 2026 virality-statistics reproduction")
    print(f"Repository root: {repo_root}")
    for name, row in result["rows"].items():
        print(
            f"[{row['status']}] {name}: rho={row['spearman_rho']:.6f}; "
            f"p={row['two_tailed_p_value']:.6g}; "
            f"CI95={tuple(round(v, 6) for v in row['bootstrap_ci95'])}; "
            f"stars={row['significance_code'] or 'none'}"
        )
    for row in result["paired_tests"]:
        print(
            f"[{row['status']}] McNemar {row['baseline_a']}/{row['baseline_b']}: "
            f"discordant={row['discordant_a_only']}/{row['discordant_b_only']}; "
            f"p={row['exact_p_value']:.6g}"
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
