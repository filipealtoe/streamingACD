#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "lightgbm==4.6.0",
#   "numpy==1.26.4",
#   "pyarrow==22.0.0",
#   "scikit-learn==1.8.0",
#   "scipy==1.16.3",
# ]
# ///
"""Freshly fit the public CIKM 2026 tabular virality baselines.

The sequence, point-process, and text models are outside this command because
their raw time-series or text inputs are not part of the public package.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import fbeta_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

METRIC_NAMES = ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")


@dataclass(frozen=True)
class Reproduction:
    name: str
    actual: dict[str, float]
    stored: dict[str, float]
    max_delta: float
    passed: bool


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    feature_columns = [
        name for name in table.column_names if name not in {"cluster_id", "psr"}
    ]
    matrix = np.column_stack(
        [
            np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=float)
            for name in feature_columns
        ]
    )
    labels = np.asarray(table["psr"].to_numpy(zero_copy_only=False), dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), labels


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.clip(predictions, 0.0, 1.0)
    metrics = {
        "spearman_rho": float(stats.spearmanr(labels, predictions).statistic),
        "r2": float(r2_score(labels, predictions)),
        "mae": float(mean_absolute_error(labels, predictions)),
    }
    for threshold, key in ((0.65, "f2_065"), (0.75, "f2_075"), (0.85, "f2_085")):
        metrics[key] = float(
            fbeta_score(
                labels >= threshold, predictions >= threshold, beta=2, zero_division=0
            )
        )
    return metrics


def compare(
    name: str,
    actual: dict[str, float],
    stored_row: dict[str, Any],
    tolerance: float,
) -> Reproduction:
    stored = {metric: float(stored_row[metric]) for metric in METRIC_NAMES}
    max_delta = max(abs(actual[metric] - stored[metric]) for metric in METRIC_NAMES)
    passed = all(
        math.isclose(actual[metric], stored[metric], rel_tol=0.0, abs_tol=tolerance)
        for metric in METRIC_NAMES
    )
    return Reproduction(
        name=name, actual=actual, stored=stored, max_delta=max_delta, passed=passed
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the public CIKM artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Absolute per-metric tolerance for the stored full-precision results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"

    matrix, labels = load_matrix(artifact_root / "features_enhanced.parquet")
    train_x, test_x, train_y, test_y = train_test_split(
        matrix,
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=None,
    )
    scaler = StandardScaler()
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)

    models = [
        (
            "Ridge",
            Ridge(alpha=0.01),
            scaled_train_x,
            scaled_test_x,
        ),
        (
            "BayesianRidge",
            BayesianRidge(alpha_1=1e-6, lambda_1=1e-6),
            scaled_train_x,
            scaled_test_x,
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=2,
                n_estimators=100,
            ),
            train_x,
            test_x,
        ),
        (
            "LightGBM",
            LGBMRegressor(
                random_state=42,
                verbose=-1,
                learning_rate=0.05,
                max_depth=6,
                n_estimators=100,
            ),
            train_x,
            test_x,
        ),
        (
            "SVR (RBF)",
            SVR(C=0.1, epsilon=0.05, gamma="auto"),
            scaled_train_x,
            scaled_test_x,
        ),
    ]
    stored_rows = {
        row["name"]: row
        for row in json.loads(
            (artifact_root / "tuned_baselines.json").read_text(encoding="utf-8")
        )
    }

    reproductions: list[Reproduction] = []
    random_generator = np.random.RandomState(42)
    random_actual = evaluate(test_y, random_generator.uniform(0.0, 1.0, len(test_y)))
    reproductions.append(
        compare("Random", random_actual, stored_rows["Random"], args.tolerance)
    )
    for name, model, model_train_x, model_test_x in models:
        model.fit(model_train_x, train_y)
        actual = evaluate(test_y, model.predict(model_test_x))
        reproductions.append(compare(name, actual, stored_rows[name], args.tolerance))

    print("CIKM 2026 public tabular-baseline reproduction")
    print(f"Repository root: {repo_root}")
    print(
        f"Dataset: {len(labels)} rows, {matrix.shape[1]} features; split={len(train_y)}/{len(test_y)}"
    )
    print(f"Absolute tolerance: {args.tolerance:g}")
    for reproduction in reproductions:
        status = "PASS" if reproduction.passed else "FAIL"
        print(
            f"[{status}] {reproduction.name}: max metric delta={reproduction.max_delta:.3e}"
        )
        for metric in METRIC_NAMES:
            print(
                f"  {metric}: actual={reproduction.actual[metric]:.17g} "
                f"stored={reproduction.stored[metric]:.17g} "
                f"delta={reproduction.actual[metric] - reproduction.stored[metric]:.3e}"
            )

    failures = [
        reproduction for reproduction in reproductions if not reproduction.passed
    ]
    print(
        f"\nSummary: {len(reproductions) - len(failures)} reproduced, {len(failures)} mismatched"
    )
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
