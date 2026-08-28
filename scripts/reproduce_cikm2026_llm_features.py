#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 19:21 PDT.
# Reason: make the three paper-facing LLM-feature F1 cells executable from a
# text-free numerical bundle.
# Artifact scope — Sérgio Pinto, 2026-08-25 18:18 WEST.
# Reason: keep the public command focused on the exact three-cell numerical
# reproduction from the checksum-bound matrix bundle.
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "scikit-learn==1.8.0",
# ]
# ///
"""Reproduce the CIKM PCA-64 + LLM + text-feature classifier row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PAPER_F1 = {"CT24": 0.694, "ClaimBuster": 0.894, "CT23": 0.846}
DATASET_KEYS = {
    "CT24": "ct24",
    "ClaimBuster": "claimbuster",
    "CT23": "ct23",
}
EXPECTED_SHAPES = {
    "X_train": (22_402, 153),
    "X_dev": (1_031, 153),
    "X_ct24": (341, 153),
    "X_claimbuster": (1_032, 153),
    "X_ct23": (318, 153),
}


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    positive = labels.astype(bool)
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "threshold": threshold,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "accuracy": (true_positive + true_negative) / len(labels),
    }


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    rows = [
        evaluate(labels, probabilities, float(threshold))
        for threshold in np.arange(0.30, 0.80, 0.05)
    ]
    return max(rows, key=lambda row: row["f1"])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/llm_features_table_reproduction_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    artifact_parent = (
        root
        / "reproducibility/source_artifacts/checkworthiness"
    )
    artifact_root = artifact_parent / "llm_features_classifier"
    with np.load(artifact_root / "matrices.npz", allow_pickle=False) as bundle:
        matrices = {name: bundle[name].copy() for name in bundle.files}
    with np.load(
        artifact_root / "reference_predictions.npz", allow_pickle=False
    ) as bundle:
        references = {name: bundle[name].copy() for name in bundle.files}

    shape_checks = {
        key: tuple(matrices[key].shape) == expected
        for key, expected in EXPECTED_SHAPES.items()
    }
    feature_names = matrices["feature_names"].tolist()
    feature_partition = {
        "text": sum(name.startswith("feat_") for name in feature_names),
        "pca64": sum(name.startswith("pca64_") for name in feature_names),
        "llm": sum(
            not name.startswith(("feat_", "pca64_")) for name in feature_names
        ),
    }
    partition_ok = feature_partition == {"text": 35, "pca64": 64, "llm": 54}

    training_features = np.vstack([matrices["X_train"], matrices["X_dev"]])
    training_labels = np.concatenate([matrices["y_train"], matrices["y_dev"]])
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(training_features)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(scaled_training, training_labels)

    results: dict[str, dict[str, Any]] = {}
    all_values_match = True
    all_reference_decisions_match = True
    for public_name, key in DATASET_KEYS.items():
        labels = matrices[f"y_{key}"]
        probabilities = classifier.predict_proba(
            scaler.transform(matrices[f"X_{key}"])
        )[:, 1]
        metrics = best_threshold(labels, probabilities)
        reference = references[key]
        decisions_match = bool(
            np.array_equal(
                probabilities >= metrics["threshold"],
                reference >= metrics["threshold"],
            )
        )
        rounded_matches = round(float(metrics["f1"]), 3) == PAPER_F1[public_name]
        all_values_match &= rounded_matches
        all_reference_decisions_match &= decisions_match
        results[public_name] = {
            **metrics,
            "paper_f1": PAPER_F1[public_name],
            "rounded_match": rounded_matches,
            "reference_decisions_match": decisions_match,
            "max_probability_delta_from_reference": float(
                np.max(np.abs(probabilities - reference))
            ),
            "n": len(labels),
        }

    status = (
        "PASS"
        if all(shape_checks.values())
        and partition_ok
        and all_values_match
        and all_reference_decisions_match
        else "FAIL"
    )
    summary = {
        "change_note": (
            "Sérgio Pinto, 2026-08-25 18:18 WEST — Freshly fitted the "
            "PCA-64 + LLM + text-feature classifier from the text-free "
            "numerical bundle."
        ),
        "method": {
            "classifier": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "scaler": "StandardScaler fitted on CT24 train+dev",
            "thresholds": [float(value) for value in np.arange(0.30, 0.80, 0.05)],
            "reported_threshold_selection": (
                "maximum F1 selected separately on each evaluation set, "
                "matching the retained paper source"
            ),
        },
        "shape_checks": shape_checks,
        "feature_partition": feature_partition,
        "results": results,
        "status": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, metrics in results.items():
        outcome = "PASS" if metrics["rounded_match"] else "FAIL"
        print(
            f"{name}: F1={metrics['f1']:.12f} @ {metrics['threshold']:.2f}; "
            f"paper={metrics['paper_f1']:.3f}; {outcome}"
        )
    print(f"VERDICT: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
