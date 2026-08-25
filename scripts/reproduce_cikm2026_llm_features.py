#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 19:21 PDT.
# Reason: make the three paper-facing LLM-feature F1 cells executable from a
# text-free numerical bundle.
# Artifact clarification — Sérgio Pinto, 2026-08-21 19:27 PDT.
# Reason: preserve the historical paper-value reproduction while also reporting
# a single threshold selected on CT24 development labels and held fixed elsewhere.
# Artifact correction — Sérgio Pinto, 2026-08-25 14:12 WEST.
# Reason: preserve the paper-run row-position result while also recomputing the
# LLM partition after complete one-to-one sentence-ID alignment.
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "polars==1.27.1",
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
import polars as pl
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


def align_llm_features(
    feature_path: Path,
    canonical_ids: list[str],
    llm_feature_names: list[str],
    context: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = pl.read_parquet(feature_path).with_columns(
        pl.col("sentence_id").cast(pl.String)
    )
    canonical = pl.DataFrame(
        {"sentence_id": [str(value) for value in canonical_ids]}
    ).with_row_index("__row_order")
    if canonical["sentence_id"].n_unique() != len(canonical):
        raise ValueError(f"{context}: canonical sentence IDs are not unique")
    if features["sentence_id"].n_unique() != len(features):
        raise ValueError(f"{context}: feature sentence IDs are not unique")
    canonical_set = set(canonical["sentence_id"].to_list())
    feature_set = set(features["sentence_id"].to_list())
    if canonical_set != feature_set:
        raise ValueError(f"{context}: sentence-ID sets differ")
    if not set(llm_feature_names).issubset(features.columns):
        raise ValueError(f"{context}: the expected LLM feature schema is incomplete")
    aligned = (
        canonical.join(features, on="sentence_id", how="left", validate="1:1")
        .sort("__row_order")
    )
    values = aligned.select(llm_feature_names)
    if values.null_count().row(0) != (0,) * len(llm_feature_names):
        raise ValueError(f"{context}: aligned LLM features contain null values")
    return values.to_numpy(), {
        "rows": len(aligned),
        "unique_sentence_ids": aligned["sentence_id"].n_unique(),
        "one_to_one_id_set_match": True,
        "source_feature_order_differed": (
            canonical["sentence_id"].to_list()
            != features["sentence_id"].to_list()
        ),
    }


def corrected_llm_partition(
    matrices: dict[str, np.ndarray], artifact_parent: Path
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Replace only the positionally combined LLM partition using ID joins."""
    corrected = {name: value.copy() for name, value in matrices.items()}
    feature_names = corrected["feature_names"].tolist()
    llm_names = [
        name for name in feature_names if not name.startswith(("feat_", "pca64_"))
    ]
    llm_indices = [feature_names.index(name) for name in llm_names]
    if len(llm_names) != 54:
        raise ValueError("unexpected LLM feature partition")

    audit: dict[str, dict[str, Any]] = {}
    split_keys = {"train": "train", "dev": "dev", "test": "ct24"}
    for split, key in split_keys.items():
        labels = pl.read_parquet(
            artifact_parent / f"ct24_labels/{split}_labels.parquet"
        )
        expected_labels = labels["label"].cast(pl.Int8).to_numpy()
        if not np.array_equal(corrected[f"y_{key}"], expected_labels):
            raise ValueError(f"CT24 {split}: retained labels changed order")
        values, audit_row = align_llm_features(
            artifact_parent / f"ct24_llm_features_v4/{split}_llm_features.parquet",
            labels["sentence_id"].cast(pl.String).to_list(),
            llm_names,
            f"CT24 {split}",
        )
        corrected[f"X_{key}"][:, llm_indices] = values
        audit[f"CT24 {split}"] = audit_row

    claimbuster = pl.read_csv(
        artifact_parent / "benchmarks/claim_buster/groundtruth.csv"
    )
    ct23 = pl.read_csv(
        artifact_parent / "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv",
        separator="\t",
    )
    benchmark_rows = {
        "claimbuster": (
            claimbuster["Sentence_id"].cast(pl.String).to_list(),
            artifact_parent
            / "benchmark_llm_features/CB_groundtruth_llm_features.parquet",
        ),
        "ct23": (
            ct23["Sentence_id"].cast(pl.String).to_list(),
            artifact_parent / "benchmark_llm_features/CT23_llm_features.parquet",
        ),
    }
    for key, (ids, path) in benchmark_rows.items():
        values, audit_row = align_llm_features(
            path, ids, llm_names, key
        )
        corrected[f"X_{key}"][:, llm_indices] = values
        audit["ClaimBuster" if key == "claimbuster" else "CT23"] = audit_row

    return corrected, audit


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

    development_scaler = StandardScaler()
    scaled_train = development_scaler.fit_transform(matrices["X_train"])
    development_classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    development_classifier.fit(scaled_train, matrices["y_train"])
    development_probabilities = development_classifier.predict_proba(
        development_scaler.transform(matrices["X_dev"])
    )[:, 1]
    development_selection = best_threshold(
        matrices["y_dev"], development_probabilities
    )
    development_threshold = float(development_selection["threshold"])

    results: dict[str, dict[str, Any]] = {}
    final_probabilities: dict[str, np.ndarray] = {}
    all_values_match = True
    all_reference_decisions_match = True
    for public_name, key in DATASET_KEYS.items():
        labels = matrices[f"y_{key}"]
        probabilities = classifier.predict_proba(
            scaler.transform(matrices[f"X_{key}"])
        )[:, 1]
        final_probabilities[public_name] = probabilities
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

    fixed_threshold_results = {
        public_name: evaluate(
            matrices[f"y_{key}"],
            final_probabilities[public_name],
            development_threshold,
        )
        for public_name, key in DATASET_KEYS.items()
    }

    corrected_matrices, alignment_audit = corrected_llm_partition(
        matrices, artifact_parent
    )
    corrected_scaler = StandardScaler()
    corrected_training = np.vstack(
        [corrected_matrices["X_train"], corrected_matrices["X_dev"]]
    )
    corrected_labels = np.concatenate(
        [corrected_matrices["y_train"], corrected_matrices["y_dev"]]
    )
    corrected_classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    corrected_classifier.fit(
        corrected_scaler.fit_transform(corrected_training), corrected_labels
    )

    corrected_development_scaler = StandardScaler()
    corrected_development_classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    corrected_development_classifier.fit(
        corrected_development_scaler.fit_transform(corrected_matrices["X_train"]),
        corrected_matrices["y_train"],
    )
    corrected_development_probabilities = (
        corrected_development_classifier.predict_proba(
            corrected_development_scaler.transform(corrected_matrices["X_dev"])
        )[:, 1]
    )
    corrected_development_selection = best_threshold(
        corrected_matrices["y_dev"], corrected_development_probabilities
    )
    corrected_threshold = float(corrected_development_selection["threshold"])
    corrected_results: dict[str, dict[str, Any]] = {}
    for public_name, key in DATASET_KEYS.items():
        labels = corrected_matrices[f"y_{key}"]
        probabilities = corrected_classifier.predict_proba(
            corrected_scaler.transform(corrected_matrices[f"X_{key}"])
        )[:, 1]
        corrected_results[public_name] = {
            "fixed_source_development_threshold": evaluate(
                labels, probabilities, corrected_threshold
            ),
            "evaluation_oracle_diagnostic": best_threshold(labels, probabilities),
            "paper_f1": PAPER_F1[public_name],
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
            "Sérgio Pinto, 2026-08-21 19:21 PDT — Freshly fitted the "
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
        "development_selected_threshold_diagnostic": {
            "selection_split": "CT24 development",
            "selection_training_split": "CT24 training",
            "threshold": development_threshold,
            "development_selection_metrics": development_selection,
            "evaluation_model_training_split": "CT24 training plus development",
            "results": fixed_threshold_results,
        },
        "corrected_sentence_id_replication": {
            "change_note": (
                "Sérgio Pinto, 2026-08-25 14:18 WEST — joined the recovered "
                "LLM partition one-to-one by sentence_id and retained the "
                "positionally combined paper-run result separately."
            ),
            "status": "COMPLETE",
            "correction": (
                "the 54 recovered LLM feature columns are joined one-to-one "
                "by sentence_id before combination with the retained 35 text "
                "and 64 PCA columns"
            ),
            "retained_partition_boundary": (
                "the text and PCA partitions remain in the canonical row order "
                "of the checksum-bound numerical bundle; that bundle does not "
                "embed their per-row sentence IDs"
            ),
            "development_threshold": corrected_threshold,
            "development_selection_metrics": corrected_development_selection,
            "alignment_audit": alignment_audit,
            "results": corrected_results,
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
