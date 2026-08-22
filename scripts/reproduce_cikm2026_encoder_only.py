#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 20:10 PDT.
# Reason: retain direct CT24/CT23 reproduction while ensuring the missing
# ClaimBuster prediction vector cannot be mistaken for reproduced evidence.
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
"""Reproduce the CIKM Encoder Only check-worthiness table row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAPER_F1 = {"CT24": 0.821, "ClaimBuster": 0.970, "CT23": 0.928}
DIRECT_THRESHOLDS = {"CT24": 0.50, "CT23": 0.50}


def evaluate(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
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
        "precision": true_positive / (true_positive + false_positive),
        "recall": true_positive / (true_positive + false_negative),
        "accuracy": (true_positive + true_negative) / len(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "n": len(labels),
    }


def evaluate_counts(counts: dict[str, int], threshold: float) -> dict[str, float | int]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    true_negative = counts["true_negative"]
    denominator = 2 * true_positive + false_positive + false_negative
    total = true_positive + false_positive + false_negative + true_negative
    return {
        "threshold": threshold,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "precision": true_positive / (true_positive + false_positive),
        "recall": true_positive / (true_positive + false_negative),
        "accuracy": (true_positive + true_negative) / total,
        **counts,
        "n": total,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/encoder_only_table_reproduction_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    checkworthiness_root = (
        root / "reproducibility/source_artifacts/checkworthiness"
    )
    bundle_root = checkworthiness_root / "encoder_only"
    matrices = np.load(
        checkworthiness_root / "llm_features_classifier/matrices.npz",
        allow_pickle=False,
    )
    audit = json.loads(
        (bundle_root / "ENCODER_RECOVERY_AUDIT.json").read_text(encoding="utf-8")
    )

    results: dict[str, dict[str, object]] = {}
    for name, label_key, probability_file in (
        ("CT24", "y_ct24", "ct24_probs.npy"),
        ("CT23", "y_ct23", "ct23_probs.npy"),
    ):
        probabilities = np.load(bundle_root / probability_file, allow_pickle=False)
        metrics = evaluate(
            matrices[label_key], probabilities, DIRECT_THRESHOLDS[name]
        )
        results[name] = {
            "evidence_type": "retained_per_example_probabilities",
            "metrics": metrics,
            "paper_f1": PAPER_F1[name],
            "rounded_match": round(float(metrics["f1"]), 3) == PAPER_F1[name],
        }

    claimbuster_evidence = audit["claimbuster_historical_record"]
    claimbuster_metrics = evaluate_counts(
        claimbuster_evidence["confusion_counts_reconstructed"],
        float(claimbuster_evidence["threshold"]),
    )
    results["ClaimBuster"] = {
        "evidence_type": "unauthenticated_historical_aggregate_reconstruction",
        "metrics": claimbuster_metrics,
        "paper_f1": PAPER_F1["ClaimBuster"],
        "direct_reproduction": False,
        "per_example_probability_vector_retained": False,
        "reported_f1_match": round(float(claimbuster_metrics["f1"]), 4)
        == claimbuster_evidence["f1_reported"],
        "reported_accuracy_match": round(float(claimbuster_metrics["accuracy"]), 4)
        == claimbuster_evidence["accuracy_reported"],
        "arithmetic_consistency": round(float(claimbuster_metrics["f1"]), 3)
        == PAPER_F1["ClaimBuster"],
        "reproduction_status": "unresolved",
    }

    direct_cells_pass = all(
        bool(results[name]["rounded_match"]) for name in ("CT24", "CT23")
    )
    aggregate_arithmetic_pass = bool(
        results["ClaimBuster"]["reported_f1_match"]
        and results["ClaimBuster"]["reported_accuracy_match"]
        and results["ClaimBuster"]["arithmetic_consistency"]
    )
    summary = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 20:10 PDT — Recomputed the CT24 and "
            "CT23 Encoder Only cells from retained per-example probabilities and "
            "classified the ClaimBuster aggregate reconstruction as unresolved."
        ),
        "paper_f1": PAPER_F1,
        "results": results,
        "direct_cells_status": "PASS" if direct_cells_pass else "FAIL",
        "claimbuster_arithmetic_status": (
            "PASS" if aggregate_arithmetic_pass else "FAIL"
        ),
        "status": (
            "PARTIAL" if direct_cells_pass and aggregate_arithmetic_pass else "FAIL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("CT24", "ClaimBuster", "CT23"):
        result = results[name]
        metrics = result["metrics"]
        print(
            f"{name}: F1={metrics['f1']:.12f}; "
            f"paper={PAPER_F1[name]:.3f}; evidence={result['evidence_type']}; "
            f"{('UNRESOLVED' if name == 'ClaimBuster' else ('PASS' if result['rounded_match'] else 'FAIL'))}"
        )
    print(f"VERDICT: {summary['status']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
