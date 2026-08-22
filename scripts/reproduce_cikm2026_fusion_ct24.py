#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 19:45 PDT.
# Reason: make the supported CT24 Fusion paper value executable from the exact
# retained component probabilities without claiming absent cross-benchmark arrays.
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
"""Reproduce the CT24 cell of the CIKM Fusion row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAPER_F1 = 0.836
THRESHOLD = 0.5


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= THRESHOLD
    positive = labels.astype(bool)
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "threshold": THRESHOLD,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "precision": true_positive / (true_positive + false_positive),
        "recall": true_positive / (true_positive + false_negative),
        "accuracy": (true_positive + true_negative) / len(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/fusion_ct24_reproduction_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    checkworthiness_root = (
        root / "reproducibility/source_artifacts/checkworthiness"
    )
    bundle_root = checkworthiness_root / "fusion_classifier"
    matrices = np.load(
        checkworthiness_root / "llm_features_classifier/matrices.npz",
        allow_pickle=False,
    )
    encoder_probabilities = np.load(
        bundle_root / "ct24_encoder_t03_probs.npy", allow_pickle=False
    )
    llm_probabilities = np.load(
        bundle_root / "ct24_llm_xgboost_test_probs.npy", allow_pickle=False
    )
    retained_fusion = np.load(
        bundle_root / "ct24_fusion_test_probs.npy", allow_pickle=False
    )
    recomputed_fusion = 0.5 * encoder_probabilities + 0.5 * llm_probabilities
    max_probability_delta = float(
        np.max(np.abs(recomputed_fusion - retained_fusion))
    )
    metrics = evaluate(matrices["y_ct24"], recomputed_fusion)
    rounded_match = round(float(metrics["f1"]), 3) == PAPER_F1
    status = (
        "PASS"
        if len(recomputed_fusion) == 341
        and max_probability_delta <= 3e-8
        and rounded_match
        else "FAIL"
    )
    summary = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 19:45 PDT — Recomputed the CT24 "
            "Fusion probabilities and paper-facing F1 from the retained "
            "temperature-0.3 encoder and XGBoost components."
        ),
        "configuration": {
            "encoder_temperature": 0.3,
            "encoder_weight": 0.5,
            "llm_xgboost_weight": 0.5,
            "threshold": THRESHOLD,
        },
        "n": len(recomputed_fusion),
        "metrics": metrics,
        "paper_f1": PAPER_F1,
        "rounded_match": rounded_match,
        "max_probability_delta_from_retained_fusion": max_probability_delta,
        "status": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"CT24 Fusion: F1={metrics['f1']:.12f}; paper={PAPER_F1:.3f}; "
        f"max probability delta={max_probability_delta:.3e}"
    )
    print(f"VERDICT: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
