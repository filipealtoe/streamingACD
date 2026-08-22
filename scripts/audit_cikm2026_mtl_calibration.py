#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 21:31 PDT.
# Reason: test the paper's approximately 0.03 threshold-calibration statement
# and distinguish it from the retained threshold-selection protocol.
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.2.5",
# ]
# ///
"""Audit the retained CT24 threshold sweep behind the reported F1 value."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER_F1 = 0.833
PAPER_APPROXIMATE_SHIFT = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_f1(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    predictions = probabilities >= threshold
    positives = labels.astype(bool)
    true_positives = int(np.sum(predictions & positives))
    false_positives = int(np.sum(predictions & ~positives))
    false_negatives = int(np.sum(~predictions & positives))
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0


def sweep(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "threshold": float(threshold),
            "f1": binary_f1(labels, probabilities, float(threshold)),
        }
        for threshold in np.arange(0.30, 0.75, 0.05)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    run_root = root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    output = args.output or root / "results/mtl_threshold_calibration_audit_2026-08-21.json"

    dev_labels = np.load(run_root / "dev_labels.npy")
    dev_probabilities = np.load(run_root / "dev_probs.npy")
    test_labels = np.load(run_root / "test_labels.npy")
    test_probabilities = np.load(run_root / "test_probs.npy")
    retained_summary = json.loads((run_root / "results.json").read_text(encoding="utf-8"))

    dev_sweep = sweep(dev_labels, dev_probabilities)
    test_sweep = sweep(test_labels, test_probabilities)
    dev_best = max(dev_sweep, key=lambda row: row["f1"])
    test_best = max(test_sweep, key=lambda row: row["f1"])
    test_at_dev_threshold = next(
        row
        for row in test_sweep
        if np.isclose(row["threshold"], dev_best["threshold"], rtol=0.0, atol=1e-12)
    )
    delta = test_at_dev_threshold["f1"] - test_best["f1"]

    retained_matches = all(
        np.isclose(recomputed[key], retained[key], rtol=0.0, atol=1e-15)
        for recomputed_rows, retained_rows in (
            (dev_sweep, retained_summary["dev_results"]),
            (test_sweep, retained_summary["test_results"]),
        )
        for recomputed, retained in zip(recomputed_rows, retained_rows, strict=True)
        for key in ("threshold", "f1")
    )
    magnitude_matches = abs(abs(delta) - PAPER_APPROXIMATE_SHIFT) <= 0.005
    reported_value_is_test_optimized = (
        round(test_best["f1"], 3) == PAPER_F1
        and not np.isclose(
            dev_best["threshold"], test_best["threshold"], rtol=0.0, atol=1e-12
        )
    )

    result = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 21:31 PDT — Recomputed the retained CT24 "
            "development and test threshold sweeps to test the approximately 0.03 "
            "sensitivity statement and its selection protocol."
        ),
        "verdict": "PARTIAL",
        "paper_claim": {
            "reported_f1": PAPER_F1,
            "threshold_selection": "development selection, final test reporting",
            "sensitivity": "a different calibration could shift F1 by approximately 0.03 in either direction",
        },
        "inputs": {
            "development_rows": int(len(dev_labels)),
            "test_rows": int(len(test_labels)),
            "thresholds": [row["threshold"] for row in dev_sweep],
            "sha256": {
                name: sha256(run_root / name)
                for name in (
                    "dev_labels.npy",
                    "dev_probs.npy",
                    "test_labels.npy",
                    "test_probs.npy",
                    "results.json",
                )
            },
        },
        "observed": {
            "development_selected_threshold": dev_best["threshold"],
            "development_f1": dev_best["f1"],
            "test_optimized_threshold": test_best["threshold"],
            "test_optimized_f1": test_best["f1"],
            "test_f1_at_development_threshold": test_at_dev_threshold["f1"],
            "signed_f1_shift": delta,
            "absolute_f1_shift": abs(delta),
        },
        "checks": {
            "retained_sweep_recomputed": "PASS" if retained_matches else "FAIL",
            "approximately_0_03_magnitude": "PASS" if magnitude_matches else "FAIL",
            "either_direction": "PARTIAL: the retained sweep directly shows the negative direction only",
            "selection_protocol": (
                "MISMATCH: the reported 0.833 value is the test-sweep optimum at 0.50; "
                "the development-selected threshold is 0.60 and yields 0.8046 on test"
                if reported_value_is_test_optimized
                else "PASS"
            ),
        },
    }
    if not retained_matches or not magnitude_matches or not reported_value_is_test_optimized:
        result["verdict"] = "FAIL"
    elif result["checks"]["either_direction"].startswith("PARTIAL"):
        result["verdict"] = "PARTIAL"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"dev threshold={dev_best['threshold']:.2f}; test-optimal threshold="
        f"{test_best['threshold']:.2f}; test F1 shift={delta:+.12f}"
    )
    print(f"VERDICT: {result['verdict']}")
    return 0 if result["verdict"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
