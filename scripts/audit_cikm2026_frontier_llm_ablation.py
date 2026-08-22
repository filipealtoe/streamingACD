#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 20:18 PDT.
# Reason: make the Opus ablation arithmetic checkable without presenting
# reconstructed aggregates as a replacement for the missing API response log.
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4"]
# ///
"""Audit the recoverable CIKM frontier-LLM ablation aggregates."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


PAPER_LLM = {
    "accuracy": 0.853,
    "f1": 0.722,
    "precision": 0.707,
    "recall": 0.739,
}


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    smaller = min(discordant_a, discordant_b)
    total = discordant_a + discordant_b
    return min(
        1.0,
        2.0 * sum(comb(total, value) for value in range(smaller + 1)) / 2**total,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/frontier_llm_ablation_audit_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    run_root = root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    labels = np.load(run_root / "test_labels.npy", allow_pickle=False).astype(int)
    mtl_probs = np.load(run_root / "test_probs.npy", allow_pickle=False)
    mtl_predicted = (mtl_probs >= 0.5).astype(int)
    n = len(labels)
    positives = int(labels.sum())
    negatives = n - positives

    llm_confusion_candidates: list[dict[str, int | float]] = []
    for true_positive in range(positives + 1):
        false_negative = positives - true_positive
        for false_positive in range(negatives + 1):
            true_negative = negatives - false_positive
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = true_positive / positives if positives else 0.0
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator else 0.0
            accuracy = (true_positive + true_negative) / n
            if {
                "accuracy": round(accuracy, 3),
                "f1": round(f1, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
            } == PAPER_LLM:
                llm_confusion_candidates.append(
                    {
                        "accuracy": accuracy,
                        "f1": f1,
                        "false_negative": false_negative,
                        "false_positive": false_positive,
                        "precision": precision,
                        "recall": recall,
                        "true_negative": true_negative,
                        "true_positive": true_positive,
                    }
                )

    llm_correct = int(
        llm_confusion_candidates[0]["true_positive"]
        + llm_confusion_candidates[0]["true_negative"]
    )
    mtl_correct = int(np.sum(mtl_predicted == labels))
    paired_candidates: list[dict[str, int | float]] = []
    for llm_only_correct in range(n + 1):
        mtl_only_correct = mtl_correct - llm_correct + llm_only_correct
        both_correct = llm_correct - llm_only_correct
        both_wrong = n - both_correct - llm_only_correct - mtl_only_correct
        if min(both_correct, mtl_only_correct, llm_only_correct, both_wrong) < 0:
            continue
        p_value = exact_mcnemar_p(llm_only_correct, mtl_only_correct)
        if min(llm_only_correct, mtl_only_correct) == 16 and round(p_value, 5) == 0.00779:
            paired_candidates.append(
                {
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "llm_only_correct": llm_only_correct,
                    "mcnemar_exact_p": p_value,
                    "mcnemar_statistic": 16,
                    "mtl_only_correct": mtl_only_correct,
                }
            )

    agreement_percentages = {
        "checkability": 82.4,
        "harm": 90.6,
        "verifiability": 77.4,
    }
    agreement_counts = {
        dimension: [
            count
            for count in range(n + 1)
            if round(100.0 * count / n, 1) == percentage
        ]
        for dimension, percentage in agreement_percentages.items()
    }
    cost_from_rounded_total = 6.16 / n * 1000
    latency_from_rounded_call_mean = 3 * 3.3
    arithmetic_pass = (
        n == 341
        and positives == 88
        and len(llm_confusion_candidates) == 1
        and len(paired_candidates) == 1
        and all(len(counts) == 1 for counts in agreement_counts.values())
        and round(latency_from_rounded_call_mean, 1) == 9.9
        and abs(cost_from_rounded_total - 18.07) < 0.01
    )

    output = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 20:18 PDT — Reconstructed the unique "
            "aggregate confusion and paired-test arithmetic implied by the paper "
            "while keeping the missing 1,023-response log explicit."
        ),
        "aggregate_arithmetic_status": "PASS" if arithmetic_pass else "FAIL",
        "dimension_agreement": {
            dimension: {
                "agreement_count_candidates": counts,
                "paper_percent": agreement_percentages[dimension],
            }
            for dimension, counts in agreement_counts.items()
        },
        "llm_aggregate_confusion_candidates": llm_confusion_candidates,
        "paper_cost_and_latency": {
            "api_calls": 1023,
            "cost_from_rounded_6_16_total_per_1000_claims": cost_from_rounded_total,
            "paper_approximate_cost_per_1000_claims": 18.07,
            "paper_latency_per_claim_seconds": 9.9,
            "paper_mean_latency_per_call_seconds": 3.3,
            "paper_observed_total_cost_usd_rounded": 6.16,
            "three_call_latency_from_rounded_mean_seconds": latency_from_rounded_call_mean,
        },
        "paired_correctness_candidates": paired_candidates,
        "public_inputs": {
            "ct24_labels_n": n,
            "ct24_negative_labels": negatives,
            "ct24_positive_labels": positives,
            "mtl_correct": mtl_correct,
            "mtl_f1_paper": 0.833,
        },
        "release_boundary": {
            "api_response_log_retained": False,
            "direct_llm_metric_reproduction": False,
            "per_sample_llm_predictions_retained": False,
            "status": "UNRESOLVED",
            "why": "The raw 1,023 API responses and per-sample Opus predictions were not found in the recovered workspace. Aggregate arithmetic cannot prove the historical inference run."
        },
        "status": "UNRESOLVED" if arithmetic_pass else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"LLM aggregate confusion candidates: {len(llm_confusion_candidates)}; "
        f"paired McNemar candidates: {len(paired_candidates)}"
    )
    print(f"Aggregate arithmetic: {'PASS' if arithmetic_pass else 'FAIL'}")
    print("Direct Opus reproduction: UNRESOLVED")
    print(f"VERDICT: {output['status']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
