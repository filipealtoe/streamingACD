#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.2",
# ]
# ///
"""Audit whether the reported CT24 shared-error percentage is reproducible."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the public CIKM artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path (default: results/checkworthiness_shared_error_audit_2026-08-21.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    artifact_root = (
        repo_root / "reproducibility" / "source_artifacts" / "checkworthiness"
    )
    llm_root = artifact_root / "llm_features_classifier"
    llm_bundle = np.load(llm_root / "matrices.npz")
    llm_labels = np.asarray(llm_bundle["y_ct24"], dtype=np.int8)
    llm_ids_path = llm_root / "ct24_sentence_ids.npy"
    llm_ids = np.asarray(np.load(llm_ids_path), dtype=np.int64)
    llm_probabilities = np.asarray(
        np.load(llm_root / "reference_predictions.npz")["ct24"], dtype=float
    )
    encoder_probs = np.asarray(
        np.load(artifact_root / "encoder_only" / "ct24_probs.npy"), dtype=float
    )
    mtl_root = (
        repo_root / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553"
    )
    mtl_labels = np.asarray(np.load(mtl_root / "test_labels.npy"), dtype=np.int8)
    mtl_ids = np.asarray(
        np.load(mtl_root / "test_sentence_ids.npy"), dtype=np.int64
    )
    mtl_probabilities = np.asarray(np.load(mtl_root / "test_probs.npy"), dtype=float)

    same_length = (
        len(llm_labels)
        == len(llm_ids)
        == len(llm_probabilities)
        == len(encoder_probs)
        == len(mtl_labels)
        == len(mtl_ids)
        == len(mtl_probabilities)
        == 341
    )
    unique_ids = (
        len(np.unique(llm_ids)) == len(llm_ids)
        and len(np.unique(mtl_ids)) == len(mtl_ids)
    )
    same_id_set = set(llm_ids.tolist()) == set(mtl_ids.tolist())
    mtl_position = {int(sentence_id): index for index, sentence_id in enumerate(mtl_ids)}
    mtl_order_for_llm = np.asarray(
        [mtl_position[int(sentence_id)] for sentence_id in llm_ids], dtype=np.int64
    )
    aligned_mtl_labels = mtl_labels[mtl_order_for_llm]
    aligned_mtl_probabilities = mtl_probabilities[mtl_order_for_llm]
    aligned_labels_match = bool(np.array_equal(llm_labels, aligned_mtl_labels))
    direct_alignment_available = bool(
        same_length and unique_ids and same_id_set and aligned_labels_match
    )
    if not direct_alignment_available:
        raise RuntimeError("CT24 sentence-ID alignment invariants failed")

    thresholds = {
        "llm_features": 0.55,
        "encoder_only": 0.50,
        "four_parallel_heads": 0.50,
    }
    errors = {
        "llm_features": (llm_probabilities >= thresholds["llm_features"])
        != llm_labels,
        "encoder_only": (encoder_probs >= thresholds["encoder_only"]) != llm_labels,
        "four_parallel_heads": (
            aligned_mtl_probabilities >= thresholds["four_parallel_heads"]
        )
        != llm_labels,
    }
    error_counts = {name: int(values.sum()) for name, values in errors.items()}
    model_names = list(errors)
    error_multiplicity = sum(errors.values())
    any_error = error_multiplicity >= 1
    at_least_two = error_multiplicity >= 2
    all_three = error_multiplicity == 3
    union_count = int(any_error.sum())
    all_three_count = int(all_three.sum())
    at_least_two_count = int(at_least_two.sum())

    pairwise: dict[str, dict[str, float | int]] = {}
    for first_index, first in enumerate(model_names):
        for second in model_names[first_index + 1 :]:
            intersection = int(np.sum(errors[first] & errors[second]))
            union = int(np.sum(errors[first] | errors[second]))
            pairwise[f"{first}__{second}"] = {
                "intersection": intersection,
                "union": union,
                "intersection_over_union_percent": 100.0 * intersection / union,
                "intersection_over_first_errors_percent": (
                    100.0 * intersection / error_counts[first]
                ),
                "intersection_over_second_errors_percent": (
                    100.0 * intersection / error_counts[second]
                ),
            }

    shared_assignments = sum(
        int(
            np.sum(
                errors[name]
                & np.logical_or.reduce(
                    [errors[other] for other in model_names if other != name]
                )
            )
        )
        for name in model_names
    )
    total_error_assignments = sum(error_counts.values())
    observed_percentages = {
        "all_three_errors_over_any_model_error": (
            100.0 * all_three_count / union_count
        ),
        "errors_from_at_least_two_models_over_any_model_error": (
            100.0 * at_least_two_count / union_count
        ),
        "model_error_assignments_shared_with_another_model": (
            100.0 * shared_assignments / total_error_assignments
        ),
        "maximum_pairwise_overlap_over_either_models_errors": max(
            max(
                pair["intersection_over_first_errors_percent"],
                pair["intersection_over_second_errors_percent"],
            )
            for pair in pairwise.values()
        ),
    }
    paper_value_reproduced = any(
        round(value) == 86 for value in observed_percentages.values()
    )
    output = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 21:26 PDT — Recovered the official CT24 "
            "sentence-ID order, joined all three retained prediction vectors, "
            "and evaluated explicit shared-error denominators."
        ),
        "paper_value_percent": 86,
        "evaluation_rows": 341,
        "retained_sources": {
            "llm_features": "labels and probabilities in official CT24 test-gold order",
            "encoder_only": "probabilities in the same label and sentence-ID order as the LLM-feature bundle",
            "four_parallel_heads": "labels, probabilities, and sentence IDs in its retained joined evaluation order",
        },
        "alignment_audit": {
            "same_vector_length": same_length,
            "unique_sentence_ids_in_both_orders": unique_ids,
            "same_sentence_id_set": same_id_set,
            "labels_match_after_sentence_id_join": aligned_labels_match,
            "mtl_unique_sentence_ids": int(len(np.unique(mtl_ids))),
            "direct_alignment_available": direct_alignment_available,
            "llm_order_sentence_ids_sha256": hashlib.sha256(
                llm_ids_path.read_bytes()
            ).hexdigest(),
            "official_ct24_test_gold_sha256": "46cbd9ed9aca07e1023c274256c40d019146a3eb635ebb69cbc6ae5ad52d124a",
        },
        "thresholds": thresholds,
        "error_counts": error_counts,
        "shared_error_counts": {
            "any_model_error": union_count,
            "at_least_two_models": at_least_two_count,
            "all_three_models": all_three_count,
            "shared_model_error_assignments": shared_assignments,
            "total_model_error_assignments": total_error_assignments,
        },
        "pairwise_overlap": pairwise,
        "observed_percentages": observed_percentages,
        "paper_value_reproduced": paper_value_reproduced,
        "status": "MISMATCH",
        "reason": (
            "The paper does not define the denominator behind 86%. After an "
            "exact sentence-ID join, none of the explicit all-model, at-least-two, "
            "error-assignment, or pairwise overlap definitions rounds to 86%."
        ),
    }
    output_path = args.output or (
        repo_root / "results" / "checkworthiness_shared_error_audit_2026-08-21.json"
    )
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("CIKM 2026 CT24 shared-error audit")
    print(f"Rows: {len(llm_labels)}")
    print("Sentence-ID join available for all three vectors: YES")
    print(f"Error counts: {error_counts}")
    for name, value in observed_percentages.items():
        print(f"{name}: {value:.4f}%")
    print(f"Result: {output_path}")
    print("VERDICT: MISMATCH")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
