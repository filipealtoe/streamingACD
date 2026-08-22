#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 20:13 PDT.
# Reason: make every paper-facing claim-normalization cell explicit and prevent
# rounded manuscript values from being confused with recovered run evidence.
# Artifact correction — Sérgio Pinto, 2026-08-21 20:40 PDT.
# Reason: distinguish the retained numerical recovery run from the different
# Model A hyperparameter tuple stated in the accepted paper.
"""Audit and reproduce the supported CIKM claim-normalization table cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_ROWS = {
    "Baseline 1": {"external": 0.4569},
    "Baseline 2": {"external": 0.4521},
    "LLM Only": {"A": 0.3423, "B": 0.3875, "C": 0.3997},
    "Approach 1": {"A": 0.4679, "B": 0.4745, "C": 0.4838},
    "Approach 2": {"A": 0.5569, "B": 0.5463, "C": 0.5691},
}

PAPER_MODEL_A_PROTOCOL = {
    "topic_clusters": 10,
    "retrieval_threshold": 0.85,
    "claim_verify_threshold": 0.65,
    "positive_examples": 4,
    "negative_examples": 2,
}

RECOVERY_MODEL_A_PROTOCOL = {
    "topic_clusters": 10,
    "retrieval_threshold": 0.85,
    "claim_verify_threshold": 0.5,
    "positive_examples": 5,
    "negative_examples": 0,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "results/claim_normalization_table_reproduction_2026-08-21.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    retained_root = root / "results/ct25_claim_normalization_lambda_2026-05-15"
    rows = [
        json.loads(line)
        for line in (retained_root / "per_sample_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    retained_summary = json.loads(
        (retained_root / "summary.json").read_text(encoding="utf-8")
    )
    meteor = sum(float(row["meteor"]) for row in rows) / len(rows)
    indices = [int(row["idx"]) for row in rows]
    numeric_match = (
        len(rows) == 300
        and sorted(indices) == list(range(300))
        and abs(meteor - float(retained_summary["avg_meteor"])) <= 1e-15
        and round(meteor, 4) == PAPER_ROWS["Approach 2"]["A"]
    )
    protocol_match = RECOVERY_MODEL_A_PROTOCOL == PAPER_MODEL_A_PROTOCOL

    results: dict[str, dict[str, object]] = {
        "Baseline 1": {
            "paper_meteor": PAPER_ROWS["Baseline 1"]["external"],
            "status": "EXTERNAL_CITATION",
        },
        "Baseline 2": {
            "paper_meteor": PAPER_ROWS["Baseline 2"]["external"],
            "status": "EXTERNAL_CITATION",
        },
    }
    unresolved: list[str] = []
    for approach in ("LLM Only", "Approach 1", "Approach 2"):
        for model in ("A", "B", "C"):
            cell = f"{approach} / Model {model}"
            if approach == "Approach 2" and model == "A":
                results[cell] = {
                    "full_precision_meteor": meteor,
                    "n": len(rows),
                    "paper_meteor": PAPER_ROWS[approach][model],
                    "paper_protocol": PAPER_MODEL_A_PROTOCOL,
                    "recovery_protocol": RECOVERY_MODEL_A_PROTOCOL,
                    "retained_input": (
                        "results/ct25_claim_normalization_lambda_2026-05-15/"
                        "per_sample_metrics.jsonl"
                    ),
                    "rounded_numeric_match": numeric_match,
                    "protocol_match": protocol_match,
                    "status": (
                        "REPRODUCED"
                        if numeric_match and protocol_match
                        else "NUMERIC_MATCH_DIFFERENT_PROTOCOL"
                        if numeric_match
                        else "MISMATCH"
                    ),
                }
                unresolved.append(cell)
            else:
                results[cell] = {
                    "paper_meteor": PAPER_ROWS[approach][model],
                    "reason": (
                        "The exact 300 per-example predictions or METEOR values "
                        "from the paper run were not found in the recovered workspace."
                    ),
                    "status": "UNRESOLVED",
                }
                unresolved.append(cell)

    directly_reproduced = sum(
        result.get("status") == "REPRODUCED" for result in results.values()
    )
    numeric_recovery_matches = sum(
        result.get("status") == "NUMERIC_MATCH_DIFFERENT_PROTOCOL"
        for result in results.values()
    )
    status = "COMPLETE" if directly_reproduced == 9 else "UNRESOLVED"
    output = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 20:40 PDT — Recomputed the retained "
            "Approach 2 / Model A value, compared its protocol with the accepted "
            "paper, and kept the numerical match separate from direct reproduction."
        ),
        "internal_cells": {
            "directly_reproduced": directly_reproduced,
            "numeric_recovery_matches": numeric_recovery_matches,
            "total": 9,
            "unresolved": len(unresolved),
        },
        "paper_rows": PAPER_ROWS,
        "results": results,
        "status": status,
        "unresolved_cells": unresolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "Approach 2 / Model A: "
        f"METEOR={meteor:.15f}; paper=0.5569; "
        f"{'NUMERIC MATCH' if numeric_match else 'MISMATCH'}; "
        f"protocol={'MATCH' if protocol_match else 'DIFFERENT'}"
    )
    print(f"Unresolved internal cells: {len(unresolved)}/9")
    print(f"VERDICT: {status}")
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
