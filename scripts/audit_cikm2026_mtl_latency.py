#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 21:31 PDT.
# Reason: bind the paper's four-head GPU latency value to the retained CUDA
# benchmark trace and state the measured batch-amortized boundary exactly.
"""Audit four-head latency from the retained CUDA benchmark trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PAPER_LIMIT_MS_PER_CLAIM = 10.0
BENCHMARKS = (
    ("ClaimBuster", "[1/2] Evaluating ClaimBuster...", "[2/2] Evaluating CT23..."),
    ("CT23", "[2/2] Evaluating CT23...", "SUMMARY: Multi-Task DeBERTa"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def benchmark_block(log: str, start_marker: str, end_marker: str) -> str:
    start = log.index(start_marker)
    end = log.index(end_marker, start + len(start_marker))
    return log[start:end]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    run_root = root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    log_path = run_root / "benchmark_eval.log"
    source_path = run_root / "scripts/evaluate_mtl_benchmarks.py"
    output = args.output or root / "results/mtl_gpu_latency_audit_2026-08-21.json"

    log = log_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")
    device_match = re.search(r"^Device:\s*(\S+)", log, flags=re.MULTILINE)
    batch_match = re.search(
        r"def get_predictions\([^\n]+batch_size=(\d+)\):", source
    )
    if device_match is None or batch_match is None:
        raise ValueError("Could not recover the CUDA device or default batch size")
    device = device_match.group(1)
    batch_size = int(batch_match.group(1))
    calls_use_default = all(
        call in source
        for call in (
            "get_predictions(model, tokenizer, cb_texts, device)",
            "get_predictions(model, tokenizer, ct23_texts, device)",
        )
    )

    observed: dict[str, dict[str, float | int | str]] = {}
    for name, start_marker, end_marker in BENCHMARKS:
        block = benchmark_block(log, start_marker, end_marker)
        sample_match = re.search(r"Samples:\s*(\d+)", block)
        progress_matches = re.findall(
            r"Inference:.*?(\d+)/(\d+).*?([0-9.]+)it/s", block
        )
        if sample_match is None or not progress_matches:
            raise ValueError(f"Could not recover {name} sample count or final throughput")
        completed, batches, batches_per_second = progress_matches[-1]
        samples = int(sample_match.group(1))
        batches = int(batches)
        completed = int(completed)
        batches_per_second = float(batches_per_second)
        elapsed_seconds = batches / batches_per_second
        milliseconds_per_claim = elapsed_seconds * 1000.0 / samples
        observed[name] = {
            "samples": samples,
            "completed_batches": completed,
            "batches": batches,
            "batch_size": batch_size,
            "final_tqdm_rate_batches_per_second": batches_per_second,
            "elapsed_seconds_derived_from_final_rate": elapsed_seconds,
            "batch_amortized_ms_per_claim": milliseconds_per_claim,
            "status": "PASS" if milliseconds_per_claim < PAPER_LIMIT_MS_PER_CLAIM else "FAIL",
        }

    passed = (
        device == "cuda"
        and calls_use_default
        and all(row["status"] == "PASS" for row in observed.values())
    )
    result = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 21:31 PDT — Parsed the retained CUDA "
            "benchmark trace and its exact evaluation source to verify the "
            "batch-amortized four-head latency boundary."
        ),
        "verdict": "PASS" if passed else "FAIL",
        "paper_claim": "less than 10 ms per claim on a single GPU",
        "device": device,
        "scope": (
            "batch-amortized benchmark loop with batch size 32; includes tokenization, "
            "host-to-device copies, model forward pass, softmax, GPU-to-CPU transfer, "
            "NumPy conversion, and progress-loop overhead; excludes model loading and "
            "dataset file reading"
        ),
        "boundary": (
            "This retained trace verifies throughput per claim in batched CUDA inference; "
            "it is not a single-request latency measurement. The GPU model is not recorded."
        ),
        "source": {
            "log": log_path.relative_to(root).as_posix(),
            "log_sha256": sha256(log_path),
            "evaluation_script": source_path.relative_to(root).as_posix(),
            "evaluation_script_sha256": sha256(source_path),
            "default_batch_size_used": calls_use_default,
        },
        "benchmarks": observed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, row in observed.items():
        print(f"{name}: {row['batch_amortized_ms_per_claim']:.6f} ms/claim")
    print(f"VERDICT: {result['verdict']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
