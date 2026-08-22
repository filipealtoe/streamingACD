#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "pyarrow==22.0.0",
#   "scikit-learn==1.8.0",
# ]
# ///
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-21 21:11 PDT | Reason: directly benchmark the paper's PSR
# sub-millisecond prediction claim on the released 42-feature matrix.
"""Benchmark single-item PSR prediction over the released 42 features.

The timed boundary includes StandardScaler.transform and model.predict for one
already-engineered feature row. It excludes feature generation, model fitting,
file I/O, and batching. BayesianRidge and SVR are benchmarked because the paper
identifies them as the strongest ranking and fit/error baselines, respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the public CIKM artifacts.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1_000,
        help="Untimed single-item calls per model.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20_000,
        help="Timed single-item calls per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path (default: results/psr_latency_benchmark_2026-08-21.json).",
    )
    return parser.parse_args()


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
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, labels


def prediction_sha256(predictions: np.ndarray) -> str:
    canonical = np.asarray(predictions, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def percentile_ms(durations_ns: list[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(durations_ns, dtype=float), percentile) / 1e6)


def benchmark_model(
    model: Any,
    scaler: StandardScaler,
    rows: list[np.ndarray],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for index in range(warmup):
        scaled = scaler.transform(rows[index % len(rows)])
        model.predict(scaled)

    durations_ns: list[int] = []
    for index in range(iterations):
        row = rows[index % len(rows)]
        started = time.perf_counter_ns()
        scaled = scaler.transform(row)
        model.predict(scaled)
        durations_ns.append(time.perf_counter_ns() - started)

    mean_ms = statistics.fmean(durations_ns) / 1e6
    p95_ms = percentile_ms(durations_ns, 95)
    return {
        "calls": iterations,
        "mean_ms": mean_ms,
        "min_ms": min(durations_ns) / 1e6,
        "p50_ms": percentile_ms(durations_ns, 50),
        "p95_ms": p95_ms,
        "p99_ms": percentile_ms(durations_ns, 99),
        "max_ms": max(durations_ns) / 1e6,
        "sub_millisecond_mean": mean_ms < 1.0,
        "sub_millisecond_p95": p95_ms < 1.0,
        "status": "PASS" if mean_ms < 1.0 and p95_ms < 1.0 else "FAIL",
    }


def hardware_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("--warmup and --iterations must be positive")

    repo_root = args.root.resolve()
    matrix_path = (
        repo_root
        / "reproducibility/source_artifacts/virality/features_enhanced.parquet"
    )
    matrix, labels = load_matrix(matrix_path)
    train_x, test_x, train_y, _ = train_test_split(
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
    rows = [row.reshape(1, -1) for row in test_x]

    models = {
        "BayesianRidge": BayesianRidge(alpha_1=1e-6, lambda_1=1e-6),
        "SVR (RBF)": SVR(C=0.1, epsilon=0.05, gamma="auto"),
    }
    expected_hashes = json.loads(
        (
            repo_root / "results/virality_tabular_reproduction_2026-08-21.json"
        ).read_text(encoding="utf-8")
    )["prediction_sha256_float64_little_endian"]

    results: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(scaled_train_x, train_y)
        prediction_hash = prediction_sha256(model.predict(scaled_test_x))
        if prediction_hash != expected_hashes[name]:
            raise RuntimeError(
                f"{name} prediction hash does not match the table reproducer"
            )
        measurement = benchmark_model(
            model,
            scaler,
            rows,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        measurement["prediction_sha256_float64_little_endian"] = prediction_hash
        results[name] = measurement

    verdict = "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL"
    output = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 21:11 PDT — Benchmarked single-item "
            "PSR prediction for the two strongest reported numeric baselines "
            "on the released 42-feature matrix."
        ),
        "claim": "PSR prediction over 42 features operates at sub-millisecond cost per item",
        "measurement_boundary": {
            "included": "StandardScaler.transform plus model.predict for one feature row",
            "excluded": ["feature generation", "model fitting", "file I/O", "batching"],
            "rows": int(len(matrix)),
            "features": int(matrix.shape[1]),
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
            "split_random_state": 42,
            "warmup_calls_per_model": args.warmup,
            "timed_calls_per_model": args.iterations,
            "acceptance": "both mean and p95 single-item latency are below 1.0 ms",
        },
        "environment": {
            "hardware": hardware_name(),
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "clock": "time.perf_counter_ns",
        },
        "models": results,
        "verdict": verdict,
    }

    output_path = args.output or (
        repo_root / "results/psr_latency_benchmark_2026-08-21.json"
    )
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("CIKM 2026 PSR single-item latency benchmark")
    print(f"Boundary: {output['measurement_boundary']['included']}")
    print(f"Hardware: {output['environment']['hardware']}")
    for name, result in results.items():
        print(
            f"[{result['status']}] {name}: mean={result['mean_ms']:.4f} ms; "
            f"p95={result['p95_ms']:.4f} ms; p99={result['p99_ms']:.4f} ms"
        )
    print(f"Result: {output_path}")
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
