from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sha256_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        checksums[filename] = digest
    return checksums


def ok(name: str, detail: str) -> Check:
    return Check(name=name, status="OK", detail=detail)


def warn(name: str, detail: str) -> Check:
    return Check(name=name, status="WARN", detail=detail)


def fail(name: str, detail: str) -> Check:
    return Check(name=name, status="FAIL", detail=detail)


def require_path(path: Path, name: str) -> Check | None:
    if path.exists():
        return None
    return fail(name, f"missing: {path}")


def check_ct24_splits() -> list[Check]:
    base = SOURCE_ROOT / "data/processed/CT24_clean"
    expected = {
        "CT24_train_clean.parquet": 22402,
        "CT24_dev_clean.parquet": 1031,
        "CT24_test_clean.parquet": 341,
    }
    checks: list[Check] = []
    for filename, expected_rows in expected.items():
        path = base / filename
        missing = require_path(path, f"CT24 split {filename}")
        if missing:
            checks.append(missing)
            continue
        rows = parquet_rows(path)
        if rows == expected_rows:
            checks.append(ok(f"CT24 split {filename}", f"{rows} rows"))
        else:
            checks.append(fail(f"CT24 split {filename}", f"expected {expected_rows}, found {rows}"))
    return checks


def check_benchmark_raw_data() -> list[Check]:
    checks: list[Check] = []
    claimbuster = SOURCE_ROOT / "data/raw/claim_buster/groundtruth.csv"
    ct23 = SOURCE_ROOT / "data/raw/check_that_23/CT23_1B_checkworthy_english_test.tsv"
    if claimbuster.exists():
        checks.append(ok("ClaimBuster groundtruth", f"{csv_rows(claimbuster)} data rows"))
    else:
        checks.append(fail("ClaimBuster groundtruth", f"missing: {claimbuster}"))
    if ct23.exists():
        checks.append(ok("CT23 test file", f"{csv_rows(ct23)} data rows"))
    else:
        checks.append(fail("CT23 test file", f"missing: {ct23}"))
    return checks


def check_us_election_raw() -> list[Check]:
    path = SOURCE_ROOT / "data/raw/us_elections_tweets.parquet"
    missing = require_path(path, "US Election raw corpus")
    if missing:
        return [missing]
    table = pq.read_table(path, columns=["created_at"])
    min_max = pc.min_max(table.column("created_at")).as_py()
    schema_names = set(pq.read_schema(path).names)
    checks = [
        ok(
            "US Election raw corpus",
            f"{table.num_rows} rows, created_at {min_max['min']} to {min_max['max']}",
        )
    ]
    if "language" in schema_names or "lang" in schema_names:
        checks.append(ok("US Election language column", "language/lang column exists"))
    else:
        checks.append(warn("US Election 87% English claim", "no language/lang column in raw parquet schema"))
    return checks


def check_canonical_pipeline_run() -> list[Check]:
    base = SOURCE_ROOT / "data/pipeline_output/streaming_full/2026-01-17_03-56"
    expected_rows = {
        "tweets.parquet": 692289,
        "clusters.parquet": 100000,
        "claims.parquet": 535,
        "cluster_timeseries.parquet": 404296,
    }
    expected_files = [
        "cluster_embeddings.npy",
        "cluster_id_to_idx.json",
        "cluster_ids_order.json",
        "summary.json",
    ]
    checks: list[Check] = []
    for filename, expected in expected_rows.items():
        path = base / filename
        missing = require_path(path, f"Pipeline run {filename}")
        if missing:
            checks.append(missing)
            continue
        rows = parquet_rows(path)
        if rows == expected:
            checks.append(ok(f"Pipeline run {filename}", f"{rows} rows"))
        else:
            checks.append(fail(f"Pipeline run {filename}", f"expected {expected}, found {rows}"))
    for filename in expected_files:
        path = base / filename
        if path.exists():
            checks.append(ok(f"Pipeline run {filename}", "present"))
        else:
            checks.append(fail(f"Pipeline run {filename}", f"missing: {path}"))
    return checks


def check_threshold_ablation() -> list[Check]:
    path = SOURCE_ROOT / "experiments/results/threshold_ablation/cluster_statistics.json"
    missing = require_path(path, "Threshold ablation JSON")
    if missing:
        return [missing]
    data = load_json(path)
    entry = data.get("0.65")
    if not isinstance(entry, dict):
        return [fail("Threshold tau=0.65", "missing key 0.65")]
    yield_pct = entry.get("cluster_yield_pct")
    mean_sim = entry.get("mean_intra_sim")
    total_tweets = entry.get("total_tweets")
    checks = [ok("Threshold ablation JSON", f"{len(data)} thresholds")]
    if yield_pct == 86.9 and abs(float(mean_sim) - 0.8685999195826681) < 1e-12 and total_tweets == 5000:
        checks.append(
            ok(
                "Threshold tau=0.65",
                "cluster_yield_pct=86.9, mean_intra_sim=0.8685999196, total_tweets=5000",
            )
        )
    else:
        checks.append(warn("Threshold tau=0.65", f"yield={yield_pct}, mean_sim={mean_sim}, total_tweets={total_tweets}"))
    return checks


def check_deberta_artifacts() -> list[Check]:
    checks: list[Check] = []
    single = SOURCE_ROOT / "experiments/results/deberta_checkworthy/deberta-v3-large/results.json"
    if single.exists():
        data = load_json(single)
        best = max(data.get("test_results", []), key=lambda row: row.get("f1", -1))
        checks.append(ok("Single DeBERTa CT24 result", f"best test F1={best.get('f1')} at threshold={best.get('threshold')}"))
    else:
        checks.append(fail("Single DeBERTa CT24 result", f"missing: {single}"))

    ensemble = SOURCE_ROOT / "lambda_backup/ubuntu/ensemble_results"
    required = [
        "seed_0/deberta-v3-large/test_probs.npy",
        "seed_123/deberta-v3-large/test_probs.npy",
        "seed_456/deberta-v3-large/test_probs.npy",
        "test_temp_0.3_probs.npy",
    ]
    missing_files = [str(ensemble / item) for item in required if not (ensemble / item).exists()]
    if missing_files:
        checks.append(fail("3-seed ensemble probability files", "missing: " + "; ".join(missing_files)))
    else:
        checks.append(ok("3-seed ensemble probability files", "seed_0, seed_123, seed_456, and temp_0.3 files present"))

    seed_456 = ensemble / "seed_456/deberta-v3-large/results.json"
    if seed_456.exists():
        best = load_json(seed_456).get("test", {}).get("best", {})
        checks.append(warn("Paper 4-head F1=0.814 conflict", f"same value appears as single-head seed_456 F1={best.get('f1')}"))
    else:
        checks.append(fail("Seed 456 result JSON", f"missing: {seed_456}"))
    return checks


def check_packaged_4head_run() -> list[Check]:
    run_dir = REPO_ROOT / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    checks: list[Check] = []
    missing = require_path(run_dir, "Packaged 4-head run")
    if missing:
        return [missing]

    result_path = run_dir / "results.json"
    benchmark_path = run_dir / "benchmark_summary.json"
    label_order_path = run_dir / "saved_label_order_summary.json"
    sha_path = run_dir / "sha256sums.txt"
    label_sha_path = run_dir / "label_order_sha256sums.txt"
    script_sha_path = run_dir / "script_sha256sums.txt"
    for path, name in (
        (result_path, "Packaged 4-head results.json"),
        (benchmark_path, "Packaged benchmark summary"),
        (label_order_path, "Packaged label-order summary"),
        (sha_path, "Packaged run checksum manifest"),
        (label_sha_path, "Packaged label-order checksum manifest"),
        (script_sha_path, "Packaged script checksum manifest"),
    ):
        missing_file = require_path(path, name)
        if missing_file:
            checks.append(missing_file)
    if any(check.status == "FAIL" for check in checks):
        return checks

    results = load_json(result_path)
    best_test = results.get("best_test", {})
    if (
        abs(float(best_test.get("f1", -1)) - 0.8333333333333333) < 1e-12
        and abs(float(best_test.get("threshold", -1)) - 0.5) < 1e-12
    ):
        checks.append(ok("4-head CT24 reproduced result", "test F1=0.8333333333333333 at threshold=0.50"))
    else:
        checks.append(fail("4-head CT24 reproduced result", f"unexpected best_test={best_test}"))

    benchmark = load_json(benchmark_path)
    claimbuster_f1 = benchmark.get("ClaimBuster", {}).get("f1")
    ct23_f1 = benchmark.get("CT23", {}).get("f1")
    if claimbuster_f1 == 0.973 and ct23_f1 == 0.9327:
        checks.append(ok("4-head cross-dataset reproduced result", "ClaimBuster F1=0.9730, CT23 F1=0.9327"))
    else:
        checks.append(warn("4-head cross-dataset reproduced result", f"ClaimBuster={claimbuster_f1}, CT23={ct23_f1}"))

    label_summary = load_json(label_order_path)
    label_best = label_summary.get("test", {}).get("best", {})
    if (
        abs(float(label_best.get("f1", -1)) - 0.8333333333333333) < 1e-12
        and abs(float(label_best.get("threshold", -1)) - 0.5) < 1e-12
    ):
        checks.append(ok("4-head saved label order", "test_labels.npy reproduces F1=0.8333333333333333"))
    else:
        checks.append(fail("4-head saved label order", f"unexpected summary={label_best}"))

    expected_files = {
        "results.json",
        "dev_probs.npy",
        "test_probs.npy",
        "training.log",
        "benchmark_eval.log",
        "benchmark_summary.json",
        "command.txt",
        "environment.txt",
    }
    run_manifest = load_sha256_manifest(sha_path)
    for filename in sorted(expected_files):
        path = run_dir / filename
        if not path.exists():
            checks.append(fail(f"4-head artifact {filename}", f"missing: {path}"))
            continue
        expected = run_manifest.get(filename)
        actual = sha256_file(path)
        if expected == actual:
            checks.append(ok(f"4-head artifact {filename}", "checksum matches"))
        else:
            checks.append(fail(f"4-head artifact {filename}", f"checksum mismatch: {actual} != {expected}"))

    model_checksum = run_manifest.get("best_model.pt")
    if model_checksum and not (run_dir / "best_model.pt").exists():
        checks.append(ok("4-head model checkpoint storage", "best_model.pt checksum recorded; file intentionally not committed"))
    else:
        checks.append(warn("4-head model checkpoint storage", "expected checksum-only storage for best_model.pt"))

    label_manifest = load_sha256_manifest(label_sha_path)
    for filename in (
        "dev_labels.npy",
        "dev_sentence_ids.npy",
        "test_labels.npy",
        "test_sentence_ids.npy",
        "saved_label_order_summary.json",
    ):
        path = run_dir / filename
        if not path.exists():
            checks.append(fail(f"4-head label-order artifact {filename}", f"missing: {path}"))
            continue
        expected = label_manifest.get(filename)
        actual = sha256_file(path)
        if expected == actual:
            checks.append(ok(f"4-head label-order artifact {filename}", "checksum matches"))
        else:
            checks.append(fail(f"4-head label-order artifact {filename}", f"checksum mismatch: {actual} != {expected}"))

    script_manifest = load_sha256_manifest(script_sha_path)
    for filename in ("scripts/finetune_deberta_mtl.py", "scripts/evaluate_mtl_benchmarks.py", "run_summary.md"):
        path = run_dir / filename
        if not path.exists():
            checks.append(fail(f"4-head packaged file {filename}", f"missing: {path}"))
            continue
        expected = script_manifest.get(filename)
        actual = sha256_file(path)
        if expected == actual:
            checks.append(ok(f"4-head packaged file {filename}", "checksum matches"))
        else:
            checks.append(fail(f"4-head packaged file {filename}", f"checksum mismatch: {actual} != {expected}"))
    return checks


def check_virality_tabular() -> list[Check]:
    path = SOURCE_ROOT / "experiments/results/virality_tuned/tuned_baselines.json"
    missing = require_path(path, "Virality tuned baselines")
    if missing:
        return [missing]
    rows = load_json(path)
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    wanted = ["BayesianRidge", "LightGBM", "SVR (RBF)"]
    checks = [ok("Virality tuned baselines", f"{len(rows)} model rows")]
    for name in wanted:
        row = by_name.get(name)
        if row:
            checks.append(ok(name, f"rho={row.get('spearman_rho')}, r2={row.get('r2')}, mae={row.get('mae')}"))
        else:
            checks.append(fail(name, "missing from tuned baselines"))
    return checks


def main() -> int:
    groups = [
        ("CT24", check_ct24_splits()),
        ("Benchmarks", check_benchmark_raw_data()),
        ("US Election Raw", check_us_election_raw()),
        ("Canonical Pipeline Run", check_canonical_pipeline_run()),
        ("Clustering", check_threshold_ablation()),
        ("DeBERTa", check_deberta_artifacts()),
        ("Packaged 4-head MTL Run", check_packaged_4head_run()),
        ("Virality", check_virality_tabular()),
    ]
    failures = 0
    warnings = 0
    for group, checks in groups:
        print(f"\n[{group}]")
        for check in checks:
            print(f"{check.status:>4}  {check.name}: {check.detail}")
            failures += check.status == "FAIL"
            warnings += check.status == "WARN"
    print(f"\nSummary: {failures} failures, {warnings} warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
