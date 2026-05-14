from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "reproducibility" / "source_artifacts"
EXTERNAL_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class LocatedPath:
    path: Path
    source: str


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a NumPy .npy file")
        major = handle.read(1)[0]
        handle.read(1)
        header_length_format = "<H" if major == 1 else "<I"
        header_length_size = struct.calcsize(header_length_format)
        header_length = struct.unpack(header_length_format, handle.read(header_length_size))[0]
        header = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
    return tuple(int(value) for value in header["shape"])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        entries[filename.strip()] = digest
    return entries


def ok(name: str, detail: str) -> Check:
    return Check(name=name, status="OK", detail=detail)


def warn(name: str, detail: str) -> Check:
    return Check(name=name, status="WARN", detail=detail)


def info(name: str, detail: str) -> Check:
    return Check(name=name, status="INFO", detail=detail)


def fail(name: str, detail: str) -> Check:
    return Check(name=name, status="FAIL", detail=detail)


def packaged_or_external(name: str, packaged: Path, external: Path | None = None) -> tuple[LocatedPath | None, Check | None]:
    if packaged.exists():
        return LocatedPath(path=packaged, source="packaged"), None
    if external is not None and external.exists():
        return LocatedPath(path=external, source="external"), warn(name, f"using external fallback: {external}")
    if external is None:
        return None, fail(name, f"missing packaged artifact: {packaged}")
    return None, fail(name, f"missing packaged artifact and external fallback: {packaged} | {external}")


def optional_external(name: str, path: Path) -> tuple[LocatedPath | None, Check]:
    if path.exists():
        return LocatedPath(path=path, source="external"), ok(name, f"external artifact available: {path}")
    return None, warn(name, f"external-only artifact unavailable: {path}")


def manifest_entry_path(filename: str) -> Path:
    relative_path = Path(filename)
    if relative_path.is_absolute():
        return relative_path
    repo_candidate = REPO_ROOT / relative_path
    if repo_candidate.exists():
        return repo_candidate
    return ARTIFACT_ROOT / relative_path


def close_enough(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance


def check_source_manifest() -> list[Check]:
    manifest_path = ARTIFACT_ROOT / "sha256sums.txt"
    if not manifest_path.exists():
        return [fail("Source artifact checksum manifest", f"missing {manifest_path}")]

    checks: list[Check] = []
    manifest = load_sha256_manifest(manifest_path)
    mismatches: list[str] = []
    missing: list[str] = []

    for filename, expected_digest in manifest.items():
        path = manifest_entry_path(filename)
        if not path.exists():
            missing.append(filename)
            continue
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            mismatches.append(filename)

    if missing:
        checks.append(fail("Source artifact checksum manifest", f"{len(missing)} missing entries: {missing[:5]}"))
    if mismatches:
        checks.append(fail("Source artifact checksum manifest", f"{len(mismatches)} checksum mismatches: {mismatches[:5]}"))
    if not missing and not mismatches:
        checks.append(ok("Source artifact checksum manifest", f"{len(manifest)} files match {manifest_path}"))
    return checks


def check_ct24_splits() -> list[Check]:
    checks: list[Check] = []
    expected_rows = {
        "train": 22402,
        "dev": 1031,
        "test": 341,
    }
    for split, expected in expected_rows.items():
        filename = f"CT24_{split}_clean.parquet"
        packaged = ARTIFACT_ROOT / "checkworthiness" / "ct24_clean" / filename
        external = EXTERNAL_ROOT / "data" / "processed" / "CT24_clean" / filename
        located, fallback = packaged_or_external(f"CT24 {split} split", packaged, external)
        if fallback:
            checks.append(fallback)
        if located is None:
            continue
        rows = parquet_rows(located.path)
        if rows == expected:
            checks.append(ok(f"CT24 {split} split", f"{rows} rows ({located.source}: {located.path})"))
        else:
            checks.append(fail(f"CT24 {split} split", f"expected {expected} rows, found {rows} in {located.path}"))
    return checks


def check_benchmark_raw_data() -> list[Check]:
    checks: list[Check] = []
    files = [
        (
            "ClaimBuster groundtruth",
            ARTIFACT_ROOT / "checkworthiness" / "benchmarks" / "claim_buster" / "groundtruth.csv",
            EXTERNAL_ROOT / "data" / "raw" / "claim_buster" / "groundtruth.csv",
            1032,
        ),
        (
            "CT23 test input",
            ARTIFACT_ROOT / "checkworthiness" / "benchmarks" / "ct23" / "CT23_1B_checkworthy_english_test.tsv",
            EXTERNAL_ROOT / "data" / "raw" / "check_that_23" / "CT23_1B_checkworthy_english_test.tsv",
            318,
        ),
        (
            "CT23 test gold",
            ARTIFACT_ROOT / "checkworthiness" / "benchmarks" / "ct23" / "CT23_1B_checkworthy_english_test_gold.tsv",
            EXTERNAL_ROOT / "data" / "raw" / "check_that_23" / "CT23_1B_checkworthy_english_test_gold.tsv",
            318,
        ),
    ]

    for name, packaged, external, expected in files:
        located, fallback = packaged_or_external(name, packaged, external)
        if fallback:
            checks.append(fallback)
        if located is None:
            continue
        rows = csv_rows(located.path)
        if rows == expected:
            checks.append(ok(name, f"{rows} rows ({located.source}: {located.path})"))
        else:
            checks.append(fail(name, f"expected {expected} rows, found {rows} in {located.path}"))
    return checks


def check_llm_feature_inputs() -> list[Check]:
    checks: list[Check] = []
    files = [
        (
            "CT24 train LLM features",
            ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "train_llm_features.parquet",
            22402,
        ),
        (
            "CT24 dev LLM features",
            ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "dev_llm_features.parquet",
            1031,
        ),
        (
            "CT24 test LLM features",
            ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "test_llm_features.parquet",
            341,
        ),
        (
            "ClaimBuster LLM features",
            ARTIFACT_ROOT / "checkworthiness" / "benchmark_llm_features" / "CB_groundtruth_llm_features.parquet",
            1032,
        ),
        (
            "CT23 LLM features",
            ARTIFACT_ROOT / "checkworthiness" / "benchmark_llm_features" / "CT23_llm_features.parquet",
            318,
        ),
    ]

    for name, path, expected in files:
        if not path.exists():
            checks.append(fail(name, f"missing packaged LLM feature file: {path}"))
            continue
        rows = parquet_rows(path)
        if rows == expected:
            checks.append(ok(name, f"{rows} rows ({path})"))
        else:
            checks.append(fail(name, f"expected {expected} rows, found {rows} in {path}"))

    ct24_checkpoint_files = [
        ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "checkpoint_train.json",
        ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "checkpoint_dev.json",
        ARTIFACT_ROOT / "checkworthiness" / "ct24_llm_features_v4" / "checkpoint_test.json",
    ]
    benchmark_checkpoint_files = [
        ARTIFACT_ROOT / "checkworthiness" / "benchmark_llm_features" / "CB_groundtruth_checkpoint.json",
        ARTIFACT_ROOT / "checkworthiness" / "benchmark_llm_features" / "CT23_checkpoint.json",
    ]
    missing_benchmark_checkpoints = [str(path) for path in benchmark_checkpoint_files if not path.exists()]
    missing_ct24_checkpoints = [str(path) for path in ct24_checkpoint_files if not path.exists()]
    if missing_benchmark_checkpoints:
        checks.append(warn("Benchmark LLM feature checkpoints", f"missing checkpoint files: {missing_benchmark_checkpoints}"))
    else:
        checks.append(ok("Benchmark LLM feature checkpoints", f"{len(benchmark_checkpoint_files)} checkpoint files available"))
    if missing_ct24_checkpoints:
        checks.append(
            info(
                "CT24 LLM feature generation checkpoints",
                f"not packaged; feature Parquets are complete and checksummed: {missing_ct24_checkpoints}",
            )
        )
    else:
        checks.append(ok("CT24 LLM feature generation checkpoints", f"{len(ct24_checkpoint_files)} checkpoint files available"))
    return checks


def check_us_election_raw() -> list[Check]:
    checks: list[Check] = []
    path = EXTERNAL_ROOT / "data" / "raw" / "us_elections_tweets.parquet"
    located, availability = optional_external("US election raw corpus", path)
    checks.append(availability)
    if located is None:
        checks.append(
            warn(
                "US election raw corpus scope",
                "raw 1.52M-tweet corpus is not packaged; date span and language mix require external storage",
            )
        )
        return checks

    table = pq.read_table(path, columns=["created_at"])
    rows = table.num_rows
    minmax = pc.min_max(table["created_at"]).as_py()
    checks.append(ok("US election raw corpus rows", f"{rows} rows"))
    checks.append(ok("US election raw corpus date span", f"{minmax['min']} to {minmax['max']}"))

    schema_names = set(pq.read_schema(path).names)
    language_column = next((column for column in ("lang", "language") if column in schema_names), None)
    if language_column is None:
        checks.append(
            info(
                "US election raw corpus language mix",
                "no lang/language column available; English-share claim should remain omitted",
            )
        )
    else:
        table_lang = pq.read_table(path, columns=[language_column])
        counts = pc.value_counts(table_lang[language_column]).to_pylist()
        counts_by_lang = {entry["values"]: entry["counts"] for entry in counts}
        english_rows = counts_by_lang.get("en", 0)
        checks.append(
            ok(
                "US election raw corpus language mix",
                f"{english_rows}/{rows} English rows by {language_column} ({english_rows / rows:.3%})",
            )
        )
    return checks


def check_clustering_subset_language_share() -> list[Check]:
    checks: list[Check] = []
    path = REPO_ROOT / "results" / "clustering_subset_language_share_2026-05-14.json"
    if not path.exists():
        return [fail("Clustering subset language share", f"missing {path}")]

    data = load_json(path)
    grouped = data["grouped"]
    source = data["source"]
    method = data["method"]
    english_share = float(grouped["english_share"])
    english_count = int(grouped["english_count"])
    sample_size = int(method["sample_size"])
    source_rows = int(source["rows"])

    if english_count == 1732 and sample_size == 2000 and source_rows == 87911 and close_enough(english_share, 0.866):
        checks.append(
            ok(
                "Clustering subset language share",
                f"{english_count}/{sample_size} English ({english_share:.1%}) over {source_rows} source rows ({path})",
            )
        )
    else:
        checks.append(
            fail(
                "Clustering subset language share",
                (
                    "expected 1732/2000 English over 87911 source rows with share 0.866; "
                    f"found {english_count}/{sample_size}, source_rows={source_rows}, share={english_share}"
                ),
            )
        )
    return checks


def check_canonical_pipeline_run() -> list[Check]:
    checks: list[Check] = []
    summary_path = ARTIFACT_ROOT / "pipeline" / "streaming_full_2026-01-17_03-56_summary.json"
    if not summary_path.exists():
        checks.append(fail("Canonical pipeline summary", f"missing packaged summary: {summary_path}"))
    else:
        summary = load_json(summary_path)
        expected_summary = {
            "total_tweets_processed": 1522909,
            "total_claims": 535,
        }
        for key, expected in expected_summary.items():
            actual = summary.get(key)
            if actual == expected:
                checks.append(ok(f"Canonical pipeline summary {key}", f"{actual} ({summary_path})"))
            else:
                checks.append(fail(f"Canonical pipeline summary {key}", f"expected {expected}, found {actual}"))

        clusters = summary.get("clusterer", {}).get("n_clusters")
        if clusters == 100000:
            checks.append(ok("Canonical pipeline summary cluster count", f"{clusters} clusters"))
        else:
            checks.append(fail("Canonical pipeline summary cluster count", f"expected 100000, found {clusters}"))

    run_dir = EXTERNAL_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
    if not run_dir.exists():
        checks.append(
            warn(
                "Canonical pipeline full outputs",
                f"large run outputs are external-only and unavailable: {run_dir}",
            )
        )
        return checks

    expected_rows = {
        "tweets.parquet": 692289,
        "clusters.parquet": 100000,
        "claims.parquet": 535,
        "cluster_timeseries.parquet": 404296,
    }
    for filename, expected in expected_rows.items():
        path = run_dir / filename
        if not path.exists():
            checks.append(fail(f"Canonical pipeline {filename}", f"missing {path}"))
            continue
        rows = parquet_rows(path)
        if rows == expected:
            checks.append(ok(f"Canonical pipeline {filename}", f"{rows} rows"))
        else:
            checks.append(fail(f"Canonical pipeline {filename}", f"expected {expected}, found {rows}"))

    for filename in ("cluster_embeddings.npy", "cluster_id_to_idx.json", "cluster_ids_order.json"):
        path = run_dir / filename
        if path.exists():
            checks.append(ok(f"Canonical pipeline {filename}", f"{path.stat().st_size} bytes at {path}"))
        else:
            checks.append(warn(f"Canonical pipeline {filename}", f"external artifact not available: {path}"))
    return checks


def check_threshold_ablation() -> list[Check]:
    checks: list[Check] = []
    packaged = ARTIFACT_ROOT / "clustering" / "cluster_statistics.json"
    external = EXTERNAL_ROOT / "experiments" / "results" / "threshold_ablation" / "cluster_statistics.json"
    located, fallback = packaged_or_external("Threshold ablation statistics", packaged, external)
    if fallback:
        checks.append(fallback)
    if located is None:
        return checks

    stats = load_json(located.path)
    threshold = stats.get("0.65")
    if threshold is None:
        return checks + [fail("Threshold 0.65 statistics", f"missing key 0.65 in {located.path}")]

    expected = {
        "cluster_yield_pct": 86.9,
        "mean_intra_sim": 0.8685999195826681,
        "total_tweets": 5000,
    }
    for key, expected_value in expected.items():
        actual = threshold.get(key)
        if isinstance(expected_value, float):
            match = close_enough(float(actual), expected_value, tolerance=1e-6)
        else:
            match = actual == expected_value
        if match:
            checks.append(ok(f"Threshold 0.65 {key}", f"{actual} ({located.source}: {located.path})"))
        else:
            checks.append(fail(f"Threshold 0.65 {key}", f"expected {expected_value}, found {actual}"))
    return checks


def check_deberta_artifacts() -> list[Check]:
    checks: list[Check] = []
    table_path = REPO_ROOT / "results" / "table3_reproduction_2026-05-12.json"
    if not table_path.exists():
        checks.append(fail("Table 3 reproduction summary", f"missing {table_path}"))
    else:
        summary = load_json(table_path)
        rows = summary.get("rows", [])
        expected_f1 = {
            "3-seed DeBERTa ensemble": 0.834,
            "4-head MTL retrain": 0.833,
            "Fusion classifier rerun": 0.836,
        }
        for row in rows:
            method = row.get("row")
            if method not in expected_f1:
                continue
            actual = round(float(row["reproduced_f1"]), 3)
            expected = expected_f1[method]
            if actual == expected:
                checks.append(ok(f"{method} F1", f"{actual:.3f} from {table_path}"))
            else:
                checks.append(fail(f"{method} F1", f"expected rounded F1 {expected:.3f}, found {actual:.3f}"))

        single_row = next((row for row in rows if row.get("row") == "Single DeBERTa"), None)
        if single_row is None:
            checks.append(warn("Single DeBERTa F1", "summary row absent; paper claim needs separate handling"))
        else:
            checks.append(
                info(
                    "Single DeBERTa F1",
                    f"rerun found {single_row['reproduced_f1']}; use this value if the baseline row is retained",
                )
            )

        pca_row = next((row for row in rows if row.get("row") == "PCA-64 + LLM + text LogReg CT24 rerun"), None)
        if pca_row is None:
            checks.append(warn("PCA + classification head F1", "summary row absent; paper claim needs separate handling"))
        elif pca_row.get("status") == "mismatch":
            checks.append(
                warn(
                    "PCA + classification head F1",
                    f"rerun found {pca_row['reproduced_f1']}; paper reports {pca_row['paper_claim_f1']}",
                )
            )

    optional_files = [
        (
            "Single DeBERTa probability file",
            EXTERNAL_ROOT
            / "experiments"
            / "results"
            / "deberta_checkworthy"
            / "deberta-v3-large"
            / "test_probs.npy",
        ),
        (
            "3-seed ensemble probability file",
            EXTERNAL_ROOT / "lambda_backup" / "ubuntu" / "ensemble_results" / "test_temp_0.3_probs.npy",
        ),
    ]
    for name, path in optional_files:
        _, availability = optional_external(name, path)
        checks.append(availability)

    fusion_path = ARTIFACT_ROOT / "checkworthiness" / "fusion_classifier" / "fusion_test_probs.npy"
    if not fusion_path.exists():
        checks.append(warn("Fusion classifier probability file", f"missing packaged artifact: {fusion_path}"))
    else:
        fusion_shape = npy_shape(fusion_path)
        if fusion_shape == (341,):
            checks.append(ok("Fusion classifier probability file", f"341 probabilities packaged at {fusion_path}"))
        else:
            checks.append(
                fail(
                    "Fusion classifier probability file",
                    f"expected shape (341,), found {fusion_shape} in {fusion_path}",
                )
            )

    inconsistent_seed = EXTERNAL_ROOT / "lambda_backup" / "ubuntu" / "ensemble_results" / "model_seed_456" / "metrics.json"
    if inconsistent_seed.exists():
        metrics = load_json(inconsistent_seed)
        f1 = metrics.get("test_metrics", {}).get("f1")
        checks.append(
            warn(
                "Seed 456 individual model metric",
                f"external metrics file reports F1={f1}; ensemble summary remains reproducible",
            )
        )
    return checks


def check_packaged_4head_run() -> list[Check]:
    checks: list[Check] = []
    run_dir = REPO_ROOT / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553"
    manifest_path = run_dir / "results.json"
    sha_path = run_dir / "sha256sums.txt"
    if not manifest_path.exists():
        return [fail("Packaged 4-head manifest", f"missing {manifest_path}")]
    if not sha_path.exists():
        return [fail("Packaged 4-head checksum manifest", f"missing {sha_path}")]

    results = load_json(manifest_path)
    best_test = max(results.get("test_results", []), key=lambda row: row.get("f1", -1), default={})
    if close_enough(float(best_test.get("f1", -1)), 0.8333333333333333):
        checks.append(ok("Packaged 4-head test F1", "0.8333333333333333"))
    else:
        checks.append(fail("Packaged 4-head test F1", f"unexpected best test metrics: {best_test}"))

    for filename, expected_digest in load_sha256_manifest(sha_path).items():
        path = run_dir / filename
        if not path.exists():
            if filename == "best_model.pt":
                checks.append(info(f"Packaged 4-head {filename}", f"checkpoint omitted from repo: {path}"))
            else:
                checks.append(fail(f"Packaged 4-head {filename}", f"missing {path}"))
            continue
        actual_digest = sha256_file(path)
        if actual_digest == expected_digest:
            checks.append(ok(f"Packaged 4-head {filename}", "sha256 matches"))
        else:
            checks.append(fail(f"Packaged 4-head {filename}", f"sha256 mismatch: {actual_digest}"))
    return checks


def check_virality_tabular() -> list[Check]:
    checks: list[Check] = []
    features_path = ARTIFACT_ROOT / "virality" / "features_enhanced.parquet"
    labels_path = ARTIFACT_ROOT / "virality" / "psr_labels.parquet"
    manifest_path = REPO_ROOT / "results" / "virality_split_manifest_2026-05-12.json"
    tuned_path = ARTIFACT_ROOT / "virality" / "tuned_baselines.json"

    expected_parquet_rows = {
        "Virality features": (features_path, 529),
        "Virality labels": (labels_path, 529),
    }
    for name, (path, expected) in expected_parquet_rows.items():
        if not path.exists():
            checks.append(fail(name, f"missing {path}"))
            continue
        rows = parquet_rows(path)
        if rows == expected:
            checks.append(ok(name, f"{rows} rows ({path})"))
        else:
            checks.append(fail(name, f"expected {expected} rows, found {rows}"))

    if not manifest_path.exists():
        checks.append(fail("Virality split manifest", f"missing {manifest_path}"))
    else:
        manifest = load_json(manifest_path)
        expected_manifest = {
            "features_rows": 529,
            "feature_columns": 42,
            "train_rows": 423,
            "test_rows": 106,
        }
        counts = manifest.get("counts", {})
        for key, expected in expected_manifest.items():
            actual = counts.get(key)
            if actual == expected:
                checks.append(ok(f"Virality split {key}", f"{actual}"))
            else:
                checks.append(fail(f"Virality split {key}", f"expected {expected}, found {actual}"))

    if not tuned_path.exists():
        checks.append(fail("Virality tuned baselines", f"missing {tuned_path}"))
        return checks

    rows = load_json(tuned_path)
    expected_models = {
        "BayesianRidge": {
            "spearman_rho": 0.5507267551704159,
            "cv_rho": 0.4697726498589779,
            "r2": 0.29336255950069234,
            "mae": 0.22561890863935924,
            "f2_075": 0.13089005235602094,
        },
        "LightGBM": {
            "spearman_rho": 0.5488122528150742,
            "cv_rho": 0.5567888025262427,
            "r2": 0.33291185194110773,
            "mae": 0.21102133849034038,
            "f2_075": 0.3217821782178218,
        },
        "SVR (RBF)": {
            "spearman_rho": 0.5424238607451446,
            "cv_rho": 0.46533491723109693,
            "r2": 0.33632071019874665,
            "mae": 0.20712320021316238,
            "f2_075": 0.3431372549019608,
        },
    }
    rows_by_model = {row["name"]: row for row in rows}
    for model, metrics in expected_models.items():
        row = rows_by_model.get(model)
        if row is None:
            checks.append(fail(f"Virality {model}", "missing model row"))
            continue
        mismatches = [
            f"{metric}: expected {expected}, found {row.get(metric)}"
            for metric, expected in metrics.items()
            if not close_enough(float(row.get(metric)), expected, tolerance=1e-12)
        ]
        if mismatches:
            checks.append(fail(f"Virality {model}", "; ".join(mismatches)))
        else:
            checks.append(ok(f"Virality {model}", "metrics match packaged tuned_baselines.json"))
    return checks


def check_unresolved_blockers() -> list[Check]:
    checks: list[Check] = []
    claim_norm_request = REPO_ROOT / "results" / "claim_normalization_filipe_request_2026-05-14.md"
    formative_request = REPO_ROOT / "results" / "formative_evaluation_filipe_request_2026-05-14.md"

    if claim_norm_request.exists():
        checks.append(
            warn(
                "Claim normalization exact Table 1 reproducibility",
                f"pending Filipe artifact check documented in {claim_norm_request}",
            )
        )
    else:
        checks.append(fail("Claim normalization exact Table 1 reproducibility", f"missing handoff doc: {claim_norm_request}"))

    if formative_request.exists():
        checks.append(
            warn(
                "Formative evaluation quantitative stats",
                f"pending anonymized Google Forms export documented in {formative_request}",
            )
        )
    else:
        checks.append(fail("Formative evaluation quantitative stats", f"missing handoff doc: {formative_request}"))
    return checks


def print_group(title: str, checks: list[Check]) -> None:
    print(f"\n## {title}")
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")


def main() -> int:
    groups = [
        ("Source Artifact Checksums", check_source_manifest()),
        ("CT24", check_ct24_splits()),
        ("Benchmarks", check_benchmark_raw_data()),
        ("LLM Feature Inputs", check_llm_feature_inputs()),
        ("US Election Raw", check_us_election_raw()),
        ("Clustering Subset Language", check_clustering_subset_language_share()),
        ("Canonical Pipeline Run", check_canonical_pipeline_run()),
        ("Clustering", check_threshold_ablation()),
        ("DeBERTa", check_deberta_artifacts()),
        ("Packaged 4-head MTL Run", check_packaged_4head_run()),
        ("Virality", check_virality_tabular()),
        ("Unresolved Paper Blockers", check_unresolved_blockers()),
    ]

    failures = 0
    warnings = 0
    infos = 0
    for title, checks in groups:
        print_group(title, checks)
        failures += sum(1 for check in checks if check.status == "FAIL")
        warnings += sum(1 for check in checks if check.status == "WARN")
        infos += sum(1 for check in checks if check.status == "INFO")

    print(f"\nSummary: {failures} failures, {warnings} warnings, {infos} info")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
