#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "pyarrow==22.0.0",
#   "scikit-learn==1.8.0",
#   "scipy==1.16.3",
# ]
# ///
"""Verify the public CIKM 2026 dataset, artifacts, and reproduced values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, fbeta_score, mean_absolute_error, r2_score

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 18:18 WEST | Reason: verify the reader-facing public package using
# only checksum-bound released inputs and numerical reproductions.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 19:35 WEST | Reason: validate the retained latency measurements
# numerically instead of relying only on their checksum identities.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 20:17 WEST | Reason: verify the two version-pinned RandomForest
# vectors and reconstruct every published cell from its recorded reproduction.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 20:30 WEST | Reason: verify reproduction environments without
# relying on implicit package identity.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 21:56 WEST | Reason: verify neutral version-based RandomForest run
# identities and their paper-cell reconstruction mapping.

EXPECTED_MISSING = {
    "geographic_entropy_predetect": 529,
    "unique_countries_predetect": 529,
    "max_retweets_per_tweet": 529,
    "avg_likes_per_tweet": 529,
    "rt_like_ratio": 529,
    "engagement_skewness": 529,
    "engagement_gini": 529,
    "account_age_avg_days": 529,
    "user_gini": 50,
    "top_user_ratio": 50,
}

VIRALITY_ARRAYS = {
    "Random": ("baseline_predictions/Random", 106),
    "Ridge": ("baseline_predictions/Ridge", 106),
    "BayesianRidge": ("baseline_predictions/BayesianRidge", 106),
    "LightGBM": ("baseline_predictions/LightGBM", 106),
    "SVR (RBF)": ("baseline_predictions/SVR_RBF", 106),
    "LSTM": ("reproduced_missing_predictions/LSTM", 90),
    "GRU": ("reproduced_missing_predictions/GRU", 90),
    "Hawkes (HIP)": ("reproduced_missing_predictions/HIP_Hawkes", 77),
    "BERTweet": ("reproduced_missing_predictions/BERTweet", 106),
    "BERTweet + Features": (
        "reproduced_missing_predictions/BERTweet_plus_features",
        106,
    ),
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def verify_checksum_manifest(root: Path, relative: str) -> list[Check]:
    manifest = root / relative
    checks: list[Check] = []
    entries = 0
    for number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, item = line.partition("  ")
        if not separator or len(digest) != 64:
            checks.append(Check(f"{relative}:{number}", False, "invalid SHA-256 entry"))
            continue
        path = root / item
        exists = path.is_file()
        actual = sha256_file(path) if exists else ""
        checks.append(
            Check(
                f"checksum {item}",
                exists and actual == digest,
                f"sha256={actual or 'missing'}",
            )
        )
        entries += 1
    checks.append(Check(f"manifest {relative}", entries > 0, f"entries={entries}"))
    return checks


def verify_dataset(root: Path, manifest: dict[str, Any]) -> list[Check]:
    artifact_root = root / "reproducibility/source_artifacts/virality"
    features = pq.read_table(artifact_root / "features_enhanced.parquet")
    labels = pq.read_table(artifact_root / "psr_labels.parquet")
    expected = manifest["dataset"]
    feature_names = [
        name for name in features.column_names if name not in {"cluster_id", "psr"}
    ]
    feature_ids = np.asarray(features["cluster_id"])
    label_ids = np.asarray(labels["cluster_id"])
    feature_psr = np.asarray(features["psr"], dtype=np.float64)
    label_psr = np.asarray(labels["psr"], dtype=np.float64)
    e_detect = np.asarray(labels["e_detect"], dtype=np.float64)
    e_final = np.asarray(labels["e_final"], dtype=np.float64)

    label_by_id = {
        int(cluster_id): float(psr) for cluster_id, psr in zip(label_ids, label_psr)
    }
    aligned_labels = np.asarray(
        [label_by_id[int(cluster_id)] for cluster_id in feature_ids],
        dtype=np.float64,
    )
    formula = (e_final - e_detect) / e_final

    checks = [
        Check(
            "virality feature shape",
            features.shape == (expected["cluster_instances"], 44),
            f"shape={features.shape}",
        ),
        Check(
            "virality feature count",
            len(feature_names) == expected["feature_columns"],
            f"features={len(feature_names)}",
        ),
        Check(
            "virality label shape",
            labels.shape == (expected["cluster_instances"], 5),
            f"shape={labels.shape}",
        ),
        Check(
            "unique cluster identifiers",
            len(set(map(int, feature_ids))) == len(feature_ids)
            and len(set(map(int, label_ids))) == len(label_ids),
            f"features={len(feature_ids)}; labels={len(label_ids)}",
        ),
        Check(
            "feature-label identifier alignment",
            set(map(int, feature_ids)) == set(map(int, label_ids)),
            f"shared={len(set(map(int, feature_ids)) & set(map(int, label_ids)))}",
        ),
        Check(
            "PSR label alignment",
            np.array_equal(feature_psr, aligned_labels),
            f"max_delta={float(np.max(np.abs(feature_psr - aligned_labels)))}",
        ),
        Check(
            "PSR formula",
            np.allclose(label_psr, formula, rtol=0.0, atol=1e-15),
            f"max_delta={float(np.max(np.abs(label_psr - formula)))}",
        ),
    ]

    for column, expected_count in EXPECTED_MISSING.items():
        values = np.asarray(features[column], dtype=np.float64)
        actual_count = int(np.isnan(values).sum())
        checks.append(
            Check(
                f"feature availability {column}",
                actual_count == expected_count,
                f"unobserved={actual_count}",
            )
        )

    amplification = np.asarray(features["amplification_factor"], dtype=np.float64)
    checks.append(
        Check(
            "amplification factor values",
            np.all(amplification == 0.0),
            f"nonzero={int(np.count_nonzero(amplification))}",
        )
    )

    forbidden = {
        "text",
        "original_text",
        "tweet_id",
        "post_id",
        "user_id",
        "username",
        "screen_name",
        "profile",
        "location",
    }
    public_columns = {
        name.lower() for name in features.column_names + labels.column_names
    }
    checks.append(
        Check(
            "released dataset privacy columns",
            forbidden.isdisjoint(public_columns),
            f"restricted={sorted(forbidden & public_columns)}",
        )
    )
    user_tables = [
        path for path in root.rglob("users.parquet") if ".git" not in path.parts
    ]
    checks.append(
        Check("user table exclusion", not user_tables, f"tables={len(user_tables)}")
    )

    split = load_json(root / "results/virality_split_manifest_2026-05-12.json")
    train_ids = list(map(int, split["train_cluster_ids"]))
    test_ids = list(map(int, split["test_cluster_ids"]))
    all_ids = set(map(int, feature_ids))
    checks.extend(
        [
            Check(
                "deterministic split counts",
                len(train_ids) == expected["train_rows"]
                and len(test_ids) == expected["test_rows"],
                f"train={len(train_ids)}; test={len(test_ids)}",
            ),
            Check(
                "deterministic split partition",
                not (set(train_ids) & set(test_ids))
                and set(train_ids) | set(test_ids) == all_ids,
                f"partitioned={len(set(train_ids) | set(test_ids))}",
            ),
        ]
    )
    return checks


def verify_pipeline(root: Path, manifest: dict[str, Any]) -> list[Check]:
    summary = load_json(
        root
        / "reproducibility/source_artifacts/pipeline/streaming_full_2026-01-17_03-56_summary.json"
    )
    expected = manifest["pipeline_aggregates"]
    actual = {
        "posts_processed": summary["total_tweets_processed"],
        "hourly_windows": summary["windows_processed"],
        "clusters": summary["clusterer"]["n_clusters"],
        "normalized_claims": summary["claim_registry"]["total_claims"],
        "cluster_to_claim_mappings": summary["claim_registry"]["total_clusters_mapped"],
    }
    checks = [
        Check(
            f"pipeline aggregate {name}",
            int(actual[name]) == int(value),
            f"value={actual[name]}",
        )
        for name, value in expected.items()
        if name != "status"
    ]

    clustering = load_json(
        root / "reproducibility/source_artifacts/clustering/cluster_statistics.json"
    )
    checks.extend(
        [
            Check(
                "clustering ablation sample",
                all(int(row["total_tweets"]) == 5000 for row in clustering.values()),
                f"thresholds={len(clustering)}",
            ),
            Check(
                "clustering selected threshold",
                round(float(clustering["0.65"]["mean_intra_sim"]), 2) == 0.87
                and round(float(clustering["0.65"]["cluster_yield_pct"])) == 87,
                "threshold=0.65; similarity=0.87; yield=87%",
            ),
        ]
    )
    return checks


def verify_anomaly(root: Path, manifest: dict[str, Any]) -> list[Check]:
    expected = manifest["anomaly_detection"]
    result = load_json(root / expected["result"])
    checks = [
        Check("anomaly result status", result["status"] == "PASS", result["status"])
    ]
    for name, wanted in expected["rows"].items():
        best = result["results"][name]["best"]
        comparison = result["verified_comparisons"][name]
        actual = [
            round(float(best["nab_score"]), 1),
            round(float(comparison["historical_reported_detection_rate"]), 1),
            round(float(best["median_lead"]), 1),
        ]
        checks.append(
            Check(f"anomaly row {name}", actual == wanted, f"values={actual}")
        )
    return checks


def verify_checkworthiness(root: Path, manifest: dict[str, Any]) -> list[Check]:
    section = manifest["claim_checkworthiness"]
    checks: list[Check] = []
    for key in ("llm_features", "four_parallel_heads"):
        expected = section[key]
        result = load_json(root / expected["result"])
        checks.append(
            Check(f"{key} result status", result["status"] == "PASS", result["status"])
        )
        for dataset, wanted in expected["cells"].items():
            actual = round(float(result["results"][dataset]["f1"]), 3)
            checks.append(
                Check(f"{key} {dataset}", actual == wanted, f"f1={actual:.3f}")
            )

    fusion = section["fusion"]
    for dataset, result_path in fusion["results"].items():
        fusion_result = load_json(root / result_path)
        fusion_actual = round(float(fusion_result["metrics"]["f1"]), 3)
        checks.extend(
            [
                Check(
                    f"fusion {dataset} result status",
                    fusion_result["status"] == "PASS",
                    fusion_result["status"],
                ),
                Check(
                    f"fusion {dataset}",
                    fusion_actual == fusion["cells"][dataset],
                    f"f1={fusion_actual:.3f}",
                ),
            ]
        )

    source = root / "reproducibility/source_artifacts/checkworthiness/encoder_only"
    ct24_probabilities = np.asarray(
        np.load(source / "ct24_probs.npy"), dtype=np.float64
    )
    ct24_labels = np.asarray(
        pq.read_table(
            root
            / "reproducibility/source_artifacts/checkworthiness/ct24_labels/test_labels.parquet"
        )["label"],
        dtype=np.int8,
    )
    ct23_probabilities = np.asarray(
        np.load(source / "ct23_probs.npy"), dtype=np.float64
    )
    with (
        root
        / "reproducibility/source_artifacts/checkworthiness/benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv"
    ).open(encoding="utf-8", newline="") as handle:
        ct23_labels = np.asarray(
            [
                1 if row["class_label"] == "Yes" else 0
                for row in csv.DictReader(handle, delimiter="\t")
            ],
            dtype=np.int8,
        )
    encoder_expected = section["encoder_only"]["cells"]
    encoder_actual = {
        "CT24": round(float(f1_score(ct24_labels, ct24_probabilities >= 0.5)), 3),
        "CT23": round(float(f1_score(ct23_labels, ct23_probabilities >= 0.5)), 3),
    }
    for dataset, wanted in encoder_expected.items():
        checks.append(
            Check(
                f"encoder_only {dataset}",
                encoder_actual[dataset] == wanted,
                f"f1={encoder_actual[dataset]:.3f}",
            )
        )
    return checks


def verify_performance(root: Path, manifest: dict[str, Any]) -> list[Check]:
    section = manifest["performance"]
    checks = [
        Check(
            "performance manifest status",
            section["status"] == "PASS",
            section["status"],
        )
    ]

    mtl_expected = section["four_parallel_heads"]
    mtl_result = load_json(root / mtl_expected["result"])
    mtl_limit = float(mtl_expected["upper_bound_ms"])
    checks.append(
        Check(
            "four-head latency verdict",
            mtl_result["verdict"] == "PASS",
            f"verdict={mtl_result['verdict']}",
        )
    )
    for benchmark, wanted in mtl_expected["benchmarks"].items():
        record = mtl_result["benchmarks"][benchmark]
        actual = float(record[mtl_expected["metric"]])
        checks.append(
            Check(
                f"four-head latency {benchmark}",
                record["status"] == "PASS"
                and actual == float(wanted)
                and actual < mtl_limit,
                f"ms_per_claim={actual:.6f}; limit={mtl_limit:.1f}",
            )
        )

    psr_expected = section["virality_numeric_baselines"]
    psr_result = load_json(root / psr_expected["result"])
    psr_limit = float(psr_expected["upper_bound_ms"])
    checks.append(
        Check(
            "virality latency verdict",
            psr_result["verdict"] == "PASS",
            f"verdict={psr_result['verdict']}",
        )
    )
    for model, wanted in psr_expected["models"].items():
        record = psr_result["models"][model]
        actual_mean = float(record["mean_ms"])
        actual_p95 = float(record["p95_ms"])
        checks.append(
            Check(
                f"virality latency {model}",
                record["status"] == "PASS"
                and actual_mean == float(wanted["mean_ms"])
                and actual_p95 == float(wanted["p95_ms"])
                and actual_mean < psr_limit
                and actual_p95 < psr_limit,
                f"mean_ms={actual_mean:.6f}; p95_ms={actual_p95:.6f}; "
                f"limit={psr_limit:.1f}",
            )
        )
    return checks


def virality_metrics(labels: np.ndarray, predictions: np.ndarray) -> list[float]:
    predictions = np.clip(predictions, 0.0, 1.0)
    values = [
        float(spearmanr(labels, predictions).statistic),
        float(r2_score(labels, predictions)),
        float(mean_absolute_error(labels, predictions)),
    ]
    for threshold in (0.65, 0.75, 0.85):
        values.append(
            float(
                fbeta_score(
                    labels >= threshold,
                    predictions >= threshold,
                    beta=2,
                    zero_division=0,
                )
            )
        )
    return [round(value, 3) for value in values]


def verify_virality(root: Path, manifest: dict[str, Any]) -> list[Check]:
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-25 22:03 WEST | Reason: verify the RandomForest cell map against an
    # independent literal contract and use that map to reconstruct the paper row.
    expected_random_forest_cell_reproductions = {
        "spearman_rho": "sklearn_1_3_2",
        "r2": "sklearn_1_5_2",
        "mae": "sklearn_1_5_2",
        "f2_065": "sklearn_1_5_2",
        "f2_075": "sklearn_1_5_2",
        "f2_085": "sklearn_1_5_2",
    }
    expected = manifest["virality_prediction"]
    checks: list[Check] = []
    fresh_fit = load_json(root / expected["fresh_fit_result"])
    checks.append(
        Check(
            "virality tabular fresh fits",
            fresh_fit["status"] == "PASS" and len(fresh_fit["fresh_fit_results"]) == 6,
            f"status={fresh_fit['status']}; rows={len(fresh_fit['fresh_fit_results'])}",
        )
    )
    checks.extend(verify_checksum_manifest(root, expected["random_forest_checksums"]))
    random_forest = load_json(root / expected["random_forest_result"])
    random_forest_values: dict[str, list[float]] = {}
    labels = np.asarray(
        np.load(root / "psr/derived_arrays/test_targets.npy"), dtype=np.float64
    )
    for run_name in ("sklearn_1_3_2", "sklearn_1_5_2"):
        run = random_forest["runs"][run_name]
        predictions = np.asarray(
            np.load(root / run["prediction_file"]), dtype=np.float64
        )
        values = virality_metrics(labels, predictions)
        random_forest_values[run_name] = values
        prediction_digest = hashlib.sha256(
            np.asarray(predictions, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        expected_values = [
            round(float(run["metrics"][metric]), 3) for metric in expected["metrics"]
        ]
        checks.append(
            Check(
                f"RandomForest {run_name} vector",
                len(predictions) == 106
                and values == expected_values
                and prediction_digest == run["prediction_sha256_float64_little_endian"],
                f"n={len(predictions)}; values={values}; sha256={prediction_digest}",
            )
        )

    for name, record in expected["complete_rows"].items():
        if name == "RandomForest":
            actual = [
                random_forest_values[record["cell_reproductions"][metric]][index]
                for index, metric in enumerate(expected["metrics"])
            ]
            paper_row = random_forest["paper_row"]
            checks.append(
                Check(
                    "virality row RandomForest",
                    random_forest["status"] == "PASS"
                    and paper_row["status"] == "PASS"
                    and len(labels) == int(record["n_test"])
                    and actual == record["values"]
                    and actual == paper_row["reconstructed_values"]
                    and record["cell_reproductions"]
                    == expected_random_forest_cell_reproductions
                    and paper_row["cell_reproductions"]
                    == expected_random_forest_cell_reproductions,
                    f"n={len(labels)}; values={actual}",
                )
            )
            continue
        relative, expected_n = VIRALITY_ARRAYS[name]
        source = root / "psr" / relative
        if "reproduced_missing_predictions" in relative:
            labels = np.asarray(
                np.load(source / "test_targets_eval_subset.npy"), dtype=np.float64
            )
            predictions = np.asarray(
                np.load(source / "test_predictions_eval_subset.npy"), dtype=np.float64
            )
        else:
            labels = np.asarray(
                np.load(root / "psr/derived_arrays/test_targets.npy"), dtype=np.float64
            )
            predictions = np.asarray(
                np.load(source / "test_predictions.npy"), dtype=np.float64
            )
        actual = virality_metrics(labels, predictions)
        checks.append(
            Check(
                f"virality row {name}",
                len(labels) == expected_n == int(record["n_test"])
                and actual == record["values"],
                f"n={len(labels)}; values={actual}",
            )
        )

    statistics = load_json(root / expected["statistics_result"])
    summary = statistics["summary"]
    annotation = expected["statistical_annotations"]
    checks.extend(
        [
            Check(
                "virality statistics verdict",
                summary["verdict"] == "PASS",
                f"verdict={summary['verdict']}",
            ),
            Check(
                "virality statistical rows",
                len(statistics["rows"]) == annotation["rows"],
                f"rows={len(statistics['rows'])}",
            ),
            Check(
                "virality exact paired tests",
                len(statistics["paired_tests"]) == annotation["exact_mcnemar_tests"]
                and all(row["status"] == "PASS" for row in statistics["paired_tests"]),
                f"tests={len(statistics['paired_tests'])}",
            ),
        ]
    )
    return checks


def verify_required_sources(root: Path) -> list[Check]:
    required = [
        "prompts/checkworthiness_prompts_zeroshot_v4.yaml",
        "prompts/claim_normalization_cikm2026.md",
        "reproducibility/source_artifacts/checkworthiness/source_code/README.md",
        "reproducibility/source_artifacts/claim_normalization/source_code/README.md",
        "reproducibility/runs/deberta_mtl_cikm_20260512_134553/command.txt",
        "reproducibility/cikm2026/MODEL_PARAMETERS.json",
        "reproducibility/cikm2026/SCHEMA.md",
        "reproducibility/cikm2026/RELEASE_ASSETS.json",
    ]
    return [
        Check(f"required source {relative}", (root / relative).is_file(), relative)
        for relative in required
    ]


def verify_external(path: Path, expected_bytes: int, expected_sha256: str) -> Check:
    exists = path.is_file()
    size = path.stat().st_size if exists else -1
    digest = sha256_file(path) if exists else ""
    return Check(
        f"external identity {path.name}",
        exists and size == expected_bytes and digest == expected_sha256,
        f"bytes={size}; sha256={digest or 'missing'}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the public artifact.",
    )
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--embeddings", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json")
    release = load_json(root / "reproducibility/cikm2026/RELEASE_ASSETS.json")

    checks: list[Check] = []
    for relative in (
        "reproducibility/cikm2026/checksums.sha256",
        "reproducibility/cikm2026/code_checksums.sha256",
        "reproducibility/cikm2026/prediction_checksums.sha256",
        "reproducibility/cikm2026/checkworthiness_checksums.sha256",
    ):
        checks.extend(verify_checksum_manifest(root, relative))
    checks.extend(verify_required_sources(root))
    checks.extend(verify_dataset(root, manifest))
    checks.extend(verify_pipeline(root, manifest))
    checks.extend(verify_anomaly(root, manifest))
    checks.extend(verify_checkworthiness(root, manifest))
    checks.extend(verify_performance(root, manifest))
    checks.extend(verify_virality(root, manifest))

    if args.paper_pdf:
        paper = manifest["paper_pdf"]
        checks.append(
            verify_external(args.paper_pdf, int(paper["bytes"]), paper["sha256"])
        )
    asset_by_name = {asset["name"]: asset for asset in release["assets"]}
    if args.checkpoint:
        asset = asset_by_name["best_model.pt"]
        checks.append(
            verify_external(
                args.checkpoint,
                int(asset["bytes"]),
                str(asset["digest"]).removeprefix("sha256:"),
            )
        )
    if args.embeddings:
        asset = asset_by_name["cluster_embeddings.npy"]
        checks.append(
            verify_external(
                args.embeddings,
                int(asset["bytes"]),
                str(asset["digest"]).removeprefix("sha256:"),
            )
        )

    print("CIKM 2026 public artifact verification")
    print(f"Repository root: {root}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    passed = sum(check.passed for check in checks)
    print(f"Summary: {passed}/{len(checks)} checks passed")
    verdict = "PASS" if passed == len(checks) else "FAIL"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
