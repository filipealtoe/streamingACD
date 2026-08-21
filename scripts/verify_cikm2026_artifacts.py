#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.2",
#   "pyarrow==25.0.1",
# ]
# ///
"""Verify the public CIKM 2026 dataset and reproducibility artifacts.

The default verification uses only files committed to the public repository. An
optional checkpoint path also verifies the external release asset by size and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CHECKPOINT_SHA256 = "5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081"
CHECKPOINT_SIZE_BYTES = 1_739_380_133
CHECKPOINT_RELEASE_URL = (
    "https://github.com/filipealtoe/streamingACD/releases/download/"
    "cikm-2026-artifact-v1/best_model.pt"
)


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_nan(column: pa.ChunkedArray) -> int:
    if not pa.types.is_floating(column.type):
        return 0
    return int(pc.sum(pc.cast(pc.is_nan(column), pa.int64())).as_py())


def verify_repository_data_boundary(repo_root: Path) -> Check:
    canonical_root = (
        repo_root
        / "psr"
        / "explainableACD"
        / "data"
        / "pipeline_output"
        / "streaming_full"
        / "2026-01-17_03-56"
    )
    excluded_user_table = canonical_root / "users.parquet"
    retained_scientific_tables = [
        canonical_root / "claims.parquet",
        canonical_root / "clusters.parquet",
        canonical_root / "cluster_timeseries.parquet",
        canonical_root / "window_results.parquet",
    ]
    restricted_user_columns = {
        "email",
        "handle",
        "phone",
        "profile_description",
        "screen_name",
        "user_id",
        "username",
    }
    identifier_schema_hits: dict[str, list[str]] = {}
    for parquet_path in sorted(canonical_root.rglob("*.parquet")):
        present = sorted(
            restricted_user_columns.intersection(pq.read_schema(parquet_path).names)
        )
        if present:
            identifier_schema_hits[parquet_path.relative_to(repo_root).as_posix()] = (
                present
            )
    missing_retained = [
        path.relative_to(repo_root).as_posix()
        for path in retained_scientific_tables
        if not path.is_file()
    ]
    return Check(
        "Repository data boundary",
        not excluded_user_table.exists()
        and not identifier_schema_hits
        and not missing_retained,
        "user-level lookup table absent="
        f"{not excluded_user_table.exists()}; direct identifier schema hits="
        f"{identifier_schema_hits or 'none'}; retained scientific tables missing="
        f"{missing_retained or 'none'}",
    )


def verify_checksums(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = repo_root / relative.strip()
        entries += 1
        if not path.is_file():
            missing.append(relative.strip())
        elif sha256_file(path) != expected:
            mismatched.append(relative.strip())
    passed = not missing and not mismatched and entries == 6
    return Check(
        "Release checksums",
        passed,
        f"{entries} entries; missing={missing or 'none'}; mismatched={mismatched or 'none'}",
    )


def verify_code_checksums(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "code_checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = repo_root / relative.strip()
        entries += 1
        if not path.is_file():
            missing.append(relative.strip())
        elif sha256_file(path) != expected:
            mismatched.append(relative.strip())
    return Check(
        "Source and configuration checksums",
        entries >= 10 and not missing and not mismatched,
        f"{entries} entries; missing={missing or 'none'}; "
        f"mismatched={mismatched or 'none'}",
    )


def value_at(document: dict[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for part in dotted_path.split("."):
        value = value[part]
    return value


def verify_parameter_manifest(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "MODEL_PARAMETERS.json"
    parameters = load_json(manifest_path)
    expected = {
        "artifact_version": "cikm-2026-artifact-v1",
        "claim_normalization.model": "mistralai/Mistral-7B-Instruct-v0.3",
        "claim_normalization.split": "test",
        "claim_normalization.row_selection.start": 0,
        "claim_normalization.row_selection.stop_exclusive": 300,
        "claim_normalization.prompt_version": "v1",
        "claim_normalization.max_new_tokens": 256,
        "claim_normalization.temperature": 0.0,
        "claim_normalization.do_sample": False,
        "claim_normalization.num_examples": 5,
        "claim_normalization.topic_clusters": 10,
        "claim_normalization.retrieval_threshold": 0.85,
        "claim_normalization.claim_verify_threshold": 0.5,
        "claim_normalization.batch_size": 8,
        "checkworthiness_soft_labels.model": (
            "mistralai/Mistral-Small-24B-Instruct-2501"
        ),
        "checkworthiness_soft_labels.max_tokens": 2048,
        "checkworthiness_soft_labels.temperature": 0.0,
        "checkworthiness_soft_labels.logprobs": True,
        "checkworthiness_soft_labels.top_logprobs": 5,
        "four_head_deberta.model": "microsoft/deberta-v3-large",
        "four_head_deberta.max_length": 128,
        "four_head_deberta.batch_size": 16,
        "four_head_deberta.gradient_accumulation_steps": 2,
        "four_head_deberta.learning_rate": 0.00001,
        "four_head_deberta.aux_learning_rate": 0.001,
        "four_head_deberta.lambda_checkability": 0.3,
        "four_head_deberta.lambda_verifiability": 0.3,
        "four_head_deberta.lambda_harm": 0.2,
        "four_head_deberta.entropy_weighting": True,
        "four_head_deberta.seed": 42,
        "frontier_llm_ablation.model": "claude-opus-4-7",
        "frontier_llm_ablation.samples": 341,
        "frontier_llm_ablation.calls": 1023,
        "frontier_llm_ablation.temperature_parameter": "omitted",
        "virality_tabular_baselines.split.test_size": 0.2,
        "virality_tabular_baselines.split.random_state": 42,
        "virality_tabular_baselines.models.Ridge.alpha": 0.01,
        "virality_tabular_baselines.models.RandomForest.n_estimators": 100,
        "virality_tabular_baselines.models.LightGBM.learning_rate": 0.05,
        "virality_tabular_baselines.models.SVR (RBF).C": 0.1,
    }
    mismatched: dict[str, dict[str, Any]] = {}
    for dotted_path, expected_value in expected.items():
        try:
            actual = value_at(parameters, dotted_path)
        except (KeyError, TypeError):
            actual = "<missing>"
        if actual != expected_value:
            mismatched[dotted_path] = {
                "actual": actual,
                "expected": expected_value,
            }
    source_paths = [
        parameters["claim_normalization"]["source_path"],
        parameters["claim_normalization"]["prompt_path"],
        parameters["checkworthiness_soft_labels"]["source_path"],
        parameters["checkworthiness_soft_labels"]["prompt_path"],
        parameters["four_head_deberta"]["source_path"],
        parameters["frontier_llm_ablation"]["source_path"],
        parameters["frontier_llm_ablation"]["prompt_path"],
        parameters["virality_tabular_baselines"]["source_path"],
    ]
    missing_paths = [
        relative for relative in source_paths if not (repo_root / relative).is_file()
    ]
    return Check(
        "Model parameter manifest",
        not mismatched and not missing_paths,
        f"{len(expected)} values checked; mismatched={mismatched or 'none'}; "
        f"missing source paths={missing_paths or 'none'}",
    )


def verify_prompt_and_source_provenance(repo_root: Path) -> list[Check]:
    claim_prompt_path = repo_root / "prompts" / "claim_normalization_cikm2026.md"
    soft_label_prompt_path = (
        repo_root / "prompts" / "checkworthiness_prompts_zeroshot_v4.yaml"
    )
    soft_label_source_path = (
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "checkworthiness"
        / "source_code"
        / "scripts"
        / "runners"
        / "run_llm_checkworthiness_v4.py"
    )
    claim_source_path = (
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "claim_normalization"
        / "source_code"
        / "scripts"
        / "run_claim_normalization_ct25.py"
    )
    claim_prompt = claim_prompt_path.read_text(encoding="utf-8")
    soft_label_prompt = soft_label_prompt_path.read_text(encoding="utf-8")
    soft_label_source = soft_label_source_path.read_text(encoding="utf-8")
    claim_source = claim_source_path.read_text(encoding="utf-8")

    claim_fragments = [
        "You are a claim normalization specialist.",
        "Extract the main factual assertion from the post",
        "Preserve the original meaning exactly",
        "Here are some examples of claim normalization:",
        "Normalized claim:",
    ]
    prompt_matches_source = all(
        fragment in claim_prompt and fragment in claim_source
        for fragment in claim_fragments
    )
    soft_label_fragments = [
        "checkability:",
        "verifiability:",
        "harm_potential:",
        "max_tokens: 2048",
        "assistant: '{\"confidence\":'",
    ]
    soft_label_complete = all(
        fragment in soft_label_prompt for fragment in soft_label_fragments
    )
    soft_label_source_fragments = [
        'MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"',
        "temperature=0.0",
        "logprobs=True",
        "top_logprobs=5",
        'response_format={"type": "json_object"}',
        'messages.append({"role": "assistant", "content": prompt_config["assistant"]})',
    ]
    soft_label_source_complete = all(
        fragment in soft_label_source for fragment in soft_label_source_fragments
    )
    credential_diagnostic_safe = (
        "OPENAI_API_KEY configured=" in claim_source
        and "TOGETHER_API_KEY configured=" in claim_source
        and "OPENAI_API_KEY[:30]" not in claim_source
        and "TOGETHER_API_KEY[:30]" not in claim_source
    )
    return [
        Check(
            "Claim-normalization prompt provenance",
            prompt_matches_source,
            f"{len(claim_fragments)} exact prompt fragments match the executable source",
        ),
        Check(
            "Check-worthiness prompt provenance",
            soft_label_complete
            and soft_label_source_complete
            and sha256_file(soft_label_prompt_path)
            == "ac167033361da470b3ce3f811f2eaa38a0b23e678e69c3a2c17a22df0e867f66",
            "v4 runner, three dimensions, 2048-token limits, assistant prefix, "
            "generation settings, and source hash checked",
        ),
        Check(
            "Claim-normalization credential diagnostics",
            credential_diagnostic_safe,
            "diagnostic reports configured status without printing credential fragments",
        ),
    ]


def verify_claim_normalization_result(repo_root: Path) -> list[Check]:
    result_root = repo_root / "results" / "ct25_claim_normalization_lambda_2026-05-15"
    summary = load_json(result_root / "summary.json")
    rows = [
        json.loads(line)
        for line in (result_root / "per_sample_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_fields = {
        "idx",
        "meteor",
        "model",
        "latency_ms",
        "input_tokens",
        "output_tokens",
    }
    text_or_identifier_fields = {
        "post",
        "text",
        "raw_text",
        "predicted_claim",
        "gold_claim",
        "post_id",
        "tweet_id",
        "user_id",
        "username",
    }
    retrieval_count = sum(str(row["model"]).startswith("retrieval@") for row in rows)
    generated_count = len(rows) - retrieval_count
    mean_meteor = sum(float(row["meteor"]) for row in rows) / len(rows)
    return [
        Check(
            "Claim-normalization retained result",
            summary["n_samples"] == len(rows) == 300
            and sorted(row["idx"] for row in rows) == list(range(300))
            and retrieval_count == 227
            and generated_count == 73
            and math.isclose(
                mean_meteor,
                float(summary["avg_meteor"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"rows={len(rows)}; retrieval={retrieval_count}; "
            f"generation={generated_count}; mean METEOR={mean_meteor:.15f}",
        ),
        Check(
            "Claim-normalization result data boundary",
            all(set(row) == expected_fields for row in rows)
            and not any(text_or_identifier_fields.intersection(row) for row in rows),
            "per-sample rows contain only index, metric, route, latency, and token counts",
        ),
    ]


def verify_claim_result_checksums(repo_root: Path) -> Check:
    result_root = repo_root / "results" / "ct25_claim_normalization_lambda_2026-05-15"
    manifest_path = result_root / "checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        path = result_root / filename.strip()
        entries += 1
        if not path.is_file():
            missing.append(filename.strip())
        elif sha256_file(path) != expected:
            mismatched.append(filename.strip())
    return Check(
        "Claim-normalization result checksums",
        entries == 4 and not missing and not mismatched,
        f"{entries} entries; missing={missing or 'none'}; "
        f"mismatched={mismatched or 'none'}",
    )


def verify_checkpoint_manifest(repo_root: Path, checkpoint: Path | None) -> list[Check]:
    artifact_root = repo_root / "reproducibility" / "cikm2026"
    checksum_line = (
        (artifact_root / "checkpoint.sha256").read_text(encoding="utf-8").strip()
    )
    checkpoint_doc = (artifact_root / "CHECKPOINT.md").read_text(encoding="utf-8")
    checks = [
        Check(
            "Checkpoint release manifest",
            checksum_line == f"{CHECKPOINT_SHA256}  best_model.pt"
            and str(CHECKPOINT_SIZE_BYTES) in checkpoint_doc.replace(",", "")
            and CHECKPOINT_SHA256 in checkpoint_doc
            and CHECKPOINT_RELEASE_URL in checkpoint_doc,
            f"best_model.pt; bytes={CHECKPOINT_SIZE_BYTES}; sha256={CHECKPOINT_SHA256}",
        )
    ]
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        exists = resolved.is_file()
        actual_size = resolved.stat().st_size if exists else None
        actual_hash = sha256_file(resolved) if exists else None
        checks.append(
            Check(
                "Downloaded checkpoint identity",
                exists
                and actual_size == CHECKPOINT_SIZE_BYTES
                and actual_hash == CHECKPOINT_SHA256,
                f"path={resolved}; exists={exists}; bytes={actual_size}; sha256={actual_hash}",
            )
        )
    return checks


def verify_dataset(repo_root: Path) -> list[Check]:
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"
    features = pq.read_table(artifact_root / "features_enhanced.parquet")
    labels = pq.read_table(artifact_root / "psr_labels.parquet")
    split = load_json(repo_root / "results" / "virality_split_manifest_2026-05-12.json")

    checks: list[Check] = []
    expected_feature_columns = split["feature_columns"]
    checks.append(
        Check(
            "Dataset shape",
            features.num_rows == labels.num_rows == 529
            and features.num_columns == 44
            and labels.num_columns == 5,
            f"features={features.num_rows}x{features.num_columns}; labels={labels.num_rows}x{labels.num_columns}",
        )
    )
    checks.append(
        Check(
            "Feature schema",
            features.column_names == ["cluster_id", *expected_feature_columns, "psr"]
            and labels.column_names
            == ["cluster_id", "detection_time", "e_detect", "e_final", "psr"],
            f"42 manifest features; feature schema has {features.num_columns - 2} inputs",
        )
    )

    feature_ids = features["cluster_id"].to_pylist()
    label_ids = labels["cluster_id"].to_pylist()
    checks.append(
        Check(
            "Cluster identifiers",
            len(set(feature_ids)) == len(set(label_ids)) == 529
            and set(feature_ids) == set(label_ids),
            f"unique features={len(set(feature_ids))}; unique labels={len(set(label_ids))}; same set={set(feature_ids) == set(label_ids)}",
        )
    )

    feature_psr = dict(zip(feature_ids, features["psr"].to_pylist(), strict=True))
    label_rows = {row["cluster_id"]: row for row in labels.to_pylist()}
    matching_psr = all(
        feature_psr[cluster_id] == row["psr"] for cluster_id, row in label_rows.items()
    )
    formula_matches = all(
        row["e_final"] > 0
        and 0 <= row["e_detect"] <= row["e_final"]
        and math.isclose(
            row["psr"],
            (row["e_final"] - row["e_detect"]) / row["e_final"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in label_rows.values()
    )
    psr_values = list(feature_psr.values())
    checks.append(
        Check(
            "Virality labels",
            matching_psr
            and formula_matches
            and min(psr_values) >= 0.0
            and max(psr_values) <= 1.0,
            f"keyed tables match={matching_psr}; formula matches={formula_matches}; range=[{min(psr_values):.6f}, {max(psr_values):.6f}]",
        )
    )

    expected_nan_counts = {
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
    actual_nan_counts = {
        name: count
        for name in features.column_names
        if (count := count_nan(features[name])) > 0
    }
    checks.append(
        Check(
            "Documented missingness",
            actual_nan_counts == expected_nan_counts,
            f"NaN counts match the data card: {actual_nan_counts}",
        )
    )

    restricted_columns = {
        "text",
        "raw_text",
        "tweet_id",
        "post_id",
        "user_id",
        "username",
        "screen_name",
        "handle",
        "profile_description",
        "location",
    }
    present_restricted = restricted_columns.intersection(
        features.column_names + labels.column_names
    )
    checks.append(
        Check(
            "Public-data boundary",
            not present_restricted,
            f"raw text/user/post identifier columns present={sorted(present_restricted) or 'none'}",
        )
    )

    counts = split["counts"]
    test_size = counts["test_rows"]
    permutation = np.random.RandomState(42).permutation(len(feature_ids))
    derived_test = [feature_ids[index] for index in permutation[:test_size]]
    derived_train = [feature_ids[index] for index in permutation[test_size:]]
    stored_train = split["train_cluster_ids"]
    stored_test = split["test_cluster_ids"]
    checks.append(
        Check(
            "Deterministic split",
            counts
            == {
                "feature_columns": 42,
                "features_rows": 529,
                "labels_rows": 529,
                "test_rows": 106,
                "train_rows": 423,
            }
            and derived_train == stored_train
            and derived_test == stored_test
            and not set(stored_train).intersection(stored_test)
            and set(stored_train + stored_test) == set(feature_ids),
            f"train={len(stored_train)}; test={len(stored_test)}; RandomState(42) membership matches",
        )
    )
    return checks


def rounded_metrics(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        round(float(row[key]), 3)
        for key in ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")
    )


def verify_result_artifacts(repo_root: Path) -> list[Check]:
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"
    tuned = {
        row["name"]: row for row in load_json(artifact_root / "tuned_baselines.json")
    }
    complete = {
        row["name"]: row for row in load_json(artifact_root / "complete_baselines.json")
    }
    table_text = (artifact_root / "latex_table.tex").read_text(encoding="utf-8")

    expected = {
        "Random": (tuned, (-0.052, -1.141, 0.381, 0.281, 0.167, 0.075)),
        "Ridge": (tuned, (0.530, 0.293, 0.222, 0.410, 0.203, 0.080)),
        "BayesianRidge": (tuned, (0.551, 0.293, 0.226, 0.413, 0.131, 0.041)),
        "RandomForest": (tuned, (0.532, 0.319, 0.217, 0.547, 0.322, 0.041)),
        "LightGBM": (tuned, (0.549, 0.333, 0.211, 0.581, 0.322, 0.081)),
        "SVR (RBF)": (tuned, (0.542, 0.336, 0.207, 0.632, 0.343, 0.041)),
        "LSTM": (complete, (0.016, -0.021, 0.280, 0.263, 0.180, 0.041)),
        "GRU": (complete, (0.082, -0.023, 0.281, 0.335, 0.149, 0.041)),
        "Hawkes (HIP)": (complete, (0.247, -3.491, 0.587, 0.000, 0.000, 0.000)),
        "BERTweet + Features": (tuned, (0.510, 0.255, 0.222, 0.541, 0.256, 0.111)),
    }

    checks: list[Check] = []
    for name, (source, expected_metrics) in expected.items():
        actual = rounded_metrics(source[name]) if name in source else ()
        checks.append(
            Check(
                f"Result artifact: {name}",
                actual == expected_metrics and name in table_text,
                f"rounded metrics={actual}; expected={expected_metrics}; present in LaTeX={name in table_text}",
            )
        )
    return checks


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root to verify (defaults to the parent of this script's directory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional downloaded best_model.pt release asset to verify.",
    )
    return parser.parse_args()


def run_check_group(name: str, check_group: Any) -> list[Check]:
    try:
        result = check_group()
    except Exception as error:  # noqa: BLE001 - malformed public artifacts must produce a readable FAIL
        return [Check(f"{name} execution", False, f"{type(error).__name__}: {error}")]
    return [result] if isinstance(result, Check) else list(result)


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    checks = [
        *run_check_group("Checksum verification", lambda: verify_checksums(repo_root)),
        *run_check_group(
            "Source checksum verification",
            lambda: verify_code_checksums(repo_root),
        ),
        *run_check_group(
            "Repository data-boundary verification",
            lambda: verify_repository_data_boundary(repo_root),
        ),
        *run_check_group(
            "Parameter-manifest verification",
            lambda: verify_parameter_manifest(repo_root),
        ),
        *run_check_group(
            "Prompt and source verification",
            lambda: verify_prompt_and_source_provenance(repo_root),
        ),
        *run_check_group(
            "Claim-normalization result verification",
            lambda: verify_claim_normalization_result(repo_root),
        ),
        *run_check_group(
            "Claim-normalization result checksum verification",
            lambda: verify_claim_result_checksums(repo_root),
        ),
        *run_check_group(
            "Checkpoint-manifest verification",
            lambda: verify_checkpoint_manifest(repo_root, args.checkpoint),
        ),
        *run_check_group("Dataset verification", lambda: verify_dataset(repo_root)),
        *run_check_group(
            "Result-artifact verification", lambda: verify_result_artifacts(repo_root)
        ),
    ]

    print("CIKM 2026 public artifact verification")
    print(f"Repository root: {repo_root}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    failures = [check for check in checks if not check.passed]
    print(f"\nSummary: {len(checks) - len(failures)} passed, {len(failures)} failed")
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
