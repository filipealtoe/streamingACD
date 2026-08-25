#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.5.2",
#   "pyarrow==25.0.1",
# ]
# ///
"""Verify the public CIKM 2026 dataset and reproducibility artifacts.

The default verification uses only files committed to the public repository.
Optional paths also verify the exact camera-ready PDF and external release assets
by byte size and SHA-256.
"""

from __future__ import annotations

import argparse
import csv
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
PAPER_PDF_SHA256 = "4d82abd01d66de5e04d7107e8c4bb21d3b3d1a7148aa235e95b78b803df78b9c"
PAPER_PDF_SIZE_BYTES = 620_920
CLUSTER_EMBEDDINGS_SHA256 = (
    "514e9f44c095b1de5175cc6c76db548e9762c3e1bcdda93b8cb8f16f09bd1619"
)
CLUSTER_EMBEDDINGS_SIZE_BYTES = 153_600_128
CLUSTER_EMBEDDINGS_SHAPE = (100_000, 384)
CHECKPOINT_RELEASE_URL = (
    "https://github.com/filipealtoe/streamingACD/releases/download/"
    "cikm-2026-artifact-v1/best_model.pt"
)

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-21 22:32 PDT | Reason: make the final verification result portable
# and machine-readable, with known unresolved checks separated from unexpected
# artifact-integrity failures.
KNOWN_FAILURE_CATEGORIES = {
    "Claim-normalization Table 2 internal completeness": "external_historical_evidence",
    "Claim-normalization prose-value completeness": "external_historical_evidence",
    "Encoder Only ClaimBuster direct completeness": "external_historical_evidence",
    "Fusion cross-benchmark direct completeness": "external_historical_evidence",
    "Frontier-LLM direct reproduction completeness": "external_historical_evidence",
    "Worked-example member-count temporal consistency": "manuscript_artifact_mismatch",
    "Paper stream-duration consistency": "manuscript_artifact_mismatch",
    "Paper pipeline-unit consistency": "manuscript_artifact_mismatch",
    "Anomaly method-to-table parameter consistency": "manuscript_artifact_mismatch",
    "Anomaly detection-rate definition consistency": "manuscript_artifact_mismatch",
    "Virality feature observation-window consistency": "manuscript_artifact_mismatch",
    "Virality split stratification consistency": "manuscript_artifact_mismatch",
    "LLM-feature threshold-selection consistency": "manuscript_artifact_mismatch",
    "Published-baseline row provenance consistency": "manuscript_artifact_mismatch",
    "Four-head training-seed consistency": "manuscript_artifact_mismatch",
    "Four-head training-schedule consistency": "manuscript_artifact_mismatch",
    "Fusion ensemble seed-scope consistency": "manuscript_artifact_mismatch",
    "Check-worthiness side-claim completeness": "manuscript_artifact_mismatch",
    "Virality single-run statistical consistency": "manuscript_artifact_mismatch",
    "CT24 shared-error 86 percent direct reproduction": "manuscript_artifact_mismatch",
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


def verify_artifact_release_coverage(repo_root: Path) -> Check:
    """Compare the public package with the manuscript's release promises."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-22 00:53 PDT | Reason: verify the availability statements in the
    # exact final PDF rather than promises that belonged to an earlier draft.
    audit = load_json(
        repo_root / "results/artifact_release_coverage_audit_2026-08-22.json"
    )
    core = audit["public_scientific_core"]
    promises = audit["promise_checks"]
    dataset = audit["dataset"]
    embeddings = audit["pipeline_output_inventory"]["cluster_embeddings.npy"]
    embedding_release_accounted = (
        embeddings["present"] is True
        and embeddings["present_in_git"] is False
        and embeddings["published_as_release_asset"] is True
        and embeddings["distribution"] == "github_release"
    )
    counts_match = (
        int(core["canonical_claims"]) == 535
        and int(core["aggregate_clusters"]) == 100_000
        and int(core["aggregate_cluster_timeseries_rows"]) == 404_296
        and int(core["virality_cluster_instances"]) == 529
        and int(dataset["features"]["rows"]) == 529
        and int(dataset["labels"]["rows"]) == 529
        and int(dataset["engineered_feature_columns"]) == 42
        and dataset["cluster_ids_match"] is True
        and int(dataset["unique_cluster_ids"]) == 529
    )
    complete = all(bool(value) for value in promises.values())
    unresolved = [name for name, value in promises.items() if not value]
    return Check(
        "Paper artifact-release coverage",
        counts_match
        and embedding_release_accounted
        and complete
        and audit["summary"]["verdict"] == "PASS",
        f"aggregate scientific counts verified={counts_match}; "
        f"cluster embeddings public via release={embedding_release_accounted}; "
        f"promises satisfied={audit['summary']['promises_satisfied']}/"
        f"{audit['summary']['promises_checked']}; missing={unresolved}",
    )


def verify_checksums(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    manifest_paths: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative_path = relative.strip()
        path = repo_root / relative_path
        manifest_paths.add(relative_path)
        if not path.is_file():
            missing.append(relative_path)
        elif sha256_file(path) != expected:
            mismatched.append(relative_path)
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:14 PDT | Reason: require the complete public evidence set
    # without making legitimate future additions fail a fixed entry count.
    required_paths = {
        "reproducibility/source_artifacts/virality/features_enhanced.parquet",
        "reproducibility/source_artifacts/virality/psr_labels.parquet",
        "reproducibility/source_artifacts/virality/tuned_baselines.json",
        "reproducibility/source_artifacts/virality/complete_baselines.json",
        "reproducibility/source_artifacts/virality/latex_table.tex",
        "results/virality_split_manifest_2026-05-12.json",
        "reproducibility/source_artifacts/anomaly/timeseries.parquet",
        "reproducibility/source_artifacts/anomaly/timeseries.metadata.json",
        "reproducibility/source_artifacts/anomaly/historical/baselines_min_rows89.json",
        "reproducibility/source_artifacts/anomaly/historical/expose_min_rows_sweep.json",
        "results/anomaly_table_reproduction_2026-08-21/checksums.sha256",
        "results/virality_tabular_reproduction_2026-08-21.json",
        "results/virality_statistics_reproduction_2026-08-21.json",
        "results/artifact_release_coverage_audit_2026-08-22.json",
        "results/psr_statistics_20260518_205609/README.md",
        "results/psr_statistics_20260518_205609/metrics.json",
        "results/psr_statistics_20260518_205609/predictions/manifest.json",
        "reproducibility/cikm2026/cluster_embeddings.metadata.json",
        "reproducibility/cikm2026/cluster_embeddings.sha256",
        "psr/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56/cluster_id_to_idx.json",
        "results/case_study_reproduction_2026-08-21.json",
        "results/checkworthiness_shared_error_audit_2026-08-21.json",
        "results/psr_latency_benchmark_2026-08-21.json",
        "results/mtl_threshold_calibration_audit_2026-08-21.json",
        "results/mtl_gpu_latency_audit_2026-08-21.json",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/train_labels.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/dev_labels.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/test_labels.parquet",
        "results/public_encoder_fusion_replication_2026-08-25/per_example_predictions.npz",
        "results/public_encoder_fusion_replication_2026-08-25/summary.json",
    }
    absent_required = sorted(required_paths - manifest_paths)
    passed = not missing and not mismatched and not absent_required
    return Check(
        "Release checksums",
        passed,
        f"{len(manifest_paths)} entries; missing={missing or 'none'}; "
        f"mismatched={mismatched or 'none'}; "
        f"unlisted required={absent_required or 'none'}",
    )


def verify_code_checksums(repo_root: Path) -> Check:
    manifest_path = repo_root / "reproducibility" / "cikm2026" / "code_checksums.sha256"
    missing: list[str] = []
    mismatched: list[str] = []
    manifest_paths: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative_path = relative.strip()
        path = repo_root / relative_path
        manifest_paths.add(relative_path)
        if not path.is_file():
            missing.append(relative_path)
        elif sha256_file(path) != expected:
            mismatched.append(relative_path)
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:48 PDT | Reason: require the new statistical and release
    # coverage entry points plus the original statistical source snapshot.
    required_paths = {
        "scripts/reproduce_cikm2026_virality_statistics.py",
        "scripts/audit_cikm2026_release_coverage.py",
        "results/psr_statistics_20260518_205609/scripts/run_psr_statistics.py",
        "scripts/reproduce_cikm2026_public_encoder_fusion.py",
        "tests/test_cikm2026_checkworthiness_reproduction.py",
    }
    return Check(
        "Source and configuration checksums",
        len(manifest_paths) >= 10
        and required_paths.issubset(manifest_paths)
        and not missing
        and not mismatched,
        f"{len(manifest_paths)} entries; required paths present="
        f"{required_paths.issubset(manifest_paths)}; missing={missing or 'none'}; "
        f"mismatched={mismatched or 'none'}",
    )


def verify_prediction_checksums(repo_root: Path) -> Check:
    """Verify every retained virality prediction and evaluation input."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 18:26 PDT | Reason: prevent silent replacement or corruption of
    # the arrays used to recompute the paper-facing virality metrics.
    manifest_path = (
        repo_root / "reproducibility" / "cikm2026" / "prediction_checksums.sha256"
    )
    missing: list[str] = []
    mismatched: list[str] = []
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = repo_root / relative.strip()
        entries += 1
        if not path.is_file():
            missing.append(relative.strip())
        elif sha256_file(path) != expected:
            mismatched.append(relative.strip())
    return Check(
        "Virality prediction checksums",
        entries == 64 and not missing and not mismatched,
        f"{entries} entries; missing={missing or 'none'}; "
        f"mismatched={mismatched or 'none'}",
    )


def verify_checkworthiness_checksums(repo_root: Path) -> Check:
    """Verify benchmark inputs and arrays used by the four-head table row."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:43 PDT | Reason: bind every input and prediction used by
    # the three paper-facing four-head F1 cells.
    manifest_path = (
        repo_root / "reproducibility" / "cikm2026" / "checkworthiness_checksums.sha256"
    )
    missing: list[str] = []
    mismatched: list[str] = []
    manifest_paths: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative_path = relative.strip()
        path = repo_root / relative_path
        manifest_paths.add(relative_path)
        if not path.is_file():
            missing.append(relative_path)
        elif sha256_file(path) != expected:
            mismatched.append(relative_path)
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:20 PDT | Reason: bind the official CT24 dev-test count audit
    # without making later legitimate additions fail a fixed exact count.
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:36 PDT | Reason: require the published-baseline provenance
    # audit and the two new numerical audits to remain checksum-bound.
    required_metadata = {
        "reproducibility/source_artifacts/checkworthiness/benchmarks/ct24_devtest_metadata.json",
        "reproducibility/source_artifacts/checkworthiness/published_baseline/PUBLISHED_BASELINE_AUDIT.json",
        "results/mtl_threshold_calibration_audit_2026-08-21.json",
        "results/mtl_gpu_latency_audit_2026-08-21.json",
        "reproducibility/source_artifacts/checkworthiness/encoder_only/HISTORICAL_RUN_EXTRACT.md",
        "reproducibility/source_artifacts/checkworthiness/fusion_classifier/HISTORICAL_RUN_EXTRACT.md",
        "reproducibility/cikm2026/checkworthiness/PAPER_PROTOCOL.json",
        "reproducibility/source_artifacts/checkworthiness/benchmark_llm_features/CB_groundtruth_llm_features.parquet",
        "reproducibility/source_artifacts/checkworthiness/benchmark_llm_features/CT23_llm_features.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_llm_features_v4/train_llm_features.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_llm_features_v4/dev_llm_features.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_llm_features_v4/test_llm_features.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/train_labels.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/dev_labels.parquet",
        "reproducibility/source_artifacts/checkworthiness/ct24_labels/test_labels.parquet",
        "results/public_encoder_fusion_replication_2026-08-25/per_example_predictions.npz",
        "results/public_encoder_fusion_replication_2026-08-25/summary.json",
    }
    return Check(
        "Check-worthiness artifact checksums",
        len(manifest_paths) >= 33
        and required_metadata.issubset(manifest_paths)
        and not missing
        and not mismatched,
        f"{len(manifest_paths)} entries; required metadata present="
        f"{required_metadata.issubset(manifest_paths)}; missing={missing or 'none'}; "
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
        "claim_normalization.negative_examples": 0,
        "claim_normalization_paper_model_a.topic_clusters": 10,
        "claim_normalization_paper_model_a.retrieval_threshold": 0.85,
        "claim_normalization_paper_model_a.claim_verify_threshold": 0.65,
        "claim_normalization_paper_model_a.positive_examples": 4,
        "claim_normalization_paper_model_a.negative_examples": 2,
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
        "four_head_deberta.training_data_counts.train_total": 22402,
        "four_head_deberta.training_data_counts.train_negative": 17003,
        "four_head_deberta.training_data_counts.train_positive": 5399,
        "four_head_deberta.training_data_counts.dev_rows": 1031,
        "four_head_deberta.training_data_counts.test_rows": 341,
        "four_head_deberta.training_data_counts.positive_class_weight_paper_rounded": 3.15,
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
        "virality_statistics.bootstrap_resamples": 1000,
        "virality_statistics.bootstrap_seed": 42,
        "virality_statistics.paired_threshold": 0.65,
        "virality_statistics.paired_test": "exact two-sided McNemar",
        "virality_statistics.spearman_p_value_test": "scipy.stats.spearmanr two-tailed",
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
        parameters["virality_statistics"]["source_path"],
        parameters["virality_statistics"]["statistical_source_snapshot"],
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
    table_summary = load_json(
        repo_root / "results/claim_normalization_table_reproduction_2026-08-21.json"
    )
    recovery_audit = load_json(
        repo_root
        / "reproducibility/source_artifacts/claim_normalization/"
        "TABLE2_RECOVERY_AUDIT.json"
    )
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    paper_table = manifest["tables"]["claim_normalization"]
    supported = table_summary["results"]["Approach 2 / Model A"]
    unresolved = table_summary["unresolved_cells"]
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
        Check(
            "Claim-normalization Model A numerical recovery boundary",
            table_summary["status"] == "UNRESOLVED"
            and table_summary["internal_cells"]
            == {
                "directly_reproduced": 0,
                "numeric_recovery_matches": 1,
                "total": 9,
                "unresolved": 9,
            }
            and supported["status"] == "NUMERIC_MATCH_DIFFERENT_PROTOCOL"
            and supported["rounded_numeric_match"]
            and not supported["protocol_match"]
            and math.isclose(
                float(supported["full_precision_meteor"]),
                mean_meteor,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and paper_table["rows"]["Approach 2"]["metrics"][0] == 0.5569
            and supported["paper_protocol"]
            == paper_table["paper_model_a_protocol"]
            and supported["recovery_protocol"]
            == paper_table["recovery_model_a_protocol"]
            and recovery_audit["internal_evidence"]["Approach 2 / Model A"][
                "per_example_metrics_sha256"
            ]
            == sha256_file(result_root / "per_sample_metrics.jsonl"),
            f"Approach 2 / Model A METEOR={mean_meteor:.15f}; "
            "paper=0.5569; numeric match=yes; protocol match=no",
        ),
        Check(
            "Claim-normalization Table 2 internal completeness",
            table_summary["status"] == "COMPLETE" and not unresolved,
            f"directly reproduced={table_summary['internal_cells']['directly_reproduced']}/9; "
            f"unresolved={unresolved}",
        ),
    ]


def verify_shared_error_analysis(repo_root: Path) -> list[Check]:
    """Verify the retained evidence boundary for the paper's 86% claim."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:26 PDT | Reason: verify the recovered sentence-ID join and
    # distinguish complete alignment from a non-reproducing percentage.
    result = load_json(
        repo_root / "results/checkworthiness_shared_error_audit_2026-08-21.json"
    )
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )["checkworthiness_shared_error_analysis"]
    alignment = result["alignment_audit"]
    alignment_recovered = (
        result["paper_value_percent"] == manifest["paper_value_percent"] == 86
        and result["evaluation_rows"] == manifest["evaluation_rows"] == 341
        and alignment["same_vector_length"]
        and alignment["unique_sentence_ids_in_both_orders"]
        and alignment["same_sentence_id_set"]
        and alignment["labels_match_after_sentence_id_join"]
        and alignment["mtl_unique_sentence_ids"] == 341
        and alignment["direct_alignment_available"]
        and result["error_counts"]
        == {"llm_features": 53, "encoder_only": 30, "four_parallel_heads": 30}
        and result["status"] == "MISMATCH"
    )
    return [
        Check(
            "CT24 shared-error alignment audit",
            alignment_recovered,
            "341 unique official sentence IDs join all three vectors; labels match after reordering",
        ),
        Check(
            "CT24 shared-error 86 percent direct reproduction",
            bool(result["paper_value_reproduced"]),
            "paper=86%; observed explicit definitions="
            + ", ".join(
                f"{name}={value:.2f}%"
                for name, value in result["observed_percentages"].items()
            ),
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


def verify_paper_pdf_manifest(repo_root: Path, paper_pdf: Path | None) -> list[Check]:
    """Bind the paper-value inventory to the exact camera-ready PDF."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-22 00:53 PDT | Reason: bind the inventory to the exact final build
    # rather than the superseded August 13 camera-ready draft.
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    metadata = manifest["paper_pdf"]
    manifest_ok = (
        metadata["file_name"] == "CIKM2026-camera-ready-2026-08-22.pdf"
        and int(metadata["bytes"]) == PAPER_PDF_SIZE_BYTES
        and int(metadata["pages"]) == 12
        and metadata["sha256"] == PAPER_PDF_SHA256
    )
    checks = [
        Check(
            "Camera-ready PDF manifest",
            manifest_ok,
            f"{metadata['file_name']}; pages={metadata['pages']}; bytes={metadata['bytes']}; "
            f"sha256={metadata['sha256']}",
        )
    ]
    if paper_pdf is not None:
        resolved = paper_pdf.expanduser().resolve()
        exists = resolved.is_file()
        actual_size = resolved.stat().st_size if exists else None
        actual_hash = sha256_file(resolved) if exists else None
        checks.append(
            Check(
                "Provided camera-ready PDF identity",
                exists
                and actual_size == PAPER_PDF_SIZE_BYTES
                and actual_hash == PAPER_PDF_SHA256,
                f"path={resolved}; exists={exists}; bytes={actual_size}; "
                f"sha256={actual_hash}",
            )
        )
    return checks


def verify_cluster_embeddings_manifest(
    repo_root: Path, embeddings: Path | None
) -> list[Check]:
    """Verify the recovered canonical embedding asset metadata and optional file."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-22 00:53 PDT | Reason: authenticate the released 384-dimensional
    # matrix without carrying forward a dimension claim absent from the final PDF.
    artifact_root = repo_root / "reproducibility/cikm2026"
    metadata = load_json(artifact_root / "cluster_embeddings.metadata.json")
    checksum_line = (
        (artifact_root / "cluster_embeddings.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    index_path = repo_root / metadata["index"]["path"]
    manifest_ok = (
        checksum_line == f"{CLUSTER_EMBEDDINGS_SHA256}  cluster_embeddings.npy"
        and int(metadata["bytes"]) == CLUSTER_EMBEDDINGS_SIZE_BYTES
        and (int(metadata["rows"]), int(metadata["retained_dimensions"]))
        == CLUSTER_EMBEDDINGS_SHAPE
        and metadata["dtype"] == "float32"
        and metadata["finite_values"] is True
        and metadata["paper_dimensions"] is None
        and sha256_file(index_path) == metadata["index"]["sha256"]
        and int(metadata["index"]["entries"]) == CLUSTER_EMBEDDINGS_SHAPE[0]
    )
    checks = [
        Check(
            "Cluster-embedding release manifest",
            manifest_ok,
            f"candidate bytes={metadata['bytes']}; shape="
            f"({metadata['rows']}, {metadata['retained_dimensions']}); "
            f"sha256={metadata['sha256']}",
        )
    ]
    if embeddings is not None:
        resolved = embeddings.expanduser().resolve()
        exists = resolved.is_file()
        actual_size = resolved.stat().st_size if exists else None
        actual_hash = sha256_file(resolved) if exists else None
        actual_shape: tuple[int, ...] | None = None
        actual_dtype: str | None = None
        finite = False
        if exists and actual_size == CLUSTER_EMBEDDINGS_SIZE_BYTES:
            matrix = np.load(resolved, mmap_mode="r", allow_pickle=False)
            actual_shape = tuple(int(value) for value in matrix.shape)
            actual_dtype = str(matrix.dtype)
            finite = bool(np.isfinite(matrix).all())
        checks.append(
            Check(
                "Downloaded cluster-embedding identity",
                exists
                and actual_size == CLUSTER_EMBEDDINGS_SIZE_BYTES
                and actual_hash == CLUSTER_EMBEDDINGS_SHA256
                and actual_shape == CLUSTER_EMBEDDINGS_SHAPE
                and actual_dtype == "float32"
                and finite,
                f"path={resolved}; exists={exists}; bytes={actual_size}; "
                f"sha256={actual_hash}; shape={actual_shape}; dtype={actual_dtype}; "
                f"finite={finite}",
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


def verify_pipeline_narrative_values(repo_root: Path) -> list[Check]:
    """Verify paper-facing corpus, pipeline, mapping, and clustering counts."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:29 PDT | Reason: cover numeric claims in the prose as well
    # as table cells, while preserving the 5,000-post clustering-ablation scope.
    canonical_root = (
        repo_root
        / "psr"
        / "explainableACD"
        / "data"
        / "pipeline_output"
        / "streaming_full"
        / "2026-01-17_03-56"
    )
    manifest = load_json(
        repo_root / "reproducibility" / "cikm2026" / "PAPER_VALUE_MANIFEST.json"
    )["pipeline_narrative_values"]
    summary = load_json(
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "pipeline"
        / "streaming_full_2026-01-17_03-56_summary.json"
    )
    window_table = pq.read_table(
        canonical_root / "window_results.parquet",
        columns=["timestamp", "tweets_processed"],
    )
    first_window = pc.min(window_table["timestamp"]).as_py()
    last_window = pc.max(window_table["timestamp"]).as_py()
    calendar_dates = {
        timestamp.date() for timestamp in window_table["timestamp"].to_pylist()
    }
    stream_expected = manifest["stream"]
    stream_passed = (
        window_table.num_rows == int(stream_expected["hourly_windows"])
        and int(pc.sum(window_table["tweets_processed"]).as_py())
        == int(stream_expected["posts_processed"])
        and first_window.isoformat() == stream_expected["first_window"]
        and last_window.isoformat() == stream_expected["last_window"]
        and len(calendar_dates) == int(stream_expected["calendar_dates_inclusive"])
        and summary["windows_processed"] == int(stream_expected["hourly_windows"])
        and summary["total_tweets_processed"]
        == int(stream_expected["posts_processed"])
    )

    canonical_expected = manifest["canonical_pipeline"]
    canonical_actual = {
        "posts_passed_gate": summary["total_tweets_passed_gate"],
        "clusters": summary["clusterer"]["n_clusters"],
        "anomaly_detections": summary["total_anomalies"],
        "normalized_claims": summary["total_claims"],
        "cluster_to_claim_mappings": summary["claim_registry"][
            "total_clusters_mapped"
        ],
    }
    canonical_parquet_counts_match = (
        pq.read_metadata(canonical_root / "claims.parquet").num_rows
        == int(canonical_expected["normalized_claims"])
        and pq.read_metadata(canonical_root / "clusters.parquet").num_rows
        == int(canonical_expected["clusters"])
        and pq.read_metadata(canonical_root / "window_results.parquet").num_rows
        == int(stream_expected["hourly_windows"])
    )

    mapping = load_json(canonical_root / "registry" / "cluster_mapping.json")
    features = pq.read_table(
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "virality"
        / "features_enhanced.parquet",
        columns=["cluster_id"],
    )
    feature_ids = {int(value) for value in features["cluster_id"].to_pylist()}
    mapped_cluster_ids = {cluster_id for cluster_id in feature_ids if str(cluster_id) in mapping}
    mapped_claim_ids = {mapping[str(cluster_id)] for cluster_id in mapped_cluster_ids}
    virality_expected = manifest["released_virality_artifact"]
    mapping_passed = (
        len(feature_ids) == int(virality_expected["cluster_instances"])
        and len(mapped_cluster_ids)
        == int(virality_expected["mapped_cluster_instances"])
        and len(mapped_claim_ids)
        == int(virality_expected["mapped_unique_canonical_claims"])
        and len(mapping) == int(canonical_expected["cluster_to_claim_mappings"])
    )

    clustering_stats = load_json(
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "clustering"
        / "cluster_statistics.json"
    )["0.65"]
    all_clustering_stats = load_json(
        repo_root
        / "reproducibility"
        / "source_artifacts"
        / "clustering"
        / "cluster_statistics.json"
    )
    clustering_expected = manifest["clustering_threshold_ablation"]
    clustering_passed = (
        clustering_stats["total_tweets"] == int(clustering_expected["sample_posts"])
        and clustering_stats["n_clusters"] == int(clustering_expected["clusters"])
        and math.isclose(
            clustering_stats["cluster_yield_pct"],
            float(clustering_expected["cluster_yield_percent"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            clustering_stats["mean_intra_sim"],
            float(clustering_expected["mean_intra_cluster_similarity"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and round(clustering_stats["cluster_yield_pct"]) == 87
        and round(clustering_stats["mean_intra_sim"], 2) == 0.87
        and all_clustering_stats["0.55"]["total_tweets"] == 5000
        and round(all_clustering_stats["0.55"]["cluster_yield_pct"], 1) == 97.2
        and round(all_clustering_stats["0.55"]["mean_intra_sim"], 2) == 0.82
        and all_clustering_stats["0.8"]["total_tweets"] == 5000
        and round(all_clustering_stats["0.8"]["cluster_yield_pct"], 1) == 27.7
        and round(all_clustering_stats["0.8"]["mean_intra_sim"], 2) == 0.95
    )

    return [
        Check(
            "Corpus and hourly replay values",
            stream_passed,
            f"posts={int(pc.sum(window_table['tweets_processed']).as_py())}; "
            f"windows={window_table.num_rows}; range={first_window.isoformat()} to "
            f"{last_window.isoformat()}; calendar dates={len(calendar_dates)}",
        ),
        Check(
            "Canonical pipeline counts",
            canonical_actual == canonical_expected and canonical_parquet_counts_match,
            f"summary={canonical_actual}; canonical parquet row counts match={canonical_parquet_counts_match}",
        ),
        Check(
            "Virality cluster-to-claim mapping",
            mapping_passed,
            f"cluster instances={len(feature_ids)}; mapped={len(mapped_cluster_ids)}; "
            f"unique canonical claims={len(mapped_claim_ids)}",
        ),
        Check(
            "Clustering threshold ablation scope and values",
            clustering_passed,
            f"sample={clustering_stats['total_tweets']}; tau=0.65; "
            f"yield={clustering_stats['cluster_yield_pct']}; "
            f"mean similarity={clustering_stats['mean_intra_sim']}; "
            "endpoints=0.55:(97.2%,0.82), 0.80:(27.7%,0.95)",
        ),
    ]


def verify_worked_example(repo_root: Path) -> list[Check]:
    """Verify the retained cluster-72106 worked-example result."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:14 PDT | Reason: separate reproduction of the published
    # registry value from its inconsistent detection-timestamp interpretation.
    result = load_json(
        repo_root / "results" / "case_study_reproduction_2026-08-21.json"
    )
    comparisons = result["comparisons"]
    supported = {
        name: passed
        for name, passed in comparisons.items()
        if name != "detection_timestamp_member_count"
    }
    actual = result["actual"]
    expected = result["paper_values"]
    return [
        Check(
            "Worked-example supported numeric fields",
            len(supported) == 18 and all(supported.values()),
            f"supported={sum(bool(value) for value in supported.values())}/{len(supported)}; "
            f"cluster={actual['cluster_id']}; status={result['status']}",
        ),
        Check(
            "Worked-example published member value",
            bool(comparisons["paper_member_value_from_registry"]),
            f"paper={expected['members_at_detection']}; "
            f"registry snapshot={actual['registry_total_tweets_at_normalization']}",
        ),
        Check(
            "Worked-example member-count temporal consistency",
            bool(comparisons["detection_timestamp_member_count"]),
            f"paper={expected['members_at_detection']}; "
            f"registry snapshot={actual['registry_total_tweets_at_normalization']}; "
            f"time-series through detection={actual['timeseries_members_through_detection']}",
        ),
    ]


def verify_paper_protocol_consistency(repo_root: Path) -> list[Check]:
    """Compare retained executable protocols with numerical method claims."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:44 PDT | Reason: distinguish exact table arithmetic from
    # method, split, timing, and seed statements that use different protocols.
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:31 PDT | Reason: verify the retained CT24 calibration shift
    # and CUDA throughput trace instead of leaving both numerical claims inferred.
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:36 PDT | Reason: expose the evaluation-set threshold search
    # and the three-source published-baseline row as protocol/provenance mismatches.
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:46 PDT | Reason: compare the exact retained MTL schedule and
    # cross-benchmark Fusion seed scope with the manuscript descriptions.
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    protocol = manifest["protocol_consistency"]
    pipeline = manifest["pipeline_narrative_values"]
    split = load_json(
        repo_root / "results/virality_split_manifest_2026-05-12.json"
    )["split_method"]
    anomaly = load_json(
        repo_root / "results/anomaly_table_reproduction_2026-08-21/summary.json"
    )
    parameters = load_json(
        repo_root / "reproducibility/cikm2026/MODEL_PARAMETERS.json"
    )
    feature_source = (
        repo_root
        / "psr/reproducibility/source_artifacts/virality/generate_enhanced_features.py"
    ).read_text(encoding="utf-8")

    stream = pipeline["stream"]
    stream_matches_paper_days = (
        int(stream["hourly_windows"]) == 24 * 24
        and int(stream["calendar_dates_inclusive"]) == 24
    )
    canonical = pipeline["canonical_pipeline"]
    release = pipeline["released_virality_artifact"]
    pipeline_units_match = (
        int(canonical["anomaly_detections"]) == 529
        and int(canonical["normalized_claims"]) == 535
        and int(release["mapped_unique_canonical_claims"]) == 535
    )

    configuration = anomaly["configuration_provenance"]
    anomaly_parameters_match = (
        configuration["method_paragraph"] == configuration["retained_table_run"]
        and configuration["method_paragraph"]
        == protocol["anomaly_expose_parameters"]["paper_method"]
    )
    rate_boundary = anomaly["detection_rate_inconsistency"]
    anomaly_rate_definition_matches = (
        rate_boundary["expose_artifact_formula"]
        == rate_boundary["baseline_artifact_formula"]
        == "strict_pre_peak"
    )

    uses_all_pre_detection_history = (
        '(pl.col("timestamp") <= det_time)' in feature_source
        and '(pl.col("created_at") <= det_time)' in feature_source
    )
    fixed_six_hour_window = (
        "timedelta(hours=6)" in feature_source
        or "timedelta(hours = 6)" in feature_source
    )
    feature_window_matches = uses_all_pre_detection_history and fixed_six_hour_window
    split_matches = split["stratify"] is not None
    retained_seed = int(parameters["four_head_deberta"]["seed"])
    seed_matches = [retained_seed] == protocol["four_head_training_seed"]["paper_value"]
    retained_training = parameters["four_head_deberta"]
    paper_training = protocol["four_head_training_schedule"]["paper_value"]
    retained_phase_epochs = [
        int(phase["epochs"]) for phase in retained_training["phase_sequence"]
    ]
    retained_effective_batch = int(retained_training["batch_size"]) * int(
        retained_training["gradient_accumulation_steps"]
    )
    training_schedule_matches = (
        retained_phase_epochs == [int(paper_training["epochs"])]
        and retained_effective_batch in {16, 32}
        and math.isclose(
            float(retained_training["learning_rate"]),
            float(paper_training["learning_rate"]),
        )
        and retained_training["scheduler"] == "cosine_annealing"
    )
    fusion_audit = load_json(
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/fusion_classifier/FUSION_RECOVERY_AUDIT.json"
    )
    fusion_loaded_seeds = [
        int(value) for value in fusion_audit["historical_run"]["loaded_seeds"]
    ]
    fusion_seed_scope_matches = len(fusion_loaded_seeds) == 3
    embedding_metadata = load_json(
        repo_root / "reproducibility/cikm2026/cluster_embeddings.metadata.json"
    )
    embedding_dimension_provenance_matches = (
        embedding_metadata["paper_dimensions"] is None
        and int(embedding_metadata["retained_dimensions"]) == 384
    )
    llm_feature_summary = load_json(
        repo_root / "results/llm_features_table_reproduction_2026-08-21.json"
    )
    llm_feature_method = llm_feature_summary["method"]
    llm_feature_threshold_matches = llm_feature_method[
        "reported_threshold_selection"
    ].startswith("one threshold selected on CT24 development")
    published_baseline = load_json(
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/published_baseline/PUBLISHED_BASELINE_AUDIT.json"
    )
    published_baseline_row_matches = published_baseline["status"] != "MISMATCH"

    frontier_statuses = [
        item["status"]
        for item in protocol["frontier_llm_side_claims"].values()
    ]
    devtest_audit = load_json(
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/benchmarks/ct24_devtest_metadata.json"
    )
    devtest_evidence_matches = (
        devtest_audit["official_source"]["archive_sha256"]
        == "e1846a6d26615b15278b4286ccbdfa22c41ad8e2c198786bf037f4feb9758a07"
        and devtest_audit["official_source"]["file_sha256"]
        == "aaf9e9427545c1213465e8857da00fb9fe4d4a7c337bab2ae0a75150fa8c01f2"
        and int(devtest_audit["observed_file"]["data_rows"]) == 318
        and int(devtest_audit["paper_value"]["dev_test_rows"]) == 317
        and int(
            devtest_audit["retained_four_head_training_selection"]["rows"]
        )
        == 1031
        and devtest_audit["status"] == "MISMATCH"
    )
    latency_claims = protocol["latency_side_claims"]
    psr_latency = load_json(
        repo_root / "results/psr_latency_benchmark_2026-08-21.json"
    )
    psr_latency_models = psr_latency["models"]
    psr_latency_direct = (
        psr_latency["verdict"] == "PASS"
        and set(psr_latency_models) == {"BayesianRidge", "SVR (RBF)"}
        and all(
            float(result["mean_ms"]) < 1.0
            and float(result["p95_ms"]) < 1.0
            and result["status"] == "PASS"
            for result in psr_latency_models.values()
        )
    )
    calibration = load_json(
        repo_root / "results/mtl_threshold_calibration_audit_2026-08-21.json"
    )
    calibration_observed = calibration["observed"]
    calibration_magnitude_direct = (
        calibration["verdict"] == "PARTIAL"
        and calibration["checks"]["retained_sweep_recomputed"] == "PASS"
        and calibration["checks"]["approximately_0_03_magnitude"] == "PASS"
        and math.isclose(
            float(calibration_observed["development_selected_threshold"]),
            0.60,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(calibration_observed["test_optimized_threshold"]),
            0.50,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(calibration_observed["signed_f1_shift"]),
            -0.028735632183908066,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    mtl_latency = load_json(
        repo_root / "results/mtl_gpu_latency_audit_2026-08-21.json"
    )
    mtl_latency_benchmarks = mtl_latency["benchmarks"]
    mtl_latency_direct = (
        mtl_latency["verdict"] == "PASS"
        and mtl_latency["device"] == "cuda"
        and mtl_latency["source"]["default_batch_size_used"] is True
        and set(mtl_latency_benchmarks) == {"ClaimBuster", "CT23"}
        and all(
            int(result["batch_size"]) == 32
            and float(result["batch_amortized_ms_per_claim"]) < 10.0
            and result["status"] == "PASS"
            for result in mtl_latency_benchmarks.values()
        )
    )
    claim_prose = manifest["claim_normalization_prose_values"]
    claim_prose_statuses = [
        claim_prose["pure_retrieval_meteor"]["status"],
        claim_prose["fine_tuned_model_a_meteor"]["status"],
    ]
    run_record = (
        repo_root / "results/ct25_claim_normalization_lambda_2026-05-15/RUN.md"
    ).read_text(encoding="utf-8")
    training_rows_verified = (
        int(claim_prose["training_rows"]["paper_value"]) == 11_374
        and "`train-eng.csv` | 11374" in run_record
    )
    return [
        Check(
            "Paper stream-duration consistency",
            stream_matches_paper_days,
            f"paper=24 contiguous days; artifact={stream['hourly_windows']} hours over "
            f"{stream['calendar_dates_inclusive']} calendar dates",
        ),
        Check(
            "Paper pipeline-unit consistency",
            pipeline_units_match,
            f"canonical anomalies={canonical['anomaly_detections']}; mappings="
            f"{canonical['cluster_to_claim_mappings']}; claims={canonical['normalized_claims']}; "
            f"release clusters={release['cluster_instances']}; release claims="
            f"{release['mapped_unique_canonical_claims']}",
        ),
        Check(
            "Anomaly method-to-table parameter consistency",
            anomaly_parameters_match,
            f"method={configuration['method_paragraph']}; table={configuration['retained_table_run']}",
        ),
        Check(
            "Anomaly detection-rate definition consistency",
            anomaly_rate_definition_matches,
            f"EXPoSE={rate_boundary['expose_artifact_formula']}; "
            f"baselines={rate_boundary['baseline_artifact_formula']}",
        ),
        Check(
            "Virality feature observation-window consistency",
            feature_window_matches,
            f"all pre-detection history={uses_all_pre_detection_history}; "
            f"fixed six-hour window={fixed_six_hour_window}",
        ),
        Check(
            "Virality split stratification consistency",
            split_matches,
            f"paper=stratified; retained split stratify={split['stratify']}",
        ),
        Check(
            "LLM-feature threshold-selection consistency",
            llm_feature_threshold_matches,
            f"paper={protocol['llm_feature_threshold_selection']['paper_value']}; "
            f"artifact={llm_feature_method['reported_threshold_selection']}; "
            f"diagnostic={protocol['llm_feature_threshold_selection']['development_threshold_diagnostic']}",
        ),
        Check(
            "Published-baseline row provenance consistency",
            published_baseline_row_matches,
            "; ".join(
                f"{name}={row['value']:.3f} from {row['source_method']}"
                for name, row in published_baseline["cell_provenance"].items()
            ),
        ),
        Check(
            "Four-head training-seed consistency",
            seed_matches,
            f"paper={protocol['four_head_training_seed']['paper_value']}; retained={[retained_seed]}",
        ),
        Check(
            "Four-head training-schedule consistency",
            training_schedule_matches,
            f"paper={paper_training}; retained phase epochs={retained_phase_epochs}, "
            f"effective batch={retained_effective_batch}, encoder LR="
            f"{retained_training['learning_rate']}, aux LR="
            f"{retained_training['aux_learning_rate']}, scheduler="
            f"{retained_training['scheduler']}",
        ),
        Check(
            "Fusion ensemble seed-scope consistency",
            fusion_seed_scope_matches,
            "paper=three-seed ensemble; CT24 retained vector=three-seed; "
            f"historical ClaimBuster/CT23 loaded seeds={fusion_loaded_seeds}",
        ),
        Check(
            "Cluster-embedding final-paper provenance",
            embedding_dimension_provenance_matches,
            f"paper dimension stated={embedding_metadata['paper_dimensions']}; retained="
            f"{embedding_metadata['retained_dimensions']}; rows="
            f"{embedding_metadata['rows']}; bytes={embedding_metadata['bytes']}",
        ),
        Check(
            "CT24 dev-test row-count evidence",
            devtest_evidence_matches,
            f"official data rows={devtest_audit['observed_file']['data_rows']}; "
            f"paper={devtest_audit['paper_value']['dev_test_rows']}; retained "
            "four-head selection rows="
            f"{devtest_audit['retained_four_head_training_selection']['rows']}",
        ),
        Check(
            "Check-worthiness side-claim completeness",
            all(
                status.startswith(("reproduced", "verified"))
                for status in frontier_statuses
            ),
            "; ".join(frontier_statuses),
        ),
        Check(
            "Four-head threshold-calibration magnitude",
            calibration_magnitude_direct,
            f"development threshold={calibration_observed['development_selected_threshold']:.2f}; "
            f"test-optimal threshold={calibration_observed['test_optimized_threshold']:.2f}; "
            f"test F1 shift={calibration_observed['signed_f1_shift']:+.12f}; "
            f"selection={calibration['checks']['selection_protocol']}",
        ),
        Check(
            "Claim-normalization prose-value completeness",
            training_rows_verified
            and not any(status.startswith("unresolved") for status in claim_prose_statuses),
            f"training rows verified={training_rows_verified}; "
            f"pure retrieval={claim_prose['pure_retrieval_meteor']['status']}; "
            f"fine-tuned Model A={claim_prose['fine_tuned_model_a_meteor']['status']}",
        ),
        Check(
            "PSR single-item latency evidence",
            psr_latency_direct,
            f"scope={latency_claims['psr_prediction']['scope']}; "
            + "; ".join(
                f"{name}: mean={result['mean_ms']:.4f} ms, p95={result['p95_ms']:.4f} ms"
                for name, result in psr_latency_models.items()
            ),
        ),
        Check(
            "Four-head GPU latency evidence",
            mtl_latency_direct,
            f"scope={latency_claims['four_head_gpu']['scope']}; "
            + "; ".join(
                f"{name}: {result['batch_amortized_ms_per_claim']:.4f} ms/claim"
                for name, result in mtl_latency_benchmarks.items()
            ),
        ),
    ]


def rounded_metrics(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        round(float(row[key]), 3)
        for key in ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")
    )


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    smaller = min(discordant_a, discordant_b)
    one_tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * one_tail)


def verify_result_artifacts(repo_root: Path) -> list[Check]:
    # CAMERA-READY ARTIFACT CORRECTION | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:58 PDT | Reason: verify the RandomForest numeric cells from
    # their actual retained/fresh sources while keeping the cross-run row
    # inconsistency as a separate failing scientific check.
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "virality"
    tuned = {
        row["name"]: row for row in load_json(artifact_root / "tuned_baselines.json")
    }
    retained = load_json(
        repo_root / "psr" / "reproduced_missing_predictions" / "manifest.json"
    )["metrics"]
    retained_tabular = load_json(
        repo_root / "psr" / "baseline_predictions" / "predictions_manifest.json"
    )["metrics"]
    paper_manifest = load_json(
        repo_root / "reproducibility" / "cikm2026" / "PAPER_VALUE_MANIFEST.json"
    )
    paper_table = paper_manifest["tables"]["virality_prediction"]
    table_text = (artifact_root / "latex_table.tex").read_text(encoding="utf-8")

    retained_names = {
        "LSTM": "LSTM",
        "GRU": "GRU",
        "Hawkes (HIP)": "HIP_Hawkes",
        "BERTweet": "BERTweet",
        "BERTweet + Features": "BERTweet_plus_features",
    }

    checks: list[Check] = []
    metric_keys = ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 18:31 PDT | Reason: verify the current paper values rather than
    # silently accepting the superseded LSTM and GRU rows.
    for name, paper_row in paper_table["rows"].items():
        expected_metrics = tuple(float(value) for value in paper_row["metrics"])
        if name in retained_names:
            source_row = retained.get(retained_names[name], {})
        else:
            source_row = tuned.get(name, {})
        if name == "RandomForest" and all(key in source_row for key in metric_keys):
            actual = (
                round(float(retained_tabular[name]["spearman_rho"]), 3),
                *(round(float(source_row[key]), 3) for key in metric_keys[1:]),
            )
            evidence_mode = "retained Spearman plus five fresh-fit cells"
        else:
            actual = (
                tuple(round(float(source_row[key]), 3) for key in metric_keys)
                if all(key in source_row for key in metric_keys)
                else ()
            )
            evidence_mode = "single source"
        rendered_line = next(
            (line for line in table_text.splitlines() if f"{name} &" in line),
            "",
        )
        normalized_line = rendered_line.replace(r"\textbf{", "").replace("}", "")
        try:
            rendered_metrics = tuple(
                float(part.strip().rstrip("\\").strip())
                for part in normalized_line.split("&")[1:7]
            )
        except ValueError:
            rendered_metrics = ()
        checks.append(
            Check(
                f"Result artifact: {name}",
                actual == expected_metrics and rendered_metrics == expected_metrics,
                f"rounded evidence={actual}; paper={expected_metrics}; "
                f"rendered={rendered_metrics}; mode={evidence_mode}",
            )
        )

    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:23 PDT | Reason: independently recompute the exact paired
    # tests and expose whether they use the same fresh predictions as Table 3.
    reproduction = load_json(
        repo_root / "results" / "virality_tabular_reproduction_2026-08-21.json"
    )
    test_targets = np.asarray(
        np.load(repo_root / "psr" / "derived_arrays" / "test_targets.npy"),
        dtype=float,
    )
    retained_prediction_paths = {
        "SVR (RBF)": repo_root
        / "psr"
        / "baseline_predictions"
        / "SVR_RBF"
        / "test_predictions.npy",
        "RandomForest": repo_root
        / "psr"
        / "baseline_predictions"
        / "RandomForest"
        / "test_predictions.npy",
        "LightGBM": repo_root
        / "psr"
        / "baseline_predictions"
        / "LightGBM"
        / "test_predictions.npy",
    }
    retained_predictions = {
        name: np.asarray(np.load(path), dtype=float)
        for name, path in retained_prediction_paths.items()
    }
    result_tests = {
        (row["baseline_a"], row["baseline_b"]): row
        for row in reproduction["paper_paired_tests_from_retained_predictions"]
    }
    paired_details: list[str] = []
    paired_passed = True
    for expected in paper_table["paired_mcnemar_exact_at_065"]:
        name_a = expected["baseline_a"]
        name_b = expected["baseline_b"]
        truth = test_targets > 0.65
        correct_a = (retained_predictions[name_a] > 0.65) == truth
        correct_b = (retained_predictions[name_b] > 0.65) == truth
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        p_value = exact_mcnemar_p(a_only, b_only)
        result_row = result_tests[(name_a, name_b)]
        row_passed = (
            round(p_value, 2) == float(expected["paper_p_rounded_2"])
            and math.isclose(
                p_value,
                float(expected["retained_prediction_p_full_precision"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                p_value,
                float(result_row["p_value"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and result_row["status"] == "PASS"
        )
        paired_passed = paired_passed and row_passed
        paired_details.append(f"{name_a}/{name_b}: discordant={a_only}/{b_only}, p={p_value:.6g}")
    checks.append(
        Check(
            "Virality retained paired-test reproduction",
            paired_passed,
            "; ".join(paired_details),
        )
    )

    paper_row_mismatches = [
        name
        for name, row in reproduction["paper_table_comparison"].items()
        if row["status"] != "PASS"
    ]
    fresh_test_mismatches = [
        f"{row['baseline_a']}/{row['baseline_b']}"
        for row in reproduction["fresh_single_run_consistency_diagnostic"]
        if row["status"] != "PASS"
    ]
    checks.append(
        Check(
            "Virality single-run statistical consistency",
            not paper_row_mismatches and not fresh_test_mismatches,
            f"table mismatches={paper_row_mismatches}; paired-test mismatches={fresh_test_mismatches}",
        )
    )
    return checks


def verify_virality_statistics(repo_root: Path) -> list[Check]:
    """Verify the freshly recomputed Table 3 statistical annotations."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 21:40 PDT | Reason: bind every Table 3 significance mark,
    # bootstrap interval, and exact paired test to retained per-example arrays.
    result = load_json(
        repo_root / "results/virality_statistics_reproduction_2026-08-21.json"
    )
    paper = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )["tables"]["virality_prediction"]
    annotations = paper["statistical_annotations"]
    rows = result["rows"]
    expected_codes = annotations["spearman_significance_codes"]
    expected_intervals = annotations["spearman_ci95_rounded_2"]

    row_names_match = set(rows) == set(expected_codes) == set(expected_intervals)
    row_details: list[str] = []
    rows_pass = row_names_match
    if row_names_match:
        for name in expected_codes:
            row = rows[name]
            actual_interval = [round(float(value), 2) for value in row["bootstrap_ci95"]]
            expected_interval = [float(value) for value in expected_intervals[name]]
            passed = (
                row["status"] == "PASS"
                and row["significance_code"] == expected_codes[name]
                and actual_interval == expected_interval
                and all(row["checks"].values())
            )
            rows_pass = rows_pass and passed
            row_details.append(
                f"{name}: stars={row['significance_code'] or 'none'}, "
                f"CI95={actual_interval}"
            )

    expected_pairs = {
        (
            row["baseline_a"].replace("SVR (RBF)", "SVR_RBF"),
            row["baseline_b"],
        ): float(row["retained_prediction_p_full_precision"])
        for row in paper["paired_mcnemar_exact_at_065"]
    }
    actual_pairs = {
        (row["baseline_a"], row["baseline_b"]): row
        for row in result["paired_tests"]
    }
    pairs_pass = set(actual_pairs) == set(expected_pairs) and all(
        actual_pairs[names]["status"] == "PASS"
        and math.isclose(
            float(actual_pairs[names]["exact_p_value"]),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for names, expected in expected_pairs.items()
    )
    protocol = result["protocol"]
    protocol_pass = (
        int(protocol["bootstrap_resamples"])
        == int(annotations["bootstrap_resamples"])
        and int(protocol["bootstrap_seed"]) == int(annotations["bootstrap_seed"])
        and math.isclose(float(protocol["paired_threshold"]), 0.65)
        and result["summary"]["verdict"] == "PASS"
    )

    return [
        Check(
            "Virality Spearman significance and CI reproduction",
            rows_pass and protocol_pass,
            f"rows={len(rows)}/11; bootstrap={protocol['bootstrap_resamples']}; "
            f"seed={protocol['bootstrap_seed']}; " + "; ".join(row_details),
        ),
        Check(
            "Virality statistical paired-test reproduction",
            pairs_pass,
            "; ".join(
                f"{name_a}/{name_b}: p={row['exact_p_value']:.6g}"
                for (name_a, name_b), row in actual_pairs.items()
            ),
        ),
    ]


def verify_anomaly_reproduction(repo_root: Path) -> list[Check]:
    """Verify the paper-facing anomaly rows and their aggregate release boundary."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:24 PDT | Reason: make the fresh anomaly-table reproduction
    # part of the normal public-package verification path.
    artifact_root = repo_root / "reproducibility" / "source_artifacts" / "anomaly"
    input_path = artifact_root / "timeseries.parquet"
    metadata = load_json(artifact_root / "timeseries.metadata.json")
    summary = load_json(
        repo_root / "results" / "anomaly_table_reproduction_2026-08-21" / "summary.json"
    )
    paper_manifest = load_json(
        repo_root / "reproducibility" / "cikm2026" / "PAPER_VALUE_MANIFEST.json"
    )
    paper_table = paper_manifest["tables"]["anomaly_detection"]

    expected_schema = ["cluster_id", "hour_offset", "tweet_count", "engagement"]
    schema = pq.read_schema(input_path).names
    table = pq.read_table(input_path, columns=expected_schema)
    input_checks = Check(
        "Anomaly aggregate input",
        metadata["artifact"] == input_path.name
        and metadata["output_sha256"] == sha256_file(input_path)
        and summary["input_sha256"] == sha256_file(input_path)
        and schema == expected_schema
        and table.num_rows == 404_296,
        f"file={metadata['artifact']}; rows={table.num_rows}; columns={schema}; "
        f"sha256={sha256_file(input_path)}",
    )

    checks = [input_checks]
    actual_paper_rows = summary.get("paper_table_rows", [])
    expected_paper_rows = list(paper_table["rows"])
    checks.append(
        Check(
            "Anomaly paper-row boundary",
            summary.get("status") == "PASS"
            and actual_paper_rows == expected_paper_rows
            and summary.get("additional_sanity_baselines") == ["Random"],
            f"paper rows={actual_paper_rows}; additional={summary.get('additional_sanity_baselines')}",
        )
    )
    for name, paper_row in paper_table["rows"].items():
        best = summary["results"][name]["best"]
        historical_rate = summary["historical_comparisons"][name][
            "historical_reported_detection_rate"
        ]
        actual = (
            round(float(best["nab_score"]), 1),
            round(float(historical_rate), 1),
            round(float(best["median_lead"]), 1),
        )
        expected = tuple(float(value) for value in paper_row["metrics"])
        checks.append(
            Check(
                f"Anomaly result artifact: {name}",
                actual == expected,
                f"fresh rounded={actual}; paper={expected}",
            )
        )
    return checks


def binary_f1(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    predictions = probs >= threshold
    positives = labels.astype(bool)
    true_positives = int(np.sum(predictions & positives))
    false_positives = int(np.sum(predictions & ~positives))
    false_negatives = int(np.sum(~predictions & positives))
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0


def verify_mtl_table_reproduction(repo_root: Path) -> list[Check]:
    """Recompute the three four-head F1 cells from retained probabilities."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:43 PDT | Reason: verify the CT24, ClaimBuster, and CT23
    # cells directly instead of trusting a historical text summary.
    run_root = repo_root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    benchmark_root = run_root / "benchmark_predictions"
    arrays = {
        "CT24": (
            np.load(run_root / "test_labels.npy"),
            np.load(run_root / "test_probs.npy"),
            [0.50],
        ),
        "ClaimBuster": (
            np.load(benchmark_root / "claimbuster_labels.npy"),
            np.load(benchmark_root / "claimbuster_probs.npy"),
            list(np.arange(0.30, 0.80, 0.05)),
        ),
        "CT23": (
            np.load(benchmark_root / "ct23_labels.npy"),
            np.load(benchmark_root / "ct23_probs.npy"),
            list(np.arange(0.30, 0.80, 0.05)),
        ),
    }
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    paper_row = manifest["tables"]["claim_checkworthiness"]["rows"][
        "4 Parallel Heads"
    ]
    paper_values = dict(
        zip(("CT24", "ClaimBuster", "CT23"), paper_row["metrics"], strict=True)
    )
    summary = load_json(repo_root / "results/mtl_table_reproduction_2026-08-21.json")
    checkpoint_summary = load_json(
        repo_root / "results/mtl_table_checkpoint_inference_2026-08-21.json"
    )

    checks: list[Check] = []
    for name, (labels, probs, thresholds) in arrays.items():
        candidates = [(binary_f1(labels, probs, float(value)), float(value)) for value in thresholds]
        actual_f1, threshold = max(candidates, key=lambda item: item[0])
        expected = float(paper_values[name])
        summary_f1 = float(summary["results"][name]["f1"])
        checkpoint_f1 = float(checkpoint_summary["results"][name]["f1"])
        checks.append(
            Check(
                f"Four-head result artifact: {name}",
                len(labels) == len(probs)
                and round(actual_f1, 3) == expected
                and math.isclose(actual_f1, summary_f1, rel_tol=0.0, abs_tol=1e-15)
                and math.isclose(
                    actual_f1,
                    checkpoint_f1,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                f"n={len(labels)}; F1={actual_f1:.12f}; threshold={threshold:.2f}; "
                f"paper={expected:.3f}",
            )
        )
    checks.append(
        Check(
            "Four-head reproduction summary",
            summary["status"] == "PASS"
            and summary["inference"] == "committed prediction arrays"
            and summary["paper_f1"] == paper_values,
            f"status={summary['status']}; inference={summary['inference']}",
        )
    )
    checks.append(
        Check(
            "Four-head checkpoint-inference summary",
            checkpoint_summary["status"] == "PASS"
            and checkpoint_summary["inference"].startswith(
                "fresh checkpoint inference on "
            )
            and checkpoint_summary["paper_f1"] == paper_values,
            f"status={checkpoint_summary['status']}; "
            f"inference={checkpoint_summary['inference']}",
        )
    )
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:51 PDT | Reason: bind the paper's training counts and class
    # ratio to the retained log rather than only repeating them in prose.
    training_log = (run_root / "training.log").read_text(encoding="utf-8")
    counts = load_json(
        repo_root / "reproducibility/cikm2026/MODEL_PARAMETERS.json"
    )["four_head_deberta"]["training_data_counts"]
    counts_verified = (
        counts["train_total"] == 22402
        and counts["train_negative"] == 17003
        and counts["train_positive"] == 5399
        and counts["dev_rows"] == 1031
        and counts["test_rows"] == 341
        and math.isclose(
            float(counts["positive_class_weight_exact"]),
            17003 / 5399,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and "Train: 22402 samples" in training_log
        and "Dev: 1031 samples" in training_log
        and "Class imbalance: 17003:5399 (weight: 3.15)" in training_log
    )
    checks.append(
        Check(
            "Four-head training-count provenance",
            counts_verified,
            "train=22,402 (17,003/5,399); dev=1,031; test=341; class weight=3.15",
        )
    )
    return checks


def verify_llm_feature_reproduction(repo_root: Path) -> list[Check]:
    """Verify the text-free bundle and the three LLM-feature F1 cells."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:21 PDT | Reason: bind the fresh LLM-feature fit to its
    # numerical inputs and independently recompute its three reported F1 cells.
    # CAMERA-READY ARTIFACT CLARIFICATION | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:27 PDT | Reason: verify the retained threshold rule and the
    # single CT24-development-threshold transfer diagnostic separately.
    artifact_root = (
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/llm_features_classifier"
    )
    matrices = np.load(artifact_root / "matrices.npz", allow_pickle=False)
    references = np.load(
        artifact_root / "reference_predictions.npz",
        allow_pickle=False,
    )
    summary = load_json(
        repo_root / "results/llm_features_table_reproduction_2026-08-21.json"
    )
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    paper_row = manifest["tables"]["claim_checkworthiness"]["rows"]["LLM features"]
    paper_values = dict(
        zip(("CT24", "ClaimBuster", "CT23"), paper_row["metrics"], strict=True)
    )

    expected_shapes = {
        "X_train": (22_402, 153),
        "y_train": (22_402,),
        "X_dev": (1_031, 153),
        "y_dev": (1_031,),
        "X_ct24": (341, 153),
        "y_ct24": (341,),
        "X_claimbuster": (1_032, 153),
        "y_claimbuster": (1_032,),
        "X_ct23": (318, 153),
        "y_ct23": (318,),
        "feature_names": (153,),
    }
    actual_shapes = {name: tuple(matrices[name].shape) for name in matrices.files}
    feature_names = matrices["feature_names"].tolist()
    restricted_names = {
        "post",
        "post_id",
        "sentence_id",
        "text",
        "tweet_id",
        "user_id",
        "username",
    }
    boundary_hits = sorted(restricted_names.intersection(feature_names))
    checks = [
        Check(
            "LLM-feature numerical bundle",
            actual_shapes == expected_shapes
            and all(
                matrices[name].dtype.kind in "fiu"
                for name in matrices.files
                if name != "feature_names"
            ),
            f"arrays={len(matrices.files)}; shapes match={actual_shapes == expected_shapes}; "
            "153 numeric features",
        ),
        Check(
            "LLM-feature data boundary",
            not boundary_hits
            and matrices["feature_names"].dtype.kind in "SU"
            and len(feature_names) == 153,
            f"raw text or identifier fields={boundary_hits or 'none'}",
        ),
    ]
    key_by_name = {
        "CT24": "ct24",
        "ClaimBuster": "claimbuster",
        "CT23": "ct23",
    }
    for name, key in key_by_name.items():
        labels = matrices[f"y_{key}"]
        probabilities = references[key]
        threshold = float(summary["results"][name]["threshold"])
        actual_f1 = binary_f1(labels, probabilities, threshold)
        expected = float(paper_values[name])
        checks.append(
            Check(
                f"LLM-feature result artifact: {name}",
                len(labels) == len(probabilities)
                and round(actual_f1, 3) == expected
                and math.isclose(
                    actual_f1,
                    float(summary["results"][name]["f1"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and summary["results"][name]["reference_decisions_match"],
                f"n={len(labels)}; F1={actual_f1:.12f}; "
                f"threshold={threshold:.2f}; paper={expected:.3f}",
            )
        )
    diagnostic = summary["development_selected_threshold_diagnostic"]
    diagnostic_threshold = float(diagnostic["threshold"])
    expected_fixed_f1 = {
        "CT24": 0.6871165644171779,
        "ClaimBuster": 0.8814229249011858,
        "CT23": 0.8019323671497585,
    }
    diagnostic_matches = True
    for name, key in key_by_name.items():
        fixed_f1 = binary_f1(
            matrices[f"y_{key}"], references[key], diagnostic_threshold
        )
        recorded_f1 = float(diagnostic["results"][name]["f1"])
        diagnostic_matches &= math.isclose(
            fixed_f1, expected_fixed_f1[name], rel_tol=0.0, abs_tol=1e-15
        ) and math.isclose(
            fixed_f1, recorded_f1, rel_tol=0.0, abs_tol=1e-15
        )
    checks.append(
        Check(
            "LLM-feature fixed development-threshold diagnostic",
            diagnostic["selection_split"] == "CT24 development"
            and round(diagnostic_threshold, 2) == 0.65
            and diagnostic_matches,
            f"selection={diagnostic['selection_split']}; "
            f"threshold={diagnostic_threshold:.2f}; fixed F1 values match",
        )
    )
    checks.append(
        Check(
            "LLM-feature fresh-fit summary",
            summary["status"] == "PASS"
            and summary["feature_partition"]
            == {"text": 35, "pca64": 64, "llm": 54}
            and paper_row["reproduction_command"]
            == "uv run scripts/reproduce_cikm2026_llm_features.py"
            and paper_row["reported_threshold_selection"]
            == summary["method"]["reported_threshold_selection"],
            f"status={summary['status']}; partition={summary['feature_partition']}",
        )
    )
    corrected = summary["corrected_sentence_id_replication"]
    corrected_alignment = corrected["alignment_audit"]
    corrected_expected = {
        "CT24": {"fixed": 0.7012987012987013, "oracle": 0.7457627118644068},
        "ClaimBuster": {
            "fixed": 0.8274231678486997,
            "oracle": 0.8865979381443299,
        },
        "CT23": {"fixed": 0.6086956521739131, "oracle": 0.7860696517412935},
    }
    corrected_values_match = all(
        math.isclose(
            float(corrected["results"][name]["fixed_source_development_threshold"]["f1"]),
            expected["fixed"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(corrected["results"][name]["evaluation_oracle_diagnostic"]["f1"]),
            expected["oracle"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for name, expected in corrected_expected.items()
    )
    alignment_complete = (
        set(corrected_alignment)
        == {"CT24 train", "CT24 dev", "CT24 test", "ClaimBuster", "CT23"}
        and all(
            row["one_to_one_id_set_match"]
            and row["rows"] == row["unique_sentence_ids"]
            and row["source_feature_order_differed"]
            for row in corrected_alignment.values()
        )
    )
    checks.append(
        Check(
            "LLM-feature sentence-ID-aligned replication",
            corrected["status"] == "COMPLETE"
            and round(float(corrected["development_threshold"]), 2) == 0.70
            and alignment_complete
            and corrected_values_match,
            "five complete one-to-one ID joins; corrected fixed and diagnostic "
            f"values verified={corrected_values_match}",
        )
    )
    return checks


def verify_public_encoder_fusion_replication(repo_root: Path) -> list[Check]:
    """Verify the public-checkpoint predictions and ID-aligned Fusion audit."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-25 14:18 WEST | Reason: bind direct public-model inference and
    # corrected cross-benchmark joins without treating a fresh mismatch as the
    # historical paper run.
    result_root = repo_root / "results/public_encoder_fusion_replication_2026-08-25"
    summary = load_json(result_root / "summary.json")
    protocol = load_json(
        repo_root / "reproducibility/cikm2026/checkworthiness/PAPER_PROTOCOL.json"
    )
    prediction_path = repo_root / summary["prediction_bundle"]["path"]
    with np.load(prediction_path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}

    required_keys = {
        "claimbuster_sentence_ids",
        "claimbuster_labels",
        "ct23_sentence_ids",
        "ct23_labels",
        *{
            f"{benchmark}_seed_{seed}_probabilities"
            for benchmark in ("claimbuster", "ct23")
            for seed in (0, 42, 456)
        },
        *{
            f"{benchmark}_{component}_probabilities"
            for benchmark in ("claimbuster", "ct23")
            for component in (
                "historical_xgboost",
                "corrected_xgboost",
                "historical_fusion",
                "corrected_fusion",
            )
        },
    }
    expected_rows = {"claimbuster": 1_032, "ct23": 318}
    arrays_complete = required_keys.issubset(arrays) and all(
        arrays[name].shape == (expected_rows[name.split("_")[0]],)
        for name in required_keys
    )
    forbidden_fragments = ("text", "user", "post", "tweet", "profile", "location")
    boundary_safe = not any(
        fragment in name.lower()
        for name in arrays
        for fragment in forbidden_fragments
    )
    identity_verified = all(
        model["model_sha256"] == model["observed_model_sha256"]
        and model["model_bytes"] == model["observed_model_bytes"]
        and len(model["revision"]) == 40
        for model in summary["model_specs"].values()
    )

    metric_checks: list[bool] = []
    for seed, benchmark_rows in summary["individual_public_encoder_models"].items():
        for benchmark_name, record in benchmark_rows.items():
            key = benchmark_name.lower()
            labels = arrays[f"{key}_labels"]
            probabilities = arrays[f"{key}_seed_{seed}_probabilities"]
            for result_name in ("fixed_threshold", "evaluation_oracle_diagnostic"):
                metric = record[result_name]
                metric_checks.append(
                    math.isclose(
                        binary_f1(labels, probabilities, float(metric["threshold"])),
                        float(metric["f1"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )

    fusion_keys = {
        "historical_positional_two_seed": "historical_fusion",
        "corrected_id_aligned_three_seed": "corrected_fusion",
    }
    for result_name, array_component in fusion_keys.items():
        for benchmark_name, record in summary["fusion"][result_name].items():
            key = benchmark_name.lower()
            labels = arrays[f"{key}_labels"]
            probabilities = arrays[f"{key}_{array_component}_probabilities"]
            for metric_name in ("fixed_threshold", "evaluation_oracle_diagnostic"):
                metric = record[metric_name]
                metric_checks.append(
                    math.isclose(
                        binary_f1(labels, probabilities, float(metric["threshold"])),
                        float(metric["f1"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )

    alignment = summary["sentence_id_alignment_audit"]
    alignment_verified = (
        set(alignment) == {"CT24 train", "CT24 dev", "ClaimBuster", "CT23"}
        and all(
            row["one_to_one_id_set_match"]
            and row["rows"] == row["unique_sentence_ids"]
            and row["source_feature_order_differed"]
            for row in alignment.values()
        )
    )
    paper_flags_truthful = (
        summary["individual_public_encoder_models"]["0"]["CT23"][
            "fixed_threshold_matches_paper_after_rounding"
        ]
        is True
        and summary["individual_public_encoder_models"]["0"]["ClaimBuster"][
            "oracle_matches_paper_after_rounding"
        ]
        is False
        and all(
            row["fixed_threshold_matches_paper_after_rounding"] is False
            and row["oracle_matches_paper_after_rounding"] is False
            for path in summary["fusion"].values()
            for row in path.values()
        )
        and protocol["paper_reported_values"]["table_5"]["Encoder Only"][
            "ClaimBuster"
        ]
        == 0.970
    )

    return [
        Check(
            "Public Encoder/Fusion prediction bundle",
            arrays_complete
            and boundary_safe
            and sha256_file(prediction_path)
            == summary["prediction_bundle"]["sha256"],
            f"arrays={len(arrays)}; required complete={arrays_complete}; "
            f"data boundary safe={boundary_safe}",
        ),
        Check(
            "Public Encoder checkpoint identity",
            identity_verified,
            "three immutable revisions verified by observed model byte size and SHA-256",
        ),
        Check(
            "Public Encoder/Fusion metric integrity",
            all(metric_checks) and alignment_verified and paper_flags_truthful,
            f"metric records checked={len(metric_checks)}; "
            f"ID alignment verified={alignment_verified}; "
            f"paper-match flags truthful={paper_flags_truthful}",
        ),
    ]


def verify_encoder_only_reproduction(repo_root: Path) -> list[Check]:
    """Verify the direct cells and the unresolved ClaimBuster boundary."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:10 PDT | Reason: bind the two directly reproduced cells
    # while preventing aggregate arithmetic from standing in for a missing vector.
    checkworthiness_root = (
        repo_root / "reproducibility/source_artifacts/checkworthiness"
    )
    bundle_root = checkworthiness_root / "encoder_only"
    matrices = np.load(
        checkworthiness_root / "llm_features_classifier/matrices.npz",
        allow_pickle=False,
    )
    ct24_probabilities = np.load(
        bundle_root / "ct24_probs.npy", allow_pickle=False
    )
    ct23_probabilities = np.load(
        bundle_root / "ct23_probs.npy", allow_pickle=False
    )
    retained_ct24 = load_json(bundle_root / "retained_ct24_results.json")
    audit = load_json(bundle_root / "ENCODER_RECOVERY_AUDIT.json")
    summary = load_json(
        repo_root / "results/encoder_only_table_reproduction_2026-08-21.json"
    )
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    encoder_row = manifest["tables"]["claim_checkworthiness"]["rows"]["Encoder"]

    ct24_f1 = binary_f1(matrices["y_ct24"], ct24_probabilities, 0.50)
    ct23_f1 = binary_f1(matrices["y_ct23"], ct23_probabilities, 0.50)

    historical = audit["claimbuster_historical_record"]
    counts = historical["confusion_counts_reconstructed"]
    true_positive = int(counts["true_positive"])
    false_positive = int(counts["false_positive"])
    false_negative = int(counts["false_negative"])
    true_negative = int(counts["true_negative"])
    claimbuster_n = true_positive + false_positive + false_negative + true_negative
    claimbuster_f1 = 2 * true_positive / (
        2 * true_positive + false_positive + false_negative
    )
    claimbuster_accuracy = (true_positive + true_negative) / claimbuster_n
    claimbuster_path = (
        checkworthiness_root / "benchmarks/claim_buster/groundtruth.csv"
    )
    with claimbuster_path.open(encoding="utf-8", newline="") as handle:
        claimbuster_rows = list(csv.DictReader(handle))
    public_positive_labels = sum(
        int(row["Verdict"]) == 1 for row in claimbuster_rows
    )

    compatible_counts = 0
    positive_labels = int(historical["positive_labels"])
    for candidate_tp in range(positive_labels + 1):
        candidate_fn = positive_labels - candidate_tp
        for candidate_fp in range(claimbuster_n - positive_labels + 1):
            candidate_tn = claimbuster_n - positive_labels - candidate_fp
            denominator = 2 * candidate_tp + candidate_fp + candidate_fn
            candidate_f1 = 2 * candidate_tp / denominator if denominator else 0.0
            candidate_accuracy = (candidate_tp + candidate_tn) / claimbuster_n
            if (
                round(candidate_f1, 4) == historical["f1_reported"]
                and round(candidate_accuracy, 4)
                == historical["accuracy_reported"]
            ):
                compatible_counts += 1

    return [
        Check(
            "Encoder Only per-example numerical bundle",
            ct24_probabilities.shape == (341,)
            and ct23_probabilities.shape == (318,)
            and ct24_probabilities.dtype.kind == "f"
            and ct23_probabilities.dtype.kind == "f",
            f"CT24 n={len(ct24_probabilities)}; CT23 n={len(ct23_probabilities)}; "
            "floating-point probabilities only",
        ),
        Check(
            "Encoder Only direct CT24 and CT23 reproduction",
            round(ct24_f1, 3) == 0.821
            and round(ct23_f1, 3) == 0.928
            and math.isclose(
                ct24_f1,
                float(summary["results"]["CT24"]["metrics"]["f1"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                ct23_f1,
                float(summary["results"]["CT23"]["metrics"]["f1"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                ct24_f1,
                float(retained_ct24["best_test"]["f1"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"CT24 F1={ct24_f1:.12f}; CT23 F1={ct23_f1:.12f}",
        ),
        Check(
            "Encoder Only ClaimBuster arithmetic and evidence boundary",
            claimbuster_n == len(claimbuster_rows) == 1032
            and true_positive + false_negative
            == positive_labels
            == public_positive_labels
            == 238
            and compatible_counts == 1
            and round(claimbuster_f1, 4) == historical["f1_reported"] == 0.9702
            and round(claimbuster_accuracy, 4)
            == historical["accuracy_reported"]
            == 0.9864
            and round(claimbuster_f1, 3) == 0.970
            and not historical["per_example_probability_vector_retained"],
            f"public n={len(claimbuster_rows)}; unique compatible confusion "
            f"matrix={compatible_counts == 1}; F1={claimbuster_f1:.12f}; "
            f"direct reproduction={historical['direct_reproduction_status']}",
        ),
        Check(
            "Encoder Only ClaimBuster direct completeness",
            historical["direct_reproduction_status"] == "reproduced"
            and historical["per_example_probability_vector_retained"],
            "UNRESOLVED: the historical per-example ClaimBuster probability vector is absent",
        ),
        Check(
            "Encoder Only result and manifest linkage",
            summary["status"] == "PARTIAL"
            and summary["direct_cells_status"] == "PASS"
            and summary["claimbuster_arithmetic_status"] == "PASS"
            and not summary["results"]["ClaimBuster"]["direct_reproduction"]
            and encoder_row["reproduction_command"]
            == "uv run scripts/reproduce_cikm2026_encoder_only.py"
            and encoder_row["metrics"] == [0.821, 0.970, 0.928]
            and encoder_row["claimbuster_recovery_audit"]
            == "reproducibility/source_artifacts/checkworthiness/encoder_only/ENCODER_RECOVERY_AUDIT.json"
            and encoder_row["claimbuster_historical_extract"]
            == "reproducibility/source_artifacts/checkworthiness/encoder_only/HISTORICAL_RUN_EXTRACT.md"
            and not audit["release_boundary"]["benchmark_text_included"]
            and not audit["release_boundary"]["post_or_user_identifiers_included"],
            "status=PARTIAL; direct cells, unresolved ClaimBuster boundary, command, and privacy linked",
        ),
    ]


def verify_fusion_ct24_reproduction(repo_root: Path) -> list[Check]:
    """Verify the exact CT24 Fusion reconstruction and its evidence boundary."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 19:45 PDT | Reason: bind the supported CT24 Fusion cell to its
    # retained numeric components without treating later checkpoints as paper evidence.
    # CAMERA-READY ARTIFACT UPDATE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-25 14:24 WEST | Reason: verify the new public-checkpoint evidence
    # while preserving the missing exact historical cross-benchmark boundary.
    bundle_root = (
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/fusion_classifier"
    )
    matrices = np.load(
        repo_root
        / "reproducibility/source_artifacts/checkworthiness/"
        "llm_features_classifier/matrices.npz",
        allow_pickle=False,
    )
    encoder = np.load(bundle_root / "ct24_encoder_t03_probs.npy", allow_pickle=False)
    llm = np.load(
        bundle_root / "ct24_llm_xgboost_test_probs.npy", allow_pickle=False
    )
    retained = np.load(
        bundle_root / "ct24_fusion_test_probs.npy", allow_pickle=False
    )
    recomputed = 0.5 * encoder + 0.5 * llm
    max_delta = float(np.max(np.abs(recomputed - retained)))
    actual_f1 = binary_f1(matrices["y_ct24"], recomputed, 0.5)
    summary = load_json(
        repo_root / "results/fusion_ct24_reproduction_2026-08-21.json"
    )
    recovery_audit = load_json(bundle_root / "FUSION_RECOVERY_AUDIT.json")
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )
    fusion_row = manifest["tables"]["claim_checkworthiness"]["rows"]["Fusion"]
    return [
        Check(
            "Fusion CT24 component reconstruction",
            len(encoder) == len(llm) == len(retained) == 341
            and max_delta <= 3e-8,
            f"n={len(recomputed)}; max probability delta={max_delta:.3e}",
        ),
        Check(
            "Fusion CT24 result artifact",
            round(actual_f1, 3) == 0.836
            and math.isclose(
                actual_f1,
                float(summary["metrics"]["f1"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and summary["status"] == "PASS",
            f"F1={actual_f1:.12f}; paper=0.836; status={summary['status']}",
        ),
        Check(
            "Fusion cross-benchmark evidence boundary",
            fusion_row["ct24_reproduction_command"]
            == "uv run scripts/reproduce_cikm2026_fusion_ct24.py"
            and fusion_row["cross_benchmark_evidence_status"].startswith(
                "fresh public-checkpoint predictions"
            )
            and fusion_row["public_checkpoint_reproduction_command"]
            == "uv run scripts/reproduce_cikm2026_public_encoder_fusion.py --inference"
            and (
                repo_root / fusion_row["public_checkpoint_evidence"]
            ).is_file()
            and fusion_row["cross_benchmark_historical_extract"]
            == "reproducibility/source_artifacts/checkworthiness/fusion_classifier/HISTORICAL_RUN_EXTRACT.md"
            and recovery_audit["historical_run"]["console_metrics"]["ClaimBuster"][
                "fusion_f1"
            ]
            == 0.9605
            and recovery_audit["historical_run"]["console_metrics"]["CT23"][
                "fusion_f1"
            ]
            == 0.9151
            and not recovery_audit["candidate_checkpoint_diagnostic"][
                "matches_paper_values_after_rounding"
            ],
            "CT24 executable; checksum-bound recovery audit preserves the "
            "ClaimBuster/CT23 evidence boundary",
        ),
        Check(
            "Fusion cross-benchmark direct completeness",
            recovery_audit["release_boundary"]["checkpoint_weights_included"]
            and recovery_audit["candidate_checkpoint_diagnostic"][
                "matches_paper_values_after_rounding"
            ],
            "UNRESOLVED: the paper-run ClaimBuster and CT23 prediction arrays or checkpoint bytes are absent",
        ),
    ]


def verify_frontier_llm_ablation(repo_root: Path) -> list[Check]:
    """Verify the recoverable Opus aggregates and expose the missing raw run."""
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-21 20:18 PDT | Reason: distinguish uniquely recoverable aggregate
    # arithmetic from the unavailable 1,023-response historical inference log.
    audit = load_json(
        repo_root / "results/frontier_llm_ablation_audit_2026-08-21.json"
    )
    manifest = load_json(
        repo_root / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json"
    )["tables"]["frontier_llm_ablation"]
    confusion = audit["llm_aggregate_confusion_candidates"]
    paired = audit["paired_correctness_candidates"]
    agreement = audit["dimension_agreement"]
    cost = audit["paper_cost_and_latency"]
    return [
        Check(
            "Frontier-LLM aggregate arithmetic",
            audit["aggregate_arithmetic_status"] == "PASS"
            and len(confusion) == 1
            and {
                key: confusion[0][key]
                for key in (
                    "true_positive",
                    "false_positive",
                    "false_negative",
                    "true_negative",
                )
            }
            == {
                "true_positive": 65,
                "false_positive": 27,
                "false_negative": 23,
                "true_negative": 226,
            }
            and len(paired) == 1
            and paired[0]["llm_only_correct"] == 16
            and paired[0]["mtl_only_correct"] == 36
            and math.isclose(
                float(paired[0]["mcnemar_exact_p"]),
                0.0077874363057435225,
                rel_tol=0.0,
                abs_tol=1e-18,
            )
            and agreement["checkability"]["agreement_count_candidates"] == [281]
            and agreement["verifiability"]["agreement_count_candidates"] == [264]
            and agreement["harm"]["agreement_count_candidates"] == [309]
            and abs(
                float(cost["cost_from_rounded_6_16_total_per_1000_claims"])
                - float(
                    manifest["rows"]["Claude Opus 4.7 zero-shot"][
                        "cost_per_1000_claims_usd_approximate"
                    ]
                )
            )
            < 0.01,
            "unique confusion=(65,27,23,226); McNemar discordant=(16,36); "
            "dimension counts=(281,264,309); rounded cost arithmetic consistent",
        ),
        Check(
            "Frontier-LLM direct reproduction completeness",
            audit["release_boundary"]["api_response_log_retained"]
            and audit["release_boundary"]["per_sample_llm_predictions_retained"]
            and audit["release_boundary"]["direct_llm_metric_reproduction"],
            "UNRESOLVED: the historical 1,023 API responses and per-example Opus predictions are absent",
        ),
    ]


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
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=None,
        help="Optional exact final camera-ready PDF to verify.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional recovered cluster_embeddings.npy asset to verify.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable report path; relative paths use the repository root.",
    )
    return parser.parse_args()


def run_check_group(name: str, check_group: Any) -> list[Check]:
    try:
        result = check_group()
    except Exception as error:  # noqa: BLE001 - malformed public artifacts must produce a readable FAIL
        return [Check(f"{name} execution", False, f"{type(error).__name__}: {error}")]
    return [result] if isinstance(result, Check) else list(result)


def write_json_report(
    output_path: Path,
    repo_root: Path,
    args: argparse.Namespace,
    checks: list[Check],
) -> None:
    replacements = {str(repo_root): "<repository-root>"}
    for label, path in (
        ("paper-pdf", args.paper_pdf),
        ("checkpoint", args.checkpoint),
        ("cluster-embeddings", args.embeddings),
    ):
        if path is not None:
            replacements[str(path.resolve())] = f"<{label}>"

    def portable_detail(detail: str) -> str:
        for local_path, replacement in replacements.items():
            detail = detail.replace(local_path, replacement)
        return detail

    failure_counts: dict[str, int] = {}
    check_rows = []
    for check in checks:
        category = (
            "verified"
            if check.passed
            else KNOWN_FAILURE_CATEGORIES.get(
                check.name, "unexpected_artifact_integrity_failure"
            )
        )
        if not check.passed:
            failure_counts[category] = failure_counts.get(category, 0) + 1
        check_rows.append(
            {
                "category": category,
                "detail": portable_detail(check.detail),
                "name": check.name,
                "status": "PASS" if check.passed else "FAIL",
            }
        )

    failed = sum(not check.passed for check in checks)
    report = {
        "change_note": "Sérgio Pinto, 2026-08-21 22:32 PDT — Added a portable machine-readable verification result with explicit failure categories.",
        "schema_version": 1,
        "inputs": {
            "checkpoint": args.checkpoint.name if args.checkpoint else None,
            "cluster_embeddings": args.embeddings.name if args.embeddings else None,
            "paper_pdf": args.paper_pdf.name if args.paper_pdf else None,
        },
        "summary": {
            "failed": failed,
            "failure_categories": failure_counts,
            "passed": len(checks) - failed,
            "total": len(checks),
            "verdict": "PASS" if failed == 0 else "FAIL",
        },
        "checks": check_rows,
    }
    destination = output_path
    if not destination.is_absolute():
        destination = repo_root / destination
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


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
            "Virality prediction checksum verification",
            lambda: verify_prediction_checksums(repo_root),
        ),
        *run_check_group(
            "Check-worthiness checksum verification",
            lambda: verify_checkworthiness_checksums(repo_root),
        ),
        *run_check_group(
            "Repository data-boundary verification",
            lambda: verify_repository_data_boundary(repo_root),
        ),
        *run_check_group(
            "Artifact-release coverage verification",
            lambda: verify_artifact_release_coverage(repo_root),
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
        *run_check_group(
            "Camera-ready PDF verification",
            lambda: verify_paper_pdf_manifest(repo_root, args.paper_pdf),
        ),
        *run_check_group(
            "Cluster-embedding manifest verification",
            lambda: verify_cluster_embeddings_manifest(repo_root, args.embeddings),
        ),
        *run_check_group("Dataset verification", lambda: verify_dataset(repo_root)),
        *run_check_group(
            "Pipeline narrative-value verification",
            lambda: verify_pipeline_narrative_values(repo_root),
        ),
        *run_check_group(
            "Worked-example verification", lambda: verify_worked_example(repo_root)
        ),
        *run_check_group(
            "Paper protocol-consistency verification",
            lambda: verify_paper_protocol_consistency(repo_root),
        ),
        *run_check_group(
            "Result-artifact verification", lambda: verify_result_artifacts(repo_root)
        ),
        *run_check_group(
            "Virality-statistics verification",
            lambda: verify_virality_statistics(repo_root),
        ),
        *run_check_group(
            "Anomaly-table verification", lambda: verify_anomaly_reproduction(repo_root)
        ),
        *run_check_group(
            "Four-head table verification",
            lambda: verify_mtl_table_reproduction(repo_root),
        ),
        *run_check_group(
            "Encoder-only table verification",
            lambda: verify_encoder_only_reproduction(repo_root),
        ),
        *run_check_group(
            "LLM-feature table verification",
            lambda: verify_llm_feature_reproduction(repo_root),
        ),
        *run_check_group(
            "Public Encoder/Fusion verification",
            lambda: verify_public_encoder_fusion_replication(repo_root),
        ),
        *run_check_group(
            "Fusion CT24 verification",
            lambda: verify_fusion_ct24_reproduction(repo_root),
        ),
        *run_check_group(
            "Frontier-LLM ablation verification",
            lambda: verify_frontier_llm_ablation(repo_root),
        ),
        *run_check_group(
            "CT24 shared-error verification",
            lambda: verify_shared_error_analysis(repo_root),
        ),
    ]

    print("CIKM 2026 public artifact verification")
    print(f"Repository root: {repo_root}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    failures = [check for check in checks if not check.passed]
    if args.json_output is not None:
        write_json_report(args.json_output, repo_root, args, checks)
    print(f"\nSummary: {len(checks) - len(failures)} passed, {len(failures)} failed")
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
