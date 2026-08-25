# Artifact change — Sérgio Pinto, 2026-08-25 13:41 WEST.
# Reason: prevent the recovered check-worthiness evidence from regressing to
# positional joins, mutable model revisions, incomplete schemas, or test-label
# threshold claims.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "reproducibility/source_artifacts/checkworthiness"
SCRIPT_PATH = ROOT / "scripts/reproduce_cikm2026_public_encoder_fusion.py"


def load_reproducer():
    spec = importlib.util.spec_from_file_location("checkworthiness_reproducer", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_protocol_is_bound_to_final_pdf() -> None:
    protocol = json.loads(
        (
            ROOT / "reproducibility/cikm2026/checkworthiness/PAPER_PROTOCOL.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["paper"]["sha256"] == (
        "4d82abd01d66de5e04d7107e8c4bb21d3b3d1a7148aa235e95b78b803df78b9c"
    )
    assert protocol["paper_reported_values"]["table_5"]["Fusion Classifier"] == {
        "CT2024": 0.836,
        "ClaimBuster": 0.961,
        "CT2023": 0.915,
    }


def test_recovered_soft_label_schemas_are_privacy_safe() -> None:
    paths = [
        ARTIFACT_ROOT / "ct24_llm_features_v4/train_llm_features.parquet",
        ARTIFACT_ROOT / "ct24_llm_features_v4/dev_llm_features.parquet",
        ARTIFACT_ROOT / "ct24_llm_features_v4/test_llm_features.parquet",
        ARTIFACT_ROOT
        / "benchmark_llm_features/CB_groundtruth_llm_features.parquet",
        ARTIFACT_ROOT / "benchmark_llm_features/CT23_llm_features.parquet",
    ]
    forbidden = {
        "text",
        "original_text",
        "post_id",
        "tweet_id",
        "user_id",
        "username",
        "screen_name",
        "profile",
        "location",
    }
    expected_rows = [22_402, 1_031, 341, 1_032, 318]
    for path, rows in zip(paths, expected_rows, strict=True):
        frame = pl.read_parquet(path)
        assert frame.shape == (rows, 61)
        assert forbidden.isdisjoint({column.lower() for column in frame.columns})
        assert frame["sentence_id"].n_unique() == rows


def test_cross_benchmark_features_require_sentence_id_join() -> None:
    claimbuster = pl.read_csv(
        ARTIFACT_ROOT / "benchmarks/claim_buster/groundtruth.csv"
    )
    claimbuster_features = pl.read_parquet(
        ARTIFACT_ROOT
        / "benchmark_llm_features/CB_groundtruth_llm_features.parquet"
    )
    ct23 = pl.read_csv(
        ARTIFACT_ROOT / "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv",
        separator="\t",
    )
    ct23_features = pl.read_parquet(
        ARTIFACT_ROOT / "benchmark_llm_features/CT23_llm_features.parquet"
    )

    for data_ids, feature_ids in (
        (
            claimbuster["Sentence_id"].cast(pl.String).to_list(),
            claimbuster_features["sentence_id"].cast(pl.String).to_list(),
        ),
        (
            ct23["Sentence_id"].cast(pl.String).to_list(),
            ct23_features["sentence_id"].cast(pl.String).to_list(),
        ),
    ):
        assert set(data_ids) == set(feature_ids)
        assert data_ids != feature_ids


def test_text_free_ct24_label_tables_are_complete() -> None:
    expected = {
        "train": (22_402, 5_399),
        "dev": (1_031, 238),
        "test": (341, 88),
    }
    for split, (rows, positives) in expected.items():
        frame = pl.read_parquet(ARTIFACT_ROOT / f"ct24_labels/{split}_labels.parquet")
        assert frame.columns == ["sentence_id", "label"]
        assert frame.shape == (rows, 2)
        assert frame["sentence_id"].n_unique() == rows
        assert int(frame["label"].sum()) == positives


def test_historical_matrices_retain_positional_misalignment() -> None:
    claimbuster = pl.read_csv(
        ARTIFACT_ROOT / "benchmarks/claim_buster/groundtruth.csv"
    )
    ct23_gold = pl.read_csv(
        ARTIFACT_ROOT
        / "benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv",
        separator="\t",
    )
    with np.load(
        ARTIFACT_ROOT / "llm_features_classifier/matrices.npz", allow_pickle=False
    ) as bundle:
        assert np.array_equal(
            bundle["y_claimbuster"],
            (claimbuster["Verdict"] == 1).cast(pl.Int8).to_numpy(),
        )
        assert np.array_equal(
            bundle["y_ct23"],
            (ct23_gold["class_label"] == "Yes").cast(pl.Int8).to_numpy(),
        )
    claimbuster_features = pl.read_parquet(
        ARTIFACT_ROOT
        / "benchmark_llm_features/CB_groundtruth_llm_features.parquet"
    )
    assert claimbuster["Sentence_id"].cast(pl.String).to_list() != (
        claimbuster_features["sentence_id"].cast(pl.String).to_list()
    )


def test_corrected_feature_alignment_uses_sentence_ids() -> None:
    module = load_reproducer()
    claimbuster = pl.read_csv(
        ARTIFACT_ROOT / "benchmarks/claim_buster/groundtruth.csv"
    )
    labels = claimbuster.select(
        pl.col("Sentence_id").cast(pl.String).alias("sentence_id"),
        (pl.col("Verdict") == 1).cast(pl.Int8).alias("label"),
    )
    features, aligned_labels, audit = module.align_features_to_labels(
        ARTIFACT_ROOT
        / "benchmark_llm_features/CB_groundtruth_llm_features.parquet",
        labels,
        "ClaimBuster test",
    )
    assert features.shape == (1_032, 9)
    assert np.array_equal(aligned_labels, labels["label"].to_numpy())
    assert audit == {
        "rows": 1_032,
        "unique_sentence_ids": 1_032,
        "one_to_one_id_set_match": True,
        "source_feature_order_differed": True,
    }


def test_model_revisions_and_weight_hashes_are_immutable() -> None:
    module = load_reproducer()
    assert set(module.MODEL_SPECS) == {0, 42, 456}
    for model in module.MODEL_SPECS.values():
        assert len(model["revision"]) == 40
        assert len(model["model_sha256"]) == 64
        assert model["model_bytes"] == 1_740_304_440


def test_evaluation_oracle_is_never_labelled_as_held_out() -> None:
    module = load_reproducer()
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.2, 0.6, 0.4, 0.8], dtype=np.float64)
    record = module.metric_record(labels, probabilities, fixed_threshold=0.5)
    assert set(record) >= {"fixed_threshold", "evaluation_oracle_diagnostic"}
    assert "held_out" not in record


def test_tokenizer_runtime_dependency_is_explicit() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"protobuf==6.31.1"' in source
    assert "use_fast=True" in source


def test_offline_rerun_preserves_observed_model_identity() -> None:
    module = load_reproducer()
    retained = module.load_retained_model_observations(
        ROOT / "results/public_encoder_fusion_replication_2026-08-25/summary.json"
    )
    assert set(retained) == {"0", "42", "456"}
    for row in retained.values():
        assert row["observed_model_sha256"] == row["model_sha256"]
        assert row["observed_model_bytes"] == row["model_bytes"]
