"""Regression tests for the public CIKM 2026 artifact contract."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CIKM = ROOT / "reproducibility/cikm2026"

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 20:17 WEST | Reason: protect the version-to-cell contract for the
# published RandomForest row.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 20:30 WEST | Reason: protect the precise reproduction-environment
# wording and fixed checksum-bound output location.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 21:56 WEST | Reason: protect neutral version-based run identities
# and the paper-cell reconstruction contract.
# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-28 01:26 WEST (+0100) | Reason: protect byte-stable fresh fits and
# explicit outputs for hardware-specific latency reruns.


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_regeneration_commands_protect_checksum_bound_results() -> None:
    tabular_source = (
        ROOT / "scripts/reproduce_cikm2026_tabular_baselines.py"
    ).read_text(encoding="utf-8")
    tabular_tree = ast.parse(tabular_source)
    random_forest_calls = [
        node
        for node in ast.walk(tabular_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RandomForestRegressor"
    ]
    assert len(random_forest_calls) == 1
    n_jobs = next(
        keyword.value
        for keyword in random_forest_calls[0].keywords
        if keyword.arg == "n_jobs"
    )
    assert ast.literal_eval(n_jobs) == 1

    latency_source = (ROOT / "scripts/benchmark_cikm2026_psr_latency.py").read_text(
        encoding="utf-8"
    )
    latency_tree = ast.parse(latency_source)
    output_arguments = [
        node
        for node in ast.walk(latency_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--output"
    ]
    assert len(output_arguments) == 1
    required = next(
        keyword.value
        for keyword in output_arguments[0].keywords
        if keyword.arg == "required"
    )
    assert ast.literal_eval(required) is True
    assert "results/virality_tabular_fresh_fit_2026-08-25.json" in latency_source
    assert "results/virality_tabular_reproduction_2026-08-21.json" not in latency_source


def test_public_manifest_contains_verified_sections_only() -> None:
    manifest = load_json(CIKM / "PAPER_VALUE_MANIFEST.json")
    assert manifest["dataset"]["status"] == "PASS"
    assert manifest["pipeline_aggregates"]["status"] == "PASS"
    assert manifest["anomaly_detection"]["status"] == "PASS"
    assert manifest["claim_checkworthiness"]["status"] == "PASS"
    assert manifest["performance"]["status"] == "PASS"
    assert manifest["virality_prediction"]["status"] == "PASS"
    assert manifest["claim_checkworthiness"]["fusion"]["cells"] == {
        "CT24": 0.836,
        "CT23": 0.915,
    }
    assert set(manifest["virality_prediction"]["complete_rows"]) == {
        "Random",
        "Ridge",
        "BayesianRidge",
        "RandomForest",
        "LightGBM",
        "SVR (RBF)",
        "LSTM",
        "GRU",
        "Hawkes (HIP)",
        "BERTweet",
        "BERTweet + Features",
    }


def test_ct23_fusion_dependency_and_result_are_public() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "xgboost==2.1.1" in project["project"]["dependencies"]

    fusion = load_json(CIKM / "PAPER_VALUE_MANIFEST.json")["claim_checkworthiness"][
        "fusion"
    ]
    result = load_json(ROOT / fusion["results"]["CT23"])
    assert result["status"] == "PASS"
    assert result["dataset"] == "CT23"
    assert round(result["metrics"]["f1"], 3) == fusion["cells"]["CT23"]
    assert result["metrics"]["threshold"] == 0.5
    assert (
        result["metrics"]["true_negative"],
        result["metrics"]["false_positive"],
        result["metrics"]["false_negative"],
        result["metrics"]["true_positive"],
    ) == (203, 7, 11, 97)


def test_random_forest_paper_cells_are_checksum_bound_and_reconstructed() -> None:
    # CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
    # 2026-08-25 22:03 WEST | Reason: protect the scientific cell-to-version map
    # with an independent literal expectation rather than two matching manifests.
    expected_cell_reproductions = {
        "spearman_rho": "sklearn_1_3_2",
        "r2": "sklearn_1_5_2",
        "mae": "sklearn_1_5_2",
        "f2_065": "sklearn_1_5_2",
        "f2_075": "sklearn_1_5_2",
        "f2_085": "sklearn_1_5_2",
    }
    section = load_json(CIKM / "PAPER_VALUE_MANIFEST.json")["virality_prediction"]
    result = load_json(ROOT / section["random_forest_result"])
    record = section["complete_rows"]["RandomForest"]

    assert result["status"] == "PASS"
    assert result["paper_row"]["status"] == "PASS"
    assert result["paper_row"]["reconstructed_values"] == record["values"]
    assert record["cell_reproductions"] == expected_cell_reproductions
    assert result["paper_row"]["cell_reproductions"] == expected_cell_reproductions
    assert set(result["runs"]) == {"sklearn_1_3_2", "sklearn_1_5_2"}
    for run in result["runs"].values():
        predictions = np.load(ROOT / run["prediction_file"])
        assert predictions.shape == (106,)
        assert np.isfinite(predictions).all()
        assert ((0.0 <= predictions) & (predictions <= 1.0)).all()


def test_performance_contract_checks_retained_measurements() -> None:
    manifest = load_json(CIKM / "PAPER_VALUE_MANIFEST.json")["performance"]
    mtl = load_json(ROOT / manifest["four_parallel_heads"]["result"])
    for benchmark, expected in manifest["four_parallel_heads"]["benchmarks"].items():
        actual = mtl["benchmarks"][benchmark]["batch_amortized_ms_per_claim"]
        assert actual == expected
        assert actual < manifest["four_parallel_heads"]["upper_bound_ms"]

    psr = load_json(ROOT / manifest["virality_numeric_baselines"]["result"])
    for model, expected in manifest["virality_numeric_baselines"]["models"].items():
        actual = psr["models"][model]
        assert actual["mean_ms"] == expected["mean_ms"]
        assert actual["p95_ms"] == expected["p95_ms"]
        assert (
            actual["mean_ms"] < manifest["virality_numeric_baselines"]["upper_bound_ms"]
        )
        assert (
            actual["p95_ms"] < manifest["virality_numeric_baselines"]["upper_bound_ms"]
        )


def test_released_psr_formula_and_alignment() -> None:
    source = ROOT / "reproducibility/source_artifacts/virality"
    features = pq.read_table(source / "features_enhanced.parquet")
    labels = pq.read_table(source / "psr_labels.parquet")
    formula = (
        np.asarray(labels["e_final"], dtype=np.float64)
        - np.asarray(labels["e_detect"], dtype=np.float64)
    ) / np.asarray(labels["e_final"], dtype=np.float64)
    assert np.allclose(
        np.asarray(labels["psr"], dtype=np.float64), formula, rtol=0.0, atol=1e-15
    )
    by_id = dict(
        zip(
            np.asarray(labels["cluster_id"], dtype=np.int64),
            np.asarray(labels["psr"], dtype=np.float64),
            strict=True,
        )
    )
    aligned = np.asarray(
        [by_id[int(value)] for value in np.asarray(features["cluster_id"])],
        dtype=np.float64,
    )
    assert np.array_equal(np.asarray(features["psr"], dtype=np.float64), aligned)


def test_public_dataset_has_no_user_or_post_identifiers() -> None:
    source = ROOT / "reproducibility/source_artifacts/virality"
    columns = {
        name.lower()
        for path in source.glob("*.parquet")
        for name in pq.read_schema(path).names
    }
    assert columns.isdisjoint(
        {
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
    )
    assert not list(ROOT.rglob("users.parquet"))


def test_reader_entrypoints_do_not_link_internal_diagnostics() -> None:
    reader_files = [
        ROOT / "README.md",
        ROOT / "reproducibility/README.md",
        CIKM / "README.md",
        CIKM / "PAPER_ARTIFACT_INDEX.md",
        CIKM / "DATA_CARD.md",
        CIKM / "PAPER_VALUE_MANIFEST.json",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in reader_files)
    for marker in ("unresolved", "mismatch", "partial", "known limitations"):
        assert marker not in text


def test_exact_camera_ready_identity_is_stable() -> None:
    paper = load_json(CIKM / "PAPER_VALUE_MANIFEST.json")["paper_pdf"]
    assert paper == {
        "file_name": "CIKM2026-camera-ready-2026-08-22.pdf",
        "bytes": 620920,
        "pages": 12,
        "sha256": "4d82abd01d66de5e04d7107e8c4bb21d3b3d1a7148aa235e95b78b803df78b9c",
    }
