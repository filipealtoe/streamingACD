"""Tests for the two-track CIKM check-worthiness GPU run contract."""

# Sérgio Pinto, 2026-08-28 01:11 WEST — protect the recovered per-run
# parameters, retained CT23 Fusion reconstruction and A10 runtime contract.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_cikm2026_checkworthiness_lambda.py"
SPEC = ROOT / "reproducibility/cikm2026/checkworthiness/LAMBDA_RUN_SPEC.json"


def load_runner():
    module_spec = importlib.util.spec_from_file_location("lambda_runner", SCRIPT)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def benchmark(runner, name: str, sentence_id: str, text: str, label: int):
    return runner.Benchmark(
        name=name,
        sentence_ids=np.asarray([sentence_id]),
        texts=[text],
        labels=np.asarray([label], dtype=np.int8),
    )


def test_machine_spec_matches_executable_constants() -> None:
    runner = load_runner()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["base_model"] == {
        "repository": runner.BASE_MODEL_REPOSITORY,
        "revision": runner.BASE_MODEL_REVISION,
    }
    assert (
        tuple(
            item["effective_rng_seed"] for item in spec["historical_training"]["runs"]
        )
        == runner.HISTORICAL_SEEDS
    )
    assert spec["tracks"]["strict"]["internal_validation"] == {
        "method": "StratifiedGroupKFold",
        "groups": "NFKC-casefolded whitespace-normalized text",
        "folds": runner.STRICT_FOLDS,
        "selected_fold": runner.STRICT_SELECTED_FOLD,
        "shuffle": True,
        "random_state": runner.STRICT_RANDOM_STATE,
    }


def test_historical_training_configuration_matches_retained_runs() -> None:
    runner = load_runner()
    seed_42 = runner.training_configuration("historical", 42)
    seed_456 = runner.training_configuration("historical", 456)
    assert seed_42["gradient_accumulation_steps"] == 2
    assert seed_42["fgm_epsilon"] == 0.5
    assert seed_456["gradient_accumulation_steps"] == 4
    assert seed_456["fgm_epsilon"] == 1.0
    for configuration in (seed_42, seed_456):
        assert configuration["fp16"] is True
        assert configuration["bf16"] is False
        assert configuration["eval_on_devtest"] is True


def test_retained_fusion_feature_order_is_stable() -> None:
    runner = load_runner()
    assert runner.FUSION_FEATURES == (
        "check_score",
        "verif_score",
        "harm_score",
        "check_p_yes",
        "verif_p_yes",
        "harm_p_yes",
        "check_entropy",
        "verif_entropy",
        "harm_entropy",
    )


def test_historical_environment_requires_a10_cuda128_torch27() -> None:
    runner = load_runner()
    packages = {"transformers": "4.44.0", "torch": "2.7.0+cu128"}
    runner.validate_historical_environment_values(
        python_major_minor=(3, 10),
        packages=packages,
        cuda_version="12.8",
        gpu_name="NVIDIA A10",
        cuda_capability=(8, 6),
    )
    with pytest.raises(RuntimeError, match="NVIDIA A10"):
        runner.validate_historical_environment_values(
            python_major_minor=(3, 10),
            packages=packages,
            cuda_version="12.8",
            gpu_name="NVIDIA A40",
            cuda_capability=(8, 6),
        )


def test_retained_ct23_fusion_reproduces_paper_cell() -> None:
    runner = load_runner()
    metrics = runner.retained_ct23_fusion(ROOT)
    assert metrics.threshold == 0.5
    assert metrics.f1 == pytest.approx(0.9150943396226415)
    assert (
        metrics.true_negative,
        metrics.false_positive,
        metrics.false_negative,
        metrics.true_positive,
    ) == (203, 7, 11, 97)


def test_retained_ct23_public_result_matches_reconstruction() -> None:
    runner = load_runner()
    metrics = runner.retained_ct23_fusion(ROOT)
    result = runner.retained_ct23_result(metrics)
    assert result["status"] == "PASS"
    assert result["paper_f1"] == 0.915
    assert result["metrics"]["f1"] == pytest.approx(0.9150943396226415)
    assert result["method"]["xgboost"]["version"] == "2.1.1"
    assert result["method"]["encoder_runs"] == [
        {"historical_label": "seed_0", "effective_rng_seed": 42},
        {"historical_label": "seed_456", "effective_rng_seed": 456},
    ]


def test_strict_split_excludes_benchmarks_and_groups_duplicate_text() -> None:
    runner = load_runner()
    rows = []
    for index in range(160):
        label = "Yes" if index % 4 == 0 else "No"
        rows.append(
            {
                "Sentence_id": f"source-{index}",
                "Text": f"Unique training sentence {index}",
                "class_label": label,
            }
        )
    rows.extend(
        [
            {
                "Sentence_id": "duplicate-a",
                "Text": "  CLAIMBUSTER   OVERLAP ",
                "class_label": "Yes",
            },
            {
                "Sentence_id": "duplicate-b",
                "Text": "ct23 overlap",
                "class_label": "No",
            },
            {
                "Sentence_id": "group-copy-1",
                "Text": "Same internal text",
                "class_label": "No",
            },
            {
                "Sentence_id": "group-copy-2",
                "Text": "same   internal TEXT",
                "class_label": "No",
            },
        ]
    )
    source = pl.DataFrame(rows)
    benchmarks = {
        "ClaimBuster": benchmark(runner, "ClaimBuster", "cb", "claimbuster overlap", 1),
        "CT23": benchmark(runner, "CT23", "ct23", "CT23 OVERLAP", 0),
    }
    train, validation, audit = runner.strict_split(source, benchmarks)
    train_text = {runner.canonical_text(value) for value in train["Text"]}
    validation_text = {runner.canonical_text(value) for value in validation["Text"]}
    external_text = {
        runner.canonical_text(value)
        for item in benchmarks.values()
        for value in item.texts
    }
    assert audit["excluded_rows"] == 2
    assert audit["train_validation_text_overlap"] == 0
    assert audit["selection_external_text_overlap"] == 0
    assert train_text.isdisjoint(validation_text)
    assert (train_text | validation_text).isdisjoint(external_text)
    assert len(train) + len(validation) == len(source) - 2


def test_run_root_inside_repository_is_rejected() -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="outside the Git repository"):
        runner.assert_run_root_outside_repo(ROOT, ROOT / "generated-run")


def test_threshold_selection_is_restricted_to_frozen_grid() -> None:
    runner = load_runner()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.4, 0.45, 0.9], dtype=np.float64)
    selected = runner.select_threshold(labels, probabilities)
    assert selected.threshold in runner.THRESHOLDS
    assert round(selected.threshold, 2) == 0.45
    assert selected.f1 == 1.0


def test_receipt_from_prior_preflight_is_rejected(tmp_path: Path) -> None:
    runner = load_runner()
    run_root = tmp_path / "run"
    model_root = run_root / "models/historical/seed_42/best_model"
    model_root.mkdir(parents=True)
    model_path = model_root / "model.safetensors"
    model_path.write_bytes(b"model bytes")
    preflight = {
        "run_contract_sha256": {
            "runner": "runner-a",
            "specification": "spec-a",
            "requirements": "requirements-a",
        },
        "tracks": {
            "historical": {
                "splits": {
                    "train": {
                        "path": "staged_data/historical/train.parquet",
                        "file_sha256": "split-a",
                        "sentence_id_sha256": "ids-a",
                        "rows": 100,
                        "positive": 25,
                    }
                }
            }
        },
    }
    runner.write_json(run_root / "preflight.json", preflight)
    base_identity = {
        "repository": runner.BASE_MODEL_REPOSITORY,
        "revision": runner.BASE_MODEL_REVISION,
        "files": {},
    }
    receipt = {
        "status": "PASS",
        "track": "historical",
        "seed": 42,
        "base_model_revision": runner.BASE_MODEL_REVISION,
        "base_model_identity": base_identity,
        "training_source_sha256": runner.TRAINER_SHA256,
        "binding": runner.receipt_binding(run_root, preflight, "historical", 42),
        "model_file_sha256": {"model.safetensors": runner.sha256_file(model_path)},
    }
    runner.write_json(run_root / "models/historical/seed_42/run_receipt.json", receipt)
    runner.verify_model_receipt(run_root, "historical", 42, preflight, base_identity)

    changed_preflight = json.loads(json.dumps(preflight))
    changed_preflight["tracks"]["historical"]["splits"]["train"]["file_sha256"] = (
        "split-b"
    )
    runner.write_json(run_root / "preflight.json", changed_preflight)
    with pytest.raises(ValueError, match="stale or invalid run receipt"):
        runner.verify_model_receipt(
            run_root, "historical", 42, changed_preflight, base_identity
        )
