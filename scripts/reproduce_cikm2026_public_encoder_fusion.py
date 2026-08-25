#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-25 13:41 WEST.
# Reason: recover direct Encoder Only and Fusion cross-benchmark evidence from
# immutable public checkpoints while separating historical positional behavior
# from the scientifically correct sentence-ID-aligned evaluation.
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub==0.31.4",
#   "numpy==1.26.4",
#   "polars==1.27.1",
#   "protobuf==6.31.1",
#   "scikit-learn==1.5.2",
#   "sentencepiece==0.2.0",
#   "torch==2.7.0",
#   "transformers==4.51.3",
#   "xgboost==3.0.2",
# ]
# ///
"""Reproduce public Encoder Only inference and cross-benchmark Fusion evidence.

The default mode evaluates an existing prediction bundle. Pass ``--inference``
to download each checksum-identified public model, run ClaimBuster and CT23
inference, and update the bundle before evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_SPECS = {
    0: {
        "repository": "sergiopinto/deberta-v3-large-claim-checkworthiness-seed0",
        "revision": "a7e8f07153c72c853ce39bb4c03b4bc817c1e39c",
        "model_sha256": "9c4c555ce0e839b109d7ca2d9df69b8b5b86fe63be2a493fe1ecb1cfa467440b",
        "model_bytes": 1_740_304_440,
    },
    42: {
        "repository": "sergiopinto/deberta-v3-large-claim-checkworthiness-seed42",
        "revision": "972bbd2a25acbe26c46358f11c55c42cfafa0090",
        "model_sha256": "74ef9ea94b3345ecf87b84e65b07ff390f8cf9d20f6defb67d7ec9a1f278b7c0",
        "model_bytes": 1_740_304_440,
    },
    456: {
        "repository": "sergiopinto/deberta-v3-large-claim-checkworthiness-seed456",
        "revision": "eda7cc7a98577ab52dab370642ca9a9ecd33acfc",
        "model_sha256": "6c76e81fd86773e516fd3d26e0d243c49c68bd88ae331bd060b41cde26fbd517",
        "model_bytes": 1_740_304_440,
    },
}

INPUT_SHA256 = {
    "benchmarks/claim_buster/groundtruth.csv": "64890c51e1092fec0d361ca770b3efccc0c889dd569b46b35922d94c44fa99ea",
    "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv": "cca096dbc4417ca42c25dd4011299de140d577d75d46ccb204c7c3ee27c452bf",
    "benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv": "bb7211548c0c1a619f7283d66a0ccc12eaf115a8124046c5ccc5e13cd9eb8f6c",
    "llm_features_classifier/matrices.npz": "de2c3e47534a0bfe81a4855bf5989ad2efaa434f20229220843080ba2862fa6e",
    "benchmark_llm_features/CB_groundtruth_llm_features.parquet": "a26ac74990d876fe91e8cd9e34fe24134969a2c4bb997079b922d7cfc99e0a7a",
    "benchmark_llm_features/CT23_llm_features.parquet": "7db1dcdc780a0991b09af84288e1abc8d907ad1c1e57358953757fc86c7a53d6",
    "ct24_llm_features_v4/train_llm_features.parquet": "ec15929b5eb6d509d32d6b32b38e0d765bbfa15609a617b4f4a085741886f218",
    "ct24_llm_features_v4/dev_llm_features.parquet": "00d0ce6698dd9d7dc3a50f56f7c644abe22e4c9b3c6f10d97ed7ececfeb05af7",
    "ct24_llm_features_v4/test_llm_features.parquet": "60252ee38453d14a43b076fe8fa50e998a9dafebdb1d51f7e81d92470a9d0e1c",
    "ct24_labels/train_labels.parquet": "9c29a60b17f14c7e171bffe8eeab46fc763f09e0ff70d0e2186b209b2ae53868",
    "ct24_labels/dev_labels.parquet": "2c31452d08fd3fa447f633c9eb4bbc072486b783a060d90b1a9d514796f17099",
    "ct24_labels/test_labels.parquet": "2831afeb6cc08d2942ae02ee50afdac2368aadba41fbabf8fd069a117408ae57",
}

PAPER_F1 = {
    "Encoder Only": {"ClaimBuster": 0.970, "CT23": 0.928},
    "Fusion Classifier": {"ClaimBuster": 0.961, "CT23": 0.915},
}

PAPER_FUSION_THRESHOLDS = {"ClaimBuster": 0.55, "CT23": 0.50}
THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(10))
FUSION_FEATURES = (
    "check_score",
    "verif_score",
    "harm_score",
    "check_entropy",
    "verif_entropy",
    "harm_entropy",
    "check_p_yes",
    "verif_p_yes",
    "harm_p_yes",
)


@dataclass(frozen=True)
class Metrics:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class Benchmark:
    name: str
    sentence_ids: np.ndarray
    texts: list[str]
    labels: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Metrics:
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"label/probability shape mismatch: {labels.shape} != {probabilities.shape}"
        )
    predicted = probabilities >= threshold
    positive = labels.astype(bool)
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return Metrics(
        threshold=float(threshold),
        f1=2 * tp / denominator if denominator else 0.0,
        precision=precision,
        recall=recall,
        accuracy=(tp + tn) / len(labels),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def oracle_diagnostic(labels: np.ndarray, probabilities: np.ndarray) -> Metrics:
    """Return an explicitly non-held-out evaluation-set threshold diagnostic."""
    return max(
        (evaluate(labels, probabilities, threshold) for threshold in THRESHOLDS),
        key=lambda row: row.f1,
    )


def temperature_scale(probabilities: list[np.ndarray], temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = []
    for values in probabilities:
        clipped = np.clip(values, 1e-8, 1 - 1e-8)
        logits = np.log(clipped / (1 - clipped))
        scaled.append(1 / (1 + np.exp(-(logits / temperature))))
    return np.mean(scaled, axis=0)


def validate_inputs(artifact_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in INPUT_SHA256.items():
        path = artifact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"artifact hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        observed[relative] = actual
    return observed


def load_benchmarks(artifact_root: Path) -> dict[str, Benchmark]:
    claimbuster = pl.read_csv(
        artifact_root / "benchmarks/claim_buster/groundtruth.csv"
    )
    cb_ids = claimbuster["Sentence_id"].cast(pl.String).to_numpy()
    cb_labels = (claimbuster["Verdict"] == 1).cast(pl.Int8).to_numpy()

    ct23_input = pl.read_csv(
        artifact_root / "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv",
        separator="\t",
    )
    ct23_gold = pl.read_csv(
        artifact_root
        / "benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv",
        separator="\t",
    )
    if len(ct23_input) != len(ct23_gold):
        raise ValueError("CT23 input and gold row counts differ")
    if ct23_input["Text"].to_list() != ct23_gold["Text"].to_list():
        raise ValueError("CT23 input and gold text order differs")
    ct23_ids = ct23_input["Sentence_id"].cast(pl.String).to_numpy()
    ct23_labels = (ct23_gold["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    benchmarks = {
        "ClaimBuster": Benchmark(
            "ClaimBuster", cb_ids, claimbuster["Text"].to_list(), cb_labels
        ),
        "CT23": Benchmark(
            "CT23", ct23_ids, ct23_input["Text"].to_list(), ct23_labels
        ),
    }
    expected = {"ClaimBuster": (1_032, 238), "CT23": (318, 108)}
    for name, benchmark in benchmarks.items():
        expected_rows, expected_positive = expected[name]
        if len(benchmark.labels) != expected_rows:
            raise ValueError(f"{name}: unexpected row count")
        if int(benchmark.labels.sum()) != expected_positive:
            raise ValueError(f"{name}: unexpected positive-label count")
        if len(set(benchmark.sentence_ids.tolist())) != expected_rows:
            raise ValueError(f"{name}: sentence IDs are not unique")
    return benchmarks


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_prediction_arrays(
    prediction_path: Path, benchmarks: dict[str, Benchmark]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if prediction_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as retained:
            arrays.update({name: retained[name] for name in retained.files})

    for name, benchmark in benchmarks.items():
        key = name.lower()
        expected_ids = benchmark.sentence_ids.astype(str)
        expected_labels = benchmark.labels.astype(np.int8)
        if f"{key}_sentence_ids" in arrays and not np.array_equal(
            arrays[f"{key}_sentence_ids"].astype(str), expected_ids
        ):
            raise ValueError(f"{name}: retained sentence-ID order changed")
        if f"{key}_labels" in arrays and not np.array_equal(
            arrays[f"{key}_labels"].astype(np.int8), expected_labels
        ):
            raise ValueError(f"{name}: retained label order changed")
        arrays[f"{key}_sentence_ids"] = expected_ids
        arrays[f"{key}_labels"] = expected_labels
    return arrays


def resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_probabilities(
    texts: list[str], model, tokenizer, device, batch_size: int
) -> np.ndarray:
    import torch

    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.append(
                torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            )
    return np.concatenate(probabilities).astype(np.float32)


def run_public_inference(
    arrays: dict[str, np.ndarray],
    prediction_path: Path,
    benchmarks: dict[str, Benchmark],
    device_name: str,
    batch_size: int,
    force_inference: bool,
) -> dict[str, dict[str, Any]]:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(device_name)
    observed_models: dict[str, dict[str, Any]] = {}
    print(f"Inference device: {device}")

    for seed, spec in MODEL_SPECS.items():
        required_keys = [
            f"{name.lower()}_seed_{seed}_probabilities" for name in benchmarks
        ]
        model_file = hf_hub_download(
            repo_id=str(spec["repository"]),
            filename="model.safetensors",
            revision=str(spec["revision"]),
        )
        model_path = Path(model_file)
        actual_size = model_path.stat().st_size
        actual_sha = sha256_file(model_path)
        if actual_size != spec["model_bytes"] or actual_sha != spec["model_sha256"]:
            raise ValueError(f"seed {seed}: downloaded model identity mismatch")
        observed_models[str(seed)] = {
            **spec,
            "observed_model_bytes": actual_size,
            "observed_model_sha256": actual_sha,
        }

        if not force_inference and all(key in arrays for key in required_keys):
            print(f"Seed {seed}: retained probabilities found; inference skipped")
            continue

        print(f"Seed {seed}: loading {spec['repository']}@{spec['revision']}")
        tokenizer = AutoTokenizer.from_pretrained(
            spec["repository"], revision=spec["revision"], use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec["repository"], revision=spec["revision"], use_safetensors=True
        ).to(device)

        for name, benchmark in benchmarks.items():
            key = f"{name.lower()}_seed_{seed}_probabilities"
            if key in arrays and not force_inference:
                continue
            print(f"Seed {seed}: {name} ({len(benchmark.texts)} rows)")
            arrays[key] = infer_probabilities(
                benchmark.texts, model, tokenizer, device, batch_size
            )
        atomic_savez(prediction_path, arrays)

        del model
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    return observed_models


def require_complete_predictions(
    arrays: dict[str, np.ndarray], benchmarks: dict[str, Benchmark]
) -> None:
    for seed in MODEL_SPECS:
        for name, benchmark in benchmarks.items():
            key = f"{name.lower()}_seed_{seed}_probabilities"
            if key not in arrays:
                raise FileNotFoundError(
                    f"missing {key}; rerun with --inference to generate it"
                )
            if arrays[key].shape != benchmark.labels.shape:
                raise ValueError(f"{key}: unexpected probability shape")
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key}: probabilities contain non-finite values")


def load_retained_model_observations(summary_path: Path) -> dict[str, dict[str, Any]]:
    """Preserve checkpoint identity observations during an offline metric rerun."""
    if not summary_path.is_file():
        return {}
    retained = json.loads(summary_path.read_text(encoding="utf-8"))
    observed = retained.get("model_specs", {})
    if set(observed) != {str(seed) for seed in MODEL_SPECS}:
        return {}
    for seed, expected in MODEL_SPECS.items():
        row = observed[str(seed)]
        if (
            row.get("revision") != expected["revision"]
            or row.get("model_sha256") != expected["model_sha256"]
            or row.get("model_bytes") != expected["model_bytes"]
            or row.get("observed_model_sha256") != expected["model_sha256"]
            or row.get("observed_model_bytes") != expected["model_bytes"]
        ):
            return {}
    return observed


def fit_xgboost(features: np.ndarray, labels: np.ndarray):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=3,
        random_state=42,
        verbosity=0,
    )
    classifier.fit(scaled, labels)
    return scaler, classifier


def align_features_to_labels(
    feature_path: Path,
    label_frame: pl.DataFrame,
    context: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Join numerical features to a canonical label order by sentence ID."""
    labels = label_frame.select(
        pl.col("sentence_id").cast(pl.String),
        pl.col("label").cast(pl.Int8),
    ).with_row_index("__row_order")
    features = pl.read_parquet(feature_path).with_columns(
        pl.col("sentence_id").cast(pl.String)
    )
    if labels["sentence_id"].n_unique() != len(labels):
        raise ValueError(f"{context}: label sentence IDs are not unique")
    if features["sentence_id"].n_unique() != len(features):
        raise ValueError(f"{context}: feature sentence IDs are not unique")
    label_ids = set(labels["sentence_id"].to_list())
    feature_ids = set(features["sentence_id"].to_list())
    if label_ids != feature_ids:
        raise ValueError(
            f"{context}: sentence-ID sets differ; "
            f"missing_features={len(label_ids - feature_ids)}, "
            f"unexpected_features={len(feature_ids - label_ids)}"
        )

    aligned = (
        labels.join(features, on="sentence_id", how="left", validate="1:1")
        .sort("__row_order")
    )
    if aligned.select(FUSION_FEATURES).null_count().row(0) != (0,) * len(
        FUSION_FEATURES
    ):
        raise ValueError(f"{context}: aligned features contain null values")
    return (
        aligned.select(FUSION_FEATURES).to_numpy(),
        aligned["label"].to_numpy(),
        {
            "rows": len(aligned),
            "unique_sentence_ids": aligned["sentence_id"].n_unique(),
            "one_to_one_id_set_match": True,
            "source_feature_order_differed": (
                labels["sentence_id"].to_list()
                != features["sentence_id"].to_list()
            ),
        },
    )


def load_fusion_components(
    artifact_root: Path, benchmarks: dict[str, Benchmark]
) -> dict[str, Any]:
    with np.load(
        artifact_root / "llm_features_classifier/matrices.npz", allow_pickle=False
    ) as bundle:
        feature_names = bundle["feature_names"].tolist()
        feature_indices = [feature_names.index(name) for name in FUSION_FEATURES]
        historical_train = np.vstack(
            [bundle["X_train"][:, feature_indices], bundle["X_dev"][:, feature_indices]]
        )
        historical_labels = np.concatenate([bundle["y_train"], bundle["y_dev"]])
        historical_evaluation = {
            "ClaimBuster": bundle["X_claimbuster"][:, feature_indices],
            "CT23": bundle["X_ct23"][:, feature_indices],
        }
        for name, key in (("ClaimBuster", "y_claimbuster"), ("CT23", "y_ct23")):
            if not np.array_equal(bundle[key], benchmarks[name].labels):
                raise ValueError(f"{name}: historical matrix labels changed order")

    corrected_train_parts: list[np.ndarray] = []
    corrected_label_parts: list[np.ndarray] = []
    alignment_audit: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        labels = pl.read_parquet(
            artifact_root / f"ct24_labels/{split}_labels.parquet"
        )
        features, split_labels, audit = align_features_to_labels(
            artifact_root / f"ct24_llm_features_v4/{split}_llm_features.parquet",
            labels,
            f"CT24 {split}",
        )
        corrected_train_parts.append(features)
        corrected_label_parts.append(split_labels)
        alignment_audit[f"CT24 {split}"] = audit
    corrected_train = np.vstack(corrected_train_parts)
    corrected_labels = np.concatenate(corrected_label_parts)

    corrected_evaluation: dict[str, np.ndarray] = {}
    benchmark_feature_files = {
        "ClaimBuster": "CB_groundtruth_llm_features.parquet",
        "CT23": "CT23_llm_features.parquet",
    }
    for name, benchmark in benchmarks.items():
        label_frame = pl.DataFrame(
            {
                "sentence_id": benchmark.sentence_ids.astype(str),
                "label": benchmark.labels.astype(np.int8),
            }
        )
        features, aligned_labels, audit = align_features_to_labels(
            artifact_root / "benchmark_llm_features" / benchmark_feature_files[name],
            label_frame,
            name,
        )
        if not np.array_equal(aligned_labels, benchmark.labels):
            raise ValueError(f"{name}: aligned labels changed benchmark order")
        corrected_evaluation[name] = features
        alignment_audit[name] = audit

    corrected_scaler, corrected_model = fit_xgboost(
        corrected_train, corrected_labels
    )
    historical_scaler, historical_model = fit_xgboost(
        historical_train, historical_labels
    )

    return {
        "corrected": {
            name: corrected_model.predict_proba(
                corrected_scaler.transform(corrected_evaluation[name])
            )[:, 1]
            for name in benchmarks
        },
        "historical_positional": {
            name: historical_model.predict_proba(
                historical_scaler.transform(historical_evaluation[name])
            )[:, 1]
            for name in benchmarks
        },
        "alignment_audit": alignment_audit,
    }


def metric_record(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fixed_threshold: float,
    paper_f1: float | None = None,
) -> dict[str, Any]:
    fixed = evaluate(labels, probabilities, fixed_threshold)
    oracle = oracle_diagnostic(labels, probabilities)
    return {
        "fixed_threshold": asdict(fixed),
        "evaluation_oracle_diagnostic": asdict(oracle),
        "paper_f1": paper_f1,
        "fixed_threshold_matches_paper_after_rounding": (
            round(fixed.f1, 3) == paper_f1 if paper_f1 is not None else None
        ),
        "oracle_matches_paper_after_rounding": (
            round(oracle.f1, 3) == paper_f1 if paper_f1 is not None else None
        ),
    }


def build_summary(
    artifact_root: Path,
    prediction_path: Path,
    arrays: dict[str, np.ndarray],
    benchmarks: dict[str, Benchmark],
    input_hashes: dict[str, str],
    observed_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    individual: dict[str, dict[str, Any]] = {}
    for seed in MODEL_SPECS:
        individual[str(seed)] = {}
        for name, benchmark in benchmarks.items():
            probabilities = arrays[f"{name.lower()}_seed_{seed}_probabilities"]
            individual[str(seed)][name] = metric_record(
                benchmark.labels,
                probabilities,
                fixed_threshold=0.5,
                paper_f1=(
                    PAPER_F1["Encoder Only"][name] if seed == 0 else None
                ),
            )

    xgboost = load_fusion_components(artifact_root, benchmarks)
    fusion: dict[str, dict[str, Any]] = {
        "historical_positional_two_seed": {},
        "corrected_id_aligned_three_seed": {},
    }
    xgboost_results: dict[str, dict[str, Any]] = {
        "historical_positional": {},
        "corrected_id_aligned": {},
    }

    for name, benchmark in benchmarks.items():
        seed_probabilities = {
            seed: arrays[f"{name.lower()}_seed_{seed}_probabilities"]
            for seed in MODEL_SPECS
        }
        historical_encoder = temperature_scale(
            [seed_probabilities[0], seed_probabilities[456]], temperature=0.7
        )
        corrected_encoder = temperature_scale(
            [seed_probabilities[0], seed_probabilities[42], seed_probabilities[456]],
            temperature=0.7,
        )
        historical_fusion = (
            0.6 * historical_encoder + 0.4 * xgboost["historical_positional"][name]
        )
        corrected_fusion = (
            0.6 * corrected_encoder + 0.4 * xgboost["corrected"][name]
        )

        xgboost_results["historical_positional"][name] = metric_record(
            benchmark.labels,
            xgboost["historical_positional"][name],
            fixed_threshold=0.5,
        )
        xgboost_results["corrected_id_aligned"][name] = metric_record(
            benchmark.labels,
            xgboost["corrected"][name],
            fixed_threshold=0.5,
        )
        fusion["historical_positional_two_seed"][name] = metric_record(
            benchmark.labels,
            historical_fusion,
            fixed_threshold=PAPER_FUSION_THRESHOLDS[name],
            paper_f1=PAPER_F1["Fusion Classifier"][name],
        )
        fusion["corrected_id_aligned_three_seed"][name] = metric_record(
            benchmark.labels,
            corrected_fusion,
            fixed_threshold=0.5,
            paper_f1=PAPER_F1["Fusion Classifier"][name],
        )

        arrays[f"{name.lower()}_historical_xgboost_probabilities"] = xgboost[
            "historical_positional"
        ][name].astype(np.float32)
        arrays[f"{name.lower()}_corrected_xgboost_probabilities"] = xgboost[
            "corrected"
        ][name].astype(np.float32)
        arrays[f"{name.lower()}_historical_fusion_probabilities"] = (
            historical_fusion.astype(np.float32)
        )
        arrays[f"{name.lower()}_corrected_fusion_probabilities"] = (
            corrected_fusion.astype(np.float32)
        )

    atomic_savez(prediction_path, arrays)
    prediction_sha = sha256_file(prediction_path)
    return {
        "change_note": (
            "Sérgio Pinto, 2026-08-25 13:41 WEST — ran immutable public "
            "Encoder Only checkpoints and separated historical positional "
            "Fusion evidence from the corrected sentence-ID-aligned replication."
        ),
        "paper_values_are_immutable_targets": True,
        "status": "fresh public-checkpoint replication complete",
        "model_specs": observed_models or MODEL_SPECS,
        "input_sha256": input_hashes,
        "prediction_bundle": {
            "path": prediction_path.relative_to(artifact_root.parents[2]).as_posix(),
            "sha256": prediction_sha,
            "contains_source_text": False,
            "contains_user_or_post_identifiers": False,
        },
        "evaluation_policy": {
            "fixed_threshold_results": "direct evaluations at a threshold fixed independently of the evaluated labels",
            "evaluation_oracle_diagnostic": "maximum F1 over the evaluated labels; retained only for historical comparison and never treated as held-out reproduction",
            "corrected_join": "one-to-one benchmark sentence-ID alignment",
            "historical_join": "row-position behavior retained only as a diagnostic of the original cross-benchmark pipeline",
        },
        "individual_public_encoder_models": individual,
        "llm_xgboost_components": xgboost_results,
        "sentence_id_alignment_audit": xgboost["alignment_audit"],
        "fusion": fusion,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_output = root / "results/public_encoder_fusion_replication_2026-08-25"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--inference",
        action="store_true",
        help="download fixed public model revisions and run missing inference",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="regenerate probabilities even when a complete bundle exists",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for inference: auto, cuda, mps, or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    artifact_root = (
        root / "reproducibility/source_artifacts/checkworthiness"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "per_example_predictions.npz"
    summary_path = args.output_dir / "summary.json"

    input_hashes = validate_inputs(artifact_root)
    benchmarks = load_benchmarks(artifact_root)
    arrays = load_prediction_arrays(prediction_path, benchmarks)
    observed_models = load_retained_model_observations(summary_path)
    if args.inference:
        observed_models = run_public_inference(
            arrays,
            prediction_path,
            benchmarks,
            args.device,
            args.batch_size,
            args.force_inference,
        )
        arrays = load_prediction_arrays(prediction_path, benchmarks)

    require_complete_predictions(arrays, benchmarks)
    summary = build_summary(
        artifact_root,
        prediction_path,
        arrays,
        benchmarks,
        input_hashes,
        observed_models,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for path_name, result in summary["fusion"].items():
        print(path_name)
        for benchmark, record in result.items():
            fixed = record["fixed_threshold"]
            oracle = record["evaluation_oracle_diagnostic"]
            print(
                f"  {benchmark}: fixed F1={fixed['f1']:.6f} "
                f"@ {fixed['threshold']:.2f}; oracle diagnostic "
                f"F1={oracle['f1']:.6f} @ {oracle['threshold']:.2f}"
            )
    print(f"Prediction bundle: {prediction_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
