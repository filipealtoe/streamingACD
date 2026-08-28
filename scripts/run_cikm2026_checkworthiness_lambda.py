#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-28 03:38 WEST.
# Reason: bind the ClaimBuster paper cells to the fresh A10 seed-42 predictions,
# recovered four-confidence inputs and exact recorded evaluation thresholds.
"""Prepare, train, and evaluate the two-track CIKM check-worthiness run."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import sys
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Sérgio Pinto, 2026-08-26 18:05 WEST — encode the two retained January
# training runs using their effective RNG seeds and per-run hyperparameters.
HISTORICAL_SEEDS = (42, 456)
STRICT_SEEDS = (0, 42, 456)
HISTORICAL_RUN_LABELS = {42: "seed_0", 456: "seed_456"}
BASE_MODEL_REPOSITORY = "microsoft/deberta-v3-large"
BASE_MODEL_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
TRAINER_SHA256 = "95b3161f6d8ca1a7b6da8a727b48e2b7c769d3f24b29a98d8936ec10a3e0cb4a"
STRICT_RANDOM_STATE = 42
STRICT_FOLDS = 10
STRICT_SELECTED_FOLD = 0
TEMPERATURE = 0.7
ENCODER_WEIGHT = 0.6
THRESHOLDS = tuple(float(value) for value in np.arange(0.30, 0.75, 0.05))

PRIVATE_INPUTS = {
    "processed/CT24_clean/CT24_train_clean.parquet": {
        "sha256": "200d674d2187021c2376ad5ef10419cfda6421a9e325f731f0347fc76db026d4",
        "rows": 22_402,
        "positive": 5_399,
    },
    "processed/CT24_clean/CT24_dev_clean.parquet": {
        "sha256": "2ff8dbb80c86e79eb1d6fbe9d5f9e994eba6ccce1304154f4d8a551f04ffc07b",
        "rows": 1_031,
        "positive": 238,
    },
    "processed/CT24_clean/CT24_test_clean.parquet": {
        "sha256": "bfd7296b8718ed62950ebf7ffa88b1f7fb4ba890eb3088f6ed000296ca76d226",
        "rows": 341,
        "positive": 88,
    },
}

PUBLIC_INPUTS = {
    "benchmarks/claim_buster/groundtruth.csv": (
        "64890c51e1092fec0d361ca770b3efccc0c889dd569b46b35922d94c44fa99ea"
    ),
    "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv": (
        "cca096dbc4417ca42c25dd4011299de140d577d75d46ccb204c7c3ee27c452bf"
    ),
    "benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv": (
        "bb7211548c0c1a619f7283d66a0ccc12eaf115a8124046c5ccc5e13cd9eb8f6c"
    ),
    "ct24_llm_features_v4/train_llm_features.parquet": (
        "ec15929b5eb6d509d32d6b32b38e0d765bbfa15609a617b4f4a085741886f218"
    ),
    "llm_features_classifier/matrices.npz": (
        "de2c3e47534a0bfe81a4855bf5989ad2efaa434f20229220843080ba2862fa6e"
    ),
    "benchmark_llm_features/CB_groundtruth_llm_features.parquet": (
        "a26ac74990d876fe91e8cd9e34fe24134969a2c4bb997079b922d7cfc99e0a7a"
    ),
    "benchmark_llm_features/CT23_llm_features.parquet": (
        "7db1dcdc780a0991b09af84288e1abc8d907ad1c1e57358953757fc86c7a53d6"
    ),
    "fusion_benchmark_features/CB_groundtruth_features_mistral-small-24b.parquet": (
        "60bdfcf5a89002b8eddfcfdf220b2d35665ce0121cc228f5ea0cd30e70858326"
    ),
    "fusion_benchmark_features/CT23_features_mistral-small-24b.parquet": (
        "07591b77a281939c852aebd5627c6e1cf36731bc84457839f6aa170ff4536327"
    ),
    "encoder_only/ct23_probs.npy": (
        "b3260befbe0200dbfe45fc7a77cc159909d1252f924dbccb02e4cbbbc514292c"
    ),
    "encoder_only/ct23_seed_456_probs.json": (
        "99bc3c16f30ec1783ba4e8bb98a822dccacc55e629a490dc98011c79b2eea82e"
    ),
    "encoder_only/claimbuster_seed_42_predictions.npz": (
        "544a6a1230464fc1c52875e50328c9b30af8bf3585163e3acc5d6b36431085ab"
    ),
}

# The order is intentional. The retained CT23 reconstruction places probability
# columns before entropy; XGBoost uses column position when resolving split
# ties, so a semantically equivalent permutation changes the paper cell by one
# prediction.
FUSION_FEATURES = (
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

CLAIMBUSTER_FUSION_FEATURES = (
    "checkability_conf",
    "verifiability_conf",
    "harm_conf",
    "avg_confidence",
)

PAPER_TARGETS = {
    "encoder_only": {"ClaimBuster": 0.970, "CT23": 0.928},
    "fusion": {"ClaimBuster": 0.961, "CT23": 0.915},
}

HISTORICAL_THRESHOLDS = {
    "encoder_only": {"ClaimBuster": 0.65, "CT23": 0.50},
    "fusion": {"ClaimBuster": 0.60, "CT23": 0.50},
}

HISTORICAL_ENVIRONMENT = {
    "python_major_minor": (3, 10),
    "transformers": "4.44.0",
    "torch": "2.7.0",
    "cuda_runtime": "12.8",
    "gpu_name": "NVIDIA A10",
    "cuda_compute_capability": (8, 6),
}


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


class Tee:
    """Write training output to both the terminal and a run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_string_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def standard_frame(frame: pl.DataFrame, context: str) -> pl.DataFrame:
    required = {"Sentence_id", "Text", "class_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}")
    result = frame.select(
        pl.col("Sentence_id").cast(pl.String),
        pl.col("Text").cast(pl.String),
        pl.col("class_label").cast(pl.String),
    )
    if result["Sentence_id"].n_unique() != len(result):
        raise ValueError(f"{context}: sentence IDs are not unique")
    labels = set(result["class_label"].unique().to_list())
    if labels != {"No", "Yes"}:
        raise ValueError(f"{context}: unexpected labels {sorted(labels)}")
    if result.null_count().row(0) != (0, 0, 0):
        raise ValueError(f"{context}: null values found")
    return result


def validate_frame(
    frame: pl.DataFrame, context: str, expected_rows: int, expected_positive: int
) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"{context}: expected {expected_rows} rows, got {len(frame)}")
    positive = int((frame["class_label"] == "Yes").sum())
    if positive != expected_positive:
        raise ValueError(
            f"{context}: expected {expected_positive} positives, got {positive}"
        )


def check_hash(path: Path, expected: str, context: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{context}: missing {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{context}: SHA-256 mismatch for {path}: {actual}")
    return actual


def artifact_root(repo_root: Path) -> Path:
    return repo_root / "reproducibility/source_artifacts/checkworthiness"


def trainer_path(repo_root: Path) -> Path:
    return (
        artifact_root(repo_root)
        / "source_code/scripts/finetuning/finetune_deberta_multimodel.py"
    )


def run_spec_path(repo_root: Path) -> Path:
    return repo_root / "reproducibility/cikm2026/checkworthiness/LAMBDA_RUN_SPEC.json"


def requirements_path(repo_root: Path) -> Path:
    return (
        repo_root / "reproducibility/cikm2026/checkworthiness/requirements-lambda.txt"
    )


def validate_public_inputs(repo_root: Path) -> dict[str, str]:
    root = artifact_root(repo_root)
    observed = {}
    for relative, expected in PUBLIC_INPUTS.items():
        observed[relative] = check_hash(
            root / relative, expected, f"public input {relative}"
        )
    observed["source_code/scripts/finetuning/finetune_deberta_multimodel.py"] = (
        check_hash(trainer_path(repo_root), TRAINER_SHA256, "training source")
    )
    return observed


def load_private_inputs(
    data_root: Path,
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    frames: dict[str, pl.DataFrame] = {}
    hashes: dict[str, str] = {}
    split_names = {"train": "train", "dev": "dev", "test": "test"}
    for relative, expected in PRIVATE_INPUTS.items():
        path = data_root / relative
        hashes[relative] = check_hash(path, expected["sha256"], relative)
        key = next(name for token, name in split_names.items() if token in path.name)
        frame = standard_frame(pl.read_parquet(path), relative)
        validate_frame(frame, relative, expected["rows"], expected["positive"])
        frames[key] = frame
    return frames, hashes


def load_benchmarks(repo_root: Path) -> dict[str, Benchmark]:
    root = artifact_root(repo_root)
    cb = pl.read_csv(root / "benchmarks/claim_buster/groundtruth.csv")
    cb_ids = cb["Sentence_id"].cast(pl.String).to_numpy()
    cb_labels = (cb["Verdict"] == 1).cast(pl.Int8).to_numpy()

    ct23_input = pl.read_csv(
        root / "benchmarks/ct23/CT23_1B_checkworthy_english_test.tsv",
        separator="\t",
    )
    ct23_gold = pl.read_csv(
        root / "benchmarks/ct23/CT23_1B_checkworthy_english_test_gold.tsv",
        separator="\t",
    )
    if len(ct23_input) != len(ct23_gold):
        raise ValueError("CT23 input and gold row counts differ")
    if ct23_input["Text"].to_list() != ct23_gold["Text"].to_list():
        raise ValueError("CT23 input and gold text order differs")
    ct23_ids = ct23_input["Sentence_id"].cast(pl.String).to_numpy()
    ct23_labels = (ct23_gold["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    result = {
        "ClaimBuster": Benchmark(
            "ClaimBuster", cb_ids, cb["Text"].to_list(), cb_labels
        ),
        "CT23": Benchmark("CT23", ct23_ids, ct23_input["Text"].to_list(), ct23_labels),
    }
    expected = {"ClaimBuster": (1_032, 238), "CT23": (318, 108)}
    for name, benchmark in result.items():
        rows, positives = expected[name]
        if len(benchmark.labels) != rows:
            raise ValueError(f"{name}: expected {rows} rows")
        if int(benchmark.labels.sum()) != positives:
            raise ValueError(f"{name}: expected {positives} positive labels")
        if len(set(benchmark.sentence_ids.tolist())) != rows:
            raise ValueError(f"{name}: sentence IDs are not unique")
    return result


def benchmark_frame(benchmark: Benchmark) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Sentence_id": benchmark.sentence_ids.astype(str),
            "Text": benchmark.texts,
            "class_label": ["Yes" if value else "No" for value in benchmark.labels],
        }
    )


def write_stage_split(
    stage_root: Path, split: str, frame: pl.DataFrame
) -> dict[str, Any]:
    path = stage_root / "CT24_clean" / f"CT24_{split}_clean.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    ids = frame["Sentence_id"].cast(pl.String).to_list()
    return {
        "path": path.relative_to(stage_root.parents[1]).as_posix(),
        "rows": len(frame),
        "positive": int((frame["class_label"] == "Yes").sum()),
        "sentence_id_sha256": stable_string_hash(ids),
        "file_sha256": sha256_file(path),
    }


def prepare_historical_stage(
    run_root: Path,
    private: dict[str, pl.DataFrame],
    benchmarks: dict[str, Benchmark],
) -> dict[str, Any]:
    stage_root = run_root / "staged_data/historical"
    return {
        "track": "historical",
        "splits": {
            "train": write_stage_split(stage_root, "train", private["train"]),
            "dev": write_stage_split(stage_root, "dev", private["dev"]),
            "dev-test": write_stage_split(
                stage_root, "dev-test", benchmark_frame(benchmarks["CT23"])
            ),
            "test": write_stage_split(stage_root, "test", private["test"]),
        },
    }


def strict_split(
    source: pl.DataFrame, benchmarks: dict[str, Benchmark]
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    benchmark_canonical = {
        name: {canonical_text(text) for text in benchmark.texts}
        for name, benchmark in benchmarks.items()
    }
    all_benchmark_texts = set().union(*benchmark_canonical.values())
    canonical = [canonical_text(value) for value in source["Text"].to_list()]
    keep = np.asarray([value not in all_benchmark_texts for value in canonical])
    excluded = source.filter(~keep)
    pool = source.filter(keep)
    groups = np.asarray([value for value, retain in zip(canonical, keep) if retain])
    labels = (pool["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=STRICT_FOLDS, shuffle=True, random_state=STRICT_RANDOM_STATE
    )
    splits = list(splitter.split(np.zeros(len(pool)), labels, groups))
    train_index, dev_index = splits[STRICT_SELECTED_FOLD]
    train = pool[train_index]
    dev = pool[dev_index]

    train_groups = {canonical_text(value) for value in train["Text"].to_list()}
    dev_groups = {canonical_text(value) for value in dev["Text"].to_list()}
    if train_groups & dev_groups:
        raise ValueError("strict split: canonical text crosses train and validation")
    if (train_groups | dev_groups) & all_benchmark_texts:
        raise ValueError("strict split: external benchmark text entered selection data")

    excluded_by_benchmark = {
        name: sum(value in texts for value in canonical)
        for name, texts in benchmark_canonical.items()
    }
    return (
        train,
        dev,
        {
            "source_rows": len(source),
            "deduplicated_pool_rows": len(pool),
            "excluded_rows": len(excluded),
            "excluded_sentence_id_sha256": stable_string_hash(
                excluded["Sentence_id"].cast(pl.String).to_list()
            ),
            "excluded_by_benchmark_text_set": excluded_by_benchmark,
            "train_rows": len(train),
            "train_positive": int((train["class_label"] == "Yes").sum()),
            "validation_rows": len(dev),
            "validation_positive": int((dev["class_label"] == "Yes").sum()),
            "train_validation_text_overlap": 0,
            "selection_external_text_overlap": 0,
            "folds": STRICT_FOLDS,
            "selected_fold": STRICT_SELECTED_FOLD,
            "random_state": STRICT_RANDOM_STATE,
        },
    )


def prepare_strict_stage(
    run_root: Path,
    private: dict[str, pl.DataFrame],
    benchmarks: dict[str, Benchmark],
) -> dict[str, Any]:
    stage_root = run_root / "staged_data/strict"
    train, dev, audit = strict_split(private["train"], benchmarks)
    return {
        "track": "strict",
        "split_audit": audit,
        "splits": {
            "train": write_stage_split(stage_root, "train", train),
            "dev": write_stage_split(stage_root, "dev", dev),
            "test": write_stage_split(stage_root, "test", dev),
        },
    }


def package_versions() -> dict[str, str]:
    names = (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "polars",
        "pyarrow",
        "safetensors",
        "scikit-learn",
        "sentencepiece",
        "torch",
        "transformers",
        "xgboost",
    )
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def seeds_for_track(track: str) -> tuple[int, ...]:
    return HISTORICAL_SEEDS if track == "historical" else STRICT_SEEDS


def validate_historical_environment_values(
    *,
    python_major_minor: tuple[int, int],
    packages: dict[str, str],
    cuda_version: str,
    gpu_name: str,
    cuda_capability: tuple[int, int],
) -> None:
    """Validate the recovered January runtime from explicit observed values."""
    expected_python = HISTORICAL_ENVIRONMENT["python_major_minor"]
    if python_major_minor != expected_python:
        raise RuntimeError(
            f"historical training requires Python {expected_python[0]}."
            f"{expected_python[1]}, got {python_major_minor[0]}."
            f"{python_major_minor[1]}"
        )
    actual_transformers = packages["transformers"]
    expected_transformers = HISTORICAL_ENVIRONMENT["transformers"]
    if actual_transformers != expected_transformers:
        raise RuntimeError(
            "historical training requires transformers "
            f"{expected_transformers}, got {actual_transformers}"
        )
    actual_torch = packages["torch"].split("+", 1)[0]
    expected_torch = HISTORICAL_ENVIRONMENT["torch"]
    if actual_torch != expected_torch:
        raise RuntimeError(
            f"historical training requires PyTorch {expected_torch}, "
            f"got {packages['torch']}"
        )
    expected_cuda = HISTORICAL_ENVIRONMENT["cuda_runtime"]
    if cuda_version != expected_cuda:
        raise RuntimeError(
            f"historical training requires CUDA runtime {expected_cuda}, "
            f"got {cuda_version}"
        )
    expected_gpu = HISTORICAL_ENVIRONMENT["gpu_name"]
    if gpu_name != expected_gpu:
        raise RuntimeError(
            f"historical training requires {expected_gpu}, got {gpu_name}"
        )
    expected_capability = HISTORICAL_ENVIRONMENT["cuda_compute_capability"]
    if cuda_capability != expected_capability:
        raise RuntimeError(
            "historical training requires CUDA compute capability "
            f"{expected_capability[0]}.{expected_capability[1]}, got "
            f"{cuda_capability[0]}.{cuda_capability[1]}"
        )


def validate_historical_environment() -> None:
    """Fail before training when the recovered January runtime is not active."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("historical training requires a CUDA GPU")
    validate_historical_environment_values(
        python_major_minor=sys.version_info[:2],
        packages=package_versions(),
        cuda_version=str(torch.version.cuda),
        gpu_name=torch.cuda.get_device_name(0),
        cuda_capability=tuple(torch.cuda.get_device_capability(0)),
    )


def environment_record(include_cuda: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }
    if include_cuda:
        import torch

        result["torch_cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["gpu_count"] = torch.cuda.device_count()
        result["gpus"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return result


def assert_run_root_outside_repo(repo_root: Path, run_root: Path) -> None:
    resolved_repo = repo_root.resolve()
    resolved_run = run_root.resolve()
    if resolved_run == resolved_repo or resolved_repo in resolved_run.parents:
        raise ValueError("--run-root must be outside the Git repository")


def preflight(repo_root: Path, data_root: Path, run_root: Path) -> dict[str, Any]:
    assert_run_root_outside_repo(repo_root, run_root)
    public_hashes = validate_public_inputs(repo_root)
    private, private_hashes = load_private_inputs(data_root)
    benchmarks = load_benchmarks(repo_root)
    historical = prepare_historical_stage(run_root, private, benchmarks)
    strict = prepare_strict_stage(run_root, private, benchmarks)
    result = {
        "change_note": (
            "Sérgio Pinto, 2026-08-25 19:11 WEST — validated the exact inputs "
            "and built separate historical and strict training stages."
        ),
        "status": "PASS",
        "repo_root": str(repo_root.resolve()),
        "data_root": str(data_root.resolve()),
        "run_root": str(run_root.resolve()),
        "run_contract_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "specification": sha256_file(run_spec_path(repo_root)),
            "requirements": sha256_file(requirements_path(repo_root)),
        },
        "public_input_sha256": public_hashes,
        "private_input_sha256": private_hashes,
        "tracks": {"historical": historical, "strict": strict},
        "environment": environment_record(),
    }
    write_json(run_root / "preflight.json", result)
    print("PREFLIGHT: PASS")
    print(f"Run root: {run_root.resolve()}")
    print(
        "Strict exclusions: "
        f"{strict['split_audit']['excluded_rows']} benchmark-overlapping rows"
    )
    return result


def load_preflight(repo_root: Path, run_root: Path) -> dict[str, Any]:
    path = run_root / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("preflight.json is missing; run preflight first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise ValueError("preflight did not pass")
    expected_contract = {
        "runner": sha256_file(Path(__file__).resolve()),
        "specification": sha256_file(run_spec_path(repo_root)),
        "requirements": sha256_file(requirements_path(repo_root)),
    }
    if value.get("run_contract_sha256") != expected_contract:
        raise ValueError("preflight was created by a different run contract")
    validate_public_inputs(repo_root)
    for track in ("historical", "strict"):
        for item in value["tracks"][track]["splits"].values():
            path = run_root / item["path"]
            check_hash(path, item["file_sha256"], f"staged {track} split")
    return value


def load_trainer(repo_root: Path, model_path: Path):
    path = trainer_path(repo_root)
    check_hash(path, TRAINER_SHA256, "training source")
    spec = importlib.util.spec_from_file_location("cikm_historical_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.MODEL_REGISTRY["deberta-v3-large"]["hf_name"] = str(model_path)
    return module


def download_base_model(run_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = run_root / "base_model"
    snapshot_download(
        repo_id=BASE_MODEL_REPOSITORY,
        revision=BASE_MODEL_REVISION,
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=(
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "spm.model",
            "tokenizer_config.json",
        ),
    )
    required = ("config.json", "spm.model", "tokenizer_config.json")
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise FileNotFoundError(f"base model snapshot is missing {missing}")
    weight_files = [
        path
        for name in ("model.safetensors", "pytorch_model.bin")
        if (path := target / name).is_file()
    ]
    if len(weight_files) != 1:
        raise ValueError("base model snapshot must contain exactly one weight file")
    identity = {
        "repository": BASE_MODEL_REPOSITORY,
        "revision": BASE_MODEL_REVISION,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [*(target / name for name in required), *weight_files]
        },
    }
    write_json(run_root / "base_model_manifest.json", identity)
    return target


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    return {
        path.relative_to(model_dir).as_posix(): sha256_file(path)
        for path in sorted(files)
    }


def training_configuration(track: str, seed: int) -> dict[str, Any]:
    configuration = {
        "model_key": "deberta-v3-large",
        "max_length": 128,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "num_epochs": 5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "eval_steps": 100,
        "save_steps": 200,
        "early_stopping_patience": 5,
        "eval_on_devtest": track == "historical",
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": False,
        "seed": seed,
        "use_focal_loss": True,
        "focal_gamma": 2.0,
        "use_llrd": True,
        "llrd_decay": 0.9,
        "use_cosine_schedule": True,
        "use_rdrop": True,
        "rdrop_alpha": 1.0,
        "use_fgm": True,
        "fgm_epsilon": 1.0,
    }
    if track == "historical":
        if seed not in HISTORICAL_SEEDS:
            raise ValueError(f"unsupported historical seed: {seed}")
        configuration.update(
            {
                "gradient_accumulation_steps": 2 if seed == 42 else 4,
                "fp16": True,
                "bf16": False,
                "fgm_epsilon": 0.5 if seed == 42 else 1.0,
            }
        )
    return configuration


def receipt_binding(
    run_root: Path,
    preflight_record: dict[str, Any],
    track: str,
    seed: int,
) -> dict[str, Any]:
    # Sérgio Pinto, 2026-08-25 19:35 WEST — bind every trained seed to the
    # exact preflight, staged splits, run contract, and training parameters.
    return {
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "run_contract_sha256": preflight_record["run_contract_sha256"],
        "staged_splits": preflight_record["tracks"][track]["splits"],
        "training_configuration": training_configuration(track, seed),
    }


def verify_model_receipt(
    run_root: Path,
    track: str,
    seed: int,
    preflight_record: dict[str, Any],
    base_identity: dict[str, Any],
) -> None:
    model_root = run_root / f"models/{track}/seed_{seed}"
    receipt_path = model_root / "run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"seed {seed}: run receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("track") != track
        or receipt.get("seed") != seed
        or receipt.get("base_model_revision") != BASE_MODEL_REVISION
        or receipt.get("base_model_identity") != base_identity
        or receipt.get("training_source_sha256") != TRAINER_SHA256
        or receipt.get("binding")
        != receipt_binding(run_root, preflight_record, track, seed)
    ):
        raise ValueError(f"seed {seed}: stale or invalid run receipt")
    for relative, expected in receipt["model_file_sha256"].items():
        check_hash(model_root / "best_model" / relative, expected, f"seed {seed}")


def train_track(repo_root: Path, run_root: Path, track: str) -> None:
    preflight_record = load_preflight(repo_root, run_root)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA GPU")
    if track == "historical":
        validate_historical_environment()
    base_model = download_base_model(run_root)
    base_model_identity = json.loads(
        (run_root / "base_model_manifest.json").read_text(encoding="utf-8")
    )
    trainer = load_trainer(repo_root, base_model)
    stage_root = run_root / f"staged_data/{track}"
    environment = environment_record(include_cuda=True)
    write_json(run_root / "environment.json", environment)

    for seed in seeds_for_track(track):
        output_dir = run_root / f"models/{track}/seed_{seed}"
        receipt_path = output_dir / "run_receipt.json"
        if receipt_path.is_file():
            verify_model_receipt(
                run_root, track, seed, preflight_record, base_model_identity
            )
            print(f"Seed {seed}: bound receipt found; skipping")
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"incomplete output exists at {output_dir}; use a new run root"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = trainer.Config(**training_configuration(track, seed))
        log_path = output_dir / "train.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            tee = Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                trainer.train_model(config, stage_root, output_dir)
        best_model = output_dir / "best_model"
        if not (best_model / "model.safetensors").is_file():
            raise FileNotFoundError(f"seed {seed}: best model was not saved")
        receipt = {
            "status": "PASS",
            "track": track,
            "seed": seed,
            "base_model_repository": BASE_MODEL_REPOSITORY,
            "base_model_revision": BASE_MODEL_REVISION,
            "base_model_identity": base_model_identity,
            "training_source_sha256": TRAINER_SHA256,
            "binding": receipt_binding(run_root, preflight_record, track, seed),
            "environment": environment,
            "model_file_sha256": model_file_hashes(best_model),
        }
        write_json(receipt_path, receipt)
        verify_model_receipt(
            run_root, track, seed, preflight_record, base_model_identity
        )
        print(f"Seed {seed}: PASS")


def infer_probabilities(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            values.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def evaluate(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Metrics:
    predictions = probabilities >= threshold
    positive = labels.astype(bool)
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    tn = int(np.sum(~predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    return Metrics(
        threshold=float(threshold),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        accuracy=float(accuracy_score(labels, predictions)),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> Metrics:
    candidates = [evaluate(labels, probabilities, value) for value in THRESHOLDS]
    return max(candidates, key=lambda value: value.f1)


def temperature_scale(probabilities: list[np.ndarray]) -> np.ndarray:
    scaled = []
    for values in probabilities:
        clipped = np.clip(values, 1e-8, 1 - 1e-8)
        logits = np.log(clipped / (1 - clipped))
        scaled.append(1 / (1 + np.exp(-(logits / TEMPERATURE))))
    return np.mean(scaled, axis=0)


def feature_frame(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path).with_columns(pl.col("sentence_id").cast(pl.String))
    if frame["sentence_id"].n_unique() != len(frame):
        raise ValueError(f"feature IDs are not unique: {path}")
    return frame


def align_features(
    features: pl.DataFrame, labels: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    canonical = labels.select(
        pl.col("Sentence_id").cast(pl.String).alias("sentence_id"),
        (pl.col("class_label") == "Yes").cast(pl.Int8).alias("label"),
    ).with_row_index("__order")
    if canonical["sentence_id"].n_unique() != len(canonical):
        raise ValueError("label IDs are not unique")
    if set(canonical["sentence_id"].to_list()) - set(features["sentence_id"].to_list()):
        raise ValueError("features do not cover every selected sentence ID")
    aligned = canonical.join(
        features, on="sentence_id", how="left", validate="1:1"
    ).sort("__order")
    if aligned.select(FUSION_FEATURES).null_count().row(0) != (0,) * len(
        FUSION_FEATURES
    ):
        raise ValueError("aligned fusion features contain null values")
    return aligned.select(FUSION_FEATURES).to_numpy(), aligned["label"].to_numpy()


def fit_xgboost(features: np.ndarray, labels: np.ndarray):
    from xgboost import XGBClassifier

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


def retained_ct23_fusion(repo_root: Path) -> Metrics:
    """Recompute the retained January CT23 Fusion paper cell."""
    root = artifact_root(repo_root)
    with np.load(
        root / "llm_features_classifier/matrices.npz", allow_pickle=False
    ) as bundle:
        feature_names = bundle["feature_names"].tolist()
        feature_indices = [feature_names.index(name) for name in FUSION_FEATURES]
        train_x = np.vstack(
            [
                bundle["X_train"][:, feature_indices],
                bundle["X_dev"][:, feature_indices],
            ]
        )
        train_y = np.concatenate([bundle["y_train"], bundle["y_dev"]])
        ct23_x = bundle["X_ct23"][:, feature_indices]
        ct23_y = bundle["y_ct23"].astype(np.int8)

    seed_42 = np.load(root / "encoder_only/ct23_probs.npy", allow_pickle=False)
    seed_456_record = json.loads(
        (root / "encoder_only/ct23_seed_456_probs.json").read_text(encoding="utf-8")
    )
    if (
        seed_456_record.get("benchmark") != "CT23"
        or seed_456_record.get("historical_label") != "seed_456"
        or seed_456_record.get("effective_rng_seed") != 456
    ):
        raise ValueError("retained seed_456 CT23 probability metadata changed")
    seed_456 = np.asarray(seed_456_record["probabilities"], dtype=np.float32)
    if seed_42.shape != ct23_y.shape or seed_456.shape != ct23_y.shape:
        raise ValueError("retained CT23 probability length changed")
    if np.any((seed_42 < 0) | (seed_42 > 1)) or np.any((seed_456 < 0) | (seed_456 > 1)):
        raise ValueError("retained CT23 probabilities are outside [0, 1]")

    scaler, classifier = fit_xgboost(train_x, train_y)
    xgboost_probabilities = classifier.predict_proba(scaler.transform(ct23_x))[:, 1]
    encoder_probabilities = temperature_scale([seed_42, seed_456])
    fusion_probabilities = (
        ENCODER_WEIGHT * encoder_probabilities
        + (1 - ENCODER_WEIGHT) * xgboost_probabilities
    )
    return evaluate(
        ct23_y,
        fusion_probabilities,
        HISTORICAL_THRESHOLDS["fusion"]["CT23"],
    )


def retained_ct23_result(metrics: Metrics) -> dict[str, Any]:
    """Build the deterministic public record for the retained CT23 cell."""
    return {
        "change_note": (
            "Sérgio Pinto, 2026-08-28 01:11 WEST — added the checksum-bound "
            "CT23 Fusion reconstruction to the public paper-value inventory."
        ),
        "status": "PASS",
        "dataset": "CT23",
        "reproduction_type": (
            "deterministic reconstruction from retained January numerical components"
        ),
        "method": {
            "encoder_runs": [
                {"historical_label": "seed_0", "effective_rng_seed": 42},
                {"historical_label": "seed_456", "effective_rng_seed": 456},
            ],
            "encoder_temperature": TEMPERATURE,
            "features": list(FUSION_FEATURES),
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "scale_pos_weight": 3,
                "random_state": 42,
                "version": "2.1.1",
            },
            "encoder_weight": ENCODER_WEIGHT,
            "xgboost_weight": 1 - ENCODER_WEIGHT,
            "threshold": HISTORICAL_THRESHOLDS["fusion"]["CT23"],
        },
        "inputs": {
            name: digest
            for name, digest in PUBLIC_INPUTS.items()
            if name
            in {
                "llm_features_classifier/matrices.npz",
                "encoder_only/ct23_probs.npy",
                "encoder_only/ct23_seed_456_probs.json",
            }
        },
        "metrics": asdict(metrics),
        "paper_f1": PAPER_TARGETS["fusion"]["CT23"],
    }


def benchmark_feature_path(root: Path, name: str) -> Path:
    filename = {
        "ClaimBuster": "CB_groundtruth_llm_features.parquet",
        "CT23": "CT23_llm_features.parquet",
    }[name]
    return root / "benchmark_llm_features" / filename


def evaluation_feature_components(
    repo_root: Path,
    run_root: Path,
    track: str,
    benchmarks: dict[str, Benchmark],
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    root = artifact_root(repo_root)
    if track == "historical":
        with np.load(
            root / "llm_features_classifier/matrices.npz", allow_pickle=False
        ) as bundle:
            feature_names = bundle["feature_names"].tolist()
            feature_indices = [feature_names.index(name) for name in FUSION_FEATURES]
            ct23_train_x = np.vstack(
                [
                    bundle["X_train"][:, feature_indices],
                    bundle["X_dev"][:, feature_indices],
                ]
            )
            train_y = np.concatenate([bundle["y_train"], bundle["y_dev"]])
            if not np.array_equal(bundle["y_ct23"], benchmarks["CT23"].labels):
                raise ValueError("CT23: historical label order changed")
            ct23_evaluation_x = bundle["X_ct23"][:, feature_indices].copy()

            score_indices = [
                feature_names.index(name)
                for name in ("check_score", "verif_score", "harm_score")
            ]
            claimbuster_scores = np.vstack(
                [
                    bundle["X_train"][:, score_indices],
                    bundle["X_dev"][:, score_indices],
                ]
            )
            claimbuster_train_x = np.column_stack(
                [claimbuster_scores, claimbuster_scores.mean(axis=1)]
            )

        retained_claimbuster = pl.read_parquet(
            root
            / "fusion_benchmark_features/CB_groundtruth_features_mistral-small-24b.parquet"
        )
        if retained_claimbuster["text"].to_list() != benchmarks["ClaimBuster"].texts:
            raise ValueError("ClaimBuster: retained Fusion text order changed")
        if not np.array_equal(
            retained_claimbuster["label"].to_numpy(),
            benchmarks["ClaimBuster"].labels,
        ):
            raise ValueError("ClaimBuster: retained Fusion label order changed")
        claimbuster_x = retained_claimbuster.select(
            CLAIMBUSTER_FUSION_FEATURES
        ).to_numpy().astype(np.float32)
        claimbuster_x = np.nan_to_num(claimbuster_x, nan=50.0)

        claimbuster_scaler, claimbuster_classifier = fit_xgboost(
            claimbuster_train_x, train_y
        )
        ct23_scaler, ct23_classifier = fit_xgboost(ct23_train_x, train_y)
        probabilities = {
            "ClaimBuster": claimbuster_classifier.predict_proba(
                claimbuster_scaler.transform(claimbuster_x)
            )[:, 1],
            "CT23": ct23_classifier.predict_proba(
                ct23_scaler.transform(ct23_evaluation_x)
            )[:, 1],
        }
        return probabilities, None, None

    train_features = feature_frame(
        root / "ct24_llm_features_v4/train_llm_features.parquet"
    )
    stage = run_root / f"staged_data/{track}/CT24_clean"
    train_labels = pl.read_parquet(stage / "CT24_train_clean.parquet")
    train_x, train_y = align_features(train_features, train_labels)

    validation_x: np.ndarray | None = None
    validation_y: np.ndarray | None = None
    validation_labels = pl.read_parquet(stage / "CT24_dev_clean.parquet")
    validation_x, validation_y = align_features(train_features, validation_labels)

    scaler, classifier = fit_xgboost(train_x, train_y)
    probabilities: dict[str, np.ndarray] = {}
    for name, benchmark in benchmarks.items():
        labels = benchmark_frame(benchmark)
        features = feature_frame(benchmark_feature_path(root, name))
        values, aligned_labels = align_features(features, labels)
        if not np.array_equal(aligned_labels, benchmark.labels):
            raise ValueError(f"{name}: feature alignment changed label order")
        probabilities[name] = classifier.predict_proba(scaler.transform(values))[:, 1]

    validation_probabilities = None
    if validation_x is not None:
        validation_probabilities = classifier.predict_proba(
            scaler.transform(validation_x)
        )[:, 1]
    return probabilities, validation_probabilities, validation_y


def retained_claimbuster_encoder_fusion(
    repo_root: Path,
) -> tuple[Metrics, Metrics, Metrics, dict[str, np.ndarray]]:
    """Recompute the ClaimBuster Encoder Only, XGBoost and Fusion cells."""
    benchmarks = load_benchmarks(repo_root)
    benchmark = benchmarks["ClaimBuster"]
    root = artifact_root(repo_root)
    with np.load(
        root / "encoder_only/claimbuster_seed_42_predictions.npz",
        allow_pickle=False,
    ) as bundle:
        sentence_ids = bundle["sentence_ids"].astype(str)
        labels = bundle["labels"].astype(np.int8)
        encoder_probabilities = bundle["probabilities"].astype(np.float64)
    if not np.array_equal(sentence_ids, benchmark.sentence_ids.astype(str)):
        raise ValueError("retained ClaimBuster sentence order changed")
    if not np.array_equal(labels, benchmark.labels):
        raise ValueError("retained ClaimBuster labels changed")
    if encoder_probabilities.shape != benchmark.labels.shape:
        raise ValueError("retained ClaimBuster probability length changed")
    if np.any((encoder_probabilities < 0) | (encoder_probabilities > 1)):
        raise ValueError("retained ClaimBuster probabilities are outside [0, 1]")

    xgboost, validation_probabilities, validation_labels = (
        evaluation_feature_components(
            repo_root, repo_root / "unused-historical-run", "historical", benchmarks
        )
    )
    if validation_probabilities is not None or validation_labels is not None:
        raise ValueError("historical ClaimBuster reconstruction returned validation data")
    xgboost_probabilities = xgboost["ClaimBuster"].astype(np.float64)
    fusion_probabilities = (
        ENCODER_WEIGHT * encoder_probabilities
        + (1 - ENCODER_WEIGHT) * xgboost_probabilities
    )
    encoder_metrics = evaluate(
        benchmark.labels,
        encoder_probabilities,
        HISTORICAL_THRESHOLDS["encoder_only"]["ClaimBuster"],
    )
    xgboost_metrics = evaluate(benchmark.labels, xgboost_probabilities, 0.45)
    fusion_metrics = evaluate(
        benchmark.labels,
        fusion_probabilities,
        HISTORICAL_THRESHOLDS["fusion"]["ClaimBuster"],
    )
    arrays = {
        "sentence_ids": benchmark.sentence_ids.astype(str),
        "labels": benchmark.labels.astype(np.int8),
        "encoder_probabilities": encoder_probabilities.astype(np.float32),
        "xgboost_probabilities": xgboost_probabilities.astype(np.float32),
        "fusion_probabilities": fusion_probabilities.astype(np.float32),
    }
    return encoder_metrics, xgboost_metrics, fusion_metrics, arrays


def retained_claimbuster_result(
    encoder_metrics: Metrics,
    xgboost_metrics: Metrics,
    fusion_metrics: Metrics,
) -> dict[str, Any]:
    """Build the checksum-bound record for the ClaimBuster paper cells."""
    return {
        "change_note": (
            "Sérgio Pinto, 2026-08-28 03:38 WEST — bound the ClaimBuster "
            "Encoder Only and Fusion cells to the fresh A10 replication."
        ),
        "status": "PASS",
        "dataset": "ClaimBuster",
        "reproduction_type": (
            "fresh A10 training with deterministic public numerical reconstruction"
        ),
        "encoder_run": {
            "historical_label": "seed_0",
            "effective_rng_seed": 42,
            "model_sha256": (
                "3765638fb1f60a87741fdd6c576faeece8be1cb520d074d08e7e2abe8c3feb0f"
            ),
            "gpu": "NVIDIA A10",
            "torch": "2.7.0+cu128",
            "transformers": "4.44.0",
        },
        "method": {
            "encoder_threshold": HISTORICAL_THRESHOLDS["encoder_only"][
                "ClaimBuster"
            ],
            "xgboost_features": list(CLAIMBUSTER_FUSION_FEATURES),
            "xgboost_threshold": 0.45,
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "scale_pos_weight": 3,
                "random_state": 42,
                "version": "2.1.1",
            },
            "encoder_weight": ENCODER_WEIGHT,
            "xgboost_weight": 1 - ENCODER_WEIGHT,
            "fusion_threshold": HISTORICAL_THRESHOLDS["fusion"]["ClaimBuster"],
        },
        "inputs": {
            name: digest
            for name, digest in PUBLIC_INPUTS.items()
            if name
            in {
                "llm_features_classifier/matrices.npz",
                "fusion_benchmark_features/CB_groundtruth_features_mistral-small-24b.parquet",
                "encoder_only/claimbuster_seed_42_predictions.npz",
            }
        },
        "metrics": {
            "encoder_only": asdict(encoder_metrics),
            "xgboost": asdict(xgboost_metrics),
            "fusion": asdict(fusion_metrics),
        },
        "paper_f1": {
            "encoder_only": PAPER_TARGETS["encoder_only"]["ClaimBuster"],
            "fusion": PAPER_TARGETS["fusion"]["ClaimBuster"],
        },
    }


def verify_model_receipts(
    run_root: Path, track: str, preflight_record: dict[str, Any]
) -> None:
    base_manifest_path = run_root / "base_model_manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError("base model manifest is missing")
    base_identity = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    for seed in seeds_for_track(track):
        verify_model_receipt(run_root, track, seed, preflight_record, base_identity)


def checksums_manifest(output_dir: Path) -> None:
    track = output_dir.name
    files = {
        "../../environment.json": output_dir.parent.parent / "environment.json",
        "../../preflight.json": output_dir.parent.parent / "preflight.json",
        "../../base_model_manifest.json": (
            output_dir.parent.parent / "base_model_manifest.json"
        ),
        "per_example_predictions.npz": output_dir / "per_example_predictions.npz",
        "summary.json": output_dir / "summary.json",
    }
    for seed in seeds_for_track(track):
        relative = f"../../models/{track}/seed_{seed}/run_receipt.json"
        files[relative] = (
            output_dir.parent.parent / f"models/{track}/seed_{seed}/run_receipt.json"
        )
    lines = []
    for relative, path in files.items():
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {relative}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate_track(
    repo_root: Path, run_root: Path, track: str, batch_size: int
) -> None:
    preflight_record = load_preflight(repo_root, run_root)
    verify_model_receipts(run_root, track, preflight_record)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarks = load_benchmarks(repo_root)
    stage_dev = standard_frame(
        pl.read_parquet(
            run_root / f"staged_data/{track}/CT24_clean/CT24_dev_clean.parquet"
        ),
        f"{track} dev",
    )
    arrays: dict[str, np.ndarray] = {
        f"{name.lower()}_sentence_ids": benchmark.sentence_ids.astype(str)
        for name, benchmark in benchmarks.items()
    }
    arrays.update(
        {
            f"{name.lower()}_labels": benchmark.labels.astype(np.int8)
            for name, benchmark in benchmarks.items()
        }
    )
    dev_probabilities: dict[int, np.ndarray] = {}
    benchmark_probabilities: dict[int, dict[str, np.ndarray]] = {}

    track_seeds = seeds_for_track(track)
    for seed in track_seeds:
        model_dir = run_root / f"models/{track}/seed_{seed}/best_model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, use_safetensors=True
        ).to(device)
        dev_probabilities[seed] = infer_probabilities(
            model, tokenizer, stage_dev["Text"].to_list(), device, batch_size
        )
        benchmark_probabilities[seed] = {}
        for name, benchmark in benchmarks.items():
            values = infer_probabilities(
                model, tokenizer, benchmark.texts, device, batch_size
            )
            benchmark_probabilities[seed][name] = values
            arrays[f"{name.lower()}_seed_{seed}_probabilities"] = values
        del model
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    xgb_external, xgb_dev, dev_labels = evaluation_feature_components(
        repo_root, run_root, track, benchmarks
    )
    for name, values in xgb_external.items():
        arrays[f"{name.lower()}_xgboost_probabilities"] = values.astype(np.float32)

    summary: dict[str, Any] = {
        "change_note": (
            "Sérgio Pinto, 2026-08-25 19:11 WEST — evaluated the checksum-bound "
            f"{track} Encoder Only and Fusion GPU run."
        ),
        "status": "PASS",
        "track": track,
        "paper_targets": PAPER_TARGETS,
        "base_model": {
            "repository": BASE_MODEL_REPOSITORY,
            "revision": BASE_MODEL_REVISION,
        },
        "seeds": list(track_seeds),
        "encoder_temperature": TEMPERATURE,
        "fusion_encoder_weight": ENCODER_WEIGHT,
        "encoder_only": {},
        "fusion": {},
    }

    if track == "strict":
        if xgb_dev is None or dev_labels is None:
            raise RuntimeError("strict validation features are missing")
        encoded_dev = temperature_scale(
            [dev_probabilities[seed] for seed in track_seeds]
        )
        fused_dev = ENCODER_WEIGHT * encoded_dev + (1 - ENCODER_WEIGHT) * xgb_dev
        encoder_selection = select_threshold(
            (stage_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy(),
            dev_probabilities[0],
        )
        fusion_selection = select_threshold(dev_labels, fused_dev)
        summary["selection"] = {
            "split": "strict internal CT24 training fold",
            "encoder_only": asdict(encoder_selection),
            "fusion": asdict(fusion_selection),
        }
        for name, benchmark in benchmarks.items():
            encoder_values = benchmark_probabilities[0][name]
            ensemble = temperature_scale(
                [benchmark_probabilities[seed][name] for seed in track_seeds]
            )
            fusion_values = (
                ENCODER_WEIGHT * ensemble + (1 - ENCODER_WEIGHT) * xgb_external[name]
            )
            arrays[f"{name.lower()}_fusion_probabilities"] = fusion_values.astype(
                np.float32
            )
            summary["encoder_only"][name] = {
                "selected_on_internal_validation": asdict(
                    evaluate(
                        benchmark.labels, encoder_values, encoder_selection.threshold
                    )
                ),
                "paper_target": PAPER_TARGETS["encoder_only"][name],
            }
            summary["fusion"][name] = {
                "selected_on_internal_validation": asdict(
                    evaluate(
                        benchmark.labels, fusion_values, fusion_selection.threshold
                    )
                ),
                "paper_target": PAPER_TARGETS["fusion"][name],
            }
    else:
        summary["selection"] = {
            "split": "recorded historical benchmark-specific thresholds",
            "thresholds": HISTORICAL_THRESHOLDS,
        }
        summary["historical_run_labels"] = {
            str(seed): HISTORICAL_RUN_LABELS[seed] for seed in track_seeds
        }
        summary["encoder_sources"] = {
            "ClaimBuster": "fresh seed_0 run (effective RNG seed 42)",
            "CT23": "retained seed_0 run (effective RNG seed 42)",
        }
        for name, benchmark in benchmarks.items():
            ensemble_values = temperature_scale(
                [benchmark_probabilities[seed][name] for seed in track_seeds]
            )
            encoder_values = benchmark_probabilities[42][name]
            arrays[f"{name.lower()}_encoder_probabilities"] = encoder_values.astype(
                np.float32
            )
            summary["encoder_only"][name] = {
                "recorded_threshold": asdict(
                    evaluate(
                        benchmark.labels,
                        encoder_values,
                        HISTORICAL_THRESHOLDS["encoder_only"][name],
                    )
                ),
                "paper_target": PAPER_TARGETS["encoder_only"][name],
            }
            fusion_encoder_values = (
                encoder_values if name == "ClaimBuster" else ensemble_values
            )
            fusion_values = ENCODER_WEIGHT * fusion_encoder_values + (
                1 - ENCODER_WEIGHT
            ) * xgb_external[name]
            arrays[f"{name.lower()}_fusion_probabilities"] = fusion_values.astype(
                np.float32
            )
            summary["fusion"][name] = {
                "recorded_threshold": asdict(
                    evaluate(
                        benchmark.labels,
                        fusion_values,
                        HISTORICAL_THRESHOLDS["fusion"][name],
                    )
                ),
                "paper_target": PAPER_TARGETS["fusion"][name],
            }

    output_dir = run_root / f"results/{track}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "per_example_predictions.npz"
    write_npz(prediction_path, arrays)
    summary["prediction_sha256"] = sha256_file(prediction_path)
    write_json(output_dir / "summary.json", summary)
    checksums_manifest(output_dir)
    print(f"{track.upper()} EVALUATION: PASS")
    print(f"Summary: {output_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--data-root", type=Path, required=True)
    preflight_parser.add_argument("--run-root", type=Path, required=True)
    subparsers.add_parser(
        "verify-retained-ct23",
        help="recompute the retained historical CT23 Fusion paper cell",
    )
    subparsers.add_parser(
        "verify-retained-claimbuster",
        help="recompute the fresh A10 ClaimBuster Encoder Only and Fusion cells",
    )

    for name in ("train", "evaluate"):
        child = subparsers.add_parser(name)
        child.add_argument("--track", choices=("historical", "strict"), required=True)
        child.add_argument("--run-root", type=Path, required=True)
        if name == "evaluate":
            child.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "verify-retained-ct23":
        metrics = retained_ct23_fusion(repo_root)
        expected = PAPER_TARGETS["fusion"]["CT23"]
        if round(metrics.f1, 3) != expected:
            raise RuntimeError(
                f"retained CT23 Fusion F1 is {metrics.f1:.6f}, expected {expected:.3f}"
            )
        result_path = repo_root / "results/fusion_ct23_reproduction_2026-08-28.json"
        write_json(result_path, retained_ct23_result(metrics))
        print("RETAINED CT23 FUSION: PASS")
        print(f"F1: {metrics.f1:.6f} -> {metrics.f1:.3f}")
        print(f"Result: {result_path}")
        return 0
    if args.command == "verify-retained-claimbuster":
        encoder, xgboost, fusion, arrays = retained_claimbuster_encoder_fusion(
            repo_root
        )
        expected_encoder = PAPER_TARGETS["encoder_only"]["ClaimBuster"]
        expected_fusion = PAPER_TARGETS["fusion"]["ClaimBuster"]
        if round(encoder.f1, 3) != expected_encoder:
            raise RuntimeError(
                f"ClaimBuster Encoder F1 is {encoder.f1:.6f}, "
                f"expected {expected_encoder:.3f}"
            )
        if round(fusion.f1, 3) != expected_fusion:
            raise RuntimeError(
                f"ClaimBuster Fusion F1 is {fusion.f1:.6f}, "
                f"expected {expected_fusion:.3f}"
            )
        output_dir = (
            repo_root / "results/claimbuster_encoder_fusion_reproduction_2026-08-28"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_npz(output_dir / "per_example_predictions.npz", arrays)
        write_json(
            output_dir / "summary.json",
            retained_claimbuster_result(encoder, xgboost, fusion),
        )
        checksums_manifest(output_dir)
        print("CLAIMBUSTER ENCODER AND FUSION: PASS")
        print(f"Encoder F1: {encoder.f1:.6f} -> {encoder.f1:.3f}")
        print(f"XGBoost F1: {xgboost.f1:.6f} -> {xgboost.f1:.3f}")
        print(f"Fusion F1: {fusion.f1:.6f} -> {fusion.f1:.3f}")
        print(f"Result: {output_dir}")
        return 0

    run_root = args.run_root.resolve()
    if args.command == "preflight":
        preflight(repo_root, args.data_root.resolve(), run_root)
    elif args.command == "train":
        train_track(repo_root, run_root, args.track)
    else:
        evaluate_track(repo_root, run_root, args.track, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
