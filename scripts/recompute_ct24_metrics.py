from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/ct24_metric_reproduction_2026-05-12.json")
THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class ArtifactMetrics:
    name: str
    probabilities_path: str
    labels_path: str
    probabilities_sha256: str
    n_samples: int
    n_positive: int
    best: ThresholdMetrics
    thresholds: list[ThresholdMetrics]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(source_root: Path) -> np.ndarray:
    path = source_root / "data/processed/CT24_clean/CT24_test_clean.parquet"
    table = pq.read_table(path, columns=["class_label"])
    labels = np.array([1 if value.as_py() == "Yes" else 0 for value in table.column("class_label")], dtype=np.int64)
    if labels.shape != (341,):
        raise ValueError(f"Expected 341 CT24 test labels, found {labels.shape[0]}")
    return labels


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> ThresholdMetrics:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    return ThresholdMetrics(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def evaluate_artifact(labels: np.ndarray, labels_path: Path, name: str, path: Path) -> ArtifactMetrics:
    probabilities = np.load(path)
    if probabilities.shape != labels.shape:
        raise ValueError(f"{name}: probability shape {probabilities.shape} does not match labels {labels.shape}")
    thresholds = [compute_metrics(labels, probabilities, threshold) for threshold in THRESHOLDS]
    best = max(thresholds, key=lambda row: (row.f1, row.threshold))
    return ArtifactMetrics(
        name=name,
        probabilities_path=str(path),
        labels_path=str(labels_path),
        probabilities_sha256=sha256_file(path),
        n_samples=int(labels.shape[0]),
        n_positive=int(np.sum(labels)),
        best=best,
        thresholds=thresholds,
    )


def build_report(source_root: Path) -> dict:
    repo_root = DEFAULT_REPO_ROOT
    clean_labels_path = source_root / "data/processed/CT24_clean/CT24_test_clean.parquet"
    clean_labels = load_labels(source_root)
    four_head_run = repo_root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    four_head_labels_path = four_head_run / "test_labels.npy"
    four_head_labels = np.load(four_head_labels_path)
    artifacts = [
        (
            "single_deberta_local",
            source_root / "experiments/results/deberta_checkworthy/deberta-v3-large/test_probs.npy",
            clean_labels,
            clean_labels_path,
        ),
        (
            "ensemble_seed_0",
            source_root / "lambda_backup/ubuntu/ensemble_results/seed_0/deberta-v3-large/test_probs.npy",
            clean_labels,
            clean_labels_path,
        ),
        (
            "ensemble_seed_123",
            source_root / "lambda_backup/ubuntu/ensemble_results/seed_123/deberta-v3-large/test_probs.npy",
            clean_labels,
            clean_labels_path,
        ),
        (
            "ensemble_seed_456",
            source_root / "lambda_backup/ubuntu/ensemble_results/seed_456/deberta-v3-large/test_probs.npy",
            clean_labels,
            clean_labels_path,
        ),
        (
            "ensemble_temp_0_3",
            source_root / "lambda_backup/ubuntu/ensemble_results/test_temp_0.3_probs.npy",
            clean_labels,
            clean_labels_path,
        ),
        (
            "four_head_mtl_retrain_2026_05_12",
            four_head_run / "test_probs.npy",
            four_head_labels,
            four_head_labels_path,
        ),
    ]
    artifact_metrics = [evaluate_artifact(labels, labels_path, name, path) for name, path, labels, labels_path in artifacts]
    return {
        "dataset": {
            "name": "CT24 check-worthiness English test",
            "default_labels_path": str(clean_labels_path),
            "n_samples": int(clean_labels.shape[0]),
            "n_positive": int(np.sum(clean_labels)),
            "positive_label": "Yes",
            "thresholds": list(THRESHOLDS),
            "label_order_note": (
                "The four-head MTL run uses the post-join row order from CT24_features joined with "
                "CT24_llm_features_v4. Its packaged test_labels.npy must be used when recomputing that row."
            ),
        },
        "artifacts": [asdict(metrics) for metrics in artifact_metrics],
        "paper_relevance": {
            "single_head_conflict": (
                "The previously reported 0.814 value is reproduced by ensemble_seed_456 at threshold 0.55, "
                "which is a single-head DeBERTa seed artifact, not a verified four-head multi-task artifact."
            ),
            "four_head_retrain": (
                "The 2026-05-12 Lambda retrain packages four-head MTL test_probs.npy and reproduces CT24 F1=0.8333 "
                "at threshold 0.50."
            ),
            "best_ensemble": "ensemble_temp_0_3",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute CT24 metrics from saved local probability artifacts.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for artifact in report["artifacts"]:
        best = artifact["best"]
        print(
            f"{artifact['name']}: best_f1={best['f1']:.4f} threshold={best['threshold']:.2f} "
            f"precision={best['precision']:.4f} recall={best['recall']:.4f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
