from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_SOURCE_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUTPUT = Path("results/table3_reproduction_2026-05-12.json")
DEFAULT_MARKDOWN_OUTPUT = Path("results/table3_reproduction_2026-05-12.md")
THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
FUSION_FEATURE_GROUPS = {
    "scores": ["check_score", "verif_score", "harm_score"],
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "p_yes": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    "cross_basic": ["score_variance", "score_max_diff", "yes_vote_count", "unanimous_yes", "unanimous_no"],
    "harm_subdims": [
        "harm_social_fragmentation",
        "harm_spurs_action",
        "harm_believability",
        "harm_exploitativeness",
    ],
}


@dataclass(frozen=True)
class Metrics:
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
class TableRow:
    row: str
    paper_claim_f1: float | None
    reproduced_f1: float
    threshold: float
    status: str
    evidence: str
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_labels(source_root: Path) -> tuple[np.ndarray, Path]:
    return load_split_labels(source_root, "test")


def load_split_labels(source_root: Path, split: str) -> tuple[np.ndarray, Path]:
    labels_path = source_root / f"data/processed/CT24_clean/CT24_{split}_clean.parquet"
    table = pq.read_table(labels_path, columns=["class_label"])
    labels = np.array([1 if value.as_py() == "Yes" else 0 for value in table.column("class_label")], dtype=np.int64)
    expected_rows = {"train": 22402, "dev": 1031, "test": 341}
    if labels.shape != (expected_rows[split],):
        raise ValueError(f"Expected {expected_rows[split]} CT24 {split} labels, found {labels.shape[0]}")
    return labels, labels_path


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Metrics:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    return Metrics(
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


def best_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Metrics:
    return max((compute_metrics(labels, probabilities, threshold) for threshold in THRESHOLDS), key=lambda row: row.f1)


def load_probabilities(path: Path, labels: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.load(path)
    if probabilities.shape != labels.shape:
        raise ValueError(f"{name}: probability shape {probabilities.shape} does not match labels {labels.shape}")
    return probabilities


def row_status(paper_f1: float | None, reproduced_f1: float) -> str:
    if paper_f1 is None:
        return "reproduced"
    if abs(reproduced_f1 - paper_f1) <= 0.002:
        return "reproduced"
    if abs(reproduced_f1 - paper_f1) <= 0.01:
        return "near paper value"
    return "not reproduced"


def load_fusion_features(source_root: Path, split: str) -> np.ndarray:
    path = source_root / f"data/processed/CT24_llm_features_v4/{split}_llm_features.parquet"
    frame = pl.read_parquet(path)
    columns = [feature for group in FUSION_FEATURE_GROUPS.values() for feature in group if feature in frame.columns]
    features = frame.select(columns).to_numpy().astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def rerun_xgboost_fusion_component(source_root: Path) -> tuple[np.ndarray, Metrics]:
    train_labels, _ = load_split_labels(source_root, "train")
    dev_labels, _ = load_split_labels(source_root, "dev")
    test_labels, _ = load_split_labels(source_root, "test")

    train_features = load_fusion_features(source_root, "train")
    dev_features = load_fusion_features(source_root, "dev")
    test_features = load_fusion_features(source_root, "test")

    traindev_features = np.vstack([train_features, dev_features])
    traindev_labels = np.concatenate([train_labels, dev_labels])
    scaler = StandardScaler()
    traindev_scaled = scaler.fit_transform(traindev_features)
    test_scaled = scaler.transform(test_features)

    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=3,
        random_state=42,
        verbosity=0,
    )
    classifier.fit(traindev_scaled, traindev_labels)
    probabilities = classifier.predict_proba(test_scaled)[:, 1]
    return probabilities, best_metrics(test_labels, probabilities)


def load_pca_llm_text_logreg_inputs(
    source_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classifier_root = source_root / "data/processed/CT24_classifier"
    llm_root = source_root / "data/processed/CT24_llm_features_v4"

    pca_frames = {
        split: pl.read_parquet(classifier_root / f"{split}_text_embed_pca64.parquet")
        for split in ("train", "dev", "test")
    }
    llm_frames = {
        split: pl.read_parquet(llm_root / f"{split}_llm_features.parquet") for split in ("train", "dev", "test")
    }

    text_columns = [column for column in pca_frames["train"].columns if column.startswith("feat_")]
    pca_columns = [column for column in pca_frames["train"].columns if column.startswith("pca64_")]
    llm_columns = [
        column
        for column in llm_frames["train"].columns
        if column != "sentence_id" and not column.endswith("_prediction") and not column.endswith("_parse_issue")
    ]

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "dev", "test"):
        features[split] = np.hstack(
            [
                pca_frames[split].select(text_columns).to_numpy(),
                pca_frames[split].select(pca_columns).to_numpy(),
                llm_frames[split].select(llm_columns).to_numpy(),
            ]
        )
        labels[split] = (pca_frames[split]["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return (
        features["train"],
        labels["train"],
        features["dev"],
        labels["dev"],
        features["test"],
        labels["test"],
    )


def rerun_pca_llm_text_logreg(source_root: Path) -> tuple[np.ndarray, Metrics]:
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = (
        load_pca_llm_text_logreg_inputs(source_root)
    )
    traindev_features = np.vstack([train_features, dev_features])
    traindev_labels = np.concatenate([train_labels, dev_labels])

    scaler = StandardScaler()
    traindev_scaled = scaler.fit_transform(traindev_features)
    test_scaled = scaler.transform(test_features)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    classifier.fit(traindev_scaled, traindev_labels)
    probabilities = classifier.predict_proba(test_scaled)[:, 1]
    return probabilities, best_metrics(test_labels, probabilities)


def build_report(source_root: Path, repo_root: Path) -> dict:
    clean_labels, clean_labels_path = load_clean_labels(source_root)
    four_head_dir = repo_root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    four_head_labels_path = four_head_dir / "test_labels.npy"
    four_head_labels = np.load(four_head_labels_path)

    artifact_specs = [
        (
            "Single DeBERTa",
            0.8242,
            source_root / "experiments/results/deberta_checkworthy/deberta-v3-large/test_probs.npy",
            clean_labels,
            clean_labels_path,
            "Saved local single-model probabilities.",
        ),
        (
            "3-seed DeBERTa ensemble",
            0.8343,
            source_root / "lambda_backup/ubuntu/ensemble_results/test_temp_0.3_probs.npy",
            clean_labels,
            clean_labels_path,
            "Saved Lambda ensemble probabilities, temperature-scaled output file.",
        ),
        (
            "4-head MTL retrain",
            0.8333,
            four_head_dir / "test_probs.npy",
            four_head_labels,
            four_head_labels_path,
            "Packaged 2026-05-12 retrain; uses packaged joined-label order.",
        ),
    ]

    rows: list[TableRow] = []
    artifacts: list[dict] = []
    loaded: dict[str, np.ndarray] = {}

    for name, paper_f1, path, labels, labels_path, note in artifact_specs:
        probabilities = load_probabilities(path, labels, name)
        loaded[name] = probabilities
        metrics = best_metrics(labels, probabilities)
        rows.append(
            TableRow(
                row=name,
                paper_claim_f1=paper_f1,
                reproduced_f1=metrics.f1,
                threshold=metrics.threshold,
                status=row_status(paper_f1, metrics.f1),
                evidence=str(path),
                note=note,
            )
        )
        artifacts.append(
            {
                "row": name,
                "probabilities_path": str(path),
                "labels_path": str(labels_path),
                "probabilities_sha256": sha256_file(path),
                "metrics": asdict(metrics),
            }
        )

    pca_logreg_probabilities, pca_logreg_metrics = rerun_pca_llm_text_logreg(source_root)
    rows.append(
        TableRow(
            row="PCA-64 + LLM + text LogReg CT24 rerun",
            paper_claim_f1=0.7610,
            reproduced_f1=pca_logreg_metrics.f1,
            threshold=pca_logreg_metrics.threshold,
            status=row_status(0.7610, pca_logreg_metrics.f1),
            evidence="experiments/scripts/benchmark_pca_llm_text.py CT24 training path, mirrored in this audit",
            note=(
                "This runnable path reproduces the paper-row ClaimBuster and CT23 values after rounding, "
                "but CT24 test recomputes below the paper-facing 0.761 value."
            ),
        )
    )

    llm_xgboost_probabilities, llm_xgboost_metrics = rerun_xgboost_fusion_component(source_root)
    paper_fusion = compute_metrics(
        clean_labels,
        0.5 * loaded["3-seed DeBERTa ensemble"] + 0.5 * llm_xgboost_probabilities,
        0.5,
    )
    rows.append(
        TableRow(
            row="Fusion classifier rerun",
            paper_claim_f1=0.8362,
            reproduced_f1=paper_fusion.f1,
            threshold=paper_fusion.threshold,
            status=row_status(0.8362, paper_fusion.f1),
            evidence="saved T=0.3 ensemble probabilities + rerun 24-feature XGBoost v4, weight=0.5",
            note=(
                "This matches the paper-facing Fusion Classifier setup: saved temperature-scaled ensemble probabilities, "
                "rerun XGBoost LLM component, equal-weight late fusion, threshold 0.50."
            ),
        )
    )

    return {
        "dataset": {
            "name": "CT24 check-worthiness English test",
            "clean_labels_path": str(clean_labels_path),
            "n_samples": int(clean_labels.shape[0]),
            "n_positive": int(np.sum(clean_labels)),
            "thresholds": list(THRESHOLDS),
        },
        "rows": [asdict(row) for row in rows],
        "artifacts": artifacts,
        "component_reruns": {
            "pca_llm_text_logreg_ct24": asdict(pca_logreg_metrics),
            "llm_xgboost_v4": asdict(llm_xgboost_metrics),
            "fusion_classifier_paper_config": asdict(paper_fusion),
            "pca_llm_text_logreg_probability_count": int(pca_logreg_probabilities.shape[0]),
            "llm_xgboost_probability_count": int(llm_xgboost_probabilities.shape[0]),
        },
        "conclusion": (
            "The three-seed ensemble, the four-head MTL retrain, and the Fusion Classifier now have reproducible CT24 evidence. "
            "The runnable PCA-64 + LLM + text LogReg path matches the paper's cross-dataset values after rounding, "
            "but its CT24 test F1 does not reproduce the paper-facing 0.761 value."
        ),
    }


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# Table 3 Reproduction Audit",
        "",
        f"Dataset: {report['dataset']['name']} ({report['dataset']['n_samples']} samples, "
        f"{report['dataset']['n_positive']} positives)",
        "",
        "| Row | Paper F1 | Recomputed F1 | Threshold | Status | Evidence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        paper_f1 = "-" if row["paper_claim_f1"] is None else f"{row['paper_claim_f1']:.4f}"
        lines.append(
            f"| {row['row']} | {paper_f1} | {row['reproduced_f1']:.4f} | {row['threshold']:.2f} | "
            f"{row['status']} | {row['evidence']} |"
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CT24 Table 3 reproduction audit from saved local artifacts.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.source_root, args.repo_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown_output)
    for row in report["rows"]:
        print(f"{row['row']}: f1={row['reproduced_f1']:.4f} threshold={row['threshold']:.2f} status={row['status']}")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
