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
DEFAULT_JSON_OUTPUT = Path("results/table3_reproduction_2026-05-12.json")
DEFAULT_MARKDOWN_OUTPUT = Path("results/table3_reproduction_2026-05-12.md")
THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))


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
    labels_path = source_root / "data/processed/CT24_clean/CT24_test_clean.parquet"
    table = pq.read_table(labels_path, columns=["class_label"])
    labels = np.array([1 if value.as_py() == "Yes" else 0 for value in table.column("class_label")], dtype=np.int64)
    if labels.shape != (341,):
        raise ValueError(f"Expected 341 CT24 test labels, found {labels.shape[0]}")
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
        (
            "Saved LLM probability artifact: logreg_llm_embed",
            0.7610,
            source_root / "experiments/results/deberta_checkworthy/logreg_llm_embed_probs.npy",
            clean_labels,
            clean_labels_path,
            "Saved local LLM-related probability artifact; not the documented XGBoost v4 run.",
        ),
        (
            "Saved LLM probability artifact: logreg_test",
            0.7610,
            source_root / "experiments/results/deberta_checkworthy/logreg_test_probs.npy",
            clean_labels,
            clean_labels_path,
            "Saved local logistic probability artifact; not the documented XGBoost v4 run.",
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

    fusion_candidates = []
    ensemble_candidates = {
        "lambda_temp_0_3": loaded["3-seed DeBERTa ensemble"],
        "old_local_ensemble": load_probabilities(
            source_root / "experiments/results/deberta_checkworthy/ensemble_test_probs.npy",
            clean_labels,
            "old_local_ensemble",
        ),
    }
    llm_candidates = {
        "logreg_llm_embed": loaded["Saved LLM probability artifact: logreg_llm_embed"],
        "logreg_test": loaded["Saved LLM probability artifact: logreg_test"],
    }

    for ensemble_name, ensemble_probs in ensemble_candidates.items():
        for llm_name, llm_probs in llm_candidates.items():
            for weight_index in range(1, 10):
                deberta_weight = weight_index / 10
                fused = deberta_weight * ensemble_probs + (1 - deberta_weight) * llm_probs
                metrics = best_metrics(clean_labels, fused)
                fusion_candidates.append(
                    {
                        "ensemble": ensemble_name,
                        "llm": llm_name,
                        "deberta_weight": deberta_weight,
                        "metrics": asdict(metrics),
                    }
                )

    best_fusion = max(fusion_candidates, key=lambda row: row["metrics"]["f1"])
    paper_like_saved_fusion = compute_metrics(
        clean_labels,
        0.5 * ensemble_candidates["lambda_temp_0_3"] + 0.5 * llm_candidates["logreg_llm_embed"],
        0.5,
    )
    rows.append(
        TableRow(
            row="Fusion classifier from saved local probability artifacts",
            paper_claim_f1=0.8362,
            reproduced_f1=best_fusion["metrics"]["f1"],
            threshold=best_fusion["metrics"]["threshold"],
            status="not cleanly reproduced",
            evidence=f"{best_fusion['ensemble']} + {best_fusion['llm']}, DeBERTa weight={best_fusion['deberta_weight']:.1f}",
            note=(
                "This audits saved probability artifacts only; the documented XGBoost v4 fusion run is not saved locally. "
                f"The paper-like saved-artifact check at weight=0.5 and threshold=0.50 gives F1={paper_like_saved_fusion.f1:.4f}."
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
        "fusion_candidates": fusion_candidates,
        "paper_like_saved_fusion": asdict(paper_like_saved_fusion),
        "conclusion": (
            "Single DeBERTa, the three-seed ensemble, and the four-head MTL retrain have usable local evidence. "
            "The saved local LLM/fusion probability artifacts do not reproduce the paper's LLM-feature or fusion values."
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
        print(
            f"{row['row']}: f1={row['reproduced_f1']:.4f} threshold={row['threshold']:.2f} "
            f"status={row['status']}"
        )
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
