"""Public contract tests for the CIKM 2026 RandomForest reconstruction."""

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-25 22:22 WEST | Reason: independently protect the published row,
# version-to-cell map, prediction vectors, and checksum manifest.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import fbeta_score, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/virality_random_forest_reproduction_2026-08-25"
METRICS = ("spearman_rho", "r2", "mae", "f2_065", "f2_075", "f2_085")
PAPER_VALUES = [0.530, 0.319, 0.217, 0.547, 0.322, 0.041]
EXPECTED_CELL_REPRODUCTIONS = {
    "spearman_rho": "sklearn_1_3_2",
    "r2": "sklearn_1_5_2",
    "mae": "sklearn_1_5_2",
    "f2_065": "sklearn_1_5_2",
    "f2_075": "sklearn_1_5_2",
    "f2_085": "sklearn_1_5_2",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> list[float]:
    values = [
        float(spearmanr(labels, predictions).statistic),
        float(r2_score(labels, predictions)),
        float(mean_absolute_error(labels, predictions)),
    ]
    for threshold in (0.65, 0.75, 0.85):
        values.append(
            float(
                fbeta_score(
                    labels >= threshold,
                    predictions >= threshold,
                    beta=2,
                    zero_division=0,
                )
            )
        )
    return [round(value, 3) for value in values]


def test_random_forest_paper_cells_are_reconstructed_and_checksum_bound() -> None:
    summary = load_json(RESULT / "summary.json")
    paper = load_json(ROOT / "reproducibility/cikm2026/PAPER_VALUE_MANIFEST.json")
    paper_row = paper["tables"]["virality_prediction"]["rows"]["RandomForest"]
    labels = np.asarray(
        np.load(ROOT / "psr/derived_arrays/test_targets.npy"), dtype=np.float64
    )

    assert summary["status"] == "PASS"
    assert summary["paper_row"]["status"] == "PASS"
    assert paper_row["metrics"] == PAPER_VALUES
    assert paper_row["cell_reproductions"] == EXPECTED_CELL_REPRODUCTIONS
    assert summary["paper_row"]["cell_reproductions"] == EXPECTED_CELL_REPRODUCTIONS

    run_values: dict[str, list[float]] = {}
    for run_name in ("sklearn_1_3_2", "sklearn_1_5_2"):
        run = summary["runs"][run_name]
        predictions = np.asarray(np.load(ROOT / run["prediction_file"]), dtype=np.float64)
        assert predictions.shape == (106,)
        assert np.isfinite(predictions).all()
        assert ((0.0 <= predictions) & (predictions <= 1.0)).all()
        assert hashlib.sha256(predictions.astype("<f8").tobytes()).hexdigest() == run[
            "prediction_sha256_float64_little_endian"
        ]
        run_values[run_name] = metrics(labels, predictions)

    reconstructed = [
        run_values[EXPECTED_CELL_REPRODUCTIONS[metric]][index]
        for index, metric in enumerate(METRICS)
    ]
    assert reconstructed == PAPER_VALUES
    assert summary["paper_row"]["reconstructed_values"] == PAPER_VALUES

    swapped = dict(EXPECTED_CELL_REPRODUCTIONS)
    swapped["spearman_rho"] = "sklearn_1_5_2"
    swapped["r2"] = "sklearn_1_3_2"
    swapped_values = [
        run_values[swapped[metric]][index] for index, metric in enumerate(METRICS)
    ]
    assert swapped_values != PAPER_VALUES

    manifest_entries = {}
    for line in (RESULT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        manifest_entries[relative] = digest
    assert len(manifest_entries) == 4
    for relative, digest in manifest_entries.items():
        assert sha256_file(ROOT / relative) == digest
