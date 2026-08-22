#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 19:08 PDT.
# Reason: make every value in the CIKM anomaly table executable from the public aggregate series.
# Artifact clarification — Sérgio Pinto, 2026-08-21 20:44 PDT.
# Reason: preserve the method-versus-table parameters and both detection-rate
# definitions whenever the generated result README is rebuilt.
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "pandas==2.3.3",
#   "polars==1.36.1",
#   "pyarrow==22.0.0",
#   "scikit-learn==1.8.0",
#   "scipy==1.16.3",
# ]
# ///
"""Reproduce the anomaly table in the accepted CIKM paper.

This focused port preserves the retained evaluation configuration exactly,
including the min_rows=89 cohort, threshold sweep, and pre-peak scoring window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.kernel_approximation import RBFSampler

MIN_ROWS = 89
FP_WEIGHT = 0.1
HISTORY_WINDOW = 6
ENGAGEMENT_WEIGHT = 0.3
EXPECTED_CLUSTERS = 820
EXPECTED_POINTS = 150_737


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Detector(Protocol):
    def update(self, value: float) -> float: ...


class RandomDetector:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def update(self, value: float) -> float:
        del value
        return float(self.rng.random())


class BayesianChangepointDetector:
    def __init__(
        self,
        max_run_length: int = 500,
        lambda_const: int = 250,
        alpha: float = 0.1,
        beta: float = 0.001,
        kappa: float = 1.0,
        mu: float = 0.0,
    ):
        self.max_run_length = max_run_length
        self.lambda_const = lambda_const
        self.alpha0 = alpha
        self.beta0 = beta
        self.kappa0 = kappa
        self.mu0 = mu
        self.run_length_probs = np.zeros((max_run_length + 2, 2))
        self.run_length_probs[0, 0] = 1.0
        self.record_number = 0
        self.previous_max_run = 1
        self._alpha0_arr = np.array([alpha])
        self._beta0_arr = np.array([beta])
        self._kappa0_arr = np.array([kappa])
        self._mu0_arr = np.array([mu])
        self.alpha = np.array([alpha])
        self.beta = np.array([beta])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])

    def _student_t_pdf(self, value: float) -> np.ndarray:
        degrees = 2 * self.alpha
        scale = np.sqrt((self.beta * (self.kappa + 1)) / (self.alpha * self.kappa))
        return stats.t.pdf(value, df=degrees, loc=self.mu, scale=scale)

    def _update_theta(self, value: float) -> None:
        mu_new = np.concatenate(
            (self._mu0_arr, (self.kappa * self.mu + value) / (self.kappa + 1))
        )
        kappa_new = np.concatenate((self._kappa0_arr, self.kappa + 1.0))
        alpha_new = np.concatenate((self._alpha0_arr, self.alpha + 0.5))
        beta_new = np.concatenate(
            (
                self._beta0_arr,
                self.beta
                + (self.kappa * (value - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )
        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def update(self, value: float) -> float:
        if self.record_number > 0:
            self.run_length_probs[:, 0] = self.run_length_probs[:, 1]
            self.run_length_probs[:, 1] = 0
        predictive = self._student_t_pdf(value)
        run_index = min(self.record_number, self.max_run_length)
        hazard = np.ones(self.record_number + 1) / float(self.lambda_const)
        self.run_length_probs[1 : run_index + 2, 1] = (
            self.run_length_probs[: run_index + 1, 0]
            * predictive[: run_index + 1]
            * (1 - hazard[: run_index + 1])
        )
        self.run_length_probs[0, 1] = np.sum(
            self.run_length_probs[: run_index + 1, 0]
            * predictive[: run_index + 1]
            * hazard[: run_index + 1]
        )
        total = self.run_length_probs[:, 1].sum()
        if total > 0:
            self.run_length_probs[:, 1] /= total
        self._update_theta(value)
        max_run = int(self.run_length_probs[:, 1].argmax())
        score = (
            1.0 - (float(max_run) / self.previous_max_run)
            if max_run < self.previous_max_run
            else 0.0
        )
        self.record_number += 1
        self.previous_max_run = max_run if max_run > 0 else 1
        return score


class EtsySkylineDetector:
    def __init__(self):
        self.values: list[float] = []

    @staticmethod
    def _tail_average(values: list[float]) -> float:
        return float(np.mean(values[-3:]))

    def _median_absolute_deviation(self) -> bool:
        if len(self.values) < 3:
            return False
        values = np.asarray(self.values)
        median = np.median(values)
        deviations = np.abs(values - median)
        median_deviation = np.median(deviations)
        return bool(median_deviation != 0 and deviations[-1] / median_deviation > 6)

    def _first_hour_average(self) -> bool:
        # Historical implementation used synthetic 5-minute timestamps. The
        # one-day/one-hour slice is therefore records [-289:-277].
        if len(self.values) < 289:
            return False
        sample = np.asarray(self.values[-289:-277])
        if sample.size < 2:
            return False
        std = np.std(sample)
        return bool(
            std != 0
            and abs(self._tail_average(self.values) - np.mean(sample)) > 3 * std
        )

    def _stddev_from_average(self) -> bool:
        if len(self.values) < 3:
            return False
        values = np.asarray(self.values)
        std = np.std(values)
        return bool(
            std != 0
            and abs(self._tail_average(self.values) - np.mean(values)) > 3 * std
        )

    def _stddev_from_moving_average(self) -> bool:
        if len(self.values) < 3:
            return False
        series = pd.Series(self.values)
        average = series.ewm(ignore_na=False, min_periods=0, adjust=True, com=50).mean()
        std = series.ewm(ignore_na=False, min_periods=0, adjust=True, com=50).std(
            bias=False
        )
        current_std = std.iloc[-1]
        return bool(
            not np.isnan(current_std)
            and current_std != 0
            and abs(series.iloc[-1] - average.iloc[-1]) > 3 * current_std
        )

    def _mean_subtraction_cumulation(self) -> bool:
        if len(self.values) < 3:
            return False
        values = np.asarray([value if value else 0 for value in self.values])
        centered = values - np.mean(values[:-1])
        std = np.std(centered[:-1])
        return bool(std != 0 and abs(centered[-1]) > 3 * std)

    def _least_squares(self) -> bool:
        if len(self.values) < 3:
            return False
        # Match the historical synthetic timestamps exactly: five-minute
        # samples starting at 2014-01-01 00:00:00 UTC-naive.
        x = 1_388_534_400.0 + np.arange(len(self.values), dtype=float) * 300.0
        y = np.asarray(self.values)
        design = np.vstack([x, np.ones(len(x))]).T
        try:
            slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            return False
        errors = y - (slope * x + intercept)
        std = np.std(errors)
        tail = np.mean(errors[-3:])
        return bool(abs(tail) > std * 3 and round(std) != 0 and round(tail) != 0)

    def _histogram_bins(self) -> bool:
        if len(self.values) < 3:
            return False
        counts, bins = np.histogram(np.asarray(self.values), bins=15)
        tail = self._tail_average(self.values)
        for index, count in enumerate(counts):
            if count > 20:
                continue
            if index == 0 and tail <= bins[0]:
                return True
            if index > 0 and bins[index] <= tail < bins[index + 1]:
                return True
        return False

    def update(self, value: float) -> float:
        self.values.append(value)
        votes = (
            self._median_absolute_deviation(),
            self._first_hour_average(),
            self._stddev_from_average(),
            self._stddev_from_moving_average(),
            self._mean_subtraction_cumulation(),
            self._least_squares(),
            self._histogram_bins(),
        )
        return sum(votes) / 8.0


class SteuberZScoreDetector:
    def __init__(self, window_small: int = 5, window_large: int = 5640):
        self.window_small = window_small
        self.window_large = window_large
        self.small: list[float] = []
        self.large: list[float] = []

    def update(self, value: float) -> float:
        self.small.append(value)
        if len(self.small) > self.window_small:
            self.small.pop(0)
        current = np.mean(self.small)
        score = 0.0
        if len(self.large) >= 2:
            std = np.std(self.large)
            if std > 0:
                score = max(0.0, float((current - np.mean(self.large)) / std))
        self.large.append(value)
        if len(self.large) > self.window_large:
            self.large.pop(0)
        return score


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    offsets: np.ndarray
    counts: np.ndarray
    engagements: np.ndarray
    signal: np.ndarray
    peak_offset: float
    lifetime_hours: float
    window_hours: float


def build_signal(counts: np.ndarray, engagements: np.ndarray) -> np.ndarray:
    maximum = float(np.max(engagements))
    history: list[float] = []
    values: list[float] = []
    for count, engagement in zip(counts, engagements, strict=True):
        history.append(float(count))
        if len(history) > HISTORY_WINDOW:
            history.pop(0)
        if len(history) == 1:
            values.append(0.0)
            continue
        baseline = sum(history[:-1]) / len(history[:-1])
        arrival_rate = float(count) / baseline if baseline > 0 else float(count + 1)
        boost = 1 + ENGAGEMENT_WEIGHT * (float(engagement) / (maximum + 1))
        values.append(arrival_rate * boost)
    return np.asarray(values, dtype=float)


def load_clusters(path: Path) -> list[Cluster]:
    frame = pl.read_parquet(path).sort(["cluster_id", "hour_offset"])
    expected_columns = {"cluster_id", "hour_offset", "tweet_count", "engagement"}
    if set(frame.columns) != expected_columns:
        raise ValueError(f"Unexpected columns: {frame.columns}")
    counts = frame.group_by("cluster_id").len()
    viable = counts.filter(pl.col("len") >= MIN_ROWS)["cluster_id"]
    frame = frame.filter(pl.col("cluster_id").is_in(viable.implode()))
    if frame.height != EXPECTED_POINTS or viable.len() != EXPECTED_CLUSTERS:
        raise ValueError(
            f"Expected {EXPECTED_CLUSTERS} clusters/{EXPECTED_POINTS} points, "
            f"found {viable.len()}/{frame.height}"
        )

    clusters: list[Cluster] = []
    for part in frame.partition_by("cluster_id", maintain_order=True):
        offsets = part["hour_offset"].to_numpy().astype(float)
        tweet_counts = part["tweet_count"].to_numpy().astype(float)
        engagements = part["engagement"].to_numpy().astype(float)
        if np.max(engagements) == 0:
            continue
        peak_offset = float(offsets[int(np.argmax(engagements))])
        lifetime = float(offsets[-1] - offsets[0])
        clusters.append(
            Cluster(
                cluster_id=int(part["cluster_id"][0]),
                offsets=offsets,
                counts=tweet_counts,
                engagements=engagements,
                signal=build_signal(tweet_counts, engagements),
                peak_offset=peak_offset,
                lifetime_hours=lifetime,
                window_hours=min(24.0, max(6.0, lifetime * 0.5)),
            )
        )
    if len(clusters) != EXPECTED_CLUSTERS:
        raise ValueError(f"Expected {EXPECTED_CLUSTERS} evaluable clusters")
    return clusters


def expose_scores(signal: np.ndarray) -> np.ndarray:
    sampler = RBFSampler(
        gamma=1.0,
        n_components=20_000,
        random_state=290,
    )
    sampler.fit([[0.0]])
    features = sampler.transform(signal.reshape(-1, 1))
    scores = np.zeros(len(signal), dtype=float)
    model = features[0].copy()
    for index in range(1, len(signal)):
        current = features[index]
        similarity = np.dot(current, model)
        model = 0.005 * current + 0.995 * model
        scores[index] = np.clip((1.0 - similarity + 0.02) / 1.04, 0.0, 1.0)
    return scores


def detector_scores(cluster: Cluster, name: str) -> np.ndarray:
    if name == "EXPoSE":
        return expose_scores(cluster.signal)
    factories: dict[str, type[Detector]] = {
        "Random": RandomDetector,
        "Bayesian Changept": BayesianChangepointDetector,
        "Etsy Skyline": EtsySkylineDetector,
        "Steuber Z-score": SteuberZScoreDetector,
    }
    detector = factories[name]()
    return np.asarray([detector.update(value) for value in cluster.signal])


def score_cluster(
    cluster: Cluster, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    mask = scores > threshold
    detections = cluster.offsets[mask]
    result: dict[str, Any] = {
        "cluster_id": cluster.cluster_id,
        "n_rows": len(cluster.offsets),
        "peak_offset": cluster.peak_offset,
        "lifetime_hours": cluster.lifetime_hours,
        "window_hours": cluster.window_hours,
        "n_detections": int(mask.sum()),
        "first_detection_offset": float(detections[0]) if len(detections) else None,
        "tp_score": 0.0,
        "fp_penalty": 0.0,
        "final_score": 0.0,
        "n_tp": 0,
        "n_fp": 0,
        "first_tp_lead_hours": None,
    }
    if not len(detections):
        return result

    window_start = cluster.peak_offset - cluster.window_hours
    first = float(detections[0])
    valid = detections[
        (detections >= window_start) & (detections <= cluster.peak_offset)
    ]
    if window_start <= first <= cluster.peak_offset:
        chosen = first
    elif len(valid):
        chosen = float(valid[0])
        result["fp_penalty"] = FP_WEIGHT
        result["n_fp"] = 1
    else:
        result["fp_penalty"] = FP_WEIGHT
        result["final_score"] = -FP_WEIGHT
        result["n_fp"] = 1
        return result

    lead = cluster.peak_offset - chosen
    result["tp_score"] = min(1.0, lead / cluster.window_hours)
    result["final_score"] = result["tp_score"] - result["fp_penalty"]
    result["n_tp"] = 1
    result["first_tp_lead_hours"] = lead
    return result


def aggregate(outcomes: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    leads = [
        row["first_tp_lead_hours"]
        for row in outcomes
        if row["first_tp_lead_hours"] is not None
    ]
    total_tp = sum(row["n_tp"] for row in outcomes)
    total_fp = sum(row["n_fp"] for row in outcomes)
    total_detections = sum(row["n_detections"] for row in outcomes)
    detected_before_peak = sum(row["tp_score"] > 0 for row in outcomes)
    count = len(outcomes)
    return {
        "threshold": threshold,
        "clusters_evaluated": count,
        "total_points": sum(row["n_rows"] for row in outcomes),
        "total_detections": total_detections,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "nab_score": 100 * sum(row["final_score"] for row in outcomes) / count,
        # The paper defines detection as strictly before the peak. Retain both
        # rates because the historical baseline artifact accidentally used the
        # inclusive rate while the EXPoSE artifact used the strict rate.
        "detection_rate": 100 * detected_before_peak / count,
        "detection_rate_strict_pre_peak": 100 * detected_before_peak / count,
        "detection_rate_inclusive_peak": 100 * total_tp / count,
        "missed_rate": 100 * (count - detected_before_peak) / count,
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "mean_lead": float(np.mean(leads)) if leads else 0.0,
        # Retain the historical field name/formula, despite this not being
        # conventional precision: the denominator is every flagged time point.
        "precision": 100 * total_tp / total_detections if total_detections else 0.0,
        "flagged_point_rate": 100
        * total_detections
        / sum(row["n_rows"] for row in outcomes),
    }


def evaluate_model(
    clusters: list[Cluster], name: str, thresholds: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scores_by_cluster = [
        (cluster, detector_scores(cluster, name)) for cluster in clusters
    ]
    threshold_results: list[dict[str, Any]] = []
    outcomes_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        outcomes = [
            score_cluster(cluster, scores, threshold)
            for cluster, scores in scores_by_cluster
        ]
        outcomes_by_threshold[threshold] = outcomes
        threshold_results.append(aggregate(outcomes, threshold))
    best = max(threshold_results, key=lambda row: row["nab_score"])
    return best, threshold_results, outcomes_by_threshold[best["threshold"]]


def compare_metric(
    actual: dict[str, Any], expected: dict[str, Any], names: list[str]
) -> float:
    return max(abs(float(actual[name]) - float(expected[name])) for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    input_path = (
        args.input
        or root / "reproducibility/source_artifacts/anomaly/timeseries.parquet"
    ).resolve()
    output_dir = (
        args.output_dir or root / "results/anomaly_table_reproduction_2026-08-21"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = load_clusters(input_path)
    configurations = {
        "EXPoSE": [0.05],
        "Random": [0.3, 0.4, 0.5, 0.6, 0.7],
        "Bayesian Changept": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "Etsy Skyline": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "Steuber Z-score": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    }

    results: dict[str, dict[str, Any]] = {}
    outcome_rows: list[dict[str, Any]] = []
    for name, thresholds in configurations.items():
        print(f"RUN {name}", flush=True)
        best, sweep, outcomes = evaluate_model(clusters, name, thresholds)
        results[name] = {
            "best": best,
            "threshold_results": sweep,
            "selection_rule": (
                "paper-table EXPoSE configuration selected on the same cohort"
                if name == "EXPoSE"
                else "maximum NAB-style score selected on the same cohort"
            ),
        }
        for row in outcomes:
            outcome_rows.append({"model": name, "threshold": best["threshold"], **row})

    historical_dir = root / "reproducibility/source_artifacts/anomaly/historical"
    expected_baselines = json.loads(
        (historical_dir / "baselines_min_rows89.json").read_text()
    )
    expected_by_name = {
        row["detector"]: row["best"] for row in expected_baselines["detectors"]
    }
    expected_expose_rows = json.loads(
        (historical_dir / "expose_min_rows_sweep.json").read_text()
    )["summaries"]
    expected_by_name["EXPoSE"] = next(
        row for row in expected_expose_rows if row["min_rows"] == MIN_ROWS
    )
    comparison_metrics = [
        "nab_score",
        "detection_rate",
        "median_lead",
        "precision",
    ]
    comparisons = {}
    for name, result in results.items():
        historical_formula = "strict_pre_peak" if name == "EXPoSE" else "inclusive_peak"
        comparison_actual = dict(result["best"])
        comparison_actual["detection_rate"] = result["best"][
            f"detection_rate_{historical_formula}"
        ]
        delta = compare_metric(
            comparison_actual, expected_by_name[name], comparison_metrics
        )
        comparisons[name] = {
            "metrics": comparison_metrics,
            "historical_detection_rate_formula": historical_formula,
            "historical_reported_detection_rate": comparison_actual["detection_rate"],
            "strict_pre_peak_detection_rate": result["best"][
                "detection_rate_strict_pre_peak"
            ],
            "max_absolute_delta": delta,
            "passed": delta <= args.tolerance,
        }

    summary = {
        "status": "PASS"
        if all(row["passed"] for row in comparisons.values())
        else "FAIL",
        "input": input_path.name,
        "input_sha256": sha256(input_path),
        "cohort": {
            "selection": "clusters with at least 89 retained hourly observations",
            "selection_status": "post-hoc; selected during camera-ready reconstruction",
            "clusters": EXPECTED_CLUSTERS,
            "points": EXPECTED_POINTS,
        },
        "score": {
            "name": "historical NAB-style score",
            "official_nab_score": False,
            "window": "min(24h, max(6h, 0.5 * retained cluster lifetime))",
            "false_positive_penalty": FP_WEIGHT,
            "threshold_selection": "same evaluated cohort; no held-out threshold set",
        },
        "detection_rate_inconsistency": {
            "paper_definition": "percentage detected strictly before peak",
            "expose_artifact_formula": "strict_pre_peak",
            "baseline_artifact_formula": "inclusive_peak",
            "camera_ready_recommendation": "use strict_pre_peak for every detector",
        },
        "signal": "arrival rate over six retained observations with engagement boost 0.3",
        "configuration_provenance": {
            "method_paragraph": {"decay": 0.01, "gamma": 0.5, "threshold": 0.1, "warmup": 30},
            "retained_table_run": {"decay": 0.005, "gamma": 1.0, "threshold": 0.05, "warmup": 0},
        },
        "results": results,
        "paper_table_rows": [
            "EXPoSE",
            "Bayesian Changept",
            "Etsy Skyline",
            "Steuber Z-score",
        ],
        "additional_sanity_baselines": ["Random"],
        "historical_comparisons": comparisons,
        "scope": "Exact reproduction of the retained aggregate table evaluation.",
    }

    summary_path = output_dir / "summary.json"
    outcomes_path = output_dir / "cluster_outcomes.parquet"
    readme_path = output_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pl.DataFrame(outcome_rows).sort(["model", "cluster_id"]).write_parquet(
        outcomes_path, compression="zstd", compression_level=9, statistics=True
    )

    def render_rows(names: list[str]) -> list[str]:
        rendered = []
        for name in names:
            result = results[name]
            best = result["best"]
            historical_rate = comparisons[name]["historical_reported_detection_rate"]
            rendered.append(
                f"| {name} | {best['threshold']:.2f} | {best['nab_score']:.4f} | "
                f"{historical_rate:.4f}% | {best['detection_rate_strict_pre_peak']:.4f}% | "
                f"{best['median_lead']:+.1f}h | "
                f"{best['flagged_point_rate']:.2f}% |"
            )
        return rendered

    paper_rows = summary["paper_table_rows"]
    additional_rows = summary["additional_sanity_baselines"]
    readme_path.write_text(
        "# CIKM anomaly-table reproduction\n\n"
        "<!-- Sérgio Pinto, 2026-08-21 20:44 PDT — documented the retained table protocol and both detection-rate boundaries. -->\n\n"
        "This fresh run reproduces the accepted paper's `min_rows=89` table "
        "from the released aggregate time series.\n\n"
        "| Detector | Threshold | NAB-style | Historical detection | Strictly pre-peak | Median lead | Flagged points |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(render_rows(paper_rows))
        + "\n\nAdditional sanity baseline (not a paper-table row):\n\n"
        "| Detector | Threshold | NAB-style | Historical detection | Strictly pre-peak | Median lead | Flagged points |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(render_rows(additional_rows))
        + "\n\nProtocol notes:\n\n"
        "- the retained cohort contains clusters with at least 89 hourly observations;\n"
        "- detector thresholds are selected by the retained table sweep on these 820 clusters;\n"
        "- the reported score is the paper's project-specific NAB-style early-detection score;\n"
        "- the output records both strictly pre-peak and inclusive-at-peak detection rates;\n"
        "- `summary.json` records both the method-paragraph and retained table-run parameters.\n\n"
        "For exact reuse of these table values, use the retained table-run parameters in\n"
        "`summary.json`: decay `0.005`, RBF gamma `1.0`, score threshold `0.05`, and no\n"
        "warm-up. The method paragraph's EXPoSE values (`0.01`, `0.5`, `0.1`, and 30\n"
        "observations) describe a different configuration.\n\n"
        "The historical table preserved the strict pre-peak rate for EXPoSE and the\n"
        "inclusive-at-peak rate for the three classical baselines. Both versions are\n"
        "included in `summary.json` and `cluster_outcomes.parquet`, so a downstream user\n"
        "can apply one definition consistently.\n",
        encoding="utf-8",
    )
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (readme_path, outcomes_path, summary_path)
        ),
        encoding="utf-8",
    )

    for name, comparison in comparisons.items():
        print(
            f"{name}: max_delta={comparison['max_absolute_delta']:.3e} "
            f"{'PASS' if comparison['passed'] else 'FAIL'}"
        )
    print(f"VERDICT: {summary['status']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
