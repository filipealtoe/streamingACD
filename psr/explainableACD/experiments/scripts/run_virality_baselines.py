#!/usr/bin/env python3
"""
Run all virality prediction baselines and evaluate on PSR (Preventable Spread Ratio).

Usage:
    python experiments/scripts/run_virality_baselines.py --data-dir data/pipeline_output/streaming_full/2026-01-17_03-56
    python experiments/scripts/run_virality_baselines.py --quick  # Fast test run with 100 samples
"""
from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.virality.baselines import (
    BaselineConfig,
    SequenceData,
    get_all_baselines,
    get_classification_baselines,
    get_sequence_baselines,
)
from src.virality.evaluate import (
    BaselineResults,
    evaluate_baseline,
    format_results_markdown,
    format_results_table,
    results_to_dataframe,
)
from src.virality.hawkes import HawkesBaseline, HawkesConfig
from src.virality.psr_dataset import PSRDataset, PSRDatasetConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "virality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run virality prediction baselines")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing pipeline output files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with limited samples",
    )
    parser.add_argument(
        "--skip-hawkes",
        action="store_true",
        help="Skip Hawkes baseline (slow)",
    )
    parser.add_argument(
        "--skip-sequence",
        action="store_true",
        help="Skip LSTM/GRU baselines",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_and_prepare_data(
    data_dir: Path,
    quick: bool = False,
) -> PSRDataset:
    logger.info(f"Loading data from {data_dir}")

    timeseries_path = data_dir / "cluster_timeseries.parquet"
    clusters_path = data_dir / "clusters.parquet"
    tweets_path = data_dir / "tweets.parquet"

    if not timeseries_path.exists():
        raise FileNotFoundError(f"Timeseries file not found: {timeseries_path}")

    dataset = PSRDataset(
        timeseries_path=timeseries_path,
        clusters_path=clusters_path if clusters_path.exists() else None,
        tweets_path=tweets_path if tweets_path.exists() else None,
        config=PSRDatasetConfig(),
    )

    return dataset


def prepare_features_and_targets(
    dataset: PSRDataset,
    quick: bool = False,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    detections = dataset.get_detections_from_triggers()
    logger.info(f"Found {len(detections)} clusters with anomaly triggers")

    if quick and len(detections) > 100:
        np.random.seed(random_state)
        indices = np.random.choice(len(detections), 100, replace=False)
        detections = [detections[i] for i in indices]
        logger.info(f"Quick mode: using {len(detections)} samples")

    features_df, labels_df = dataset.build_dataset(detections)
    logger.info(f"Extracted {len(features_df)} samples with {len(features_df.columns)} features")

    return features_df, labels_df


def split_data(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    list[int],
    list[int],
]:
    feature_cols = [
        c for c in features_df.columns if c not in ["cluster_id", "detection_time", "psr"]
    ]

    X = features_df.select(feature_cols).to_numpy().astype(np.float32)
    y = features_df["psr"].to_numpy().astype(np.float32)
    cluster_ids = features_df["cluster_id"].to_list()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_clusters = [cluster_ids[i] for i in train_idx]
    test_clusters = [cluster_ids[i] for i in test_idx]

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"PSR distribution - Train mean: {y_train.mean():.3f}, Test mean: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, train_clusters, test_clusters


def run_tabular_baselines(
    X_train: NDArray[np.float32],
    X_test: NDArray[np.float32],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    config: BaselineConfig,
) -> list[BaselineResults]:
    results = []

    baselines = get_all_baselines(config)
    for baseline in baselines:
        logger.info(f"Training {baseline.name}...")
        baseline.fit(X_train, y_train, X_test, y_test)
        y_pred = baseline.predict(X_test)
        result = evaluate_baseline(y_test, y_pred, baseline.name)
        results.append(result)
        logger.info(f"  MAE: {result.regression.mae:.4f}, Spearman: {result.regression.spearman_rho:.4f}")

    return results


def run_classification_baselines(
    X_train: NDArray[np.float32],
    X_test: NDArray[np.float32],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    config: BaselineConfig,
    thresholds: list[float] | None = None,
) -> list[BaselineResults]:
    thresholds = thresholds or [0.25, 0.5, 0.75]
    results = []

    baselines = get_classification_baselines(config, thresholds)
    for baseline in baselines:
        logger.info(f"Training {baseline.name}...")
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)

        y_pred_continuous = y_pred.astype(np.float32)
        if baseline.is_classifier:
            proba = baseline.predict_proba(X_test)
            if proba is not None:
                y_pred_continuous = proba

        result = evaluate_baseline(y_test, y_pred_continuous, baseline.name, thresholds)
        results.append(result)

    return results


def run_sequence_baselines(
    dataset: PSRDataset,
    features_df: pl.DataFrame,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    config: BaselineConfig,
) -> list[BaselineResults]:
    results = []
    detections = dataset.get_detections_from_triggers()

    det_map = {d.cluster_id: d for d in detections}
    train_detections = [det_map[c] for c in train_clusters if c in det_map]
    test_detections = [det_map[c] for c in test_clusters if c in det_map]

    train_seqs, train_lens, train_ids = dataset.get_sequences(train_detections)
    test_seqs, test_lens, test_ids = dataset.get_sequences(test_detections)

    if len(train_seqs) < 10 or len(test_seqs) < 5:
        logger.warning("Not enough sequences for LSTM/GRU training, skipping")
        return results

    train_id_to_psr = dict(zip(train_clusters, y_train))
    test_id_to_psr = dict(zip(test_clusters, y_test))

    train_targets = np.array([train_id_to_psr.get(c, 0.5) for c in train_ids], dtype=np.float32)
    test_targets = np.array([test_id_to_psr.get(c, 0.5) for c in test_ids], dtype=np.float32)

    train_data = SequenceData(sequences=train_seqs, lengths=train_lens, targets=train_targets)
    test_data = SequenceData(sequences=test_seqs, lengths=test_lens, targets=test_targets)

    baselines = get_sequence_baselines(config)
    for baseline in baselines:
        logger.info(f"Training {baseline.name}...")
        baseline.fit_sequence(train_data, test_data)
        y_pred = baseline.predict_sequence(test_data)
        result = evaluate_baseline(test_targets, y_pred, baseline.name)
        results.append(result)
        logger.info(f"  MAE: {result.regression.mae:.4f}, Spearman: {result.regression.spearman_rho:.4f}")

    return results


def run_hawkes_baseline(
    dataset: PSRDataset,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    config: HawkesConfig | None = None,
) -> BaselineResults | None:
    detections = dataset.get_detections_from_triggers()
    det_map = {d.cluster_id: d for d in detections}

    try:
        all_arrivals = dataset.get_tweet_arrivals(detections)
    except ValueError as e:
        logger.warning(f"Cannot run Hawkes baseline: {e}")
        return None

    train_arrivals = {c: all_arrivals[c] for c in train_clusters if c in all_arrivals}
    test_arrivals = {c: all_arrivals[c] for c in test_clusters if c in all_arrivals}

    if len(train_arrivals) < 10 or len(test_arrivals) < 5:
        logger.warning("Not enough tweet arrivals for Hawkes baseline, skipping")
        return None

    train_offsets = {}
    for c in train_arrivals:
        if c in det_map:
            times = train_arrivals[c]
            first_time = times[0] if len(times) > 0 else 0
            train_offsets[c] = times[-1] - first_time if len(times) > 0 else 0

    test_offsets = {}
    for c in test_arrivals:
        if c in det_map:
            times = test_arrivals[c]
            first_time = times[0] if len(times) > 0 else 0
            test_offsets[c] = times[-1] - first_time if len(times) > 0 else 0

    train_psr = {c: float(y_train[i]) for i, c in enumerate(train_clusters) if c in train_arrivals}

    logger.info(f"Training Hawkes on {len(train_arrivals)} clusters...")
    hawkes = HawkesBaseline(config)
    hawkes.fit(train_arrivals, train_offsets, train_psr)

    y_pred = hawkes.predict(test_arrivals, test_offsets)

    test_y = np.array([y_test[test_clusters.index(c)] for c in test_arrivals], dtype=np.float32)

    result = evaluate_baseline(test_y, y_pred, hawkes.name)
    logger.info(f"  MAE: {result.regression.mae:.4f}, Spearman: {result.regression.spearman_rho:.4f}")

    return result


def save_results(
    results: list[BaselineResults],
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = results_to_dataframe(results)
    results_df.write_csv(output_dir / "baselines_results.csv")
    results_df.write_parquet(output_dir / "baselines_results.parquet")

    features_df.write_parquet(output_dir / "virality_features.parquet")
    labels_df.write_parquet(output_dir / "psr_labels.parquet")

    with open(output_dir / "results_table.txt", "w") as f:
        f.write(format_results_table(results))

    with open(output_dir / "results_table.md", "w") as f:
        f.write(format_results_markdown(results))

    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Virality Prediction Baselines")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quick mode: {args.quick}")

    dataset = load_and_prepare_data(args.data_dir, args.quick)
    features_df, labels_df = prepare_features_and_targets(dataset, args.quick, args.random_state)

    X_train, X_test, y_train, y_test, train_clusters, test_clusters = split_data(
        features_df, labels_df, args.test_size, args.random_state
    )

    config = BaselineConfig(random_state=args.random_state)
    all_results: list[BaselineResults] = []

    logger.info("\n--- Tabular Regression Baselines ---")
    tabular_results = run_tabular_baselines(X_train, X_test, y_train, y_test, config)
    all_results.extend(tabular_results)

    logger.info("\n--- Classification Baselines ---")
    class_results = run_classification_baselines(X_train, X_test, y_train, y_test, config)
    all_results.extend(class_results)

    if not args.skip_sequence:
        logger.info("\n--- Sequence Baselines (LSTM/GRU) ---")
        seq_results = run_sequence_baselines(
            dataset, features_df, train_clusters, test_clusters, y_train, y_test, config
        )
        all_results.extend(seq_results)

    if not args.skip_hawkes:
        logger.info("\n--- Hawkes Intensity Baseline ---")
        hawkes_result = run_hawkes_baseline(
            dataset, train_clusters, test_clusters, y_train, y_test
        )
        if hawkes_result:
            all_results.append(hawkes_result)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    print("\n" + format_results_table(all_results))

    save_results(all_results, features_df, labels_df, args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
