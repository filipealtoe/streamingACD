#!/usr/bin/env python3
"""
Optimize virality prediction features for PSR (Preventable Spread Ratio).

This script:
1. Extracts ALL available features without leakage
2. Computes geographic/temporal features from pre-detection tweets only
3. Runs greedy forward selection to find minimal feature subset
4. Reports performance on metrics matching the IJCAI LaTeX table

Usage:
    python experiments/scripts/optimize_virality_features.py --data-dir data/pipeline_output/streaming_full/2026-01-17_03-56
    python experiments/scripts/optimize_virality_features.py --quick  # Fast test run
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, fbeta_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "virality_feature_selection"


# =============================================================================
# FEATURE EXTRACTION (LEAK-FREE)
# =============================================================================


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_geographic: bool = True
    include_account_age: bool = True
    include_engagement_distribution: bool = True
    include_temporal_patterns: bool = True


def extract_timeseries_features(
    ts_df: pl.DataFrame,
    cluster_id: int,
    detection_time: datetime,
) -> dict[str, float]:
    """Extract features from cluster timeseries up to detection time (no leakage)."""
    pre_detect = ts_df.filter(
        (pl.col("cluster_id") == cluster_id) & (pl.col("timestamp") <= detection_time)
    ).sort("timestamp")

    if pre_detect.is_empty():
        return {}

    features: dict[str, float] = {}

    # Cumulative activity
    features["cumulative_tweets"] = float(pre_detect["tweet_count"].sum())
    features["cumulative_engagement"] = float(pre_detect["engagement"].sum())
    features["windows_since_start"] = float(len(pre_detect))

    # Mean activity per window
    n_windows = len(pre_detect)
    features["mean_tweets_per_window"] = features["cumulative_tweets"] / max(n_windows, 1)
    features["mean_engagement_per_window"] = features["cumulative_engagement"] / max(n_windows, 1)

    # Growth rate and acceleration
    engagement_list = pre_detect["engagement"].to_list()
    if len(engagement_list) >= 2:
        prev = engagement_list[-2] if engagement_list[-2] > 0 else 1
        features["growth_rate"] = engagement_list[-1] / prev
    else:
        features["growth_rate"] = 1.0

    if len(engagement_list) >= 3:
        prev_prev = engagement_list[-3] if engagement_list[-3] > 0 else 1
        prev_growth = engagement_list[-2] / prev_prev
        features["acceleration"] = features["growth_rate"] - prev_growth
    else:
        features["acceleration"] = 0.0

    # Engagement velocity
    first_ts = pre_detect["timestamp"].min()
    last_ts = pre_detect["timestamp"].max()
    hours_elapsed = (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else 1.0
    features["engagement_velocity"] = (
        features["cumulative_engagement"] / hours_elapsed if hours_elapsed > 0 else features["cumulative_engagement"]
    )

    # User authority features
    features["max_followers_seen"] = float(pre_detect["max_followers"].max() or 0)
    features["avg_followers_seen"] = float(pre_detect["avg_followers"].mean() or 0)
    features["total_unique_users"] = float(pre_detect["unique_users"].sum())
    features["total_verified"] = float(pre_detect["verified_count"].sum())
    features["verified_ratio"] = (
        features["total_verified"] / features["total_unique_users"]
        if features["total_unique_users"] > 0 else 0.0
    )

    # Anomaly features
    last_row = pre_detect.tail(1)
    z_score = last_row["z_score"].item() if "z_score" in last_row.columns else None
    features["z_score_at_detect"] = z_score if z_score is not None else 0.0

    z_scores = pre_detect["z_score"].drop_nulls() if "z_score" in pre_detect.columns else None
    features["max_z_score_seen"] = float(z_scores.max()) if z_scores is not None and len(z_scores) > 0 else 0.0

    if "z_score_count" in last_row.columns:
        val = last_row["z_score_count"].item()
        features["z_score_count_at_detect"] = val if val is not None else 0.0
    else:
        features["z_score_count_at_detect"] = 0.0

    if "z_score_engagement" in last_row.columns:
        val = last_row["z_score_engagement"].item()
        features["z_score_engagement_at_detect"] = val if val is not None else 0.0
    else:
        features["z_score_engagement_at_detect"] = 0.0

    if "kleinberg_state" in last_row.columns:
        val = last_row["kleinberg_state"].item()
        features["kleinberg_state_at_detect"] = float(val) if val is not None else 0.0
    else:
        features["kleinberg_state_at_detect"] = 0.0

    # Temporal patterns
    features["hour_of_day"] = float(detection_time.hour)
    features["is_weekend"] = float(detection_time.weekday() >= 5)

    return features


def extract_tweet_features(
    tweets_df: pl.DataFrame,
    cluster_id: int,
    detection_time: datetime,
    config: FeatureConfig,
) -> dict[str, float]:
    """Extract features from individual tweets up to detection time (no leakage)."""
    pre_detect = tweets_df.filter(
        (pl.col("cluster_id") == cluster_id) & (pl.col("created_at") <= detection_time)
    )

    if pre_detect.is_empty():
        return {}

    features: dict[str, float] = {}

    # Geographic features (computed from pre-detection tweets only!)
    if config.include_geographic:
        countries = pre_detect.filter(
            pl.col("country").is_not_null() & (pl.col("country") != "")
        )["country"]

        if len(countries) > 0:
            country_counts = countries.value_counts()
            total = len(countries)

            features["unique_countries_predetect"] = float(country_counts.height)

            usa_count = country_counts.filter(pl.col("country") == "United States")["count"].sum()
            features["usa_ratio_predetect"] = float(usa_count) / total if total > 0 else 0.0

            # Geographic entropy
            probs = country_counts["count"].to_numpy() / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features["geographic_entropy_predetect"] = float(entropy)

            features["has_international_spread"] = float(country_counts.height > 1)
        else:
            features["unique_countries_predetect"] = 0.0
            features["usa_ratio_predetect"] = 0.0
            features["geographic_entropy_predetect"] = 0.0
            features["has_international_spread"] = 0.0

    # Account age features
    if config.include_account_age:
        join_dates = pre_detect.filter(pl.col("user_join_date").is_not_null())["user_join_date"]
        if len(join_dates) > 0:
            # Account age in days at time of tweet
            ref_date = detection_time
            ages = [(ref_date - jd).days for jd in join_dates.to_list()]
            features["account_age_avg_days"] = float(np.mean(ages))
            features["account_age_std_days"] = float(np.std(ages)) if len(ages) > 1 else 0.0
            features["account_age_min_days"] = float(np.min(ages))
        else:
            features["account_age_avg_days"] = 0.0
            features["account_age_std_days"] = 0.0
            features["account_age_min_days"] = 0.0

    # Engagement distribution features
    if config.include_engagement_distribution:
        rt_counts = pre_detect["retweet_count_at_collection"].to_numpy()
        like_counts = pre_detect["likes_at_collection"].to_numpy()

        if len(rt_counts) > 0:
            features["avg_retweets_per_tweet"] = float(np.mean(rt_counts))
            features["max_retweets_per_tweet"] = float(np.max(rt_counts))
            features["avg_likes_per_tweet"] = float(np.mean(like_counts))

            # RT to like ratio
            total_rt = np.sum(rt_counts)
            total_likes = np.sum(like_counts)
            features["rt_like_ratio"] = total_rt / (total_likes + 1)

            # Engagement skewness (are there outlier viral tweets?)
            total_engagement = rt_counts + like_counts
            if len(total_engagement) > 2 and np.std(total_engagement) > 0:
                features["engagement_skewness"] = float(stats.skew(total_engagement))
            else:
                features["engagement_skewness"] = 0.0
        else:
            features["avg_retweets_per_tweet"] = 0.0
            features["max_retweets_per_tweet"] = 0.0
            features["avg_likes_per_tweet"] = 0.0
            features["rt_like_ratio"] = 0.0
            features["engagement_skewness"] = 0.0

    # Temporal pattern features
    if config.include_temporal_patterns:
        hours = pre_detect["created_at"].dt.hour().to_numpy()
        if len(hours) > 0:
            # Posting hour entropy
            hour_counts = np.bincount(hours, minlength=24)
            hour_probs = hour_counts / len(hours)
            hour_probs = hour_probs[hour_probs > 0]
            features["posting_hour_entropy"] = float(-np.sum(hour_probs * np.log2(hour_probs + 1e-10)))

            # Cluster similarity (embedding coherence)
            if "cluster_similarity" in pre_detect.columns:
                sims = pre_detect["cluster_similarity"].drop_nulls()
                if len(sims) > 0:
                    features["cluster_similarity_avg"] = float(sims.mean())
                    features["cluster_similarity_min"] = float(sims.min())
                else:
                    features["cluster_similarity_avg"] = 0.0
                    features["cluster_similarity_min"] = 0.0
            else:
                features["cluster_similarity_avg"] = 0.0
                features["cluster_similarity_min"] = 0.0
        else:
            features["posting_hour_entropy"] = 0.0
            features["cluster_similarity_avg"] = 0.0
            features["cluster_similarity_min"] = 0.0

    return features


def build_feature_matrix(
    data_dir: Path,
    config: FeatureConfig | None = None,
    quick: bool = False,
) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """
    Build complete feature matrix with PSR labels.

    Returns:
        features_df: DataFrame with cluster_id, features, psr
        feature_names: List of feature column names
    """
    config = config or FeatureConfig()

    logger.info(f"Loading data from {data_dir}")
    ts_df = pl.read_parquet(data_dir / "cluster_timeseries.parquet")
    tweets_df = pl.read_parquet(data_dir / "tweets.parquet")

    # Get detections (clusters that triggered anomaly, detected before peak)
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    first_triggers = triggered.group_by("cluster_id").agg(
        pl.col("timestamp").min().alias("detection_time")
    )

    logger.info(f"Found {len(first_triggers)} clusters with anomaly triggers")

    # Filter to detections before peak
    valid_detections = []
    for row in first_triggers.iter_rows(named=True):
        cluster_id = row["cluster_id"]
        detection_time = row["detection_time"]

        cluster_ts = ts_df.filter(pl.col("cluster_id") == cluster_id)
        peak_engagement = cluster_ts["engagement"].max()
        peak_row = cluster_ts.filter(pl.col("engagement") == peak_engagement).head(1)
        peak_time = peak_row["timestamp"].item()

        if detection_time <= peak_time:
            # Compute PSR
            pre_detect = cluster_ts.filter(pl.col("timestamp") <= detection_time)
            e_detect = pre_detect["engagement"].sum()
            e_final = cluster_ts["engagement"].sum()
            psr = (e_final - e_detect) / e_final if e_final > 0 else 1.0

            valid_detections.append({
                "cluster_id": cluster_id,
                "detection_time": detection_time,
                "e_detect": e_detect,
                "e_final": e_final,
                "psr": psr,
            })

    logger.info(f"Valid detections (before peak): {len(valid_detections)}")

    if quick and len(valid_detections) > 100:
        np.random.seed(42)
        indices = np.random.choice(len(valid_detections), 100, replace=False)
        valid_detections = [valid_detections[i] for i in indices]
        logger.info(f"Quick mode: using {len(valid_detections)} samples")

    # Extract features for each detection
    all_features = []
    for i, det in enumerate(valid_detections):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing detection {i+1}/{len(valid_detections)}")

        cluster_id = det["cluster_id"]
        detection_time = det["detection_time"]

        # Timeseries features
        ts_feats = extract_timeseries_features(ts_df, cluster_id, detection_time)

        # Tweet-level features
        tweet_feats = extract_tweet_features(tweets_df, cluster_id, detection_time, config)

        # Combine
        row = {
            "cluster_id": cluster_id,
            "psr": det["psr"],
            **ts_feats,
            **tweet_feats,
        }
        all_features.append(row)

    features_df = pl.DataFrame(all_features)

    # Get feature column names (exclude cluster_id and psr)
    feature_cols = [c for c in features_df.columns if c not in ["cluster_id", "psr"]]

    logger.info(f"Extracted {len(feature_cols)} features for {len(features_df)} samples")

    return features_df, feature_cols


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    spearman_rho: float
    r2: float
    mae: float
    f2_065: float
    f2_075: float
    f2_085: float

    def to_dict(self) -> dict[str, float]:
        return {
            "spearman_rho": self.spearman_rho,
            "r2": self.r2,
            "mae": self.mae,
            "f2_065": self.f2_065,
            "f2_075": self.f2_075,
            "f2_085": self.f2_085,
        }


def evaluate_predictions(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
) -> EvaluationResult:
    """Evaluate predictions with all metrics from the LaTeX table."""
    # Regression metrics
    if len(y_true) > 2 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        rho = 0.0

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Classification metrics at different thresholds
    def compute_f2(threshold: float) -> float:
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        if y_true_binary.sum() == 0 or (len(y_true_binary) - y_true_binary.sum()) == 0:
            return 0.0
        return fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0)

    return EvaluationResult(
        spearman_rho=float(rho),
        r2=float(r2),
        mae=float(mae),
        f2_065=compute_f2(0.65),
        f2_075=compute_f2(0.75),
        f2_085=compute_f2(0.85),
    )


# =============================================================================
# FEATURE SELECTION
# =============================================================================


def greedy_forward_selection(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    feature_names: list[str],
    max_features: int = 10,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[list[str], list[float]]:
    """
    Greedy forward feature selection using RandomForest and cross-validated Spearman ρ.

    Returns:
        selected_features: List of selected feature names in order
        scores: CV scores at each step
    """
    selected_indices: list[int] = []
    selected_features: list[str] = []
    scores: list[float] = []
    remaining = set(range(X.shape[1]))

    logger.info(f"Starting greedy forward selection (max {max_features} features)")

    for step in range(max_features):
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            # Try adding this feature
            trial_indices = selected_indices + [idx]
            X_trial = X[:, trial_indices]

            # Fit model and evaluate with cross-validation
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
            )

            # Custom scorer for Spearman correlation
            def spearman_scorer(estimator, X_test, y_test):
                y_pred = estimator.predict(X_test)
                if np.std(y_pred) < 1e-8:
                    return 0.0
                rho, _ = stats.spearmanr(y_test, y_pred)
                return rho

            cv_scores = cross_val_score(model, X_trial, y, cv=cv_folds, scoring=spearman_scorer)
            mean_score = cv_scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_features.append(feature_names[best_idx])
        remaining.remove(best_idx)
        scores.append(best_score)

        logger.info(f"Step {step+1}: Added '{feature_names[best_idx]}' (CV Spearman ρ = {best_score:.4f})")

        # Early stopping if score plateaus
        if len(scores) >= 3 and scores[-1] - scores[-3] < 0.01:
            logger.info("Early stopping: score plateau detected")
            break

    return selected_features, scores


def evaluate_feature_subsets(
    X_train: NDArray[np.float32],
    X_test: NDArray[np.float32],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    feature_names: list[str],
    selected_features: list[str],
    random_state: int = 42,
) -> list[dict]:
    """Evaluate models with different feature subsets."""
    results = []

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=random_state, n_jobs=-1, verbosity=0
        ),
    }

    # Evaluate with all features
    logger.info("\n=== All Features ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = evaluate_predictions(y_test, y_pred)
        logger.info(f"{name}: Spearman ρ = {result.spearman_rho:.3f}, MAE = {result.mae:.3f}")
        results.append({
            "model": name,
            "feature_set": "all",
            "n_features": len(feature_names),
            **result.to_dict(),
        })

    # Evaluate with selected features
    if selected_features:
        selected_idx = [feature_names.index(f) for f in selected_features]
        X_train_sel = X_train[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]

        logger.info(f"\n=== Selected Features ({len(selected_features)}) ===")
        for name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)
            result = evaluate_predictions(y_test, y_pred)
            logger.info(f"{name}: Spearman ρ = {result.spearman_rho:.3f}, MAE = {result.mae:.3f}")
            results.append({
                "model": name,
                "feature_set": "selected",
                "n_features": len(selected_features),
                "features": selected_features,
                **result.to_dict(),
            })

    return results


def run_xgb_classifier_evaluation(
    X_train: NDArray[np.float32],
    X_test: NDArray[np.float32],
    y_train: NDArray[np.float32],
    y_test: NDArray[np.float32],
    feature_names: list[str],
    selected_features: list[str] | None = None,
    thresholds: list[float] = [0.65, 0.75, 0.85],
    random_state: int = 42,
) -> list[dict]:
    """Evaluate XGBClassifier at different thresholds."""
    results = []

    if selected_features:
        selected_idx = [feature_names.index(f) for f in selected_features]
        X_train_use = X_train[:, selected_idx]
        X_test_use = X_test[:, selected_idx]
        feature_set = "selected"
    else:
        X_train_use = X_train
        X_test_use = X_test
        feature_set = "all"

    for tau in thresholds:
        y_train_binary = (y_train >= tau).astype(int)
        y_test_binary = (y_test >= tau).astype(int)

        if y_train_binary.sum() < 5 or (len(y_train_binary) - y_train_binary.sum()) < 5:
            logger.warning(f"Skipping threshold {tau}: insufficient samples")
            continue

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_train_use, y_train_binary)

        # Get probabilities for evaluation
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]

        # Evaluate as regression (predict probability as PSR proxy)
        result = evaluate_predictions(y_test, y_pred_proba)

        # Also get F2 for this specific threshold
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        f2 = fbeta_score(y_test_binary, y_pred_binary, beta=2, zero_division=0)

        results.append({
            "model": f"XGBClassifier(τ={tau})",
            "feature_set": feature_set,
            "threshold": tau,
            "spearman_rho": result.spearman_rho,
            "mae": result.mae,
            "f2": f2,
        })

        logger.info(f"XGBClassifier(τ={tau}): Spearman ρ = {result.spearman_rho:.3f}, F2 = {f2:.3f}")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optimize virality prediction features")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--max-features", type=int, default=10, help="Max features to select")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Virality Feature Optimization")
    logger.info("=" * 60)

    # Build feature matrix
    features_df, feature_cols = build_feature_matrix(args.data_dir, quick=args.quick)

    # Prepare data
    X = features_df.select(feature_cols).to_numpy().astype(np.float32)
    y = features_df["psr"].to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"PSR distribution - Train: {y_train.mean():.3f} ± {y_train.std():.3f}")
    logger.info(f"PSR distribution - Test: {y_test.mean():.3f} ± {y_test.std():.3f}")

    # Greedy forward selection
    logger.info("\n" + "=" * 60)
    logger.info("GREEDY FORWARD FEATURE SELECTION")
    logger.info("=" * 60)
    selected_features, selection_scores = greedy_forward_selection(
        X_train, y_train, feature_cols,
        max_features=args.max_features,
        random_state=args.random_state,
    )

    logger.info(f"\nSelected features ({len(selected_features)}):")
    for i, (feat, score) in enumerate(zip(selected_features, selection_scores)):
        logger.info(f"  {i+1}. {feat} (cumulative CV ρ = {score:.4f})")

    # Evaluate all models
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    all_results = []

    # Regression baselines
    reg_results = evaluate_feature_subsets(
        X_train, X_test, y_train, y_test,
        feature_cols, selected_features,
        random_state=args.random_state,
    )
    all_results.extend(reg_results)

    # Classification baselines
    clf_results = run_xgb_classifier_evaluation(
        X_train, X_test, y_train, y_test,
        feature_cols, selected_features,
        random_state=args.random_state,
    )
    all_results.extend(clf_results)

    # Print final results table (matching LaTeX format)
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (LaTeX Table Format)")
    logger.info("=" * 60)

    print("\n| Baseline | Spearman ρ | R² | MAE | F2(τ=0.65) | F2(τ=0.75) | F2(τ=0.85) |")
    print("|----------|------------|-----|-----|------------|------------|------------|")

    for r in reg_results:
        if r["feature_set"] == "selected":
            print(f"| {r['model']} (selected) | {r['spearman_rho']:.3f} | {r['r2']:.3f} | {r['mae']:.3f} | {r['f2_065']:.3f} | {r['f2_075']:.3f} | {r['f2_085']:.3f} |")

    # Save results
    output = {
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "selection_scores": selection_scores,
        "results": all_results,
        "config": {
            "data_dir": str(args.data_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_samples": len(X),
            "n_features": len(feature_cols),
        },
    }

    with open(args.output_dir / "feature_selection_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    features_df.write_parquet(args.output_dir / "features_leakfree.parquet")

    logger.info(f"\nResults saved to {args.output_dir}")

    # Feature importance for best model
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE (RandomForest)")
    logger.info("=" * 60)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=args.random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    importance = dict(zip(feature_cols, rf.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 15 features by importance:")
    for feat, imp in sorted_importance[:15]:
        marker = "★" if feat in selected_features else " "
        print(f"  {marker} {feat}: {imp:.4f}")

    return output


if __name__ == "__main__":
    main()
