#!/usr/bin/env python3
"""
Generate enhanced leak-free features for virality prediction.

Changes from original:
1. REMOVES useless features (zero variance, low correlation)
2. ADDS new engineered features based on research:
   - Burstiness (temporal heterogeneity)
   - Early velocity ratio
   - Engagement per follower (efficiency)
   - Is post-peak detection
   - Amplification factor
   - Engagement momentum (jerk)

Usage:
    python experiments/scripts/generate_enhanced_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "virality_feature_selection"

# Features to DROP (zero variance or low predictive power)
DROP_FEATURES = [
    "total_verified",        # All zeros
    "verified_ratio",        # All zeros
    "usa_ratio_predetect",   # All zeros
    "account_age_min_days",  # ρ=0.004, useless
    "cluster_similarity_min", # ρ=-0.008, useless
    "cluster_similarity_avg", # ρ=-0.059, weak
    "account_age_std_days",  # ρ=-0.024, weak
    "avg_retweets_per_tweet", # ρ=0.004, useless
    "has_international_spread", # ρ=-0.019, weak (redundant with geo_entropy)
]


def compute_burstiness(timestamps: list) -> float:
    """
    Compute burstiness coefficient from inter-arrival times.

    Burstiness B = (σ - μ) / (σ + μ) where σ=std, μ=mean of inter-arrival times.
    B ∈ [-1, 1]: B=1 is maximally bursty, B=0 is Poisson, B=-1 is periodic.

    Reference: Goh & Barabási, EPL 2008
    """
    if len(timestamps) < 3:
        return 0.0

    timestamps = sorted(timestamps)
    inter_arrivals = np.diff([(t - timestamps[0]).total_seconds() for t in timestamps])

    if len(inter_arrivals) < 2:
        return 0.0

    mu = np.mean(inter_arrivals)
    sigma = np.std(inter_arrivals)

    if mu + sigma == 0:
        return 0.0

    return (sigma - mu) / (sigma + mu)


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient for concentration measurement."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GENERATING ENHANCED FEATURES")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    tweets_df = pl.read_parquet(DATA_DIR / "tweets.parquet")
    ts_df = pl.read_parquet(DATA_DIR / "cluster_timeseries.parquet")
    clusters_df = pl.read_parquet(DATA_DIR / "clusters.parquet")

    logger.info(f"  Tweets: {len(tweets_df)}")
    logger.info(f"  Timeseries rows: {len(ts_df)}")
    logger.info(f"  Clusters: {len(clusters_df)}")

    # Get detection times and compute PSR correctly
    logger.info("Computing detection times and PSR...")

    # Find triggered clusters
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    detection_times_df = triggered.group_by("cluster_id").agg([
        pl.col("timestamp").min().alias("detection_time"),
    ])

    # Compute PSR properly: only include detections BEFORE peak
    valid_detections = []

    for row in detection_times_df.iter_rows(named=True):
        cluster_id = row["cluster_id"]
        detection_time = row["detection_time"]

        cluster_ts = ts_df.filter(pl.col("cluster_id") == cluster_id).sort("timestamp")

        if len(cluster_ts) == 0:
            continue

        # Find peak time
        peak_engagement = cluster_ts["engagement"].max()
        peak_row = cluster_ts.filter(pl.col("engagement") == peak_engagement).head(1)
        peak_time = peak_row["timestamp"].item()

        # Only include if detection is before or at peak (valid early detection)
        if detection_time <= peak_time:
            # Compute PSR correctly: cumulative engagement
            pre_detect = cluster_ts.filter(pl.col("timestamp") <= detection_time)
            e_detect = pre_detect["engagement"].sum()
            e_final = cluster_ts["engagement"].sum()
            psr = (e_final - e_detect) / e_final if e_final > 0 else 1.0

            valid_detections.append({
                "cluster_id": cluster_id,
                "detection_time": detection_time,
                "psr": float(psr),
            })

    psr_df = pl.DataFrame(valid_detections)
    cluster_ids = psr_df["cluster_id"].to_list()
    logger.info(f"  Valid detections (before peak): {len(cluster_ids)}")

    # Build detection time lookup from valid detections
    detection_times = {d["cluster_id"]: d["detection_time"] for d in valid_detections}

    # =========================================================================
    # COMPUTE FEATURES
    # =========================================================================
    features_list = []

    for cluster_id in cluster_ids:
        det_time = detection_times[cluster_id]

        # Filter data to pre-detection only
        cluster_ts = ts_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("timestamp") <= det_time)
        ).sort("timestamp")

        cluster_tweets = tweets_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("created_at") <= det_time)
        )

        if len(cluster_ts) == 0:
            continue

        features = {"cluster_id": cluster_id}

        # -----------------------------------------------------------------
        # TEMPORAL FEATURES (keeping the good ones)
        # -----------------------------------------------------------------
        features["cumulative_tweets"] = cluster_ts["tweet_count"].sum()
        features["cumulative_engagement"] = cluster_ts["engagement"].sum()
        features["windows_since_start"] = len(cluster_ts)
        features["mean_tweets_per_window"] = cluster_ts["tweet_count"].mean()
        features["mean_engagement_per_window"] = cluster_ts["engagement"].mean()

        # Growth dynamics
        if len(cluster_ts) >= 2:
            eng = cluster_ts["engagement"].to_numpy()
            features["growth_rate"] = eng[-1] / eng[0] if eng[0] > 0 else 1.0

            if len(eng) >= 3:
                velocity = np.diff(eng)
                features["acceleration"] = velocity[-1] - velocity[0] if len(velocity) >= 2 else 0.0
                features["engagement_velocity"] = np.mean(np.abs(velocity))

                # NEW: Jerk (rate of change of acceleration)
                if len(velocity) >= 2:
                    accel = np.diff(velocity)
                    features["engagement_jerk"] = np.mean(np.abs(accel)) if len(accel) > 0 else 0.0
                else:
                    features["engagement_jerk"] = 0.0
            else:
                features["acceleration"] = 0.0
                features["engagement_velocity"] = 0.0
                features["engagement_jerk"] = 0.0
        else:
            features["growth_rate"] = 1.0
            features["acceleration"] = 0.0
            features["engagement_velocity"] = 0.0
            features["engagement_jerk"] = 0.0

        # -----------------------------------------------------------------
        # NEW: BURSTINESS FEATURES
        # -----------------------------------------------------------------
        if len(cluster_tweets) >= 3:
            tweet_times = cluster_tweets["created_at"].to_list()
            features["burstiness"] = compute_burstiness(tweet_times)

            # Inter-arrival time stats
            timestamps_sorted = sorted(tweet_times)
            inter_arrivals = np.array([
                (timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds()
                for i in range(len(timestamps_sorted)-1)
            ])
            if len(inter_arrivals) > 0:
                features["inter_arrival_mean"] = np.mean(inter_arrivals)
                features["inter_arrival_std"] = np.std(inter_arrivals)
                features["inter_arrival_cv"] = (
                    np.std(inter_arrivals) / np.mean(inter_arrivals)
                    if np.mean(inter_arrivals) > 0 else 0.0
                )
            else:
                features["inter_arrival_mean"] = 0.0
                features["inter_arrival_std"] = 0.0
                features["inter_arrival_cv"] = 0.0
        else:
            features["burstiness"] = 0.0
            features["inter_arrival_mean"] = 0.0
            features["inter_arrival_std"] = 0.0
            features["inter_arrival_cv"] = 0.0

        # -----------------------------------------------------------------
        # NEW: EARLY VELOCITY RATIO
        # -----------------------------------------------------------------
        if len(cluster_ts) >= 2:
            # Engagement in first 25% of windows vs at detection
            n_early = max(1, len(cluster_ts) // 4)
            early_eng = cluster_ts["engagement"][:n_early].sum()
            total_eng = cluster_ts["engagement"].sum()
            features["early_velocity_ratio"] = early_eng / total_eng if total_eng > 0 else 0.0

            # Engagement in last window vs total
            features["final_window_ratio"] = (
                cluster_ts["engagement"][-1] / total_eng if total_eng > 0 else 0.0
            )
        else:
            features["early_velocity_ratio"] = 1.0
            features["final_window_ratio"] = 1.0

        # -----------------------------------------------------------------
        # NEW: IS POST-PEAK (already peaked before detection?)
        # -----------------------------------------------------------------
        if len(cluster_ts) >= 2:
            eng = cluster_ts["engagement"].to_numpy()
            peak_idx = np.argmax(eng)
            features["is_post_peak"] = 1.0 if peak_idx < len(eng) - 1 else 0.0
            features["peak_position_ratio"] = peak_idx / (len(eng) - 1) if len(eng) > 1 else 1.0
        else:
            features["is_post_peak"] = 0.0
            features["peak_position_ratio"] = 1.0

        # -----------------------------------------------------------------
        # USER AUTHORITY FEATURES (keeping)
        # -----------------------------------------------------------------
        features["max_followers_seen"] = cluster_ts["max_followers"].max()
        features["avg_followers_seen"] = cluster_ts["avg_followers"].mean()
        features["total_unique_users"] = cluster_ts["unique_users"].sum()

        # -----------------------------------------------------------------
        # NEW: ENGAGEMENT EFFICIENCY
        # -----------------------------------------------------------------
        total_followers = features["avg_followers_seen"] * features["total_unique_users"]
        features["engagement_per_follower"] = (
            features["cumulative_engagement"] / total_followers
            if total_followers > 0 else 0.0
        )

        # -----------------------------------------------------------------
        # NEW: AMPLIFICATION FACTOR
        # -----------------------------------------------------------------
        if len(cluster_tweets) > 0 and "retweet_count" in cluster_tweets.columns:
            total_retweets = cluster_tweets["retweet_count"].sum()
            features["amplification_factor"] = (
                total_retweets / features["total_unique_users"]
                if features["total_unique_users"] > 0 else 0.0
            )
        else:
            features["amplification_factor"] = 0.0

        # -----------------------------------------------------------------
        # ANOMALY SIGNAL FEATURES (keeping)
        # -----------------------------------------------------------------
        features["z_score_at_detect"] = cluster_ts["z_score"][-1] if len(cluster_ts) > 0 else 0.0
        features["max_z_score_seen"] = cluster_ts["z_score"].max()
        features["z_score_count_at_detect"] = cluster_ts["z_score_count"][-1] if len(cluster_ts) > 0 else 0.0
        features["z_score_engagement_at_detect"] = cluster_ts["z_score_engagement"][-1] if len(cluster_ts) > 0 else 0.0
        features["kleinberg_state_at_detect"] = cluster_ts["kleinberg_state"][-1] if len(cluster_ts) > 0 else 0.0

        # -----------------------------------------------------------------
        # GEOGRAPHIC FEATURES (keeping good ones)
        # -----------------------------------------------------------------
        if len(cluster_tweets) > 0 and "user_location_country" in cluster_tweets.columns:
            countries = cluster_tweets["user_location_country"].drop_nulls().to_list()
            if len(countries) > 0:
                from collections import Counter
                country_counts = Counter(countries)
                total = sum(country_counts.values())
                probs = np.array(list(country_counts.values())) / total
                features["geographic_entropy_predetect"] = -np.sum(probs * np.log(probs + 1e-10))
                features["unique_countries_predetect"] = len(country_counts)
            else:
                features["geographic_entropy_predetect"] = 0.0
                features["unique_countries_predetect"] = 0
        else:
            features["geographic_entropy_predetect"] = np.nan
            features["unique_countries_predetect"] = np.nan

        # -----------------------------------------------------------------
        # TEMPORAL PATTERN FEATURES (keeping - highest predictive group!)
        # -----------------------------------------------------------------
        if len(cluster_ts) > 0:
            det_hour = det_time.hour
            det_weekday = det_time.weekday()
            features["hour_of_day"] = det_hour
            features["is_weekend"] = 1.0 if det_weekday >= 5 else 0.0
            features["is_morning"] = 1.0 if 6 <= det_hour < 12 else 0.0
            features["is_evening"] = 1.0 if 18 <= det_hour < 24 else 0.0

        # Posting hour entropy
        if len(cluster_tweets) > 0:
            hours = [t.hour for t in cluster_tweets["created_at"].to_list()]
            hour_counts = np.zeros(24)
            for h in hours:
                hour_counts[h] += 1
            hour_probs = hour_counts / (hour_counts.sum() + 1e-10)
            features["posting_hour_entropy"] = -np.sum(hour_probs * np.log(hour_probs + 1e-10))
        else:
            features["posting_hour_entropy"] = 0.0

        # -----------------------------------------------------------------
        # ENGAGEMENT DISTRIBUTION FEATURES (keeping good ones)
        # -----------------------------------------------------------------
        if len(cluster_tweets) > 0 and "like_count" in cluster_tweets.columns:
            likes = cluster_tweets["like_count"].to_numpy()
            retweets = cluster_tweets["retweet_count"].to_numpy() if "retweet_count" in cluster_tweets.columns else np.zeros_like(likes)

            features["max_retweets_per_tweet"] = np.max(retweets) if len(retweets) > 0 else 0.0
            features["avg_likes_per_tweet"] = np.mean(likes) if len(likes) > 0 else 0.0

            total_rt = np.sum(retweets)
            total_likes = np.sum(likes)
            features["rt_like_ratio"] = total_rt / (total_likes + 1) if total_likes > 0 else 0.0

            # Engagement skewness
            engagement = likes + retweets
            if np.std(engagement) > 0:
                from scipy.stats import skew
                features["engagement_skewness"] = skew(engagement)
            else:
                features["engagement_skewness"] = 0.0

            # NEW: Engagement Gini (concentration)
            features["engagement_gini"] = compute_gini(engagement)
        else:
            features["max_retweets_per_tweet"] = np.nan
            features["avg_likes_per_tweet"] = np.nan
            features["rt_like_ratio"] = np.nan
            features["engagement_skewness"] = np.nan
            features["engagement_gini"] = np.nan

        # -----------------------------------------------------------------
        # ACCOUNT AGE FEATURES (only keeping avg, dropping min/std)
        # -----------------------------------------------------------------
        if len(cluster_tweets) > 0 and "user_created_at" in cluster_tweets.columns:
            user_ages = []
            for tweet_time, user_created in zip(
                cluster_tweets["created_at"].to_list(),
                cluster_tweets["user_created_at"].to_list()
            ):
                if user_created is not None:
                    age_days = (tweet_time - user_created).days
                    user_ages.append(age_days)

            if len(user_ages) > 0:
                features["account_age_avg_days"] = np.mean(user_ages)
            else:
                features["account_age_avg_days"] = np.nan
        else:
            features["account_age_avg_days"] = np.nan

        # -----------------------------------------------------------------
        # NEW: USER CONCENTRATION
        # -----------------------------------------------------------------
        if len(cluster_tweets) > 0 and "user_id" in cluster_tweets.columns:
            user_tweet_counts = cluster_tweets.group_by("user_id").agg(
                pl.count().alias("tweet_count")
            )["tweet_count"].to_numpy()
            features["user_gini"] = compute_gini(user_tweet_counts)
            features["top_user_ratio"] = (
                np.max(user_tweet_counts) / np.sum(user_tweet_counts)
                if np.sum(user_tweet_counts) > 0 else 0.0
            )
        else:
            features["user_gini"] = np.nan
            features["top_user_ratio"] = np.nan

        features_list.append(features)

    # Convert to DataFrame
    features_df = pl.DataFrame(features_list)

    # Add PSR
    features_df = features_df.join(
        psr_df.select(["cluster_id", "psr"]),
        on="cluster_id",
        how="inner"
    )

    # Remove any features in DROP_FEATURES that might have been added
    existing_drop = [f for f in DROP_FEATURES if f in features_df.columns]
    if existing_drop:
        features_df = features_df.drop(existing_drop)

    logger.info(f"\n=== FINAL FEATURE SET ===")
    logger.info(f"Samples: {len(features_df)}")
    feature_cols = [c for c in features_df.columns if c not in ["cluster_id", "psr"]]
    logger.info(f"Features: {len(feature_cols)}")

    # List all features
    logger.info("\nFeatures included:")
    for i, col in enumerate(sorted(feature_cols), 1):
        logger.info(f"  {i:2}. {col}")

    # Save
    output_path = OUTPUT_DIR / "features_enhanced.parquet"
    features_df.write_parquet(output_path)
    logger.info(f"\nSaved to: {output_path}")

    # Quick correlation check for new features
    logger.info("\n=== NEW FEATURE CORRELATIONS WITH PSR ===")
    from scipy import stats as sp_stats

    new_features = [
        "burstiness", "inter_arrival_cv", "early_velocity_ratio",
        "final_window_ratio", "is_post_peak", "peak_position_ratio",
        "engagement_per_follower", "amplification_factor", "engagement_jerk",
        "engagement_gini", "user_gini", "top_user_ratio",
        "is_morning", "is_evening"
    ]

    psr = features_df["psr"].to_numpy()
    for col in new_features:
        if col in features_df.columns:
            vals = features_df[col].to_numpy()
            mask = ~np.isnan(vals)
            if mask.sum() > 10:
                rho, p = sp_stats.spearmanr(vals[mask], psr[mask])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                logger.info(f"  {col:<25}: ρ={rho:+.3f}{sig}")


if __name__ == "__main__":
    main()
