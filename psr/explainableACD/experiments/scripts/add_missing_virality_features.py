"""
Add missing tweet-level and user features to features_enhanced.parquet.

Computes features that were missing from original extraction:

**Tweet-level stats (6 features):**
  1. max_retweets_per_tweet
  2. avg_likes_per_tweet
  3. rt_like_ratio
  4. engagement_skewness
  5. engagement_gini
  6. account_age_avg_days

**Extended author/user features (8 features):**
  7. user_following_avg (following count)
  8. user_follower_following_ratio
  9. user_verified_ratio (fraction of verified users)
  10. user_has_description_ratio
  11. user_has_location_ratio
  12. tweet_retweet_ratio (fraction of retweets)
  13. tweet_quote_ratio (fraction of quotes)
  14. tweet_reply_ratio (fraction of replies)

**Additional engagement stats (4 features):**
  15. avg_quote_count_per_tweet
  16. avg_reply_count_per_tweet
  17. avg_view_count_per_tweet
  18. avg_bookmark_count_per_tweet

Usage:
    python experiments/scripts/add_missing_virality_features.py \
        --features experiments/results/virality_feature_selection/features_enhanced.parquet \
        --tweets data/pipeline_output/phase1_temporal/2026-01-28_08-55/tweets.parquet \
        --output experiments/results/virality_feature_selection/features_enhanced_v2.parquet
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import skew

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="experiments/results/virality_feature_selection/features_enhanced.parquet",
    )
    parser.add_argument(
        "--tweets",
        default="data/pipeline_output/phase1_temporal/2026-01-28_08-55/tweets.parquet",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/virality_feature_selection/features_enhanced_v2.parquet",
    )
    args = parser.parse_args()

    # Load existing features
    features_df = pl.read_parquet(args.features)
    logger.info(f"Loaded {len(features_df)} clusters with {len(features_df.columns)} features")

    # Load tweets
    tweets_df = pl.read_parquet(args.tweets)
    logger.info(f"Loaded {len(tweets_df)} tweets")

    # Filter to clusters in features_df
    cluster_ids = set(features_df["cluster_id"].to_list())
    tweets_df = tweets_df.filter(pl.col("cluster_id").is_in(list(cluster_ids)))
    logger.info(f"Filtered to {len(tweets_df)} tweets in {len(cluster_ids)} clusters")

    # Compute new features per cluster
    new_features = []

    for cluster_id in cluster_ids:
        cluster_tweets = tweets_df.filter(pl.col("cluster_id") == cluster_id)

        if len(cluster_tweets) == 0:
            # No tweets for this cluster (shouldn't happen, but handle it)
            new_features.append({
                "cluster_id": cluster_id,
                "max_retweets_per_tweet": 0.0,
                "avg_likes_per_tweet": 0.0,
                "rt_like_ratio": 0.0,
                "engagement_skewness": 0.0,
                "engagement_gini": 0.0,
                "account_age_avg_days": 0.0,
                "user_following_avg": 0.0,
                "user_follower_following_ratio": 0.0,
                "user_verified_ratio": 0.0,
                "user_has_description_ratio": 0.0,
                "user_has_location_ratio": 0.0,
                "tweet_retweet_ratio": 0.0,
                "tweet_quote_ratio": 0.0,
                "tweet_reply_ratio": 0.0,
                "avg_quote_count_per_tweet": 0.0,
                "avg_reply_count_per_tweet": 0.0,
                "avg_view_count_per_tweet": 0.0,
                "avg_bookmark_count_per_tweet": 0.0,
            })
            continue

        # Extract per-tweet metrics
        rts = cluster_tweets["retweet_count_at_collection"].to_numpy()
        likes = cluster_tweets["likes_at_collection"].to_numpy()
        engagement_per_tweet = rts + likes

        # Feature 1: max_retweets_per_tweet
        max_retweets_per_tweet = float(np.max(rts)) if len(rts) > 0 else 0.0

        # Feature 2: avg_likes_per_tweet
        avg_likes_per_tweet = float(np.mean(likes)) if len(likes) > 0 else 0.0

        # Feature 3: rt_like_ratio
        total_rts = np.sum(rts)
        total_likes = np.sum(likes)
        rt_like_ratio = (
            float(total_rts / total_likes) if total_likes > 0 else 0.0
        )

        # Feature 4: engagement_skewness
        if len(engagement_per_tweet) >= 3:
            engagement_skewness = float(skew(engagement_per_tweet))
        else:
            engagement_skewness = 0.0

        # Feature 5: engagement_gini
        if len(engagement_per_tweet) >= 2:
            engagement_gini = compute_gini(engagement_per_tweet)
        else:
            engagement_gini = 0.0

        # Feature 6: account_age_avg_days
        join_dates = cluster_tweets["user_join_date"].drop_nulls()
        if len(join_dates) > 0:
            # Compute age at time of first tweet in cluster
            first_tweet_time = cluster_tweets["created_at"].min()
            if first_tweet_time is not None:
                ages_days = [
                    (first_tweet_time - jd).total_seconds() / 86400.0
                    for jd in join_dates
                ]
                account_age_avg_days = float(np.mean(ages_days))
            else:
                account_age_avg_days = 0.0
        else:
            account_age_avg_days = 0.0

        # Feature 7-8: User following stats
        following = cluster_tweets["user_following_count"].drop_nulls().to_numpy()
        followers = cluster_tweets["user_followers_count"].drop_nulls().to_numpy()
        user_following_avg = float(np.mean(following)) if len(following) > 0 else 0.0

        if len(following) > 0 and len(followers) > 0 and len(following) == len(followers):
            # Ratio per user, then average
            ratios = followers / (following + 1)  # +1 to avoid division by zero
            user_follower_following_ratio = float(np.mean(ratios))
        else:
            user_follower_following_ratio = 0.0

        # Feature 9: Verified ratio
        verified = cluster_tweets["user_verified"].to_numpy()
        user_verified_ratio = float(np.mean(verified)) if len(verified) > 0 else 0.0

        # Feature 10-11: Profile completeness
        descriptions = cluster_tweets["user_description"].drop_nulls()
        locations = cluster_tweets["user_location"].drop_nulls()
        user_has_description_ratio = float(len(descriptions) / len(cluster_tweets))
        user_has_location_ratio = float(len(locations) / len(cluster_tweets))

        # Feature 12-14: Tweet type ratios
        tweet_types = cluster_tweets["tweet_type"].to_list()
        n_tweets = len(tweet_types)
        tweet_retweet_ratio = float(tweet_types.count("retweet") / n_tweets) if n_tweets > 0 else 0.0
        tweet_quote_ratio = float(tweet_types.count("quote") / n_tweets) if n_tweets > 0 else 0.0

        is_reply = cluster_tweets["is_reply"].to_numpy()
        tweet_reply_ratio = float(np.mean(is_reply)) if len(is_reply) > 0 else 0.0

        # Feature 15-18: Additional engagement metrics
        quote_counts = cluster_tweets["quote_count_at_collection"].drop_nulls().to_numpy()
        reply_counts = cluster_tweets["reply_count_at_collection"].drop_nulls().to_numpy()
        view_counts = cluster_tweets["view_count_at_collection"].drop_nulls().to_numpy()
        bookmark_counts = cluster_tweets["bookmark_count_at_collection"].drop_nulls().to_numpy()

        avg_quote_count_per_tweet = float(np.mean(quote_counts)) if len(quote_counts) > 0 else 0.0
        avg_reply_count_per_tweet = float(np.mean(reply_counts)) if len(reply_counts) > 0 else 0.0
        avg_view_count_per_tweet = float(np.mean(view_counts)) if len(view_counts) > 0 else 0.0
        avg_bookmark_count_per_tweet = float(np.mean(bookmark_counts)) if len(bookmark_counts) > 0 else 0.0

        new_features.append({
            "cluster_id": cluster_id,
            "max_retweets_per_tweet": max_retweets_per_tweet,
            "avg_likes_per_tweet": avg_likes_per_tweet,
            "rt_like_ratio": rt_like_ratio,
            "engagement_skewness": engagement_skewness,
            "engagement_gini": engagement_gini,
            "account_age_avg_days": account_age_avg_days,
            "user_following_avg": user_following_avg,
            "user_follower_following_ratio": user_follower_following_ratio,
            "user_verified_ratio": user_verified_ratio,
            "user_has_description_ratio": user_has_description_ratio,
            "user_has_location_ratio": user_has_location_ratio,
            "tweet_retweet_ratio": tweet_retweet_ratio,
            "tweet_quote_ratio": tweet_quote_ratio,
            "tweet_reply_ratio": tweet_reply_ratio,
            "avg_quote_count_per_tweet": avg_quote_count_per_tweet,
            "avg_reply_count_per_tweet": avg_reply_count_per_tweet,
            "avg_view_count_per_tweet": avg_view_count_per_tweet,
            "avg_bookmark_count_per_tweet": avg_bookmark_count_per_tweet,
        })

    new_features_df = pl.DataFrame(new_features)
    logger.info(f"Computed {len(new_features_df.columns) - 1} new features for {len(new_features_df)} clusters")

    # Join with existing features
    merged_df = features_df.join(new_features_df, on="cluster_id", how="left")
    logger.info(f"Merged: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Check for NaNs in new features
    for col in new_features_df.columns:
        if col != "cluster_id":
            null_count = merged_df[col].null_count()
            if null_count > 0:
                logger.warning(f"  {col}: {null_count} nulls — filling with 0")
                merged_df = merged_df.with_columns(pl.col(col).fill_null(0.0))

    # Save
    merged_df.write_parquet(args.output)
    logger.info(f"Saved to {args.output}")
    logger.info(f"Final: {len(merged_df)} clusters × {len(merged_df.columns)} features")


if __name__ == "__main__":
    main()
