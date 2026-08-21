# CIKM 2026 Virality Dataset Schema

The schema below is read directly from the two committed Parquet files. `NaN` counts refer to IEEE floating-point NaN values;
Parquet null counts are zero for every column.

## Labels: `psr_labels.parquet`

| Column | Parquet type | NaN/null count | Meaning |
|---|---|---:|---|
| `cluster_id` | `int64` | 0 | Cluster-instance identifier |
| `detection_time` | `timestamp[us]` | 0 | Start timestamp of the first anomaly-triggering 60-minute window |
| `e_detect` | `int64` | 0 | Cumulative engagement at detection |
| `e_final` | `int64` | 0 | Final cumulative engagement in the observed stream |
| `psr` | `double` | 0 | `(e_final - e_detect) / e_final` |

## Feature matrix: `features_enhanced.parquet`

The file contains `cluster_id`, the 42 features below, and `psr`, for 44 columns in total.

| Feature | Parquet type | NaN rows | Feature family |
|---|---|---:|---|
| `cumulative_tweets` | `int64` | 0 | Activity volume |
| `cumulative_engagement` | `int64` | 0 | Engagement volume |
| `windows_since_start` | `int64` | 0 | Temporal history |
| `mean_tweets_per_window` | `double` | 0 | Activity rate |
| `mean_engagement_per_window` | `double` | 0 | Engagement rate |
| `growth_rate` | `double` | 0 | Growth dynamics |
| `acceleration` | `double` | 0 | Growth dynamics |
| `engagement_velocity` | `double` | 0 | Growth dynamics |
| `engagement_jerk` | `double` | 0 | Growth dynamics |
| `burstiness` | `double` | 0 | Inter-arrival dynamics |
| `inter_arrival_mean` | `double` | 0 | Inter-arrival dynamics |
| `inter_arrival_std` | `double` | 0 | Inter-arrival dynamics |
| `inter_arrival_cv` | `double` | 0 | Inter-arrival dynamics |
| `early_velocity_ratio` | `double` | 0 | Early activity profile |
| `final_window_ratio` | `double` | 0 | Early activity profile |
| `is_post_peak` | `double` | 0 | Peak-relative timing |
| `peak_position_ratio` | `double` | 0 | Peak-relative timing |
| `max_followers_seen` | `int64` | 0 | Audience aggregate |
| `avg_followers_seen` | `double` | 0 | Audience aggregate |
| `total_unique_users` | `int64` | 0 | Participation aggregate |
| `engagement_per_follower` | `double` | 0 | Engagement efficiency |
| `amplification_factor` | `double` | 0 | Amplification dynamics |
| `z_score_at_detect` | `double` | 0 | Anomaly state |
| `max_z_score_seen` | `double` | 0 | Anomaly state |
| `z_score_count_at_detect` | `double` | 0 | Anomaly state |
| `z_score_engagement_at_detect` | `double` | 0 | Anomaly state |
| `kleinberg_state_at_detect` | `int64` | 0 | Burst-state indicator |
| `geographic_entropy_predetect` | `double` | 529 | Geographic aggregate; unavailable in this release |
| `unique_countries_predetect` | `double` | 529 | Geographic aggregate; unavailable in this release |
| `hour_of_day` | `int64` | 0 | Detection-time calendar feature |
| `is_weekend` | `double` | 0 | Detection-time calendar feature |
| `is_morning` | `double` | 0 | Detection-time calendar feature |
| `is_evening` | `double` | 0 | Detection-time calendar feature |
| `posting_hour_entropy` | `double` | 0 | Temporal diversity |
| `max_retweets_per_tweet` | `double` | 529 | Post-level engagement aggregate; unavailable in this release |
| `avg_likes_per_tweet` | `double` | 529 | Post-level engagement aggregate; unavailable in this release |
| `rt_like_ratio` | `double` | 529 | Post-level engagement aggregate; unavailable in this release |
| `engagement_skewness` | `double` | 529 | Engagement distribution; unavailable in this release |
| `engagement_gini` | `double` | 529 | Engagement distribution; unavailable in this release |
| `account_age_avg_days` | `double` | 529 | Account aggregate; unavailable in this release |
| `user_gini` | `double` | 50 | Participation concentration |
| `top_user_ratio` | `double` | 50 | Participation concentration |

The trailing columns are:

| Column | Parquet type | NaN/null count | Meaning |
|---|---|---:|---|
| `cluster_id` | `int64` | 0 | Cluster-instance identifier |
| `psr` | `double` | 0 | Virality label, exactly matching `psr_labels.parquet` by `cluster_id` |

## Training-time missing-value handling

The packaged tabular baseline source converts NaN, positive infinity, and negative infinity to `0.0` before splitting and fitting:

```python
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

Any comparison should either use the same rule or state its alternative preprocessing explicitly.
