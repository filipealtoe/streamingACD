import sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
df = pd.read_parquet(sys.argv[1]); df["day"] = pd.to_datetime(df.t0).dt.date.astype(str)
for v in ["max_followers","median_followers","stream_volume_24h","tweets_after","engagement_after","new_users_after","hours_to_k","account_age_days","early_engagement"]:
    df["l_"+v] = np.log1p(df[v])
ctx = "l_stream_volume_24h + day_index"
who = "l_median_followers + l_account_age_days + url_share + mention_share + mean_words + similarity"
def fit(y, x, rhs, label):
    m = smf.ols(f"{y} ~ {x}" + (" + " + rhs if rhs else ""), df).fit(cov_type="cluster", cov_kwds={"groups": df.day})
    lo, hi = m.conf_int().loc[x]
    print(f"  {label:<46} {m.params[x]:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
print("Spearman with tweets_after:", {v: round(spearmanr(df[v], df.tweets_after)[0], 3) for v in ["max_followers","unique_users","hours_to_k","url_share","similarity","early_engagement"]})
print("Spearman with engagement_after:", {v: round(spearmanr(df[v], df.engagement_after)[0], 3) for v in ["max_followers","unique_users","hours_to_k","early_engagement"]})
for y in ["l_tweets_after", "l_new_users_after", "l_engagement_after"]:
    print(f"\nOutcome {y} (day-clustered 95% intervals)")
    print(" treatment: log max followers of the first 10 posters")
    fit(y, "l_max_followers", "", "unadjusted")
    fit(y, "l_max_followers", f"{ctx} + {who} + unique_users", "adjusted for context and early who/what")
    fit(y, "l_max_followers", f"{ctx} + {who} + unique_users + l_hours_to_k", "also holding early speed fixed")
    print(" treatment: distinct users among the first 10 tweets")
    fit(y, "unique_users", "", "unadjusted")
    fit(y, "unique_users", f"{ctx} + {who}", "adjusted for context and early who/what")
    fit(y, "unique_users", f"{ctx} + {who} + l_hours_to_k", "also holding early speed fixed")
print("\nPlacebo: 'effect' of distinct users on a variable fixed before the narrative (stream volume 24h before)")
fit("l_stream_volume_24h", "unique_users", f"day_index + {who}", "adjusted")
print("\nLeft-truncation check: narratives whose first tweet came after day 3 only")
df["first_day"] = df.day_index - df.hours_to_k/24
sub = df[df.first_day > 3]; print(" n =", len(sub), " spearman(day_index, hours_to_k): all", round(spearmanr(df.day_index, df.hours_to_k)[0],2), " subset", round(spearmanr(sub.day_index, sub.hours_to_k)[0],2))
print(" unique_users distribution:", df.unique_users.value_counts().sort_index().to_dict())
