"""
Virality prediction benchmark: all paper baselines on features_enhanced.parquet.

Benchmarks:
  - Random (constant-mean baseline)
  - BayesianRidge (paper best regression: ρ=0.551)
  - SVR-RBF (paper best F2@0.65=0.632)
  - Ridge
  - LightGBM (current production model)
  - RandomForest
  - XGBoost

Metrics (Table 2 format):
  Spearman ρ ↑ | R² ↑ | MAE ↓ | F2@0.65 ↑ | F2@0.75 ↑ | F2@0.85 ↑

Usage:
    python experiments/scripts/benchmark_virality_baselines.py \
        --features experiments/results/virality_feature_selection/features_enhanced.parquet \
        --seed 42 --test-fraction 0.20
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, fbeta_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION (matches optimize_virality_features.py exactly)
# =============================================================================


@dataclass
class EvaluationResult:
    spearman_rho: float
    r2: float
    mae: float
    f2_065: float
    f2_075: float
    f2_085: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_predictions(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
) -> EvaluationResult:
    """Evaluate predictions with all Table 2 metrics."""
    if len(y_true) > 2 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        rho = 0.0

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

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
# MODELS
# =============================================================================


def get_models() -> dict:
    """Return all benchmark models. SVR needs StandardScaler (applied externally)."""
    models = {
        "BayesianRidge": BayesianRidge(max_iter=500),
        "Ridge": Ridge(alpha=1.0),
        "SVR-RBF": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    if lgb is not None:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        logger.warning("LightGBM not installed — skipping")

    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        logger.warning("XGBoost not installed — skipping")

    return models


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="experiments/results/virality_feature_selection/features_enhanced.parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--output", default="experiments/results/virality_benchmark.json")
    args = parser.parse_args()

    # Load data
    df = pl.read_parquet(args.features)
    logger.info(f"Loaded {len(df)} clusters, {len(df.columns)} columns")

    feature_cols = [c for c in df.columns if c not in ["cluster_id", "psr"]]

    # Drop columns that are 100% NaN (features not computed in this dataset)
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    nan_counts = np.isnan(X_raw).sum(axis=0)
    valid_cols_mask = nan_counts < len(df)  # Keep cols with < 100% NaN
    feature_cols = [c for i, c in enumerate(feature_cols) if valid_cols_mask[i]]
    logger.info(f"Dropped {(~valid_cols_mask).sum()} fully-NaN columns, keeping {len(feature_cols)} features")

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df["psr"].to_numpy().astype(np.float32)

    # Impute remaining NaNs with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.isnan(col).any():
            median_val = np.nanmedian(col)
            X[:, col_idx] = np.where(np.isnan(col), median_val, col)
            logger.info(f"  Imputed {np.isnan(col).sum()} NaNs in feature {col_idx} with median={median_val:.3f}")

    # Final validation check
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    logger.info(f"Final dataset: {len(y)} samples × {X.shape[1]} features")

    # Train/test split (stratified-ish: shuffle with seed)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(y))
    n_test = int(len(y) * args.test_fraction)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Split: {len(y_train)} train / {len(y_test)} test")

    # Scale features (required for SVR, beneficial for Ridge/BayesianRidge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random baseline (predict train mean)
    mean_pred = np.full(len(y_test), y_train.mean())
    random_result = evaluate_predictions(y_test, mean_pred)
    logger.info(f"[Random] ρ={random_result.spearman_rho:.3f} MAE={random_result.mae:.3f}")

    # Benchmark all models
    results = {"Random": random_result.to_dict()}

    # Models that need scaled input
    needs_scaling = {"SVR-RBF", "BayesianRidge", "Ridge"}

    for name, model in get_models().items():
        logger.info(f"Training {name}...")
        if name in needs_scaling:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled).astype(np.float32)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test).astype(np.float32)

        result = evaluate_predictions(y_test, preds)
        results[name] = result.to_dict()
        logger.info(
            f"[{name}] ρ={result.spearman_rho:.3f} R²={result.r2:.3f} "
            f"MAE={result.mae:.3f} F2@.65={result.f2_065:.3f} "
            f"F2@.75={result.f2_075:.3f} F2@.85={result.f2_085:.3f}"
        )

    # Print LaTeX-style table
    print("\n" + "=" * 90)
    print(f"{'Model':<16} {'ρ':>8} {'R²':>8} {'MAE':>8} {'F2@.65':>8} {'F2@.75':>8} {'F2@.85':>8}")
    print("-" * 90)
    # Sort by spearman_rho descending
    for name in sorted(results, key=lambda k: results[k]["spearman_rho"], reverse=True):
        r = results[name]
        print(
            f"{name:<16} {r['spearman_rho']:>8.3f} {r['r2']:>8.3f} {r['mae']:>8.3f} "
            f"{r['f2_065']:>8.3f} {r['f2_075']:>8.3f} {r['f2_085']:>8.3f}"
        )
    print("=" * 90)

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
