#!/usr/bin/env python3
"""
Run expanded baselines WITH hyperparameter tuning for IJCAI paper.

Models:
- Random (trivial baseline)
- Tree-based: ExtraTrees, HistGradientBoosting, GradientBoosting, LightGBM, CatBoost, RandomForest, XGBoost
- Linear: Ridge, Lasso, ElasticNet, Huber, BayesianRidge
- Kernel: SVR (RBF)
- Neural: MLP
- Ensemble: Voting, Stacking

Each model gets a hyperparameter grid search with 5-fold CV.

Usage:
    python experiments/scripts/run_tuned_baselines.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    Ridge,
)
from sklearn.metrics import fbeta_score, make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use enhanced features if available
FEATURES_ENHANCED = PROJECT_ROOT / "experiments" / "results" / "virality_feature_selection" / "features_enhanced.parquet"
FEATURES_ORIGINAL = PROJECT_ROOT / "experiments" / "results" / "virality_feature_selection" / "features_leakfree.parquet"
FEATURES_PATH = FEATURES_ENHANCED if FEATURES_ENHANCED.exists() else FEATURES_ORIGINAL
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "virality_tuned"


@dataclass
class BaselineResult:
    name: str
    spearman_rho: float
    cv_rho: float
    r2: float
    mae: float
    f2_065: float
    f2_075: float
    f2_085: float
    best_params: dict = None
    notes: str = ""


def evaluate(y_true: NDArray, y_pred: NDArray) -> dict:
    """Compute all metrics."""
    y_pred = np.clip(y_pred, 0, 1)

    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        rho = 0.0

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    def f2(tau):
        yt = (y_true >= tau).astype(int)
        yp = (y_pred >= tau).astype(int)
        if yt.sum() == 0 or (len(yt) - yt.sum()) == 0:
            return 0.0
        return fbeta_score(yt, yp, beta=2, zero_division=0)

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "r2": float(r2),
        "mae": float(mae),
        "f2_065": f2(0.65),
        "f2_075": f2(0.75),
        "f2_085": f2(0.85),
    }


def spearman_scorer(y_true, y_pred):
    """Custom scorer for Spearman correlation."""
    rho, _ = stats.spearmanr(y_true, y_pred)
    return rho if not np.isnan(rho) else 0.0


def log_result(result: BaselineResult) -> None:
    """Log result in compact format."""
    params_str = f", params={result.best_params}" if result.best_params else ""
    logger.info(
        f"  {result.name}: ρ={result.spearman_rho:.3f}, CV_ρ={result.cv_rho:.3f}, "
        f"R²={result.r2:.3f}, MAE={result.mae:.3f}, F2(.65)={result.f2_065:.3f}{params_str}"
    )


def tune_and_evaluate(
    name: str,
    model,
    param_grid: dict,
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    cv: int = 5,
) -> BaselineResult:
    """Run grid search and evaluate on test set."""
    scorer = make_scorer(spearman_scorer)

    if param_grid:
        grid = GridSearchCV(
            model, param_grid, scoring=scorer, cv=cv, n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        cv_rho = grid.best_score_
        best_params = grid.best_params_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        # Get CV score
        y_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=cv)
        cv_rho, _ = stats.spearmanr(y_train, y_pred_cv)
        cv_rho = cv_rho if not np.isnan(cv_rho) else 0.0
        best_params = None

    # Evaluate on test
    y_pred = best_model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    return BaselineResult(
        name=name,
        cv_rho=cv_rho,
        best_params=best_params,
        **metrics,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TUNED BASELINES FOR IJCAI")
    logger.info("=" * 70)

    # Load data
    features_type = "ENHANCED" if "enhanced" in str(FEATURES_PATH) else "ORIGINAL"
    logger.info(f"Loading {features_type} features from {FEATURES_PATH}")
    df = pl.read_parquet(FEATURES_PATH)

    feature_cols = [c for c in df.columns if c not in ["cluster_id", "psr"]]
    X = df.select(feature_cols).to_numpy()
    y = df["psr"].to_numpy()

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Samples: {len(y)}, Features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for linear/kernel/neural models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # =========================================================================
    # RANDOM BASELINE
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("RANDOM BASELINE")
    logger.info("=" * 50)

    np.random.seed(42)
    y_pred_random = np.random.uniform(0, 1, len(y_test))
    metrics = evaluate(y_test, y_pred_random)
    result = BaselineResult(name="Random", cv_rho=0.0, **metrics, notes="Trivial baseline")
    log_result(result)
    results.append(result)

    # =========================================================================
    # TREE-BASED MODELS
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("TREE-BASED MODELS (with tuning)")
    logger.info("=" * 50)

    # RandomForest
    logger.info("Tuning RandomForest...")
    result = tune_and_evaluate(
        "RandomForest",
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, None],
            "min_samples_split": [2, 5, 10],
        },
        X_train, y_train, X_test, y_test,
    )
    log_result(result)
    results.append(result)

    # ExtraTrees
    logger.info("Tuning ExtraTrees...")
    result = tune_and_evaluate(
        "ExtraTrees",
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, None],
            "min_samples_split": [2, 5, 10],
        },
        X_train, y_train, X_test, y_test,
    )
    log_result(result)
    results.append(result)

    # XGBoost
    logger.info("Tuning XGBoost...")
    result = tune_and_evaluate(
        "XGBoost",
        XGBRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        X_train, y_train, X_test, y_test,
    )
    log_result(result)
    results.append(result)

    # HistGradientBoosting
    logger.info("Tuning HistGradientBoosting...")
    result = tune_and_evaluate(
        "HistGradientBoosting",
        HistGradientBoostingRegressor(random_state=42),
        {
            "max_depth": [6, 10, None],
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
        X_train, y_train, X_test, y_test,
    )
    log_result(result)
    results.append(result)

    # GradientBoosting
    logger.info("Tuning GradientBoosting...")
    result = tune_and_evaluate(
        "GradientBoosting",
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        },
        X_train, y_train, X_test, y_test,
    )
    log_result(result)
    results.append(result)

    # LightGBM
    try:
        import lightgbm as lgb
        logger.info("Tuning LightGBM...")
        result = tune_and_evaluate(
            "LightGBM",
            lgb.LGBMRegressor(random_state=42, verbose=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [6, 10, -1],
                "learning_rate": [0.05, 0.1],
            },
            X_train, y_train, X_test, y_test,
        )
        log_result(result)
        results.append(result)
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # CatBoost (fully manual due to sklearn compatibility issues)
    try:
        from catboost import CatBoostRegressor
        from sklearn.model_selection import KFold

        logger.info("Tuning CatBoost (manual grid search)...")

        best_cv_rho = -1
        best_params = None

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for iterations in [100, 200]:
            for depth in [4, 6, 8]:
                for lr in [0.05, 0.1]:
                    # Manual 5-fold CV
                    fold_rhos = []
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]

                        model = CatBoostRegressor(
                            iterations=iterations, depth=depth, learning_rate=lr,
                            random_state=42, verbose=0
                        )
                        model.fit(X_tr, y_tr)
                        y_pred_val = model.predict(X_val)
                        rho, _ = stats.spearmanr(y_val, y_pred_val)
                        fold_rhos.append(rho if not np.isnan(rho) else 0.0)

                    cv_rho = np.mean(fold_rhos)

                    if cv_rho > best_cv_rho:
                        best_cv_rho = cv_rho
                        best_params = {"iterations": iterations, "depth": depth, "learning_rate": lr}

        # Train final model with best params
        best_model = CatBoostRegressor(
            iterations=best_params["iterations"],
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            random_state=42, verbose=0
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        result = BaselineResult(
            name="CatBoost", cv_rho=best_cv_rho, best_params=best_params, **metrics
        )
        log_result(result)
        results.append(result)
    except ImportError:
        logger.warning("CatBoost not installed, skipping")

    # =========================================================================
    # LINEAR MODELS
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("LINEAR MODELS (with tuning)")
    logger.info("=" * 50)

    # Ridge
    logger.info("Tuning Ridge...")
    result = tune_and_evaluate(
        "Ridge",
        Ridge(),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # Lasso
    logger.info("Tuning Lasso...")
    result = tune_and_evaluate(
        "Lasso",
        Lasso(random_state=42, max_iter=5000),
        {"alpha": [0.001, 0.01, 0.1, 1.0]},
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # ElasticNet
    logger.info("Tuning ElasticNet...")
    result = tune_and_evaluate(
        "ElasticNet",
        ElasticNet(random_state=42, max_iter=5000),
        {
            "alpha": [0.001, 0.01, 0.1],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # Huber
    logger.info("Tuning Huber...")
    result = tune_and_evaluate(
        "Huber",
        HuberRegressor(max_iter=500),
        {"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [0.0001, 0.001, 0.01]},
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # BayesianRidge
    logger.info("Tuning BayesianRidge...")
    result = tune_and_evaluate(
        "BayesianRidge",
        BayesianRidge(),
        {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
        },
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # =========================================================================
    # KERNEL METHODS
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("KERNEL METHODS (with tuning)")
    logger.info("=" * 50)

    # SVR
    logger.info("Tuning SVR...")
    result = tune_and_evaluate(
        "SVR (RBF)",
        SVR(),
        {
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": ["scale", "auto"],
        },
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # =========================================================================
    # NEURAL NETWORK
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("NEURAL NETWORK (with tuning)")
    logger.info("=" * 50)

    # MLP
    logger.info("Tuning MLP...")
    result = tune_and_evaluate(
        "MLP",
        MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
        {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "alpha": [0.001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01],
        },
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log_result(result)
    results.append(result)

    # =========================================================================
    # ENSEMBLE METHODS (use best params from above)
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("ENSEMBLE METHODS")
    logger.info("=" * 50)

    # Get best tree and linear models for ensemble
    rf_best = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    xgb_best = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    ridge_best = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    # Voting
    logger.info("Running Voting...")
    voting = VotingRegressor([
        ("rf", rf_best),
        ("xgb", xgb_best),
        ("ridge", ridge_best),
    ])
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    y_pred_cv = cross_val_predict(voting, X_train, y_train, cv=5)
    cv_rho, _ = stats.spearmanr(y_train, y_pred_cv)
    metrics = evaluate(y_test, y_pred)
    result = BaselineResult(name="Voting (RF+XGB+Ridge)", cv_rho=cv_rho, **metrics)
    log_result(result)
    results.append(result)

    # Stacking
    logger.info("Running Stacking...")
    stacking = StackingRegressor(
        estimators=[
            ("rf", rf_best),
            ("xgb", xgb_best),
            ("ridge", ridge_best),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_pred_cv = cross_val_predict(stacking, X_train, y_train, cv=3)  # Fewer CV to speed up
    cv_rho, _ = stats.spearmanr(y_train, y_pred_cv)
    metrics = evaluate(y_test, y_pred)
    result = BaselineResult(name="Stacking (RF+XGB+Ridge)", cv_rho=cv_rho, **metrics)
    log_result(result)
    results.append(result)

    # =========================================================================
    # TEXT BASELINES (BERTweet)
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("TEXT BASELINES")
    logger.info("=" * 50)

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        # Load tweets data
        DATA_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
        tweets_df = pl.read_parquet(DATA_DIR / "tweets.parquet")
        logger.info(f"  Loaded {len(tweets_df)} tweets")

        # Get cluster IDs from features
        cluster_ids = df["cluster_id"].to_list()

        # Get representative text per cluster (first tweet)
        cluster_texts = {}
        for cid in cluster_ids:
            cluster_tweets = tweets_df.filter(pl.col("cluster_id") == cid)
            if len(cluster_tweets) > 0 and "text" in cluster_tweets.columns:
                text = cluster_tweets["text"][0]
                if text:
                    cluster_texts[cid] = str(text)[:512]  # Truncate

        logger.info(f"  Clusters with text: {len(cluster_texts)}")

        if len(cluster_texts) > 100:
            # Load BERTweet
            logger.info("  Loading BERTweet model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
            model.eval()

            # Generate embeddings
            logger.info("  Generating embeddings...")
            embeddings = {}
            with torch.no_grad():
                for cid, text in cluster_texts.items():
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                    embeddings[cid] = emb

            # Build aligned datasets
            train_indices = []
            test_indices = []
            train_emb = []
            test_emb = []

            # Get original train/test split cluster IDs
            np.random.seed(42)
            all_indices = np.arange(len(cluster_ids))
            np.random.shuffle(all_indices)
            split_idx = int(0.8 * len(all_indices))
            train_idx_set = set(all_indices[:split_idx])
            test_idx_set = set(all_indices[split_idx:])

            for i, cid in enumerate(cluster_ids):
                if cid in embeddings:
                    if i in train_idx_set:
                        train_indices.append(i)
                        train_emb.append(embeddings[cid])
                    elif i in test_idx_set:
                        test_indices.append(i)
                        test_emb.append(embeddings[cid])

            if len(train_emb) > 50 and len(test_emb) > 10:
                train_emb = np.array(train_emb)
                test_emb = np.array(test_emb)
                y_train_text = y[train_indices]
                y_test_text = y[test_indices]

                # BERTweet only
                logger.info("  Running BERTweet (text-only)...")
                xgb_text = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
                xgb_text.fit(train_emb, y_train_text)
                y_pred = xgb_text.predict(test_emb)
                metrics = evaluate(y_test_text, y_pred)
                y_pred_cv = cross_val_predict(xgb_text, train_emb, y_train_text, cv=5)
                cv_rho, _ = stats.spearmanr(y_train_text, y_pred_cv)
                result = BaselineResult(name="BERTweet", cv_rho=cv_rho, **metrics, notes="Text-only")
                log_result(result)
                results.append(result)

                # BERTweet + Features
                logger.info("  Running BERTweet + Features...")
                X_train_features = X[train_indices]
                X_test_features = X[test_indices]
                X_train_combined = np.hstack([train_emb, X_train_features])
                X_test_combined = np.hstack([test_emb, X_test_features])

                xgb_combined = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
                xgb_combined.fit(X_train_combined, y_train_text)
                y_pred = xgb_combined.predict(X_test_combined)
                metrics = evaluate(y_test_text, y_pred)
                y_pred_cv = cross_val_predict(xgb_combined, X_train_combined, y_train_text, cv=5)
                cv_rho, _ = stats.spearmanr(y_train_text, y_pred_cv)
                result = BaselineResult(name="BERTweet + Features", cv_rho=cv_rho, **metrics, notes="Text + engineered features")
                log_result(result)
                results.append(result)
            else:
                logger.warning(f"  Insufficient text data: train={len(train_emb)}, test={len(test_emb)}")
        else:
            logger.warning(f"  Insufficient clusters with text: {len(cluster_texts)}")

    except ImportError as e:
        logger.warning(f"  Transformers not available: {e}")
    except Exception as e:
        logger.error(f"  Text baselines failed: {e}")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    results.sort(key=lambda r: r.spearman_rho, reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS (sorted by Spearman ρ)")
    logger.info("=" * 70)

    print("\n| Baseline | Test ρ | CV ρ | R² | MAE | F2(.65) | F2(.75) | F2(.85) |")
    print("|----------|--------|------|-----|-----|---------|---------|---------|")
    for r in results:
        r2_str = f"{r.r2:.3f}" if r.r2 > -10 else "--"
        print(f"| {r.name} | {r.spearman_rho:.3f} | {r.cv_rho:.3f} | {r2_str} | {r.mae:.3f} | {r.f2_065:.3f} | {r.f2_075:.3f} | {r.f2_085:.3f} |")

    # Print best params separately
    logger.info("\n=== BEST HYPERPARAMETERS ===")
    for r in results:
        if r.best_params:
            logger.info(f"  {r.name}: {r.best_params}")

    # Save results
    results_dict = [asdict(r) for r in results]
    with open(OUTPUT_DIR / "tuned_baselines.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    # CSV (without params for readability)
    import csv
    csv_rows = [
        {
            "name": r.name,
            "test_rho": r.spearman_rho,
            "cv_rho": r.cv_rho,
            "r2": r.r2,
            "mae": r.mae,
            "f2_065": r.f2_065,
            "f2_075": r.f2_075,
            "f2_085": r.f2_085,
        }
        for r in results
    ]
    with open(OUTPUT_DIR / "tuned_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")

    # Best model summary
    best = results[0]
    logger.info(f"\n🏆 Best Model: {best.name}")
    logger.info(f"   Test ρ = {best.spearman_rho:.3f}, CV ρ = {best.cv_rho:.3f}")
    if best.best_params:
        logger.info(f"   Best params: {best.best_params}")


if __name__ == "__main__":
    main()
