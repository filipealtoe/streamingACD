#!/usr/bin/env python3
"""
Run ALL virality prediction baselines for IJCAI paper.

Baselines:
1. Random (trivial lower bound)
2. Ridge (linear regression)
3. RandomForest (optimized)
4. XGBoost (optimized)
5. XGBClassifier (at multiple thresholds)
6. LSTM (sequence model)
7. GRU (sequence model)
8. Hawkes/HIP (point process)
9. BERTweet (text-only)
10. BERTweet + Features (ViralBERT-style)
11. SentenceBERT (text-only)

Usage:
    python experiments/scripts/run_complete_baselines.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import fbeta_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEATURES_PATH = PROJECT_ROOT / "experiments" / "results" / "virality_feature_selection" / "features_leakfree.parquet"
DATA_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "virality_complete"


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class BaselineResult:
    name: str
    spearman_rho: float
    r2: float
    mae: float
    f2_065: float
    f2_075: float
    f2_085: float
    notes: str = ""


def evaluate(y_true: NDArray, y_pred: NDArray) -> dict:
    """Compute all metrics."""
    # Clip predictions to [0, 1]
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


def log_metrics(name: str, metrics: dict, indent: int = 2) -> None:
    """Log all metrics in a compact format."""
    prefix = " " * indent
    logger.info(
        f"{prefix}{name}: ρ={metrics['spearman_rho']:.3f}, R²={metrics['r2']:.3f}, "
        f"MAE={metrics['mae']:.3f}, F2(.65)={metrics['f2_065']:.3f}, "
        f"F2(.75)={metrics['f2_075']:.3f}, F2(.85)={metrics['f2_085']:.3f}"
    )


# =============================================================================
# FEATURE-BASED BASELINES
# =============================================================================

def run_feature_baselines(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
) -> list[BaselineResult]:
    """Run Ridge, RandomForest, XGBoost baselines."""
    results = []

    # Random baseline
    logger.info("Running Random baseline...")
    y_pred_random = np.random.uniform(0, 1, len(y_test))
    metrics = evaluate(y_test, y_pred_random)
    results.append(BaselineResult(name="Random", **metrics, notes="Trivial baseline"))

    # Ridge
    logger.info("Running Ridge baseline...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    results.append(BaselineResult(name="Ridge", **metrics))
    log_metrics("Ridge", metrics)

    # RandomForest (optimized params from exhaustive search)
    logger.info("Running RandomForest baseline...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    results.append(BaselineResult(name="RandomForest", **metrics, notes="Optimized hyperparams"))
    log_metrics("RandomForest", metrics)

    # XGBoost
    logger.info("Running XGBoost baseline...")
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    results.append(BaselineResult(name="XGBoost", **metrics))
    log_metrics("XGBoost", metrics)

    # XGBClassifier at multiple thresholds
    for tau in [0.65, 0.75, 0.85]:
        logger.info(f"Running XGBClassifier(τ={tau})...")
        y_train_bin = (y_train >= tau).astype(int)

        if y_train_bin.sum() < 5 or (len(y_train_bin) - y_train_bin.sum()) < 5:
            logger.warning(f"  Skipping τ={tau}: insufficient samples")
            continue

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train_bin)
        y_pred = clf.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred)
        results.append(BaselineResult(name=f"XGBClassifier(τ={tau})", **metrics))
        log_metrics(f"XGBClassifier(τ={tau})", metrics)

    return results


# =============================================================================
# SEQUENCE BASELINES (LSTM/GRU)
# =============================================================================

def run_sequence_baselines(
    timeseries_path: Path,
    features_df: pl.DataFrame,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray,
    y_test: NDArray,
) -> list[BaselineResult]:
    """Run LSTM and GRU baselines on cluster time series."""
    results = []

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not available, skipping sequence baselines")
        return results

    logger.info("Loading timeseries for sequence models...")
    ts_df = pl.read_parquet(timeseries_path)

    # Get detection times
    detection_times = {}
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    for row in triggered.group_by("cluster_id").agg(
        pl.col("timestamp").min().alias("detection_time")
    ).iter_rows(named=True):
        detection_times[row["cluster_id"]] = row["detection_time"]

    def get_sequence(cluster_id: int, max_len: int = 24) -> NDArray | None:
        """Extract sequence features for a cluster up to detection."""
        if cluster_id not in detection_times:
            return None

        det_time = detection_times[cluster_id]
        cluster_ts = ts_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("timestamp") <= det_time)
        ).sort("timestamp")

        if len(cluster_ts) < 3:
            return None

        # Features: tweet_count, engagement, unique_users (normalized)
        seq = cluster_ts.select([
            "tweet_count", "engagement", "unique_users"
        ]).to_numpy().astype(np.float32)

        # Normalize
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)

        # Pad/truncate to max_len
        if len(seq) > max_len:
            seq = seq[-max_len:]
        elif len(seq) < max_len:
            padding = np.zeros((max_len - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([padding, seq])

        return seq

    # Build sequences
    logger.info("Building sequences...")
    max_len = 24

    X_train_seq, y_train_seq = [], []
    for i, cid in enumerate(train_clusters):
        seq = get_sequence(cid, max_len)
        if seq is not None:
            X_train_seq.append(seq)
            y_train_seq.append(y_train[i])

    X_test_seq, y_test_seq = [], []
    for i, cid in enumerate(test_clusters):
        seq = get_sequence(cid, max_len)
        if seq is not None:
            X_test_seq.append(seq)
            y_test_seq.append(y_test[i])

    if len(X_train_seq) < 50 or len(X_test_seq) < 10:
        logger.warning(f"Insufficient sequences: train={len(X_train_seq)}, test={len(X_test_seq)}")
        return results

    X_train_seq = np.stack(X_train_seq)
    y_train_seq = np.array(y_train_seq, dtype=np.float32)
    X_test_seq = np.stack(X_test_seq)
    y_test_seq = np.array(y_test_seq, dtype=np.float32)

    logger.info(f"Sequences: train={len(X_train_seq)}, test={len(X_test_seq)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define models
    class LSTMModel(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return torch.sigmoid(self.fc(h_n[-1])).squeeze(-1)

    class GRUModel(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, h_n = self.gru(x)
            return torch.sigmoid(self.fc(h_n[-1])).squeeze(-1)

    def train_model(model, X_train, y_train, X_test, epochs=50):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(device)
            predictions = model(X_test_t).cpu().numpy()

        return predictions

    # LSTM
    logger.info("Training LSTM...")
    lstm = LSTMModel()
    y_pred_lstm = train_model(lstm, X_train_seq, y_train_seq, X_test_seq)
    metrics = evaluate(y_test_seq, y_pred_lstm)
    results.append(BaselineResult(name="LSTM", **metrics, notes="2-layer, hidden=64"))
    log_metrics("LSTM", metrics)

    # GRU
    logger.info("Training GRU...")
    gru = GRUModel()
    y_pred_gru = train_model(gru, X_train_seq, y_train_seq, X_test_seq)
    metrics = evaluate(y_test_seq, y_pred_gru)
    results.append(BaselineResult(name="GRU", **metrics, notes="2-layer, hidden=64"))
    log_metrics("GRU", metrics)

    return results


# =============================================================================
# HAWKES BASELINE
# =============================================================================

def run_hawkes_baseline(
    tweets_path: Path,
    timeseries_path: Path,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray,
    y_test: NDArray,
) -> BaselineResult | None:
    """Run Hawkes/HIP baseline."""
    logger.info("Running Hawkes (HIP) baseline...")

    from src.virality.hawkes import HawkesBaseline, HawkesConfig

    # Load data
    tweets_df = pl.read_parquet(tweets_path)
    ts_df = pl.read_parquet(timeseries_path)

    # Get detection times
    detection_times = {}
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    for row in triggered.group_by("cluster_id").agg(
        pl.col("timestamp").min().alias("detection_time")
    ).iter_rows(named=True):
        detection_times[row["cluster_id"]] = row["detection_time"]

    def get_arrivals(cluster_id: int) -> NDArray | None:
        """Get tweet arrival times for a cluster up to detection."""
        if cluster_id not in detection_times:
            return None

        det_time = detection_times[cluster_id]
        cluster_tweets = tweets_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("created_at") <= det_time)
        ).sort("created_at")

        if len(cluster_tweets) < 5:
            return None

        timestamps = cluster_tweets["created_at"].to_list()
        t0 = timestamps[0]
        arrivals = np.array([(t - t0).total_seconds() for t in timestamps], dtype=np.float64)

        return arrivals

    # Build arrival data
    logger.info("  Building arrival data...")
    train_arrivals = {}
    train_offsets = {}
    train_psr = {}

    for i, cid in enumerate(train_clusters):
        arrivals = get_arrivals(cid)
        if arrivals is not None and len(arrivals) >= 5:
            train_arrivals[cid] = arrivals
            train_offsets[cid] = arrivals[-1]  # Detection offset is last arrival
            train_psr[cid] = float(y_train[i])

    test_arrivals = {}
    test_offsets = {}
    test_indices = []  # Map back to y_test

    for i, cid in enumerate(test_clusters):
        arrivals = get_arrivals(cid)
        if arrivals is not None and len(arrivals) >= 5:
            test_arrivals[cid] = arrivals
            test_offsets[cid] = arrivals[-1]
            test_indices.append(i)

    logger.info(f"  Train clusters with arrivals: {len(train_arrivals)}")
    logger.info(f"  Test clusters with arrivals: {len(test_arrivals)}")

    if len(train_arrivals) < 20 or len(test_arrivals) < 10:
        logger.warning("  Insufficient arrival data for Hawkes")
        return None

    # Fit and predict
    hawkes = HawkesBaseline(HawkesConfig())
    hawkes.fit(train_arrivals, train_offsets, train_psr)

    y_pred = hawkes.predict(test_arrivals, test_offsets)
    y_true = np.array([y_test[i] for i in test_indices], dtype=np.float32)

    metrics = evaluate(y_true, y_pred)
    log_metrics("Hawkes", metrics)

    return BaselineResult(name="Hawkes (HIP)", **metrics, notes="Custom EM implementation")


# =============================================================================
# TRANSFORMER HAWKES BASELINE
# =============================================================================

def run_transformer_hawkes_baseline(
    tweets_path: Path,
    timeseries_path: Path,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray,
    y_test: NDArray,
) -> BaselineResult | None:
    """Run Transformer Hawkes Process baseline (Zuo et al., ICML 2020)."""
    logger.info("Running Transformer Hawkes baseline...")

    try:
        from src.virality.transformer_hawkes import TransformerHawkesBaseline, THPConfig
    except ImportError as e:
        logger.warning(f"Transformer Hawkes not available: {e}")
        return None

    # Load data
    tweets_df = pl.read_parquet(tweets_path)
    ts_df = pl.read_parquet(timeseries_path)

    # Get detection times
    detection_times = {}
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    for row in triggered.group_by("cluster_id").agg(
        pl.col("timestamp").min().alias("detection_time")
    ).iter_rows(named=True):
        detection_times[row["cluster_id"]] = row["detection_time"]

    def get_arrivals(cluster_id: int) -> NDArray | None:
        """Get tweet arrival times for a cluster up to detection."""
        if cluster_id not in detection_times:
            return None

        det_time = detection_times[cluster_id]
        cluster_tweets = tweets_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("created_at") <= det_time)
        ).sort("created_at")

        if len(cluster_tweets) < 5:
            return None

        timestamps = cluster_tweets["created_at"].to_list()
        t0 = timestamps[0]
        arrivals = np.array([(t - t0).total_seconds() for t in timestamps], dtype=np.float64)

        return arrivals

    # Build arrival data
    logger.info("  Building arrival data for Transformer Hawkes...")
    train_arrivals = {}
    train_psr = {}

    for i, cid in enumerate(train_clusters):
        arrivals = get_arrivals(cid)
        if arrivals is not None and len(arrivals) >= 5:
            train_arrivals[cid] = arrivals
            train_psr[cid] = float(y_train[i])

    test_arrivals = {}
    test_indices = []

    for i, cid in enumerate(test_clusters):
        arrivals = get_arrivals(cid)
        if arrivals is not None and len(arrivals) >= 5:
            test_arrivals[cid] = arrivals
            test_indices.append(i)

    logger.info(f"  Train clusters: {len(train_arrivals)}, Test clusters: {len(test_arrivals)}")

    if len(train_arrivals) < 50 or len(test_arrivals) < 10:
        logger.warning("  Insufficient arrival data for Transformer Hawkes")
        return None

    # Split some training for validation
    train_cids = list(train_arrivals.keys())
    np.random.seed(42)
    np.random.shuffle(train_cids)
    val_size = len(train_cids) // 5
    val_cids = train_cids[:val_size]
    train_cids = train_cids[val_size:]

    val_arrivals = {k: train_arrivals[k] for k in val_cids}
    val_psr = {k: train_psr[k] for k in val_cids}
    train_arrivals_final = {k: train_arrivals[k] for k in train_cids}
    train_psr_final = {k: train_psr[k] for k in train_cids}

    # Train
    config = THPConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        epochs=50,
        batch_size=32,
        patience=10,
    )
    thp = TransformerHawkesBaseline(config)
    thp.fit(train_arrivals_final, train_psr_final, val_arrivals, val_psr)

    # Predict
    y_pred = thp.predict(test_arrivals)
    y_true = np.array([y_test[i] for i in test_indices], dtype=np.float32)

    metrics = evaluate(y_true, y_pred)
    log_metrics("Transformer Hawkes", metrics)

    return BaselineResult(
        name="Transformer Hawkes",
        **metrics,
        notes="Zuo et al. ICML 2020"
    )


# =============================================================================
# TEXT BASELINES
# =============================================================================

def run_text_baselines(
    tweets_path: Path,
    timeseries_path: Path,
    features_df: pl.DataFrame,
    train_clusters: list[int],
    test_clusters: list[int],
    y_train: NDArray,
    y_test: NDArray,
    X_train_features: NDArray,
    X_test_features: NDArray,
) -> list[BaselineResult]:
    """Run BERTweet and SentenceBERT baselines."""
    results = []

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning("Transformers not available, skipping text baselines")
        return results

    logger.info("Loading tweets for text baselines...")
    tweets_df = pl.read_parquet(tweets_path)
    ts_df = pl.read_parquet(timeseries_path)

    # Get detection times
    detection_times = {}
    triggered = ts_df.filter(pl.col("is_anomaly_trigger"))
    for row in triggered.group_by("cluster_id").agg(
        pl.col("timestamp").min().alias("detection_time")
    ).iter_rows(named=True):
        detection_times[row["cluster_id"]] = row["detection_time"]

    def get_cluster_text(cluster_id: int) -> str | None:
        """Get representative text for a cluster (first 5 tweets concatenated)."""
        if cluster_id not in detection_times:
            return None

        det_time = detection_times[cluster_id]
        cluster_tweets = tweets_df.filter(
            (pl.col("cluster_id") == cluster_id) &
            (pl.col("created_at") <= det_time)
        ).sort("created_at").head(5)

        if len(cluster_tweets) == 0:
            return None

        texts = cluster_tweets["text"].to_list()
        return " ".join([t for t in texts if t])

    # Build text data
    logger.info("  Building text data...")
    train_texts, train_indices = [], []
    for i, cid in enumerate(train_clusters):
        text = get_cluster_text(cid)
        if text:
            train_texts.append(text)
            train_indices.append(i)

    test_texts, test_indices = [], []
    for i, cid in enumerate(test_clusters):
        text = get_cluster_text(cid)
        if text:
            test_texts.append(text)
            test_indices.append(i)

    logger.info(f"  Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    if len(train_texts) < 50 or len(test_texts) < 10:
        logger.warning("  Insufficient text data")
        return results

    y_train_text = np.array([y_train[i] for i in train_indices], dtype=np.float32)
    y_test_text = np.array([y_test[i] for i in test_indices], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def embed_texts(texts: list[str], model_name: str) -> NDArray:
        """Embed texts using BERTweet or SentenceBERT."""
        if "bertweet" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            embeddings = []
            batch_size = 16
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    outputs = model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_emb)

            return np.vstack(embeddings).astype(np.float32)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
            embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)
            return embeddings.astype(np.float32)

    # BERTweet
    logger.info("  Running BERTweet...")
    try:
        train_emb = embed_texts(train_texts, "vinai/bertweet-base")
        test_emb = embed_texts(test_texts, "vinai/bertweet-base")

        xgb = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(train_emb, y_train_text)
        y_pred = xgb.predict(test_emb)
        metrics = evaluate(y_test_text, y_pred)
        results.append(BaselineResult(name="BERTweet", **metrics, notes="Text-only"))
        log_metrics("BERTweet", metrics, indent=4)

        # BERTweet + Features (ViralBERT-style)
        logger.info("  Running BERTweet + Features...")
        X_train_features_subset = np.array([X_train_features[i] for i in train_indices])
        X_test_features_subset = np.array([X_test_features[i] for i in test_indices])

        X_train_combined = np.hstack([train_emb, X_train_features_subset])
        X_test_combined = np.hstack([test_emb, X_test_features_subset])

        xgb2 = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0)
        xgb2.fit(X_train_combined, y_train_text)
        y_pred = xgb2.predict(X_test_combined)
        metrics = evaluate(y_test_text, y_pred)
        results.append(BaselineResult(name="BERTweet + Features", **metrics, notes="ViralBERT-style"))
        log_metrics("BERTweet + Features", metrics, indent=4)

    except Exception as e:
        logger.error(f"  BERTweet failed: {e}")

    # SentenceBERT
    logger.info("  Running SentenceBERT...")
    try:
        train_emb = embed_texts(train_texts, "paraphrase-multilingual-mpnet-base-v2")
        test_emb = embed_texts(test_texts, "paraphrase-multilingual-mpnet-base-v2")

        xgb = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(train_emb, y_train_text)
        y_pred = xgb.predict(test_emb)
        metrics = evaluate(y_test_text, y_pred)
        results.append(BaselineResult(name="SentenceBERT", **metrics, notes="Text-only"))
        log_metrics("SentenceBERT", metrics, indent=4)

    except Exception as e:
        logger.error(f"  SentenceBERT failed: {e}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("COMPLETE VIRALITY BASELINES FOR IJCAI")
    logger.info("=" * 70)

    # Load features
    logger.info(f"Loading features from {FEATURES_PATH}")
    features_df = pl.read_parquet(FEATURES_PATH)

    feature_cols = [c for c in features_df.columns if c not in ["cluster_id", "psr"]]
    X = features_df.select(feature_cols).to_numpy().copy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = features_df["psr"].to_numpy().copy().astype(np.float32)
    cluster_ids = features_df["cluster_id"].to_list()

    logger.info(f"Samples: {len(X)}, Features: {len(feature_cols)}")

    # Split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_clusters = [cluster_ids[i] for i in train_idx]
    test_clusters = [cluster_ids[i] for i in test_idx]

    all_results: list[BaselineResult] = []

    # Feature-based baselines
    logger.info("\n" + "=" * 50)
    logger.info("FEATURE-BASED BASELINES")
    logger.info("=" * 50)
    feature_results = run_feature_baselines(X_train, X_test, y_train, y_test)
    all_results.extend(feature_results)

    # Sequence baselines
    logger.info("\n" + "=" * 50)
    logger.info("SEQUENCE BASELINES (LSTM/GRU)")
    logger.info("=" * 50)
    seq_results = run_sequence_baselines(
        DATA_DIR / "cluster_timeseries.parquet",
        features_df, train_clusters, test_clusters, y_train, y_test
    )
    all_results.extend(seq_results)

    # Hawkes baseline
    logger.info("\n" + "=" * 50)
    logger.info("HAWKES (HIP) BASELINE")
    logger.info("=" * 50)
    hawkes_result = run_hawkes_baseline(
        DATA_DIR / "tweets.parquet",
        DATA_DIR / "cluster_timeseries.parquet",
        train_clusters, test_clusters, y_train, y_test
    )
    if hawkes_result:
        all_results.append(hawkes_result)

    # Transformer Hawkes baseline
    logger.info("\n" + "=" * 50)
    logger.info("TRANSFORMER HAWKES BASELINE (Zuo et al. ICML 2020)")
    logger.info("=" * 50)
    thp_result = run_transformer_hawkes_baseline(
        DATA_DIR / "tweets.parquet",
        DATA_DIR / "cluster_timeseries.parquet",
        train_clusters, test_clusters, y_train, y_test
    )
    if thp_result:
        all_results.append(thp_result)

    # Text baselines
    logger.info("\n" + "=" * 50)
    logger.info("TEXT BASELINES")
    logger.info("=" * 50)
    text_results = run_text_baselines(
        DATA_DIR / "tweets.parquet",
        DATA_DIR / "cluster_timeseries.parquet",
        features_df, train_clusters, test_clusters,
        y_train, y_test, X_train, X_test
    )
    all_results.extend(text_results)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    # Sort by Spearman rho
    all_results.sort(key=lambda x: x.spearman_rho, reverse=True)

    print("\n| Baseline | Spearman ρ | R² | MAE | F2(τ=0.65) | F2(τ=0.75) | F2(τ=0.85) |")
    print("|----------|------------|-----|-----|------------|------------|------------|")
    for r in all_results:
        rho = f"{r.spearman_rho:.3f}" if r.spearman_rho > 0.01 else "--"
        r2 = f"{r.r2:.3f}" if r.r2 > -0.5 else "--"
        print(f"| {r.name} | {rho} | {r2} | {r.mae:.3f} | {r.f2_065:.3f} | {r.f2_075:.3f} | {r.f2_085:.3f} |")

    # Save results
    results_dict = [
        {
            "name": r.name,
            "spearman_rho": r.spearman_rho,
            "r2": r.r2,
            "mae": r.mae,
            "f2_065": r.f2_065,
            "f2_075": r.f2_075,
            "f2_085": r.f2_085,
            "notes": r.notes,
        }
        for r in all_results
    ]

    with open(OUTPUT_DIR / "complete_baselines.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save as CSV for LaTeX
    results_df = pl.DataFrame(results_dict)
    results_df.write_csv(OUTPUT_DIR / "complete_baselines.csv")

    logger.info(f"\nResults saved to {OUTPUT_DIR}")

    return all_results


if __name__ == "__main__":
    main()
