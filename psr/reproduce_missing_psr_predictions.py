from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import fbeta_score, mean_absolute_error, r2_score
from xgboost import XGBRegressor


ROOT = Path("/Users/sergiopinto/repro/filipe_psr_virality_handoff_2026-05-18")
DERIVED = ROOT / "derived_arrays"
RAW = Path("/Users/sergiopinto/explainableACD/data/pipeline_output/streaming_full/2026-01-17_03-56")
FEATURES = ROOT / "reproducibility/source_artifacts/virality/features_enhanced.parquet"
COMPLETE_FEATURES = ROOT / "explainableACD/experiments/results/virality_feature_selection/features_leakfree.parquet"
OUT = ROOT / "reproduced_missing_predictions"


@dataclass
class PredictionResult:
    name: str
    valid_indices: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, float]
    cluster_ids: np.ndarray | None = None


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, 1)
    rho, spearman_p = stats.spearmanr(y_true, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)

    def f2(tau: float) -> float:
        y_true_bin = (y_true >= tau).astype(int)
        y_pred_bin = (y_pred >= tau).astype(int)
        return float(fbeta_score(y_true_bin, y_pred_bin, beta=2, zero_division=0))

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
        "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else 1.0,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "f2_065": f2(0.65),
        "f2_075": f2(0.75),
        "f2_085": f2(0.85),
    }


def save_result(result: PredictionResult, test_cluster_ids: np.ndarray) -> None:
    target_dir = OUT / result.name
    target_dir.mkdir(parents=True, exist_ok=True)
    result_cluster_ids = result.cluster_ids if result.cluster_ids is not None else test_cluster_ids
    full = np.full(len(result_cluster_ids), np.nan, dtype=np.float64)
    full[result.valid_indices] = np.clip(result.predictions, 0, 1)
    mask = np.zeros(len(result_cluster_ids), dtype=bool)
    mask[result.valid_indices] = True

    np.save(target_dir / "test_predictions.npy", full)
    np.save(target_dir / "test_predictions_eval_subset.npy", np.clip(result.predictions, 0, 1))
    np.save(target_dir / "valid_mask.npy", mask)
    np.save(target_dir / "valid_indices.npy", result.valid_indices.astype(np.int64))
    np.save(target_dir / "test_cluster_ids.npy", result_cluster_ids.astype(np.int64))
    np.save(target_dir / "valid_cluster_ids.npy", result_cluster_ids[result.valid_indices].astype(np.int64))
    np.save(target_dir / "test_targets_eval_subset.npy", result.targets)
    (target_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")


def load_sklearn_split(feature_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(feature_path, columns=["cluster_id", "psr"])
    cluster_ids = df["cluster_id"].to_numpy(dtype=np.int64)
    y = df["psr"].to_numpy(dtype=np.float64)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    return cluster_ids[train_idx], cluster_ids[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h_n[-1])).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return torch.sigmoid(self.fc(h_n[-1])).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def train_sequence_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int = 50,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test).float().to(device)).cpu().numpy()


def build_detection_times(ts_df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    triggered = ts_df[ts_df["is_anomaly_trigger"]]
    grouped = triggered.groupby("cluster_id")["timestamp"].min()
    return {int(k): v for k, v in grouped.items()}


def build_sequences(
    ts_df: pd.DataFrame,
    detection_times: dict[int, pd.Timestamp],
    cluster_ids: np.ndarray,
    targets: np.ndarray,
    max_len: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    y_values = []
    valid_indices = []

    for i, cid_raw in enumerate(cluster_ids):
        cid = int(cid_raw)
        if cid not in detection_times:
            continue
        cluster_ts = ts_df[(ts_df["cluster_id"] == cid) & (ts_df["timestamp"] <= detection_times[cid])].sort_values(
            "timestamp"
        )
        if len(cluster_ts) < 3:
            continue
        seq = cluster_ts[["tweet_count", "engagement", "unique_users"]].to_numpy(dtype=np.float32)
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
        if len(seq) > max_len:
            seq = seq[-max_len:]
        elif len(seq) < max_len:
            seq = np.vstack([np.zeros((max_len - len(seq), seq.shape[1]), dtype=np.float32), seq])
        sequences.append(seq)
        y_values.append(targets[i])
        valid_indices.append(i)

    return np.stack(sequences), np.asarray(y_values, dtype=np.float32), np.asarray(valid_indices, dtype=np.int64)


def run_sequences(ts_df: pd.DataFrame, manifest: dict[str, object]) -> list[PredictionResult]:
    train_cluster_ids, test_cluster_ids, y_train, y_test, _, _ = load_sklearn_split(COMPLETE_FEATURES)
    detection_times = build_detection_times(ts_df)

    x_train, y_train_seq, _ = build_sequences(ts_df, detection_times, train_cluster_ids, y_train)
    x_test, y_test_seq, test_valid_indices = build_sequences(ts_df, detection_times, test_cluster_ids, y_test)
    manifest["sequence_counts"] = {
        "train": int(len(x_train)),
        "test": int(len(x_test)),
    }

    results = []
    for seed in [42]:
        set_seed(seed)
        lstm_pred = train_sequence_model(LSTMModel(), x_train, y_train_seq, x_test, seed)
        set_seed(seed)
        gru_pred = train_sequence_model(GRUModel(), x_train, y_train_seq, x_test, seed)
        results.append(
            PredictionResult(
                "LSTM",
                test_valid_indices,
                lstm_pred,
                y_test_seq,
                evaluate(y_test_seq, lstm_pred),
                cluster_ids=test_cluster_ids,
            )
        )
        results.append(
            PredictionResult(
                "GRU",
                test_valid_indices,
                gru_pred,
                y_test_seq,
                evaluate(y_test_seq, gru_pred),
                cluster_ids=test_cluster_ids,
            )
        )
    return results


def load_hawkes_baseline():
    hawkes_path = Path("/Users/sergiopinto/explainableACD/src/virality/hawkes.py")
    spec = importlib.util.spec_from_file_location("local_hawkes", hawkes_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load hawkes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_hawkes"] = module
    spec.loader.exec_module(module)
    return module.HawkesBaseline, module.HawkesConfig


def build_arrivals(
    tweets_df: pd.DataFrame,
    detection_times: dict[int, pd.Timestamp],
    cluster_ids: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float], list[int]]:
    arrivals_by_cluster = {}
    offsets = {}
    psr_by_cluster = {}
    valid_indices = []
    for i, cid_raw in enumerate(cluster_ids):
        cid = int(cid_raw)
        if cid not in detection_times:
            continue
        cluster_tweets = tweets_df[
            (tweets_df["cluster_id"] == cid) & (tweets_df["created_at"] <= detection_times[cid])
        ].sort_values("created_at")
        if len(cluster_tweets) < 5:
            continue
        timestamps = cluster_tweets["created_at"].to_list()
        t0 = timestamps[0]
        arrivals = np.array([(t - t0).total_seconds() for t in timestamps], dtype=np.float64)
        arrivals_by_cluster[cid] = arrivals
        offsets[cid] = float(arrivals[-1])
        psr_by_cluster[cid] = float(targets[i])
        valid_indices.append(i)
    return arrivals_by_cluster, offsets, psr_by_cluster, valid_indices


def run_hawkes(ts_df: pd.DataFrame, tweets_df: pd.DataFrame, manifest: dict[str, object]) -> PredictionResult:
    train_cluster_ids, test_cluster_ids, y_train, y_test, _, _ = load_sklearn_split(COMPLETE_FEATURES)
    detection_times = build_detection_times(ts_df)
    train_arrivals, train_offsets, train_psr, _ = build_arrivals(tweets_df, detection_times, train_cluster_ids, y_train)
    test_arrivals, test_offsets, _, valid_indices = build_arrivals(tweets_df, detection_times, test_cluster_ids, y_test)
    HawkesBaseline, HawkesConfig = load_hawkes_baseline()
    hawkes = HawkesBaseline(HawkesConfig())
    valid_psr = [v for v in train_psr.values() if not np.isnan(v)]
    hawkes._fallback_psr = float(np.median(valid_psr)) if valid_psr else 0.5
    pred = hawkes.predict(test_arrivals, test_offsets)
    y_true = np.asarray([y_test[i] for i in valid_indices], dtype=np.float32)
    manifest["hawkes_counts"] = {
        "train": len(train_arrivals),
        "test": len(test_arrivals),
    }
    return PredictionResult(
        "HIP_Hawkes",
        np.asarray(valid_indices, dtype=np.int64),
        pred,
        y_true,
        evaluate(y_true, pred),
        cluster_ids=test_cluster_ids,
    )


def run_bertweet(tweets_df: pd.DataFrame, manifest: dict[str, object]) -> list[PredictionResult]:
    from transformers import AutoModel, AutoTokenizer

    feature_df = pd.read_parquet(FEATURES)
    feature_cols = [c for c in feature_df.columns if c not in {"cluster_id", "psr"}]
    x_all = feature_df[feature_cols].to_numpy(dtype=np.float64)
    x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = feature_df["psr"].to_numpy(dtype=np.float64)
    cluster_ids = feature_df["cluster_id"].to_numpy()

    cluster_texts = {}
    grouped = tweets_df.groupby("cluster_id", sort=False)
    for cid in cluster_ids:
        if cid not in grouped.groups:
            continue
        row = tweets_df.iloc[grouped.groups[cid][0]]
        text = row.get("text")
        if isinstance(text, str) and text:
            cluster_texts[int(cid)] = text[:512]

    np.random.seed(42)
    all_indices = np.arange(len(cluster_ids))
    np.random.shuffle(all_indices)
    split_idx = int(0.8 * len(all_indices))
    train_idx_set = set(all_indices[:split_idx])
    test_idx_set = set(all_indices[split_idx:])
    train_indices = []
    test_indices = []
    train_texts = []
    test_texts = []
    for i, cid in enumerate(cluster_ids):
        cid_int = int(cid)
        if cid_int not in cluster_texts:
            continue
        if i in train_idx_set:
            train_indices.append(i)
            train_texts.append(cluster_texts[cid_int])
        elif i in test_idx_set:
            test_indices.append(i)
            test_texts.append(cluster_texts[cid_int])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
    model.eval()

    def embed(texts: list[str]) -> np.ndarray:
        embeddings = []
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
                print(f"BERTweet embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.vstack(embeddings).astype(np.float32)

    train_emb = embed(train_texts)
    test_emb = embed(test_texts)
    y_train_text = y_all[train_indices]
    y_test_text = y_all[test_indices]

    xgb_text = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    xgb_text.fit(train_emb, y_train_text)
    pred_text = xgb_text.predict(test_emb)

    x_train_features = x_all[train_indices]
    x_test_features = x_all[test_indices]
    xgb_combined = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    xgb_combined.fit(np.hstack([train_emb, x_train_features]), y_train_text)
    pred_combined = xgb_combined.predict(np.hstack([test_emb, x_test_features]))

    bertweet_test_cluster_ids = cluster_ids[test_indices].astype(np.int64)
    valid_positions = np.arange(len(test_indices), dtype=np.int64)
    manifest["bertweet_counts"] = {
        "clusters_with_text": len(cluster_texts),
        "train": len(train_texts),
        "test": len(test_texts),
    }
    return [
        PredictionResult(
            "BERTweet",
            valid_positions,
            pred_text,
            y_test_text,
            evaluate(y_test_text, pred_text),
            cluster_ids=bertweet_test_cluster_ids,
        ),
        PredictionResult(
            "BERTweet_plus_features",
            valid_positions,
            pred_combined,
            y_test_text,
            evaluate(y_test_text, pred_combined),
            cluster_ids=bertweet_test_cluster_ids,
        ),
    ]


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "method": "Regenerated missing PSR baseline predictions from local raw virality artifacts.",
        "raw_data": str(RAW),
        "derived_arrays": str(DERIVED),
        "complete_feature_source_for_lstm_gru_hawkes": str(COMPLETE_FEATURES),
        "tuned_feature_source_for_bertweet": str(FEATURES),
        "note": "test_predictions.npy is length 106 for each baseline. It contains NaN where the original script did not evaluate a cluster. Use valid_mask.npy/test_predictions_eval_subset.npy for metric replication.",
    }
    _, test_cluster_ids, _, _, _, _ = load_sklearn_split(COMPLETE_FEATURES)
    print("Loading raw parquet files...", flush=True)
    ts_df = pd.read_parquet(
        RAW / "cluster_timeseries.parquet",
        columns=["cluster_id", "timestamp", "tweet_count", "engagement", "unique_users", "is_anomaly_trigger"],
    )
    tweets_df = pd.read_parquet(RAW / "tweets.parquet", columns=["cluster_id", "created_at", "text"])

    manifest["metrics"] = {}

    def record(result: PredictionResult) -> None:
        save_result(result, test_cluster_ids)
        manifest["metrics"][result.name] = {
            "valid_n": int(len(result.valid_indices)),
            **result.metrics,
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({result.name: manifest["metrics"][result.name]}, indent=2), flush=True)

    print("Running sequence baselines...", flush=True)
    for result in run_sequences(ts_df, manifest):
        record(result)

    print("Running HIP Hawkes baseline...", flush=True)
    record(run_hawkes(ts_df, tweets_df, manifest))

    print("Running BERTweet baselines...", flush=True)
    for result in run_bertweet(tweets_df, manifest):
        record(result)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
