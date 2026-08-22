#!/usr/bin/env python3
# Artifact change — Sérgio Pinto, 2026-08-21 19:32 PDT.
# Reason: make the three paper-facing four-head F1 cells independently executable.
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.2.5",
#   "polars==1.27.1",
#   "protobuf==6.33.4",
#   "scikit-learn==1.6.1",
#   "sentencepiece==0.2.1",
#   "torch==2.7.0",
#   "transformers==4.51.3",
# ]
# ///
"""Reproduce the CIKM four-head check-worthiness table row.

Without arguments, this recomputes all three F1 values from the committed
probability arrays. With ``--checkpoint``, it first performs fresh inference on
ClaimBuster and CT23 using the release checkpoint and packaged public inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoConfig, AutoModel, AutoTokenizer


CHECKPOINT_SHA256 = "5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081"
CHECKPOINT_SIZE = 1_739_380_133
PAPER_F1 = {"CT24": 0.833, "ClaimBuster": 0.973, "CT23": 0.933}


class MultiTaskDeBERTa(nn.Module):
    """DeBERTa encoder with the four heads used by the retained run."""

    def __init__(self, config: Any, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))
        self.aux_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )
                for name in ("checkability", "verifiability", "harm")
            }
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.main_head(outputs.last_hidden_state[:, 0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probs >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    candidates = [evaluate(labels, probs, float(value)) for value in np.arange(0.30, 0.80, 0.05)]
    return max(candidates, key=lambda row: row["f1"])


def load_benchmark(root: Path, name: str) -> tuple[list[str], np.ndarray]:
    benchmark_root = root / "reproducibility/source_artifacts/checkworthiness/benchmarks"
    if name == "ClaimBuster":
        frame = pl.read_csv(benchmark_root / "claim_buster/groundtruth.csv")
        labels = (frame["Verdict"].to_numpy() == 1).astype(np.int64)
    else:
        frame = pl.read_csv(
            benchmark_root / "ct23/CT23_1B_checkworthy_english_test_gold.tsv",
            separator="\t",
        )
        labels = (frame["class_label"].to_numpy() == "Yes").astype(np.int64)
    return frame["Text"].to_list(), labels


def infer(
    model: MultiTaskDeBERTa,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            logits = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def load_checkpoint_model(root: Path, checkpoint_path: Path, device: torch.device):
    if checkpoint_path.stat().st_size != CHECKPOINT_SIZE or sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint size or SHA-256 does not match the published release asset")
    base_root = root / "reproducibility/source_artifacts/checkworthiness/deberta_v3_large_base"
    config = AutoConfig.from_pretrained(base_root, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_root, local_files_only=True, use_fast=False)
    model = MultiTaskDeBERTa(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    run_root = root / "reproducibility/runs/deberta_mtl_cikm_20260512_134553"
    prediction_root = run_root / "benchmark_predictions"
    arrays = {
        "CT24": (np.load(run_root / "test_labels.npy"), np.load(run_root / "test_probs.npy")),
        "ClaimBuster": (
            np.load(prediction_root / "claimbuster_labels.npy"),
            np.load(prediction_root / "claimbuster_probs.npy"),
        ),
        "CT23": (
            np.load(prediction_root / "ct23_labels.npy"),
            np.load(prediction_root / "ct23_probs.npy"),
        ),
    }

    inference = "committed prediction arrays"
    if args.checkpoint:
        if args.device == "auto":
            selected = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            selected = args.device
        device = torch.device(selected)
        model, tokenizer = load_checkpoint_model(root, args.checkpoint.resolve(), device)
        for name in ("ClaimBuster", "CT23"):
            texts, labels = load_benchmark(root, name)
            fresh_probs = infer(model, tokenizer, texts, device, args.batch_size)
            stored_labels, stored_probs = arrays[name]
            if not np.array_equal(labels, stored_labels):
                raise ValueError(f"{name} label order differs from the committed evidence")
            max_delta = float(np.max(np.abs(fresh_probs - stored_probs)))
            if max_delta > 1e-5:
                raise ValueError(f"{name} fresh probabilities differ by {max_delta:.3e}")
            arrays[name] = (labels, fresh_probs)
        inference = f"fresh checkpoint inference on {device}"

    results = {
        "CT24": evaluate(*arrays["CT24"], threshold=0.50),
        "ClaimBuster": best_threshold(*arrays["ClaimBuster"]),
        "CT23": best_threshold(*arrays["CT23"]),
    }
    passed = True
    for name, result in results.items():
        rounded = round(result["f1"], 3)
        row_passed = rounded == PAPER_F1[name]
        passed = passed and row_passed
        print(
            f"{name}: F1={result['f1']:.12f} @ {result['threshold']:.2f}; "
            f"paper={PAPER_F1[name]:.3f}; {'PASS' if row_passed else 'FAIL'}"
        )
    summary = {
        "change_note": (
            "Sérgio Pinto, 2026-08-21 19:38 PDT — Recomputed all three "
            "four-head F1 cells from checksum-bound prediction arrays."
        ),
        "status": "PASS" if passed else "FAIL",
        "inference": inference,
        "paper_f1": PAPER_F1,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT: {summary['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
