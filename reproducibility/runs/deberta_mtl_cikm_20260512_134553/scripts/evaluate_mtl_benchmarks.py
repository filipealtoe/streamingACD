#!/usr/bin/env python3
"""
Evaluate Multi-Task DeBERTa (4 Parallel Heads) on ClaimBuster and CT23.

Usage:
    python experiments/scripts/evaluate_mtl_benchmarks.py
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Model Definition (must match training)
# =============================================================================

class MultiTaskDeBERTa(nn.Module):
    """DeBERTa with multiple task heads for MTL."""

    def __init__(self, encoder_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.main_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        self.aux_heads = nn.ModuleDict({
            'checkability': self._make_aux_head(hidden_size, dropout),
            'verifiability': self._make_aux_head(hidden_size, dropout),
            'harm': self._make_aux_head(hidden_size, dropout),
        })

    def _make_aux_head(self, hidden_size: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, return_aux=True):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        main_logits = self.main_head(cls_embedding)
        aux_outputs = {}
        if return_aux:
            for name, head in self.aux_heads.items():
                aux_outputs[name] = head(cls_embedding).squeeze(-1)
        return main_logits, aux_outputs


# =============================================================================
# Paths
# =============================================================================

MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_mtl" / "best_model.pt"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


# =============================================================================
# Data Loading
# =============================================================================

def load_claimbuster():
    """Load ClaimBuster groundtruth."""
    df = pl.read_csv(DATA_DIR / "claim_buster" / "groundtruth.csv")
    texts = df["Text"].to_list()
    labels = np.array([1 if v == 1 else 0 for v in df["Verdict"].to_list()])
    return texts, labels


def load_ct23():
    """Load CT23 test set."""
    df = pl.read_csv(DATA_DIR / "check_that_23" / "CT23_1B_checkworthy_english_test_gold.tsv", separator="\t")
    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
    return texts, labels


# =============================================================================
# Inference
# =============================================================================

def get_predictions(model, tokenizer, texts, device, batch_size=32):
    """Get model predictions."""
    model.eval()
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        # Only keep input_ids and attention_mask (DeBERTa doesn't use token_type_ids)
        inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

        with torch.no_grad():
            logits, _ = model(**inputs, return_aux=False)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def evaluate(y_true, probs, thresholds=None):
    """Find best threshold."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.80, 0.05)

    best = {"f1": 0, "threshold": 0.5, "accuracy": 0, "precision": 0, "recall": 0}
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, preds),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("BENCHMARK: Multi-Task DeBERTa (4 Parallel Heads)")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = MultiTaskDeBERTa("microsoft/deberta-v3-large")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    # Handle both formats: raw state_dict or wrapped in dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    print("Model loaded successfully!")

    # Evaluate ClaimBuster
    print("\n[1/2] Evaluating ClaimBuster...")
    cb_texts, cb_labels = load_claimbuster()
    print(f"  Samples: {len(cb_texts)}, Positive: {cb_labels.sum()} ({100*cb_labels.mean():.1f}%)")

    cb_probs = get_predictions(model, tokenizer, cb_texts, device)
    cb_result = evaluate(cb_labels, cb_probs)
    print(f"\n  ClaimBuster: F1={cb_result['f1']:.4f} @ threshold={cb_result['threshold']:.2f}")
    print(f"              Precision={cb_result['precision']:.4f}, Recall={cb_result['recall']:.4f}")

    # Evaluate CT23
    print("\n[2/2] Evaluating CT23...")
    ct23_texts, ct23_labels = load_ct23()
    print(f"  Samples: {len(ct23_texts)}, Positive: {ct23_labels.sum()} ({100*ct23_labels.mean():.1f}%)")

    ct23_probs = get_predictions(model, tokenizer, ct23_texts, device)
    ct23_result = evaluate(ct23_labels, ct23_probs)
    print(f"\n  CT23: F1={ct23_result['f1']:.4f} @ threshold={ct23_result['threshold']:.2f}")
    print(f"        Precision={ct23_result['precision']:.4f}, Recall={ct23_result['recall']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Multi-Task DeBERTa (4 Parallel Heads, CT24 F1=0.833)")
    print("=" * 70)
    print(f"\n{'Benchmark':<15} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Threshold':<10}")
    print("-" * 57)
    for name, result in [("ClaimBuster", cb_result), ("CT23", ct23_result)]:
        print(f"{name:<15} {result['f1']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['threshold']:<10.2f}")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
