#!/usr/bin/env python3
"""
Benchmark PCA-64 + LLM + Text Features Model on ClaimBuster and CT23.

This script properly replicates the CT24 feature pipeline for benchmarks:
1. Extract 35 text features (using extract_text_features.py logic)
2. Generate PCA-64 embeddings (using CT24's fitted PCA)
3. Use LLM features from benchmark_llm_features/
4. Train LogisticRegression on CT24, evaluate on benchmarks

Usage:
    python experiments/scripts/benchmark_pca_llm_text.py
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CT24_CLASSIFIER_DIR = DATA_DIR / "processed" / "CT24_classifier"
CT24_LLM_DIR = DATA_DIR / "processed" / "CT24_llm_features_v4"
BENCHMARK_LLM_DIR = DATA_DIR / "processed" / "benchmark_llm_features"

# Raw benchmark data
CB_RAW = DATA_DIR / "raw" / "claim_buster" / "groundtruth.csv"
CT23_RAW = DATA_DIR / "raw" / "check_that_23" / "CT23_1B_checkworthy_english_test_gold.tsv"


# =============================================================================
# Text Feature Extraction (from extract_text_features.py)
# =============================================================================

SPELLED_NUMBERS = r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b"


def has_number(text: str) -> bool:
    return bool(re.search(r"\d+", text) or re.search(SPELLED_NUMBERS, text.lower()))


def has_precise_number(text: str) -> bool:
    patterns = [r"\d+\.\d+", r"\d+\s*%", r"\bpercent\b", r"\$\s*\d+", r"\d+\s*dollars?\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_large_scale(text: str) -> bool:
    return bool(re.search(r"\b(million|billion|trillion)s?\b", text.lower()))


def count_numbers(text: str) -> int:
    return len(re.findall(r"\d+", text)) + len(re.findall(SPELLED_NUMBERS, text.lower()))


def has_range(text: str) -> bool:
    patterns = [r"\bfrom\s+\d+\s+to\s+\d+", r"\bbetween\s+\d+\s+and\s+\d+", r"\d+\s*[-–—]\s*\d+", r"\d+\s+to\s+\d+"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_delta(text: str) -> bool:
    patterns = [r"\b(by|up|down)\s+\d+", r"\b(twice|half|double|triple|quadruple)d?\b", r"\b(doubled|tripled|halved)\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_specific_year(text: str) -> bool:
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def has_relative_time(text: str) -> bool:
    patterns = [r"\b(last|this|next)\s+(year|month|week|decade)\b", r"\byesterday\b",
                r"\b(in the past|over the past|in the last)\s+\d+", r"\b\d+\s*(years?|months?|weeks?|days?)\s+ago\b", r"\brecently\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_temporal_anchor(text: str) -> bool:
    patterns = [r"\bsince\s+(19|20)\d{2}\b", r"\bsince\s+(the|last|january|february|march|april|may|june|july|august|september|october|november|december)\b",
                r"\b(until|before|after)\s+(19|20)\d{2}\b", r"\b(before|after)\s+the\s+\w+\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_source_attribution(text: str) -> bool:
    patterns = [r"\baccording to\b", r"\breport\s+(says?|shows?|found|indicates?)\b",
                r"\bstudy\s+(says?|shows?|found|indicates?)\b", r"\bdata\s+(says?|shows?|found|indicates?)\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_evidence_noun(text: str) -> bool:
    return bool(re.search(r"\b(report|data|study|survey|statistics|records?|analysis|research|evidence|findings?)\b", text.lower()))


def has_official_source(text: str) -> bool:
    agencies = [r"\b(CBO|FBI|CDC|BLS|CIA|DOJ|EPA|IRS|NSA|DEA|ATF|DHS|FEMA)\b", r"\b(Congress|Senate|House|Pentagon|White House)\b",
                r"\b(Federal Reserve|Treasury|State Department)\b", r"\b(United Nations|UN|NATO|WHO|IMF|World Bank)\b"]
    return any(re.search(p, text) for p in agencies)


def has_said_claimed(text: str) -> bool:
    return bool(re.search(r"\b(said|says|claimed|claims|stated|states|reported|reports)\s+that\b", text.lower()))


def has_comparative(text: str) -> bool:
    return bool(re.search(r"\b(more|less|fewer|higher|lower|better|worse|greater|larger|smaller)\s+than\b", text.lower()))


def has_superlative(text: str) -> bool:
    return bool(re.search(r"\b(first|last|worst|best|highest|lowest|largest|smallest|biggest|greatest|most|least|record)\b", text.lower()))


def has_ranking(text: str) -> bool:
    return bool(re.search(r"\b(ranked|ranking|number one|number two|top\s+\d+|bottom\s+\d+|#\d+)\b", text.lower()))


def has_increase_decrease(text: str) -> bool:
    return bool(re.search(r"\b(increased?|decreased?|grew|grown|fell|fallen|rose|risen|dropped|gained|lost|surged|plunged|spiked|declined?)\b", text.lower()))


def has_voted(text: str) -> bool:
    return bool(re.search(r"\b(voted?|voting)\s+(for|against|to|on)\b", text.lower()))


def has_negation_claim(text: str) -> bool:
    patterns = [r"\b(did not|didn't|does not|doesn't|has not|hasn't|have not|haven't)\b",
                r"\b(never|no one|nobody|nothing|none of)\b", r"\b(failed to|refused to|denied)\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_first_person_stance(text: str) -> bool:
    return bool(re.search(r"\b(i think|i believe|i feel|in my opinion|in my view|my view is|i\'m convinced)\b", text.lower()))


def has_desire_intent(text: str) -> bool:
    return bool(re.search(r"\b(i want|i hope|i wish|we should|we must|we need to|we have to|should be|ought to)\b", text.lower()))


def has_future_modal(text: str) -> bool:
    return bool(re.search(r"\b(will be|will have|going to|gonna|expect to|plan to|intend to)\b", text.lower()))


def has_hedge(text: str) -> bool:
    return bool(re.search(r"\b(probably|likely|possibly|perhaps|maybe|seems|appears|might|could be|may be|sort of|kind of)\b", text.lower()))


def has_vague_quantifier(text: str) -> bool:
    return bool(re.search(r"\b(many people|some people|a lot of|lots of|everyone knows|people say|they say)\b", text.lower()))


def has_rhetorical_filler(text: str) -> bool:
    patterns = [r"\blet me\b", r"\blet\'s\b", r"\b(look|listen),", r"\bhere\'s the (thing|deal|point)\b", r"\byou know what\b"]
    return any(re.search(p, text.lower()) for p in patterns)


def has_fact_assertion(text: str) -> bool:
    return bool(re.search(r"\b(the fact is|the truth is|the reality is|fact of the matter|in fact)\b", text.lower()))


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


def has_transcript_artifact(text: str) -> bool:
    patterns = [r"\[.+\]", r"^[A-Z]+:", r"\b(APPLAUSE|LAUGHTER|CROSSTALK)\b"]
    return any(re.search(p, text) for p in patterns)


def compute_word_count(text: str) -> int:
    return len(text.split())


def compute_avg_word_length(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def compute_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def extract_text_features(text: str) -> dict:
    """Extract all 35 text features."""
    return {
        "feat_has_number": int(has_number(text)),
        "feat_has_precise_number": int(has_precise_number(text)),
        "feat_has_large_scale": int(has_large_scale(text)),
        "feat_number_count": count_numbers(text),
        "feat_has_range": int(has_range(text)),
        "feat_has_delta": int(has_delta(text)),
        "feat_has_specific_year": int(has_specific_year(text)),
        "feat_has_relative_time": int(has_relative_time(text)),
        "feat_has_temporal_anchor": int(has_temporal_anchor(text)),
        "feat_has_source_attribution": int(has_source_attribution(text)),
        "feat_has_evidence_noun": int(has_evidence_noun(text)),
        "feat_has_official_source": int(has_official_source(text)),
        "feat_has_said_claimed": int(has_said_claimed(text)),
        "feat_has_comparative": int(has_comparative(text)),
        "feat_has_superlative": int(has_superlative(text)),
        "feat_has_ranking": int(has_ranking(text)),
        "feat_has_increase_decrease": int(has_increase_decrease(text)),
        "feat_has_voted": int(has_voted(text)),
        "feat_has_negation_claim": int(has_negation_claim(text)),
        "feat_has_first_person_stance": int(has_first_person_stance(text)),
        "feat_has_desire_intent": int(has_desire_intent(text)),
        "feat_has_future_modal": int(has_future_modal(text)),
        "feat_has_hedge": int(has_hedge(text)),
        "feat_has_vague_quantifier": int(has_vague_quantifier(text)),
        "feat_has_rhetorical_filler": int(has_rhetorical_filler(text)),
        "feat_has_fact_assertion": int(has_fact_assertion(text)),
        "feat_is_question": int(is_question(text)),
        "feat_has_transcript_artifact": int(has_transcript_artifact(text)),
        "feat_word_count": compute_word_count(text),
        "feat_avg_word_length": round(compute_avg_word_length(text), 2),
        "feat_alpha_ratio": round(compute_alpha_ratio(text), 3),
        "feat_has_number_and_time": int(has_number(text) and (has_specific_year(text) or has_relative_time(text))),
        "feat_has_number_and_comparative": int(has_number(text) and has_comparative(text)),
        "feat_has_change_and_time": int(has_increase_decrease(text) and (has_specific_year(text) or has_relative_time(text))),
        "feat_has_source_and_number": int((has_source_attribution(text) or has_evidence_noun(text) or has_official_source(text)) and has_number(text)),
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_claimbuster() -> tuple[list[str], np.ndarray]:
    """Load ClaimBuster groundtruth dataset."""
    df = pl.read_csv(CB_RAW)
    texts = df["Text"].to_list()
    # ClaimBuster labels: -1=NFS, 0=UFS, 1=CFS → binary: CFS (1) is positive
    labels = np.array([1 if v == 1 else 0 for v in df["Verdict"].to_list()])
    return texts, labels


def load_ct23() -> tuple[list[str], np.ndarray]:
    """Load CheckThat 2023 test dataset."""
    df = pl.read_csv(CT23_RAW, separator="\t")
    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
    return texts, labels


def load_ct24_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CT24 text_embed_pca64 + LLM features."""
    # Text + PCA-64 features
    train_pca = pl.read_parquet(CT24_CLASSIFIER_DIR / "train_text_embed_pca64.parquet")
    dev_pca = pl.read_parquet(CT24_CLASSIFIER_DIR / "dev_text_embed_pca64.parquet")
    test_pca = pl.read_parquet(CT24_CLASSIFIER_DIR / "test_text_embed_pca64.parquet")

    # LLM features (v4)
    train_llm = pl.read_parquet(CT24_LLM_DIR / "train_llm_features.parquet")
    dev_llm = pl.read_parquet(CT24_LLM_DIR / "dev_llm_features.parquet")
    test_llm = pl.read_parquet(CT24_LLM_DIR / "test_llm_features.parquet")

    # Feature columns
    text_feat_cols = [c for c in train_pca.columns if c.startswith("feat_")]
    pca_cols = [c for c in train_pca.columns if c.startswith("pca64_")]
    llm_feat_cols = [c for c in train_llm.columns if c not in ["sentence_id"] and not c.endswith("_prediction") and not c.endswith("_parse_issue")]

    print(f"  Text features: {len(text_feat_cols)}")
    print(f"  PCA features: {len(pca_cols)}")
    print(f"  LLM features: {len(llm_feat_cols)}")

    def combine_features(pca_df, llm_df):
        X_text = pca_df.select(text_feat_cols).to_numpy()
        X_pca = pca_df.select(pca_cols).to_numpy()
        X_llm = llm_df.select(llm_feat_cols).to_numpy()
        return np.hstack([X_text, X_pca, X_llm])

    X_train = combine_features(train_pca, train_llm)
    X_dev = combine_features(dev_pca, dev_llm)
    X_test = combine_features(test_pca, test_llm)

    y_train = (train_pca["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev_pca["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test_pca["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, y_train, X_dev, y_dev, X_test, y_test, text_feat_cols, pca_cols, llm_feat_cols


def generate_benchmark_features(texts: list[str], llm_df: pl.DataFrame,
                                  text_feat_cols: list[str], pca_cols: list[str],
                                  llm_feat_cols: list[str]) -> np.ndarray:
    """Generate features for benchmark dataset."""
    from sentence_transformers import SentenceTransformer

    # 1. Extract text features
    print("  Extracting text features...")
    text_features = [extract_text_features(t) for t in texts]
    text_df = pl.DataFrame(text_features)
    X_text = text_df.select([c for c in text_feat_cols if c in text_df.columns]).to_numpy()

    # 2. Generate embeddings and apply PCA
    print("  Generating embeddings...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Load PCA model fitted on CT24
    print("  Applying PCA...")
    with open(CT24_CLASSIFIER_DIR / "pca_models.pkl", "rb") as f:
        pca_models = pickle.load(f)
    pca_64 = pca_models["pca_64"]
    X_pca = pca_64.transform(embeddings)

    # 3. LLM features
    print("  Processing LLM features...")
    # Filter to only numeric LLM columns that exist
    available_llm_cols = [c for c in llm_feat_cols if c in llm_df.columns]
    X_llm = llm_df.select(available_llm_cols).to_numpy()

    # Handle NaN in LLM features
    X_llm = np.nan_to_num(X_llm, nan=0.0)

    print(f"  Feature dimensions: text={X_text.shape[1]}, pca={X_pca.shape[1]}, llm={X_llm.shape[1]}")

    return np.hstack([X_text, X_pca, X_llm])


def evaluate(y_true: np.ndarray, probs: np.ndarray, thresholds: list[float] = None) -> dict:
    """Evaluate with threshold search."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.80, 0.05).tolist()

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
    print("=" * 70)
    print("BENCHMARK: PCA-64 + LLM + Text Features")
    print("=" * 70)

    # Load CT24 training data
    print("\n[1/5] Loading CT24 features...")
    X_train, y_train, X_dev, y_dev, X_test, y_test, text_feat_cols, pca_cols, llm_feat_cols = load_ct24_features()

    # Combine train + dev for final model
    X_traindev = np.vstack([X_train, X_dev])
    y_traindev = np.concatenate([y_train, y_dev])
    print(f"  Train+Dev: {X_traindev.shape[0]} samples, {X_traindev.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")

    # Scale features
    print("\n[2/5] Scaling features...")
    scaler = StandardScaler()
    X_traindev_scaled = scaler.fit_transform(X_traindev)
    X_test_scaled = scaler.transform(X_test)

    # Train LogisticRegression
    print("\n[3/5] Training LogisticRegression...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_traindev_scaled, y_traindev)

    # Evaluate on CT24 test
    ct24_probs = clf.predict_proba(X_test_scaled)[:, 1]
    ct24_result = evaluate(y_test, ct24_probs)
    print(f"\n  CT24 Test: F1={ct24_result['f1']:.4f} @ threshold={ct24_result['threshold']:.2f}")
    print(f"            Precision={ct24_result['precision']:.4f}, Recall={ct24_result['recall']:.4f}")

    # Load and process ClaimBuster
    print("\n[4/5] Processing ClaimBuster...")
    cb_texts, cb_labels = load_claimbuster()
    cb_llm = pl.read_parquet(BENCHMARK_LLM_DIR / "CB_groundtruth_llm_features.parquet")
    X_cb = generate_benchmark_features(cb_texts, cb_llm, text_feat_cols, pca_cols, llm_feat_cols)
    X_cb_scaled = scaler.transform(X_cb)

    cb_probs = clf.predict_proba(X_cb_scaled)[:, 1]
    cb_result = evaluate(cb_labels, cb_probs)
    print(f"\n  ClaimBuster: F1={cb_result['f1']:.4f} @ threshold={cb_result['threshold']:.2f}")
    print(f"              Precision={cb_result['precision']:.4f}, Recall={cb_result['recall']:.4f}")

    # Load and process CT23
    print("\n[5/5] Processing CT23...")
    ct23_texts, ct23_labels = load_ct23()
    ct23_llm = pl.read_parquet(BENCHMARK_LLM_DIR / "CT23_llm_features.parquet")
    X_ct23 = generate_benchmark_features(ct23_texts, ct23_llm, text_feat_cols, pca_cols, llm_feat_cols)
    X_ct23_scaled = scaler.transform(X_ct23)

    ct23_probs = clf.predict_proba(X_ct23_scaled)[:, 1]
    ct23_result = evaluate(ct23_labels, ct23_probs)
    print(f"\n  CT23: F1={ct23_result['f1']:.4f} @ threshold={ct23_result['threshold']:.2f}")
    print(f"        Precision={ct23_result['precision']:.4f}, Recall={ct23_result['recall']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PCA-64 + LLM + Text Features (LogReg)")
    print("=" * 70)
    print(f"\n{'Benchmark':<15} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Threshold':<10}")
    print("-" * 57)
    for name, result in [("CT24 Test", ct24_result), ("ClaimBuster", cb_result), ("CT23", ct23_result)]:
        print(f"{name:<15} {result['f1']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['threshold']:<10.2f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
