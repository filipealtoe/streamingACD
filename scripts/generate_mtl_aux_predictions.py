"""Generate MTL main + auxiliary head predictions on the CT24 test-gold split.

Loads `best_model.pt` (MultiTaskDeBERTa with DeBERTa-v3-large backbone), runs
all 341 CT24 test samples in post-join row order (aligned with
`test_sentence_ids.npy`), and saves four `.npy` arrays:

    main_probs.npy            (341,) float32 — P(check-worthy) from softmax(main_logits)
    aux_checkability.npy      (341,) float32 — sigmoid output from checkability head
    aux_verifiability.npy     (341,) float32 — sigmoid output from verifiability head
    aux_harm.npy              (341,) float32 — sigmoid output from harm head

Sanity check: F1@0.50 from regenerated main_probs matches the F1@0.50 from the
shipped test_probs.npy (within 1e-4). If not, the script exits non-zero.

Usage:
    python scripts/generate_mtl_aux_predictions.py --out-dir results/llm_ablation_<ts>/mtl_predictions
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553"
CHECKPOINT = REPO_ROOT / "data" / "4-heads_model" / "best_model.pt"
TSV = REPO_ROOT / "data" / "CT24_checkworthy_english_test_gold.tsv"
FINETUNE_SCRIPT = RUN_DIR / "scripts" / "finetune_deberta_mtl.py"
TEST_SENTENCE_IDS = RUN_DIR / "test_sentence_ids.npy"
TEST_LABELS = RUN_DIR / "test_labels.npy"
TEST_PROBS_REFERENCE = RUN_DIR / "test_probs.npy"


def import_mtl_module():
    """Import MultiTaskDeBERTa from the run-bundle script as a module."""
    spec = importlib.util.spec_from_file_location("finetune_deberta_mtl", FINETUNE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["finetune_deberta_mtl"] = module
    spec.loader.exec_module(module)
    return module


def f1_at(threshold: float, probs: np.ndarray, labels: np.ndarray) -> tuple[float, int, int, int, int]:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, tp, fp, tn, fn


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    mtl_mod = import_mtl_module()
    cfg = mtl_mod.Config()
    print(f"[cfg] model={cfg.model_name}  max_length={cfg.max_length}")

    sentence_ids = np.load(TEST_SENTENCE_IDS)
    labels = np.load(TEST_LABELS)
    ref_probs = np.load(TEST_PROBS_REFERENCE)
    n = sentence_ids.shape[0]
    assert labels.shape == (n,) and ref_probs.shape == (n,), \
        f"shape mismatch: ids={sentence_ids.shape} labels={labels.shape} ref={ref_probs.shape}"
    print(f"[data] N={n}  positives={int(labels.sum())}")

    df = pl.read_csv(TSV, separator="\t")
    cols = set(df.columns)
    assert {"Sentence_id", "Text", "class_label"}.issubset(cols), f"TSV cols: {df.columns}"
    id_to_text = dict(zip(df["Sentence_id"].cast(pl.Int64).to_list(), df["Text"].to_list()))
    id_to_class = dict(zip(df["Sentence_id"].cast(pl.Int64).to_list(), df["class_label"].to_list()))

    texts = [id_to_text[int(sid)] for sid in sentence_ids]
    tsv_labels = np.array(
        [1 if id_to_class[int(sid)] == "Yes" else 0 for sid in sentence_ids],
        dtype=np.int64,
    )
    assert np.array_equal(tsv_labels, labels), \
        "Label mismatch between TSV-joined order and shipped test_labels.npy"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    model = mtl_mod.MultiTaskDeBERTa(encoder_name=cfg.model_name, num_labels=2, dropout=0.1)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''} (total {len(missing)})")
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''} (total {len(unexpected)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[device] {device}")

    bs = 16
    all_main = np.empty(n, dtype=np.float32)
    all_aux = {k: np.empty(n, dtype=np.float32) for k in ("checkability", "verifiability", "harm")}

    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = texts[i:i + bs]
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=cfg.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            main_logits, aux = model(input_ids, attn, return_aux=True)
            p1 = F.softmax(main_logits, dim=-1)[:, 1].cpu().numpy()
            all_main[i:i + len(chunk)] = p1
            for k, v in aux.items():
                all_aux[k][i:i + len(chunk)] = v.cpu().numpy()
            print(f"[infer] {min(i + bs, n)}/{n}")

    np.save(out_dir / "main_probs.npy", all_main)
    np.save(out_dir / "aux_checkability.npy", all_aux["checkability"])
    np.save(out_dir / "aux_verifiability.npy", all_aux["verifiability"])
    np.save(out_dir / "aux_harm.npy", all_aux["harm"])

    ref_f1, ref_tp, ref_fp, ref_tn, ref_fn = f1_at(0.5, ref_probs, labels)
    new_f1, new_tp, new_fp, new_tn, new_fn = f1_at(0.5, all_main, labels)
    max_abs_delta = float(np.max(np.abs(all_main - ref_probs)))
    print(f"[sanity] reference F1@0.50={ref_f1:.6f}  tp/fp/tn/fn={ref_tp}/{ref_fp}/{ref_tn}/{ref_fn}")
    print(f"[sanity] new       F1@0.50={new_f1:.6f}  tp/fp/tn/fn={new_tp}/{new_fp}/{new_tn}/{new_fn}")
    print(f"[sanity] max |new - ref|={max_abs_delta:.6e}")

    if abs(ref_f1 - new_f1) > 1e-4:
        print(
            f"[FAIL] F1 mismatch: ref={ref_f1:.6f} vs new={new_f1:.6f} "
            f"(delta={abs(ref_f1 - new_f1):.6e} > 1e-4)"
        )
        return 1

    print(f"[OK] aux predictions saved to {out_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(main(args.out_dir))
