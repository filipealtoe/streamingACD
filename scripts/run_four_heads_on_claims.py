"""Run the 4-Parallel-Heads checkpoint over the 535 released claim texts.

Mirrors the inference protocol used by scripts/generate_mtl_aux_predictions.py
(same tokenizer, max_length=128, batch_size=16, softmax->[:,1] for overall,
sigmoid aux outputs in [0,1], threshold 0.5 for binary decisions).

Outputs (to --out-dir):
    claim_scores.parquet
    claim_scores.csv
    inference_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import transformers as _tfm


REPO_ROOT  = Path(__file__).resolve().parent.parent
RUN_DIR    = REPO_ROOT / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553"
CHECKPOINT = REPO_ROOT / "data" / "4-heads_model" / "best_model.pt"
CHECKSUM   = REPO_ROOT / "data" / "4-heads_model" / "best_model.pt.sha256"
FINETUNE_SCRIPT = RUN_DIR / "scripts" / "finetune_deberta_mtl.py"
CLAIMS_PARQUET  = REPO_ROOT / "psr" / "explainableACD" / "data" / "pipeline_output" / "streaming_full" / "2026-01-17_03-56" / "claims.parquet"

EXPECTED_SHA = "5f61837bbeb2b513ca7c49ab5901a6a107dbe275200ac0396777ce375271f081"


def import_mtl_module():
    spec = importlib.util.spec_from_file_location("finetune_deberta_mtl", FINETUNE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["finetune_deberta_mtl"] = module
    spec.loader.exec_module(module)
    return module


def verify_checkpoint_sha(path: Path, expected: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != expected:
        raise RuntimeError(f"Checkpoint SHA mismatch: got {digest}, expected {expected}")
    return digest


def summarize(name: str, arr: np.ndarray) -> dict:
    above = float((arr >= 0.5).mean() * 100.0)
    return {
        "head": name,
        "mean": float(arr.mean()),
        "p50": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "pct_above_0_5": above,
    }


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[preflight] verifying checkpoint SHA-256...")
    digest = verify_checkpoint_sha(CHECKPOINT, EXPECTED_SHA)
    print(f"[preflight] SHA OK: {digest}")

    print("[load] importing MultiTaskDeBERTa class")
    mtl_mod = import_mtl_module()
    cfg = mtl_mod.Config()
    print(f"[cfg] model={cfg.model_name}  max_length={cfg.max_length}")

    print(f"[load] claims from {CLAIMS_PARQUET}")
    claims = pd.read_parquet(CLAIMS_PARQUET)
    assert "claim_text" in claims.columns and "claim_id" in claims.columns, \
        f"claims.parquet missing required columns: {claims.columns.tolist()}"
    n = len(claims)
    print(f"[load] N={n}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print("[load] instantiating model and loading checkpoint")
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
        print(f"[load] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[device] {device}")

    texts = claims["claim_text"].astype(str).tolist()
    bs = 16
    all_main = np.empty(n, dtype=np.float32)
    all_aux  = {k: np.empty(n, dtype=np.float32) for k in ("checkability", "verifiability", "harm")}

    t0 = time.time()
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
            done = min(i + bs, n)
            print(f"[infer] {done}/{n}  ({(time.time()-t0):.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"[infer] complete in {elapsed:.1f}s")

    out = pd.DataFrame({
        "claim_id": claims["claim_id"].astype(str).values,
        "claim_text": claims["claim_text"].astype(str).values,
        "checkability_score":  all_aux["checkability"],
        "verifiability_score": all_aux["verifiability"],
        "harm_score":          all_aux["harm"],
        "overall_score":       all_main,
        "checkability_decision":  (all_aux["checkability"]  >= 0.5),
        "verifiability_decision": (all_aux["verifiability"] >= 0.5),
        "harm_decision":          (all_aux["harm"]          >= 0.5),
        "overall_decision":       (all_main                  >= 0.5),
    })
    out.to_parquet(out_dir / "claim_scores.parquet", index=False)
    out.to_csv(out_dir / "claim_scores.csv", index=False)
    print(f"[save] claim_scores written")

    summary = {
        "checkability":  summarize("checkability",  all_aux["checkability"]),
        "verifiability": summarize("verifiability", all_aux["verifiability"]),
        "harm":          summarize("harm",          all_aux["harm"]),
        "overall":       summarize("overall",       all_main),
    }
    overall_above = summary["overall"]["pct_above_0_5"]
    flag = None
    if overall_above < 30.0:
        flag = "Overall above-0.5 share is below 30%: unexpectedly conservative for anomaly-filtered claims."
    elif overall_above > 95.0:
        flag = "Overall above-0.5 share is above 95%: unexpectedly aggressive — verify the checkpoint matches the published one."

    meta = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": digest,
        "n_claims": int(n),
        "tokenizer": cfg.model_name,
        "transformers_version": _tfm.__version__,
        "torch_version": torch.__version__,
        "max_length": int(cfg.max_length),
        "batch_size": int(bs),
        "decision_threshold": 0.5,
        "device": str(device),
        "wall_clock_seconds": round(elapsed, 2),
        "per_head_summary": summary,
        "overall_decision_counts": {
            "check_worthy": int((all_main >= 0.5).sum()),
            "not_check_worthy": int((all_main < 0.5).sum()),
        },
        "sanity_flag": flag,
    }
    (out_dir / "inference_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n[summary]")
    for k in ("checkability", "verifiability", "harm", "overall"):
        s = summary[k]
        print(f"  {k:<14}: mean={s['mean']:.3f}, p50={s['p50']:.3f}, % above 0.5: {s['pct_above_0_5']:5.1f}%")
    if flag:
        print(f"\n[flag] {flag}")
    else:
        print("\n[flag] overall above-0.5 share is within expected range")

    print(f"\n[done] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(main(args.out_dir))
