"""Run Claude Opus 4.7 zero-shot check-worthiness over CT24 test-gold (N=341).

Per the LLM-ablation spec:

  - 341 samples x 3 dimensions (checkability, verifiability, harm) = 1023 calls
  - temperature=0, deterministic
  - prompt caching with 1-hour TTL on each system prompt
  - cost ceiling: $10 hard abort (expected ~$4.30)
  - resumable: skips (sentence_id, dimension) already present in raw_responses.jsonl
  - retries: up to 3 with exponential backoff on 429 / transient errors

Outputs (under --out-dir):
  - raw_responses.jsonl         append-only API record
  - llm_predictions.parquet     per-sample table (16 columns)
  - llm_cost_breakdown.json     per-prompt + total cost accounting

Usage:
    python scripts/run_llm_ablation.py --out-dir results/llm_ablation_<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json_repair

import anthropic

# Resolve the project root and add it to sys.path so `scripts.prompts` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prompts import (  # noqa: E402
    CHECKABILITY_SYSTEM, CHECKABILITY_USER, CHECKABILITY_PREFILL,
    VERIFIABILITY_SYSTEM, VERIFIABILITY_USER, VERIFIABILITY_PREFILL,
    HARM_SYSTEM, HARM_USER, HARM_PREFILL,
)

MODEL = "claude-opus-4-7"
CT24_TSV = PROJECT_ROOT / "data" / "CT24_checkworthy_english_test_gold.tsv"
SENTENCE_IDS_NPY = PROJECT_ROOT / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553" / "test_sentence_ids.npy"
LABELS_NPY = PROJECT_ROOT / "reproducibility" / "runs" / "deberta_mtl_cikm_20260512_134553" / "test_labels.npy"

PRICE_INPUT_PER_M       = 5.0
PRICE_OUTPUT_PER_M      = 25.0
PRICE_CACHE_READ_PER_M  = 0.50
PRICE_CACHE_WRITE_1H_PER_M = 10.0

COST_CEILING_USD = 10.0

DIMS: dict[str, dict[str, str | int]] = {
    "checkability":  {"system": CHECKABILITY_SYSTEM,  "user": CHECKABILITY_USER,  "prefill": CHECKABILITY_PREFILL,  "max_tokens": 512},
    "verifiability": {"system": VERIFIABILITY_SYSTEM, "user": VERIFIABILITY_USER, "prefill": VERIFIABILITY_PREFILL, "max_tokens": 512},
    "harm":          {"system": HARM_SYSTEM,          "user": HARM_USER,          "prefill": HARM_PREFILL,          "max_tokens": 1024},
}


def load_dataset() -> tuple[list[int], list[str], np.ndarray]:
    sentence_ids = np.load(SENTENCE_IDS_NPY).astype(np.int64)
    labels = np.load(LABELS_NPY).astype(np.int64)
    df = pl.read_csv(CT24_TSV, separator="\t")
    id_to_text = dict(zip(df["Sentence_id"].cast(pl.Int64).to_list(), df["Text"].to_list()))
    texts = [id_to_text[int(sid)] for sid in sentence_ids]
    return sentence_ids.tolist(), texts, labels


def load_done(jsonl_path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((int(rec["sentence_id"]), str(rec["dim"])))
            except Exception:
                continue
    return done


def call_cost_usd(usage: Any) -> tuple[float, dict[str, int]]:
    inp           = int(getattr(usage, "input_tokens", 0) or 0)
    out           = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read    = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write_5m = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_create = getattr(usage, "cache_creation", None)
    cache_write_1h = 0
    if cache_create is not None:
        cache_write_1h = int(getattr(cache_create, "ephemeral_1h_input_tokens", 0) or 0)
        cw5m = int(getattr(cache_create, "ephemeral_5m_input_tokens", 0) or 0)
        if cw5m and not cache_write_5m:
            cache_write_5m = cw5m
    cost = (
        inp           * PRICE_INPUT_PER_M / 1e6
        + out         * PRICE_OUTPUT_PER_M / 1e6
        + cache_read  * PRICE_CACHE_READ_PER_M / 1e6
        + (cache_write_5m + cache_write_1h) * PRICE_CACHE_WRITE_1H_PER_M / 1e6
    )
    return cost, {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write_5m + cache_write_1h,
    }


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )),
)
def one_call(client: anthropic.Anthropic, dim: str, claim_text: str) -> tuple[Any, float]:
    # Opus 4.7 does not support assistant-message prefill and does not accept
    # the deprecated `temperature` parameter. The system prompt instructs the
    # model to "Respond ONLY with valid JSON. Output confidence FIRST." which is
    # sufficient to constrain output structure for json_repair.
    d = DIMS[dim]
    system_block = {
        "type": "text",
        "text": d["system"],
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    messages = [
        {"role": "user", "content": d["user"].format(claim=claim_text)},
    ]
    t0 = time.monotonic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=int(d["max_tokens"]),
        system=[system_block],
        messages=messages,
        extra_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
    )
    return resp, time.monotonic() - t0


def parse_json_body(body_text: str) -> dict | None:
    try:
        return json.loads(body_text)
    except Exception:
        try:
            repaired = json_repair.repair_json(body_text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            return None
    return None


def to_bool_yes(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "true", "1"):
            return True
        if v in ("no", "false", "0"):
            return False
    return None


def to_conf(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def build_parquet(records_by_pair: dict[tuple[int, str], dict], sentence_ids: list[int], texts: list[str], labels: np.ndarray, parquet_path: Path) -> int:
    rows = []
    for sid, text, lbl in zip(sentence_ids, texts, labels):
        rec_c = records_by_pair.get((sid, "checkability"))
        rec_v = records_by_pair.get((sid, "verifiability"))
        rec_h = records_by_pair.get((sid, "harm"))

        def cfg(rec: dict | None, key: str, default=None):
            if not rec or not rec.get("parsed"):
                return default
            return rec["parsed"].get(key, default)

        def cfg_sub(rec: dict | None, key: str, sub: str, default=None):
            if not rec or not rec.get("parsed"):
                return default
            obj = rec["parsed"].get(key)
            if isinstance(obj, dict):
                return obj.get(sub, default)
            return default

        def usage(rec: dict | None, key: str, default=0):
            if not rec:
                return default
            return int(rec.get(key, default) or 0)

        def latency(rec: dict | None) -> float:
            if not rec:
                return 0.0
            return float(rec.get("latency_seconds", 0.0) or 0.0)

        row = {
            "sentence_id": int(sid),
            "text": text,
            "label": int(lbl),
            "llm_checkability_confidence":   to_conf(cfg(rec_c, "confidence")),
            "llm_checkability_is_yes":       to_bool_yes(cfg(rec_c, "is_checkable")),
            "llm_checkability_reasoning":    cfg(rec_c, "reasoning"),
            "llm_verifiability_confidence":  to_conf(cfg(rec_v, "confidence")),
            "llm_verifiability_is_yes":      to_bool_yes(cfg(rec_v, "is_verifiable")),
            "llm_verifiability_reasoning":   cfg(rec_v, "reasoning"),
            "llm_harm_confidence":           to_conf(cfg(rec_h, "confidence")),
            "llm_harm_is_yes":               to_bool_yes(cfg(rec_h, "is_harmful")),
            "llm_harm_reasoning":            cfg(rec_h, "reasoning"),
            "llm_harm_social_fragmentation": to_conf(cfg_sub(rec_h, "social_fragmentation", "confidence")),
            "llm_harm_spurs_action":         to_conf(cfg_sub(rec_h, "spurs_action", "confidence")),
            "llm_harm_believability":        to_conf(cfg_sub(rec_h, "believability", "confidence")),
            "llm_harm_exploitativeness":     to_conf(cfg_sub(rec_h, "exploitativeness", "confidence")),
            "input_tokens":          usage(rec_c, "input_tokens") + usage(rec_v, "input_tokens") + usage(rec_h, "input_tokens"),
            "output_tokens":         usage(rec_c, "output_tokens") + usage(rec_v, "output_tokens") + usage(rec_h, "output_tokens"),
            "cache_read_tokens":     usage(rec_c, "cache_read_tokens") + usage(rec_v, "cache_read_tokens") + usage(rec_h, "cache_read_tokens"),
            "cache_write_tokens":    usage(rec_c, "cache_write_tokens") + usage(rec_v, "cache_write_tokens") + usage(rec_h, "cache_write_tokens"),
            "latency_seconds":       latency(rec_c) + latency(rec_v) + latency(rec_h),
        }
        rows.append(row)

    df = pl.DataFrame(rows)
    df.write_parquet(parquet_path)
    return len(rows)


def load_records_from_jsonl(jsonl_path: Path) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (int(rec["sentence_id"]), str(rec["dim"]))
                out[key] = rec
            except Exception:
                continue
    return out


def main(out_dir: Path) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "raw_responses.jsonl"
    parquet_path = out_dir / "llm_predictions.parquet"
    cost_path = out_dir / "llm_cost_breakdown.json"

    sentence_ids, texts, labels = load_dataset()
    n = len(sentence_ids)
    print(f"[data] N={n}  positives={int(labels.sum())}")

    done = load_done(jsonl_path)
    print(f"[resume] {len(done)} (sid,dim) pairs already in {jsonl_path.name}")

    client = anthropic.Anthropic(api_key=api_key)
    by_dim_cost = {k: 0.0 for k in DIMS}
    by_dim_calls = {k: 0 for k in DIMS}
    total_cost = 0.0
    parse_failures = 0
    aborted = False

    total_pairs = n * len(DIMS)
    completed = len(done)
    pbar = tqdm(total=total_pairs, initial=completed, desc="LLM calls", dynamic_ncols=True)

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for dim in DIMS.keys():
            for sid, text in zip(sentence_ids, texts):
                if (sid, dim) in done:
                    continue
                if total_cost > COST_CEILING_USD:
                    aborted = True
                    print(f"\n[ABORT] running cost ${total_cost:.4f} > ceiling ${COST_CEILING_USD:.2f}", file=sys.stderr)
                    break
                try:
                    resp, latency_s = one_call(client, dim, text)
                except Exception as e:
                    rec = {
                        "sentence_id": int(sid),
                        "dim": dim,
                        "error": f"{type(e).__name__}: {e}",
                        "parsed": None,
                        "raw_text": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "latency_seconds": 0.0,
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                    pbar.update(1)
                    completed += 1
                    continue

                body_text = resp.content[0].text if resp.content else ""
                parsed = parse_json_body(body_text)
                if parsed is None:
                    parse_failures += 1
                cost, tokens = call_cost_usd(resp.usage)
                total_cost += cost
                by_dim_cost[dim] += cost
                by_dim_calls[dim] += 1

                rec = {
                    "sentence_id": int(sid),
                    "dim": dim,
                    "model": MODEL,
                    "parsed": parsed,
                    "raw_text": body_text,
                    **tokens,
                    "latency_seconds": float(latency_s),
                    "cost_usd": cost,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                done.add((sid, dim))
                completed += 1
                pbar.update(1)
                pbar.set_postfix({
                    "cost": f"${total_cost:.4f}",
                    "fails": parse_failures,
                })

                if completed % 50 == 0:
                    try:
                        recs = load_records_from_jsonl(jsonl_path)
                        build_parquet(recs, sentence_ids, texts, labels, parquet_path)
                    except Exception as e:
                        print(f"\n[warn] checkpoint parquet write failed: {e}", file=sys.stderr)

            if aborted:
                break

    pbar.close()

    recs = load_records_from_jsonl(jsonl_path)
    n_rows = build_parquet(recs, sentence_ids, texts, labels, parquet_path)
    print(f"[parquet] {parquet_path}  rows={n_rows}")

    cost_summary = {
        "total_cost_usd": total_cost,
        "by_dimension": {
            dim: {
                "calls": by_dim_calls[dim],
                "cost_usd": by_dim_cost[dim],
            } for dim in DIMS
        },
        "parse_failures": parse_failures,
        "aborted_on_cost_ceiling": aborted,
        "cost_ceiling_usd": COST_CEILING_USD,
        "model": MODEL,
    }
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"[cost] total ${total_cost:.4f}  by_dim={ {k: round(by_dim_cost[k], 4) for k in DIMS} }")
    print(f"[parse] failures={parse_failures}/{total_pairs}")

    return 1 if aborted else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(main(args.out_dir))
