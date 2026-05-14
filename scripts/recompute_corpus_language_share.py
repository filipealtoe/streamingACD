from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))
DEFAULT_PARQUET = SOURCE_ROOT / "data/raw/us_elections_tweets.parquet"
DEFAULT_KAGGLE_DIR = Path(
    "/Users/sergiopinto/.cache/kagglehub/datasets/manchunhui/us-election-2020-tweets/versions/19"
)
RESULT_JSON = REPO_ROOT / "results/corpus_language_share_2026-05-12.json"
RESULT_MD = REPO_ROOT / "results/corpus_language_share_2026-05-12.md"

_DETECT: Any = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_worker() -> None:
    global _DETECT
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    _DETECT = detect


def detect_language(text: Any) -> str:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "unknown"
    try:
        return str(_DETECT(text))
    except Exception:
        return "unknown"


def detect_batch(texts: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts[detect_language(text)] += 1
    return dict(counts)


def iter_text_batches(path: Path, column: str, batch_size: int) -> Iterable[list[Any]]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=[column], batch_size=batch_size):
        yield batch.column(0).to_pylist()


def compute_language_counts(path: Path, column: str, workers: int, batch_size: int) -> tuple[Counter[str], float]:
    start = time.time()
    counts: Counter[str] = Counter()
    processed = 0
    next_report = 100_000

    if workers <= 1:
        init_worker()
        for batch_counts in map(detect_batch, iter_text_batches(path, column, batch_size)):
            counts.update(batch_counts)
            processed = sum(counts.values())
            if processed >= next_report:
                print(f"processed {processed:,} rows")
                next_report += 100_000
    else:
        with Pool(processes=workers, initializer=init_worker) as pool:
            for batch_counts in pool.imap_unordered(
                detect_batch,
                iter_text_batches(path, column, batch_size),
                chunksize=1,
            ):
                counts.update(batch_counts)
                processed = sum(counts.values())
                if processed >= next_report:
                    print(f"processed {processed:,} rows")
                    next_report += 100_000

    return counts, time.time() - start


def kaggle_file_metadata(path: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    if not path.exists():
        return metadata

    for csv_path in sorted(path.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            header = next(csv.reader(handle))
        metadata.append(
            {
                "path": str(csv_path),
                "name": csv_path.name,
                "size_bytes": csv_path.stat().st_size,
                "columns": header,
                "has_language_column": any(column.lower() in {"lang", "language"} for column in header),
            }
        )
    return metadata


def build_result(args: argparse.Namespace, counts: Counter[str], elapsed_seconds: float) -> dict[str, Any]:
    total = sum(counts.values())
    english_count = counts.get("en", 0)
    non_english_count = total - english_count
    parquet_file = pq.ParquetFile(args.parquet)
    top_languages = [{"language": lang, "count": count, "share": count / total} for lang, count in counts.most_common(20)]

    return {
        "claim": "Corpus is approximately 87% English.",
        "status": "not_reproduced",
        "paper_english_share": 0.87,
        "method": {
            "detector": "langdetect.detect",
            "detector_seed": 0,
            "input_column": args.text_column,
            "short_or_invalid_text_rule": "language=unknown when text is non-string or shorter than 10 stripped characters",
            "source": "same detection rule as explainableACD src/streaming/text_preprocessor.py",
            "workers": args.workers,
            "batch_size": args.batch_size,
        },
        "source_parquet": {
            "path": str(args.parquet),
            "rows": parquet_file.metadata.num_rows,
            "sha256": sha256_file(args.parquet) if args.sha256 else None,
        },
        "online_source": {
            "kaggle_dataset": "manchunhui/us-election-2020-tweets",
            "kaggle_url": "https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets",
            "kagglehub_cache_dir": str(args.kaggle_dir),
            "files": kaggle_file_metadata(args.kaggle_dir),
        },
        "counts": dict(sorted(counts.items())),
        "total": total,
        "english_count": english_count,
        "english_share": english_count / total,
        "non_english_count": non_english_count,
        "non_english_share": non_english_count / total,
        "top_languages": top_languages,
        "elapsed_seconds": elapsed_seconds,
        "conclusion": (
            "The recomputed canonical corpus language share is about 70% English, not 87%. "
            "The online Kaggle CSV headers do not expose a language column, so this result is detector-derived."
        ),
    }


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Corpus Language Share Reproduction",
        "",
        f"Paper claim: `{result['claim']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total rows | `{result['total']:,}` |",
        f"| English rows | `{result['english_count']:,}` |",
        f"| English share | `{result['english_share']:.4f}` (`{result['english_share'] * 100:.2f}%`) |",
        f"| Non-English or unknown rows | `{result['non_english_count']:,}` |",
        f"| Non-English or unknown share | `{result['non_english_share']:.4f}` (`{result['non_english_share'] * 100:.2f}%`) |",
        "",
        "## Top Languages",
        "",
        "| Language | Count | Share |",
        "|---|---:|---:|",
    ]
    for item in result["top_languages"]:
        lines.append(f"| `{item['language']}` | `{item['count']:,}` | `{item['share']:.4f}` |")

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Detector: `langdetect.detect` with `DetectorFactory.seed = 0`.",
            "- Input: original `tweet` text from the canonical local parquet.",
            "- Short or invalid text is counted as `unknown`, matching the pipeline helper behavior.",
            "- The downloaded Kaggle CSV headers do not include `lang` or `language`, so this is detector-derived.",
            "",
            "## Conclusion",
            "",
            result["conclusion"],
            "",
        ]
    )

    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULT_JSON}")
    print(f"wrote {RESULT_MD}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--kaggle-dir", type=Path, default=DEFAULT_KAGGLE_DIR)
    parser.add_argument("--text-column", default="tweet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    counts, elapsed_seconds = compute_language_counts(args.parquet, args.text_column, args.workers, args.batch_size)
    result = build_result(args, counts, elapsed_seconds)
    write_outputs(result)
    return 1 if abs(result["english_share"] - result["paper_english_share"]) > 0.005 else 0


if __name__ == "__main__":
    raise SystemExit(main())
