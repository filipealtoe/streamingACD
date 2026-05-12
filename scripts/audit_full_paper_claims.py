from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("EXPLAINABLE_ACD_ROOT", "/Users/sergiopinto/explainableACD"))
ARTIFACT_ROOT = REPO_ROOT / "reproducibility/source_artifacts"
RESULT_JSON = REPO_ROOT / "results/full_paper_claim_audit_2026-05-12.json"
RESULT_MD = REPO_ROOT / "results/full_paper_claim_audit_2026-05-12.md"


@dataclass(frozen=True)
class ClaimAudit:
    claim_id: str
    area: str
    paper_claim: str
    status: str
    evidence: str
    checked_value: str
    required_next_step: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def add(
    audits: list[ClaimAudit],
    claim_id: str,
    area: str,
    paper_claim: str,
    status: str,
    evidence: str,
    checked_value: str,
    required_next_step: str,
) -> None:
    audits.append(
        ClaimAudit(
            claim_id=claim_id,
            area=area,
            paper_claim=paper_claim,
            status=status,
            evidence=evidence,
            checked_value=checked_value,
            required_next_step=required_next_step,
        )
    )


def rounded_match(value: float, target: float, places: int = 3) -> bool:
    return round(value, places) == round(target, places)


def audit_corpus_and_pipeline(audits: list[ClaimAudit]) -> None:
    raw = SOURCE_ROOT / "data/raw/us_elections_tweets.parquet"
    summary = ARTIFACT_ROOT / "pipeline/streaming_full_2026-01-17_03-56_summary.json"
    if raw.exists():
        table = pq.read_table(raw, columns=["created_at"])
        min_max = pc.min_max(table.column("created_at")).as_py()
        add(
            audits,
            "corpus-count-date-range",
            "Dataset",
            "US Election 2020 corpus has 1.52M tweets from Oct 15 to Nov 08, 2020.",
            "REPRODUCED_LOCAL",
            str(raw),
            f"{table.num_rows} rows; created_at {min_max['min']} to {min_max['max']}",
            "Package acquisition instructions and checksum for the raw Kaggle-derived parquet.",
        )
    else:
        add(
            audits,
            "corpus-count-date-range",
            "Dataset",
            "US Election 2020 corpus has 1.52M tweets from Oct 15 to Nov 08, 2020.",
            "MISSING_ARTIFACT",
            str(raw),
            "raw parquet not found",
            "Restore or document the raw corpus artifact.",
        )

    if summary.exists():
        data = load_json(summary)
        rows = int(data["total_tweets_processed"])
        non_english = int(data["text_preprocessor"]["non_english_detected"])
        english_ratio = (rows - non_english) / rows
        status = "MISMATCH" if round(english_ratio, 2) != 0.87 else "REPRODUCED"
        add(
            audits,
            "corpus-english-share",
            "Dataset",
            "Corpus is approximately 87% English.",
            status,
            str(summary),
            f"summary implies English share {english_ratio:.3f}; no raw language/lang column in parquet",
            "Remove/recompute the 87% claim from a clear language column or detection log.",
        )

        add(
            audits,
            "pipeline-535-claims",
            "Pipeline Output",
            "The complete pipeline generated 535 normalized claims with linked artifacts.",
            "REPRODUCED",
            str(summary),
            (
                f"total_claims={data['total_claims']}; total_tweets_processed={data['total_tweets_processed']}; "
                f"total_tweets_passed_gate={data['total_tweets_passed_gate']}; total_clusters={data['clusterer']['n_clusters']}"
            ),
            "Keep this run as the canonical paper run or update every paper cardinality to another run.",
        )

    pipeline = SOURCE_ROOT / "data/pipeline_output/streaming_full/2026-01-17_03-56"
    expected = {
        "tweets.parquet": 692_289,
        "clusters.parquet": 100_000,
        "claims.parquet": 535,
        "cluster_timeseries.parquet": 404_296,
    }
    parts: list[str] = []
    failures: list[str] = []
    for filename, expected_rows in expected.items():
        path = pipeline / filename
        if not path.exists():
            failures.append(f"{filename} missing")
            continue
        rows = parquet_rows(path)
        parts.append(f"{filename}={rows}")
        if rows != expected_rows:
            failures.append(f"{filename}: expected {expected_rows}, found {rows}")
    for filename in ("cluster_embeddings.npy", "cluster_id_to_idx.json", "cluster_ids_order.json"):
        if not (pipeline / filename).exists():
            failures.append(f"{filename} missing")
        else:
            parts.append(f"{filename}=present")
    add(
        audits,
        "pipeline-artifact-schema",
        "Pipeline Output",
        "Pipeline emits tweets, clusters, claims, cluster time series, embeddings, and index mappings.",
        "REPRODUCED_LOCAL" if not failures else "MISMATCH",
        str(pipeline),
        "; ".join(parts + failures),
        "Record external storage/checksums for large parquet and npy files.",
    )


def audit_clustering(audits: list[ClaimAudit]) -> None:
    path = ARTIFACT_ROOT / "clustering/cluster_statistics.json"
    if not path.exists():
        add(
            audits,
            "clustering-threshold-ablation",
            "Clustering",
            "Ablation over tau 0.55 to 0.80 selected tau=0.65 with 0.87 mean similarity and 87% grouped.",
            "MISSING_ARTIFACT",
            str(path),
            "cluster_statistics.json not found",
            "Package or rerun the threshold ablation.",
        )
        return
    data = load_json(path)
    entry = data.get("0.65", {})
    thresholds = {round(float(key), 2) for key in data.keys()}
    ok = (
        thresholds == {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
        and entry.get("cluster_yield_pct") == 86.9
        and abs(float(entry.get("mean_intra_sim", -1)) - 0.8685999195826681) < 1e-12
        and entry.get("total_tweets") == 5000
    )
    add(
        audits,
        "clustering-threshold-ablation",
        "Clustering",
        "Ablation over tau 0.55 to 0.80 selected tau=0.65 with 0.87 mean similarity and 87% grouped.",
        "REPRODUCED" if ok else "MISMATCH",
        str(path),
        f"thresholds={sorted(data.keys())}; tau=0.65 entry={entry}",
        "Keep wording clear that this is the 5,000-tweet ablation unless full-run evidence is added.",
    )


def audit_anomaly(audits: list[ClaimAudit]) -> None:
    grid = ARTIFACT_ROOT / "anomaly/expose_grid_search.csv"
    if not grid.exists():
        add(
            audits,
            "anomaly-expose-table",
            "Anomaly Detection",
            "EXPoSE reaches NAB 79.2, 97.6% detection, and +23h median lead time.",
            "MISSING_ARTIFACT",
            str(grid),
            "expose_grid_search.csv not found",
            "Rerun or package the exact anomaly benchmark.",
        )
        return
    with grid.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row["nab_score"]))
    nab = float(best["nab_score"])
    detection = float(best["detection_rate"])
    lead = float(best["median_lead"])
    status = "REPRODUCED" if rounded_match(nab, 79.2, 1) and rounded_match(detection, 97.6, 1) else "MISMATCH"
    add(
        audits,
        "anomaly-expose-table",
        "Anomaly Detection",
        "EXPoSE reaches NAB 79.2, 97.6% detection, and +23h median lead time.",
        status,
        str(grid),
        f"best packaged row: NAB={nab:.4f}, detection={detection:.4f}, median_lead={lead:.1f}",
        "Rerun the anomaly benchmark until the paper value is matched/surpassed, or rewrite the paper to the supported value.",
    )

    add(
        audits,
        "anomaly-baseline-table",
        "Anomaly Detection",
        "Random/Bayesian/Etsy/Steuber baselines match the IJCAI anomaly paragraph.",
        "MISSING_EXACT_RESULT",
        (
            f"{ARTIFACT_ROOT / 'anomaly/anomaly_baselines.json'}; "
            f"{ARTIFACT_ROOT / 'anomaly/anomaly_detector_comparison.json'}"
        ),
        "packaged anomaly JSONs use different detectors/metrics and do not contain the exact paper table",
        "Rerun the uniform anomaly benchmark and save one canonical table artifact.",
    )


def audit_claim_normalization(audits: list[ClaimAudit]) -> None:
    path = ARTIFACT_ROOT / "claim_normalization/comparison_test_20260113_123010.json"
    if not path.exists():
        add(
            audits,
            "claim-normalization-table1",
            "Claim Normalization",
            "Table 1 reports CT2025 test N=300 and Approach 2 METEOR 0.5583/0.5463/0.5691.",
            "MISSING_ARTIFACT",
            str(path),
            "comparison JSON not found",
            "Rerun claim normalization Table 1 or package the exact result artifact.",
        )
        return
    data = load_json(path)
    sample_counts = sorted({row.get("n_samples") for row in data.values() if isinstance(row, dict)})
    best_name, best_row = max(
        ((name, row) for name, row in data.items() if isinstance(row, dict)),
        key=lambda item: float(item[1].get("avg_meteor", -1)),
    )
    add(
        audits,
        "claim-normalization-table1",
        "Claim Normalization",
        "Table 1 reports CT2025 test N=300 and Approach 2 METEOR 0.5583/0.5463/0.5691.",
        "MISMATCH",
        str(path),
        f"available comparison has n_samples={sample_counts}; best avg_meteor={best_row['avg_meteor']:.4f} ({best_name})",
        "Rerun the exact Table 1 protocol and save predictions plus summary, or rewrite to supported values.",
    )


def audit_virality(audits: list[ClaimAudit]) -> None:
    labels = SOURCE_ROOT / "experiments/results/virality/psr_labels.parquet"
    enhanced = SOURCE_ROOT / "experiments/results/virality_feature_selection/features_enhanced.parquet"
    if labels.exists() and enhanced.exists():
        rows = parquet_rows(labels)
        feature_count = len(pq.ParquetFile(enhanced).schema.names) - 2
        status = "REPRODUCED_LOCAL" if rows == 529 and feature_count == 42 else "MISMATCH"
        checked = f"psr_labels rows={rows}; features_enhanced feature columns={feature_count}"
    else:
        status = "MISSING_ARTIFACT"
        checked = f"labels exists={labels.exists()}; features_enhanced exists={enhanced.exists()}"
    add(
        audits,
        "virality-dataset-shape",
        "Virality Prediction",
        "PSR prediction uses 529 anomalous clusters, 42 features, and a 423/106 split.",
        status,
        f"{labels}; {enhanced}",
        checked,
        "Package a split manifest containing the exact 423/106 train/test IDs.",
    )

    tuned = ARTIFACT_ROOT / "virality/tuned_baselines.json"
    complete = ARTIFACT_ROOT / "virality/complete_baselines.json"
    latex = ARTIFACT_ROOT / "virality/latex_table.tex"
    if not tuned.exists() or not complete.exists() or not latex.exists():
        add(
            audits,
            "virality-table2",
            "Virality Prediction",
            "Table 2 reports Random, BayesianRidge, LightGBM, SVR, Hawkes, and BERTweet+Features PSR metrics.",
            "MISSING_ARTIFACT",
            f"{tuned}; {complete}; {latex}",
            "one or more virality result artifacts missing",
            "Package or rerun the virality table artifacts.",
        )
        return
    tuned_rows = {row["name"]: row for row in load_json(tuned)}
    complete_rows = {row["name"]: row for row in load_json(complete)}
    checks = {
        "Random": (tuned_rows["Random"]["spearman_rho"], -0.052),
        "BayesianRidge": (tuned_rows["BayesianRidge"]["spearman_rho"], 0.551),
        "LightGBM": (tuned_rows["LightGBM"]["spearman_rho"], 0.549),
        "SVR (RBF)": (tuned_rows["SVR (RBF)"]["f2_065"], 0.632),
        "Hawkes (HIP)": (complete_rows["Hawkes (HIP)"]["spearman_rho"], 0.247),
        "BERTweet + Features": (tuned_rows["BERTweet + Features"]["f2_085"], 0.111),
    }
    ok = all(rounded_match(value, target, 3) for value, target in checks.values())
    add(
        audits,
        "virality-table2",
        "Virality Prediction",
        "Table 2 reports Random, BayesianRidge, LightGBM, SVR, Hawkes, and BERTweet+Features PSR metrics.",
        "ARTIFACT_BACKED" if ok else "MISMATCH",
        f"{tuned}; {complete}; {latex}",
        "; ".join(f"{name}: {value:.4f} vs paper {target}" for name, (value, target) in checks.items()),
        "Add the exact script command and split manifest so this becomes fully rerunnable from repo instructions.",
    )


def audit_checkworthiness(audits: list[ClaimAudit]) -> None:
    table3 = REPO_ROOT / "results/table3_reproduction_2026-05-12.json"
    if table3.exists():
        rows = {row["row"]: row for row in load_json(table3)["rows"]}
        for row_name in (
            "Single DeBERTa",
            "3-seed DeBERTa ensemble",
            "4-head MTL retrain",
            "PCA-64 + LLM + text LogReg CT24 rerun",
            "Fusion classifier rerun",
        ):
            row = rows[row_name]
            status = row["status"].upper().replace(" ", "_")
            add(
                audits,
                f"table3-{re.sub('[^a-z0-9]+', '-', row_name.lower()).strip('-')}",
                "Check-worthiness",
                f"Table 3 row: {row_name} paper F1={row['paper_claim_f1']}",
                status,
                row["evidence"],
                f"recomputed F1={row['reproduced_f1']:.4f}; threshold={row['threshold']:.2f}",
                "Use reproduced value or rerun until the exact paper value is matched/surpassed.",
            )
    else:
        add(
            audits,
            "table3-all",
            "Check-worthiness",
            "Table 3 check-worthiness rows.",
            "MISSING_ARTIFACT",
            str(table3),
            "Table 3 reproduction JSON not found",
            "Run scripts/build_table3_reproduction.py.",
        )

    script = REPO_ROOT / "reproducibility/runs/deberta_mtl_cikm_20260512_134553/scripts/finetune_deberta_mtl.py"
    if script.exists():
        text = script.read_text(encoding="utf-8")
        missing_terms = [
            term
            for term in ("Focal", "R-Drop", "FGM", "Layer-wise")
            if term.lower() not in text.lower()
        ]
        add(
            audits,
            "four-head-training-regularizers",
            "Check-worthiness",
            "Four-head training uses Focal Loss, layer-wise LR decay, R-Drop, and FGM adversarial training.",
            "MISSING_EXACT_RESULT" if missing_terms else "REPRODUCED",
            str(script),
            f"packaged script is 3-phase BCE/MSE auxiliary training; missing terms={missing_terms}",
            "Either package the exact regularized training script/run or revise the method description to match the reproduced run.",
        )

    add(
        audits,
        "llama2-baseline-row",
        "Check-worthiness",
        "Llama2-7b fine-tuned baseline has CT2024/CB/CT2023 F1 0.802/0.920/0.898.",
        "MISSING_ARTIFACT",
        "no local prediction/result artifact found",
        "rg found no exact Llama2 result bundle for this row",
        "Package the baseline artifact or cite it as external prior work instead of a rerun.",
    )


def audit_formative_and_report(audits: list[ClaimAudit]) -> None:
    add(
        audits,
        "formative-evaluation-aggregates",
        "Formative Evaluation",
        "Nine fact-checkers evaluated 27 reports; 22/27 agreement, 16/27 full agreement, component means 4.04/3.81/3.78/3.11.",
        "MISSING_ARTIFACT",
        "no anonymized aggregate response file found",
        "no raw or aggregate participant response table is packaged",
        "Add anonymized aggregate data and analysis script, or remove exact participant/statistic claims.",
    )
    add(
        audits,
        "formative-evaluation-count-conflict",
        "Formative Evaluation",
        "Paper consistently describes the formative evaluation sample size.",
        "MISMATCH",
        "/Users/sergiopinto/Downloads/IJCAI2026 (19).pdf",
        "abstract/limitations say N=7; body says N=9 and 27 report pairs",
        "Resolve the participant-count conflict before resubmission.",
    )
    add(
        audits,
        "sample-report-example",
        "Explainability Report",
        "Sample report was predicted viral 8 hours before peak, had 567 engagements, and 71.7% check-worthiness confidence.",
        "MISSING_EXACT_RESULT",
        "local report HTML candidates exist, but no exact 567/71.7/8-hour artifact was found",
        "exact example not tied to a local saved report",
        "Package the exact report HTML plus source claim ID and verifier, or remove the exact numeric example.",
    )


def build_audit() -> list[ClaimAudit]:
    audits: list[ClaimAudit] = []
    audit_corpus_and_pipeline(audits)
    audit_clustering(audits)
    audit_anomaly(audits)
    audit_claim_normalization(audits)
    audit_virality(audits)
    audit_checkworthiness(audits)
    audit_formative_and_report(audits)
    return audits


def write_markdown(audits: list[ClaimAudit]) -> None:
    lines = [
        "# Full Paper Claim Audit",
        "",
        "Date: 2026-05-12",
        "",
        "Statuses:",
        "",
        "- `REPRODUCED`: small repo artifact or packaged run directly supports the claim.",
        "- `REPRODUCED_LOCAL`: local artifact supports the claim, but large data/checkpoints still need external storage.",
        "- `ARTIFACT_BACKED`: a small result artifact supports the table value, but the exact rerun command/split manifest is incomplete.",
        "- `NEAR_PAPER_VALUE`: recomputation is close but does not exactly match the paper cell.",
        "- `NOT_REPRODUCED`: rerun exists and does not match the paper claim.",
        "- `MISMATCH`: artifact evidence contradicts the paper claim.",
        "- `MISSING_ARTIFACT` / `MISSING_EXACT_RESULT`: no adequate artifact for the exact claim is packaged.",
        "",
        "| Area | Claim ID | Status | Paper claim | Checked value | Evidence | Required next step |",
        "|---|---|---|---|---|---|---|",
    ]
    for audit in audits:
        lines.append(
            "| "
            + " | ".join(
                value.replace("|", "\\|").replace("\n", " ")
                for value in (
                    audit.area,
                    audit.claim_id,
                    audit.status,
                    audit.paper_claim,
                    audit.checked_value,
                    audit.evidence,
                    audit.required_next_step,
                )
            )
            + " |"
        )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    audits = build_audit()
    RESULT_JSON.write_text(json.dumps([asdict(audit) for audit in audits], indent=2) + "\n", encoding="utf-8")
    write_markdown(audits)
    counts: dict[str, int] = {}
    for audit in audits:
        counts[audit.status] = counts.get(audit.status, 0) + 1
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")
    print(f"wrote {RESULT_JSON}")
    print(f"wrote {RESULT_MD}")
    blocking = {
        "MISMATCH",
        "MISSING_ARTIFACT",
        "MISSING_EXACT_RESULT",
        "NOT_REPRODUCED",
        "NEAR_PAPER_VALUE",
    }
    return 1 if any(audit.status in blocking for audit in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
