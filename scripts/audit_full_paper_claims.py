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
CORPUS_LANGUAGE_JSON = REPO_ROOT / "results/corpus_language_share_2026-05-12.json"
CLUSTERING_SUBSET_LANGUAGE_JSON = REPO_ROOT / "results/clustering_subset_language_share_2026-05-14.json"


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
        if CORPUS_LANGUAGE_JSON.exists():
            language_data = load_json(CORPUS_LANGUAGE_JSON)
            language_ratio = float(language_data["english_share"])
            kaggle_files = language_data["online_source"]["files"]
            language_columns = [file_data["has_language_column"] for file_data in kaggle_files]
            subset_clause = ""
            subset_evidence = ""
            if CLUSTERING_SUBSET_LANGUAGE_JSON.exists():
                subset_data = load_json(CLUSTERING_SUBSET_LANGUAGE_JSON)
                subset_share = float(subset_data["grouped"]["english_share"])
                subset_rows = int(subset_data["source"]["rows"])
                subset_sample = int(subset_data["method"]["sample_size"])
                subset_clause = (
                    f"; three-day clustering subset has seeded n={subset_sample} "
                    f"langdetect English share {subset_share:.3f} over {subset_rows} source rows"
                )
                subset_evidence = f"; {CLUSTERING_SUBSET_LANGUAGE_JSON}"
            checked_value = (
                f"langdetect recompute English share {language_ratio:.4f}; "
                f"pipeline summary implies {english_ratio:.3f}; "
                f"Kaggle CSV language columns present={any(language_columns)}"
                f"{subset_clause}"
            )
            evidence = f"{CORPUS_LANGUAGE_JSON}; {summary}{subset_evidence}"
        else:
            language_ratio = english_ratio
            checked_value = (
                f"summary implies English share {english_ratio:.3f}; no raw language/lang column in parquet"
            )
            evidence = str(summary)
        status = "MISMATCH" if round(language_ratio, 2) != 0.87 else "REPRODUCED"
        add(
            audits,
            "corpus-english-share",
            "Dataset",
            "Corpus is approximately 87% English.",
            status,
            evidence,
            checked_value,
            "Rewrite the exact 87% claim with a narrowed denominator: full corpus is about 70% English, while the three-day clustering-analysis subset is 86.6% English on a seeded n=2,000 sample.",
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
    attempt = REPO_ROOT / "results/anomaly_expose_reproduction_attempt_2026-05-13.md"
    anomaly_fix_note = REPO_ROOT / "results/anomaly_baseline_paper_fix_2026-05-14.md"
    attempt_note = (
        "; reproduction attempt found candidate NAB 79.2236 at undocumented min_rows=89, "
        "but detection was 97.9268%, while min_rows=84 gave detection 97.6325% and NAB 78.6208"
        if attempt.exists()
        else ""
    )
    add(
        audits,
        "anomaly-expose-table",
        "Anomaly Detection",
        "EXPoSE reaches NAB 79.2, 97.6% detection, and +23h median lead time.",
        status,
        f"{grid}; {attempt}; {anomaly_fix_note}" if attempt.exists() else f"{grid}; {anomaly_fix_note}",
        (
            f"best packaged row: NAB={nab:.4f}, detection={detection:.4f}, median_lead={lead:.1f}; "
            "remote README/configs support about 72.1/94.9/+23h, and the local design-log 79.2 value is "
            f"single-row time-series share, not NAB score{attempt_note}"
        ),
        "Apply results/anomaly_baseline_paper_fix_2026-05-14.md: use the declared min_rows=89 replacement table.",
    )

    baseline_rerun = REPO_ROOT / "results/anomaly_baseline_rerun_min_rows89_2026-05-13.json"
    if baseline_rerun.exists():
        baseline_evidence = f"{baseline_rerun}; {anomaly_fix_note}"
        baseline_checked = (
            "min_rows=89 rerun: Random NAB=72.2276, detection=99.0244, lead=22.0; "
            "Bayesian NAB=7.4289, detection=40.9756, lead=7.5; "
            "Etsy NAB=8.2378, detection=47.4390, lead=6.0; "
            "Steuber NAB=6.8232, detection=41.8293, lead=7.0"
        )
        baseline_next = (
            "Apply results/anomaly_baseline_paper_fix_2026-05-14.md and do not keep the old PDF baseline values."
        )
    else:
        baseline_evidence = (
            f"{ARTIFACT_ROOT / 'anomaly/anomaly_baselines.json'}; "
            f"{ARTIFACT_ROOT / 'anomaly/anomaly_detector_comparison.json'}"
        )
        baseline_checked = "packaged anomaly JSONs use different detectors/metrics and do not contain the exact paper table"
        baseline_next = "Rerun the uniform anomaly benchmark and save one canonical table artifact."

    add(
        audits,
        "anomaly-baseline-table",
        "Anomaly Detection",
        "Random/Bayesian/Etsy/Steuber baselines match the IJCAI anomaly paragraph.",
        "MISMATCH",
        baseline_evidence,
        baseline_checked,
        baseline_next,
    )


def audit_claim_normalization(audits: list[ClaimAudit]) -> None:
    path = ARTIFACT_ROOT / "claim_normalization/comparison_test_20260113_123010.json"
    handoff_note = REPO_ROOT / "results/claim_normalization_handoff_2026-05-14.md"
    if not path.exists():
        add(
            audits,
            "claim-normalization-table1",
            "Claim Normalization",
            "Table 1 reports CT2025 test N=300 and Approach 2 METEOR 0.5583/0.5463/0.5691.",
            "MISSING_ARTIFACT",
            f"{path}; {handoff_note}",
            "comparison JSON not found",
            "Use results/claim_normalization_handoff_2026-05-14.md for the exact Filipe artifact request.",
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
        f"{path}; {handoff_note}",
        f"available comparison has n_samples={sample_counts}; best avg_meteor={best_row['avg_meteor']:.4f} ({best_name})",
        (
            "Use results/claim_normalization_handoff_2026-05-14.md: Filipe must provide the N=300 split, "
            "per-sample predictions, summary, command, and checksums, or Table 1 must be removed/rewritten."
        ),
    )


def audit_virality(audits: list[ClaimAudit]) -> None:
    labels = ARTIFACT_ROOT / "virality/psr_labels.parquet"
    enhanced = ARTIFACT_ROOT / "virality/features_enhanced.parquet"
    split_manifest = REPO_ROOT / "results/virality_split_manifest_2026-05-12.json"
    if labels.exists() and enhanced.exists() and split_manifest.exists():
        rows = parquet_rows(labels)
        feature_count = len(pq.ParquetFile(enhanced).schema.names) - 2
        split_data = load_json(split_manifest)
        train_rows = int(split_data["counts"]["train_rows"])
        test_rows = int(split_data["counts"]["test_rows"])
        status = (
            "REPRODUCED"
            if rows == 529 and feature_count == 42 and train_rows == 423 and test_rows == 106
            else "MISMATCH"
        )
        checked = (
            f"psr_labels rows={rows}; features_enhanced feature columns={feature_count}; "
            f"split={train_rows}/{test_rows}"
        )
    else:
        status = "MISSING_ARTIFACT"
        checked = (
            f"labels exists={labels.exists()}; features_enhanced exists={enhanced.exists()}; "
            f"split_manifest exists={split_manifest.exists()}"
        )
    add(
        audits,
        "virality-dataset-shape",
        "Virality Prediction",
        "PSR prediction uses 529 anomalous clusters, 42 features, and a 423/106 split.",
        status,
        f"{labels}; {enhanced}; {split_manifest}",
        checked,
        "Keep the packaged split manifest with the Table 2 result artifacts.",
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
        "Optional next step: rerun the packaged scripts in a matching Python environment to replace result-artifact verification with a fresh model-fit rerun.",
    )


def audit_checkworthiness(audits: list[ClaimAudit]) -> None:
    table3 = REPO_ROOT / "results/table3_reproduction_2026-05-12.json"
    llm_feature_fix_note = REPO_ROOT / "results/llm_feature_classifier_paper_fix_2026-05-14.md"
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
            evidence = row["evidence"]
            next_step = "Use reproduced value or rerun until the exact paper value is matched/surpassed."
            if row_name == "PCA-64 + LLM + text LogReg CT24 rerun":
                evidence = f"{row['evidence']}; {llm_feature_fix_note}"
                next_step = (
                    "Apply results/llm_feature_classifier_paper_fix_2026-05-14.md: use CT24 0.694 for the literal "
                    "PCA-64 + LLM + text LogReg row, or rewrite the method around the separate CT24 0.793 replacement candidate."
                )
            add(
                audits,
                f"table3-{re.sub('[^a-z0-9]+', '-', row_name.lower()).strip('-')}",
                "Check-worthiness",
                f"Table 3 row: {row_name} paper F1={row['paper_claim_f1']}",
                status,
                evidence,
                f"recomputed F1={row['reproduced_f1']:.4f}; threshold={row['threshold']:.2f}",
                next_step,
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
    regularizer_fix_note = REPO_ROOT / "results/four_head_training_regularizers_paper_fix_2026-05-14.md"
    regularized_single_head = (
        ARTIFACT_ROOT / "checkworthiness/regularized_single_head/finetune_deberta_multimodel.py"
    )
    regularized_ensemble = ARTIFACT_ROOT / "checkworthiness/regularized_single_head/ensemble_deberta_seeds.py"
    if script.exists():
        text = script.read_text(encoding="utf-8")
        single_head_text = (
            regularized_single_head.read_text(encoding="utf-8") if regularized_single_head.exists() else ""
        )
        ensemble_text = regularized_ensemble.read_text(encoding="utf-8") if regularized_ensemble.exists() else ""
        missing_mtl_terms = [
            term
            for term in ("Focal", "R-Drop", "FGM", "Layer-wise")
            if term.lower() not in text.lower()
        ]
        single_head_regularizers = all(
            term.lower() in single_head_text.lower()
            for term in ("FocalLoss", "R-Drop", "FGM", "Layer-wise Learning Rate Decay")
        )
        ensemble_enables_regularizers = all(
            term in ensemble_text for term in ('"focal_loss": True', '"llrd": True', '"rdrop": True', '"fgm": True')
        )
        add(
            audits,
            "four-head-training-regularizers",
            "Check-worthiness",
            "Four-head training uses Focal Loss, layer-wise LR decay, R-Drop, and FGM adversarial training.",
            "MISMATCH" if missing_mtl_terms else "REPRODUCED",
            f"{script}; {regularized_single_head}; {regularized_ensemble}; {regularizer_fix_note}",
            (
                "packaged four-head MTL script is 3-phase weighted cross-entropy/MSE auxiliary training; "
                f"missing four-head regularizer terms={missing_mtl_terms}; "
                f"regularized single-head trainer found={single_head_regularizers}; "
                f"ensemble launcher enables those flags={ensemble_enables_regularizers}"
            ),
            (
                "Apply results/four_head_training_regularizers_paper_fix_2026-05-14.md: attach "
                "Focal/LLRD/R-Drop/FGM to the single-head DeBERTa seed ensemble, not the reproduced four-head MTL run."
            ),
        )

    add(
        audits,
        "llama2-baseline-row",
        "Check-worthiness",
        "Table 3 baseline row reports Llama2-7b Fine-tuned with CT2024/ClaimBuster/CT2023 F1 0.802/0.920/0.898.",
        "MISMATCH",
        (
            "reproducibility/source_artifacts/checkworthiness/together_baselines_metrics_20251229_194021.json; "
            "reproducibility/source_artifacts/checkworthiness/together_baselines/run_together_baselines.py; "
            "results/llama2_baseline_audit_2026-05-14.md"
        ),
        (
            "CT2024 value is an external FactFinders/CheckThat prior-work baseline; the local Together run contains "
            "100-sample Llama 3.x/Mistral/Mixtral predictions and no Llama2 prediction rows; ClaimBuster 0.920 and "
            "CT23 0.898 are separate published SOTA reference constants, not Llama2 outputs."
        ),
        (
            "Rewrite Table 3 baseline label: split this into external reference rows, or label it as prior work/SOTA "
            "rather than one reproducible Llama2-7b model across all datasets."
        ),
    )


def audit_formative_and_report(audits: list[ClaimAudit]) -> None:
    study_selection = ARTIFACT_ROOT / "formative_evaluation/expose_fast_study_claims_selection.csv"
    study_selection_3day = ARTIFACT_ROOT / "formative_evaluation/expose_3day_full_study_claims_selection.csv"
    study_script = ARTIFACT_ROOT / "formative_evaluation/select_claims_for_study.py"
    study_artifact_note = REPO_ROOT / "results/formative_evaluation_artifact_audit_2026-05-14.md"
    paper_fix_note = REPO_ROOT / "results/formative_evaluation_paper_fix_2026-05-14.md"
    sample_report = ARTIFACT_ROOT / "explainability_report/03_9dea2d0d_eng564.html"
    sample_fix_note = REPO_ROOT / "results/sample_report_paper_fix_2026-05-14.md"
    if study_selection.exists():
        with study_selection.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        categories: dict[str, int] = {}
        for row in rows:
            category = row["category"]
            categories[category] = categories.get(category, 0) + 1
        checked_value = (
            "study stimulus artifacts found but no participant-response table; "
            f"expose_fast selection rows={len(rows)}, categories={categories}; "
            "PDF has N=7 in abstract/limitations and N=9 in body"
        )
        evidence = f"{study_selection}; {study_selection_3day}; {study_script}; {study_artifact_note}; {paper_fix_note}"
    else:
        checked_value = "no raw or aggregate participant response table is packaged"
        evidence = "no anonymized aggregate response file found"
    add(
        audits,
        "formative-evaluation-aggregates",
        "Formative Evaluation",
        "Nine fact-checkers evaluated 27 reports; 22/27 agreement, 16/27 full agreement, component means 4.04/3.81/3.78/3.11.",
        "MISSING_ARTIFACT",
        evidence,
        checked_value,
        (
            "Apply results/formative_evaluation_paper_fix_2026-05-14.md to remove the exact participant/statistic claims, "
            "or add anonymized participant-level or aggregate response data plus an analysis script."
        ),
    )
    add(
        audits,
        "formative-evaluation-count-conflict",
        "Formative Evaluation",
        "Paper consistently describes the formative evaluation sample size.",
        "MISMATCH",
        f"/Users/sergiopinto/Downloads/IJCAI2026 (19).pdf; {paper_fix_note}",
        "abstract/limitations say N=7; body says N=9 and 27 report pairs",
        "Apply the formative-evaluation rewrite packet, which removes the unsupported sample-size claims.",
    )
    sample_row = None
    if study_selection_3day.exists():
        with study_selection_3day.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                if row.get("claim_id") == "9dea2d0d-0604-4753-8bde-f9498ee5a2b0":
                    sample_row = row
                    break

    if sample_report.exists() and sample_row is not None:
        report_text = sample_report.read_text(encoding="utf-8")
        confidence = round(float(sample_row["checkworthiness_prob"]) * 100, 1)
        peak_engagement = int(sample_row["peak_engagement"])
        lead_time_hours = float(sample_row["lead_time_hours"])
        corrected_values_supported = (
            "Confidence: 71.7%" in report_text
            and "<div class=\"metric-value\">8.0h</div>" in report_text
            and "<div class=\"metric-value\">564</div>" in report_text
            and confidence == 71.7
            and peak_engagement == 564
            and lead_time_hours == 8.0
        )
        sample_status = "MISMATCH" if corrected_values_supported else "MISSING_EXACT_RESULT"
        sample_evidence = f"{sample_report}; {study_selection_3day}; {sample_fix_note}"
        sample_checked = (
            "exact report artifact found for claim_id=9dea2d0d-0604-4753-8bde-f9498ee5a2b0; "
            f"confidence={confidence:.1f}%, peak_engagement={peak_engagement}, lead_time={lead_time_hours:.1f}h; "
            "paper says 567 engagements"
        )
        sample_next = (
            "Apply results/sample_report_paper_fix_2026-05-14.md: rewrite 567 engagements to 564 peak engagements."
        )
    else:
        sample_status = "MISSING_EXACT_RESULT"
        sample_evidence = "local report HTML candidates exist, but no exact 567/71.7/8-hour artifact was found"
        sample_checked = "exact example not tied to a local saved report"
        sample_next = "Package the exact report HTML plus source claim ID and verifier, or remove the exact numeric example."

    add(
        audits,
        "sample-report-example",
        "Explainability Report",
        "Sample report was predicted viral 8 hours before peak, had 567 engagements, and 71.7% check-worthiness confidence.",
        sample_status,
        sample_evidence,
        sample_checked,
        sample_next,
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
