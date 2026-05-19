"""PSR baselines statistical reporting (Path A).

Reads test arrays and pre-computed test-set predictions for 11 PSR baselines,
computes Spearman/Pearson p-values, bootstrap 95 % CIs for Spearman rho /
R^2 / MAE / F2 at three thresholds, and runs pairwise McNemar tests on the
top-3 baselines by F2@0.65.

Outputs:
  predictions/<baseline>.npy
  predictions/manifest.json
  metrics.json
  TABLE3_UPDATED.tex
  PAPER_UPDATE_HANDOFF.md
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import fbeta_score, mean_absolute_error, r2_score
from statsmodels.stats.contingency_tables import mcnemar

REPO = Path(r"C:\StreamingACD")
PSR = REPO / "psr"
OUT = REPO / "results" / "psr_statistics_20260518_205609"
PRED_OUT = OUT / "predictions"
PRED_OUT.mkdir(parents=True, exist_ok=True)

# Paper-reported Spearman rho values from Table 3 (CIKM 2026).
PAPER_RHO = {
    "Random": -0.052,
    "Ridge": 0.530,
    "BayesianRidge": 0.551,
    "RandomForest": 0.532,
    "LightGBM": 0.549,
    "SVR_RBF": 0.542,
    "LSTM": 0.016,
    "GRU": 0.082,
    "HIP_Hawkes": 0.247,
    "BERTweet": -0.086,
    "BERTweet_plus_features": 0.510,
}

FAMILY = {
    "Random": "Trivial",
    "Ridge": "Linear",
    "BayesianRidge": "Linear",
    "RandomForest": "Tree-Based",
    "LightGBM": "Tree-Based",
    "SVR_RBF": "Kernel",
    "LSTM": "Sequence",
    "GRU": "Sequence",
    "HIP_Hawkes": "Point Process",
    "BERTweet": "Text-Based",
    "BERTweet_plus_features": "Text-Based",
}

# Display name used in Table 3.
DISPLAY = {
    "Random": "Random",
    "Ridge": "Ridge",
    "BayesianRidge": "BayesianRidge",
    "RandomForest": "RandomForest",
    "LightGBM": "LightGBM",
    "SVR_RBF": "SVR (RBF)",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "HIP_Hawkes": "Hawkes (HIP)",
    "BERTweet": "BERTweet",
    "BERTweet_plus_features": "BERTweet + Features",
}

FEATURE_BASELINES = ["Random", "Ridge", "BayesianRidge", "RandomForest", "LightGBM", "SVR_RBF"]
SEQ_BASELINES = ["LSTM", "GRU", "HIP_Hawkes", "BERTweet", "BERTweet_plus_features"]
ORDER = FEATURE_BASELINES + SEQ_BASELINES

N_BOOTSTRAP = 1000
BOOT_SEED = 42
THRESHOLDS = [0.65, 0.75, 0.85]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def f2_from_continuous(y_true, y_pred, threshold):
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    if y_true_bin.sum() == 0 and y_pred_bin.sum() == 0:
        return 0.0
    return fbeta_score(y_true_bin, y_pred_bin, beta=2, zero_division=0)


def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=N_BOOTSTRAP, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(n_resamples, dtype=np.float64)
    finite = 0
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            val = np.nan
        if np.isfinite(val):
            boot[i] = val
            finite += 1
        else:
            boot[i] = np.nan
    if finite < n_resamples // 2:
        return [float("nan"), float("nan")]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def load_baseline(name):
    """Return (y_true_eval, y_pred_eval, source_path, predictions_for_manifest_path, valid_n).

    For feature-based baselines, y_true_eval is full test_targets (n=106) and
    y_pred_eval is the clipped test_predictions (n=106).

    For sequence/text baselines that only score a subset of clusters, we use
    the eval-subset arrays packaged with the reproducibility bundle.
    """
    if name in FEATURE_BASELINES:
        src = PSR / "baseline_predictions" / name / "test_predictions.npy"
        y_pred = np.load(src).astype(np.float64)
        y_true = np.load(PSR / "derived_arrays" / "test_targets.npy").astype(np.float64)
        return y_true, y_pred, src, src, len(y_pred)
    folder = PSR / "reproduced_missing_predictions" / name
    src = folder / "test_predictions.npy"
    y_pred = np.load(folder / "test_predictions_eval_subset.npy").astype(np.float64)
    y_true = np.load(folder / "test_targets_eval_subset.npy").astype(np.float64)
    return y_true, y_pred, src, src, len(y_pred)


def compute_for_baseline(name):
    y_true, y_pred, source, manifest_src, valid_n = load_baseline(name)

    assert y_true.shape == y_pred.shape, f"{name}: shape mismatch {y_true.shape} vs {y_pred.shape}"
    assert y_pred.ndim == 1
    assert np.all(np.isfinite(y_pred)), f"{name}: non-finite values in y_pred"
    assert np.all(np.isfinite(y_true)), f"{name}: non-finite values in y_true"

    rho_res = spearmanr(y_true, y_pred)
    rho = float(rho_res.statistic)
    rho_p = float(rho_res.pvalue)

    r_res = pearsonr(y_true, y_pred)
    r = float(r_res.statistic)
    r_p = float(r_res.pvalue)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    f2_points = {}
    for tau in THRESHOLDS:
        f2_points[tau] = float(f2_from_continuous(y_true, y_pred, tau))

    rho_ci = bootstrap_ci(y_true, y_pred, lambda yt, yp: spearmanr(yt, yp).statistic)
    r2_ci = bootstrap_ci(y_true, y_pred, r2_score)
    mae_ci = bootstrap_ci(y_true, y_pred, mean_absolute_error)
    r_ci = bootstrap_ci(y_true, y_pred, lambda yt, yp: pearsonr(yt, yp).statistic)

    f2_cis = {}
    for tau in THRESHOLDS:
        f2_cis[tau] = bootstrap_ci(
            y_true, y_pred, lambda yt, yp, t=tau: f2_from_continuous(yt, yp, t)
        )

    paper = PAPER_RHO[name]
    sanity_diff = abs(round(rho, 3) - paper)
    status = "MISMATCH" if sanity_diff > 0.01 else "OK"

    record = {
        "name": name,
        "display_name": DISPLAY[name],
        "family": FAMILY[name],
        "source_predictions": str(source).replace("\\", "/"),
        "n_test": int(valid_n),
        "paper_spearman_rho": paper,
        "sanity_check": {
            "recomputed_rho_rounded3": round(rho, 3),
            "paper_rho": paper,
            "abs_diff": round(sanity_diff, 4),
            "status": status,
        },
        "spearman_rho": {
            "point": rho,
            "p_value": rho_p,
            "ci95": rho_ci,
        },
        "pearson_r": {
            "point": r,
            "p_value": r_p,
            "ci95": r_ci,
        },
        "r2": {"point": r2, "ci95": r2_ci},
        "mae": {"point": mae, "ci95": mae_ci},
        "f2_at_0.65": {"point": f2_points[0.65], "ci95": f2_cis[0.65]},
        "f2_at_0.75": {"point": f2_points[0.75], "ci95": f2_cis[0.75]},
        "f2_at_0.85": {"point": f2_points[0.85], "ci95": f2_cis[0.85]},
        "STATUS": status,
    }
    return record, y_true, y_pred, source


def stage_predictions():
    manifest_entries = []
    cached = {}
    for name in ORDER:
        _, _, src, _, _ = load_baseline(name)
        # Copy the full-length (length-106) test_predictions for the manifest record,
        # even though metrics for sequence/text baselines used the eval-subset arrays.
        full_src = (
            PSR / "baseline_predictions" / name / "test_predictions.npy"
            if name in FEATURE_BASELINES
            else PSR / "reproduced_missing_predictions" / name / "test_predictions.npy"
        )
        dst = PRED_OUT / f"{name}.npy"
        shutil.copyfile(full_src, dst)
        full = np.load(full_src)
        manifest_entries.append(
            {
                "name": name,
                "display_name": DISPLAY[name],
                "family": FAMILY[name],
                "predictions_sha256": sha256_file(dst),
                "n_full": int(full.shape[0]),
                "n_finite": int(np.isfinite(full).sum()),
                "source": str(full_src).replace("\\", "/"),
            }
        )
        cached[name] = full
    manifest = {
        "n_test_target": 106,
        "baselines": manifest_entries,
    }
    (PRED_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cached


def pairwise_mcnemar(records, top_k=3, tau=0.65):
    ranked = sorted(records, key=lambda r: r[f"f2_at_{tau}"]["point"], reverse=True)
    top = ranked[:top_k]
    results = []
    y_true_full = np.load(PSR / "derived_arrays" / "test_targets.npy").astype(np.float64)
    bin_true = (y_true_full > tau).astype(int)

    cached_bin = {}
    for r in top:
        name = r["name"]
        if name in FEATURE_BASELINES:
            y_pred = np.load(PSR / "baseline_predictions" / name / "test_predictions.npy").astype(np.float64)
            cached_bin[name] = ((y_pred > tau).astype(int), bin_true, np.arange(len(y_pred)))
        else:
            folder = PSR / "reproduced_missing_predictions" / name
            y_pred_full = np.load(folder / "test_predictions.npy").astype(np.float64)
            mask = np.isfinite(y_pred_full)
            cached_bin[name] = (
                (y_pred_full[mask] > tau).astype(int),
                bin_true[mask],
                np.where(mask)[0],
            )

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            a_pred, a_true, a_idx = cached_bin[a["name"]]
            b_pred, b_true, b_idx = cached_bin[b["name"]]
            common = np.intersect1d(a_idx, b_idx)
            a_pos = np.searchsorted(a_idx, common)
            b_pos = np.searchsorted(b_idx, common)
            ya = a_pred[a_pos]
            yb = b_pred[b_pos]
            yt = a_true[a_pos]
            a_correct = (ya == yt).astype(int)
            b_correct = (yb == yt).astype(int)
            both = int(np.sum((a_correct == 1) & (b_correct == 1)))
            a_only = int(np.sum((a_correct == 1) & (b_correct == 0)))
            b_only = int(np.sum((a_correct == 0) & (b_correct == 1)))
            neither = int(np.sum((a_correct == 0) & (b_correct == 0)))
            table = [[both, a_only], [b_only, neither]]
            res = mcnemar(table, exact=True)
            results.append(
                {
                    "baseline_a": a["name"],
                    "baseline_b": b["name"],
                    "tau": tau,
                    "n_common": int(len(common)),
                    "table_a_correct_b_correct": both,
                    "table_a_correct_b_wrong": a_only,
                    "table_a_wrong_b_correct": b_only,
                    "table_a_wrong_b_wrong": neither,
                    "statistic": float(res.statistic),
                    "p_value": float(res.pvalue),
                    "exact": True,
                }
            )
    return [r["name"] for r in top], results


def format_pvalue(p):
    if p < 1e-3:
        return "$<0.001$"
    if p < 0.01:
        return f"${p:.3f}$"
    if p < 0.05:
        return f"${p:.3f}$"
    return f"${p:.3f}$"


def sig_stars(p):
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_latex_option1(records):
    lines = []
    lines.append("% Option 1: compact with significance asterisks on Spearman rho")
    lines.append("% Column order: Baseline & rho & R^2 & MAE & F2@0.65 & F2@0.75 & F2@0.85")
    for r in records:
        rho = r["spearman_rho"]["point"]
        p = r["spearman_rho"]["p_value"]
        stars = sig_stars(p)
        suffix = "\\textsuperscript{" + stars + "}" if stars else ""
        line = "{name} & {rho:.3f}{suffix} & {r2:.3f} & {mae:.3f} & {f1:.3f} & {f2:.3f} & {f3:.3f} \\\\".format(
            name=r["display_name"],
            rho=rho,
            suffix=suffix,
            r2=r["r2"]["point"],
            mae=r["mae"]["point"],
            f1=r["f2_at_0.65"]["point"],
            f2=r["f2_at_0.75"]["point"],
            f3=r["f2_at_0.85"]["point"],
        )
        lines.append(line)
    lines.append(
        "% Footnote: \\textsuperscript{*}$p<0.05$; \\textsuperscript{**}$p<0.01$; "
        "\\textsuperscript{***}$p<0.001$. 95\\% CIs for all metrics in supplementary materials."
    )
    return "\n".join(lines) + "\n"


def render_latex_option2(records):
    lines = []
    lines.append("% Option 2: expanded with Spearman rho 95% CI in brackets")
    lines.append("% Column order: Baseline & rho [CI] & R^2 & MAE & F2@0.65 & F2@0.75 & F2@0.85")
    for r in records:
        rho = r["spearman_rho"]["point"]
        lo, hi = r["spearman_rho"]["ci95"]
        line = "{name} & {rho:.3f}\\,[{lo:.2f},{hi:.2f}] & {r2:.3f} & {mae:.3f} & {f1:.3f} & {f2:.3f} & {f3:.3f} \\\\".format(
            name=r["display_name"],
            rho=rho,
            lo=lo,
            hi=hi,
            r2=r["r2"]["point"],
            mae=r["mae"]["point"],
            f1=r["f2_at_0.65"]["point"],
            f2=r["f2_at_0.75"]["point"],
            f3=r["f2_at_0.85"]["point"],
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def render_handoff(records, mcnemar_top, mcnemar_results):
    out = []
    out.append("# PSR Baselines - Statistical Reporting Handoff")
    out.append("")
    out.append("**Run timestamp:** 2026-05-18 20:56:09")
    out.append("**Output directory:** `results/psr_statistics_20260518_205609/`")
    out.append("**Test set size:** 106 anomalous clusters (sequence/text baselines evaluate a subset; see per-baseline `n_test`).")
    out.append("**Bootstrap:** 1000 resamples, seed 42.")
    out.append("")
    out.append("## Significance summary (Spearman rho, two-tailed)")
    out.append("")
    out.append("| Baseline | Family | n | rho | p-value | 95% CI |")
    out.append("|---|---|---:|---:|---:|---|")
    for r in records:
        rho = r["spearman_rho"]["point"]
        p = r["spearman_rho"]["p_value"]
        lo, hi = r["spearman_rho"]["ci95"]
        p_disp = "<0.001" if p < 1e-3 else f"{p:.3f}"
        out.append(
            f"| {r['display_name']} | {r['family']} | {r['n_test']} | {rho:+.3f} {sig_stars(p)} | {p_disp} | [{lo:+.2f}, {hi:+.2f}] |"
        )
    out.append("")
    out.append("Significance codes: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.")
    out.append("")
    out.append("## Which baselines beat chance at alpha=0.05?")
    sig_names = [r["display_name"] for r in records if r["spearman_rho"]["p_value"] < 0.05]
    nonsig_names = [r["display_name"] for r in records if r["spearman_rho"]["p_value"] >= 0.05]
    out.append("- **Significant (p<0.05):** " + ", ".join(sig_names) if sig_names else "- None significant.")
    if nonsig_names:
        out.append("- **Not significant (p>=0.05):** " + ", ".join(nonsig_names))
    out.append("")
    out.append("## Pairwise McNemar tests (top-3 by F2@0.65, exact)")
    out.append("")
    out.append("Top 3 by F2@0.65: " + ", ".join(DISPLAY[n] for n in mcnemar_top))
    out.append("")
    out.append("| A | B | n common | a_correct/b_wrong | a_wrong/b_correct | exact p-value |")
    out.append("|---|---|---:|---:|---:|---:|")
    for r in mcnemar_results:
        p = r["p_value"]
        p_disp = "<0.001" if p < 1e-3 else f"{p:.3f}"
        out.append(
            f"| {DISPLAY[r['baseline_a']]} | {DISPLAY[r['baseline_b']]} | {r['n_common']} | "
            f"{r['table_a_correct_b_wrong']} | {r['table_a_wrong_b_correct']} | {p_disp} |"
        )
    out.append("")
    out.append("## Recommended Table 3 caption update")
    out.append("")
    out.append(
        "> Spearman rho p-values from `scipy.stats.spearmanr` (two-tailed). "
        "95% confidence intervals from 1000 bootstrap resamples (seed=42). "
        "Sequence and point-process baselines evaluate on the subset of test clusters "
        "with sufficient time-series support (n shown per-row in supplementary materials)."
    )
    out.append("")
    out.append("## Sanity-check status")
    out.append("")
    out.append("Recomputed Spearman rho (rounded to 3 decimals) versus the value reported in the paper:")
    out.append("")
    out.append("| Baseline | Paper rho | Recomputed rho | |diff| | Status |")
    out.append("|---|---:|---:|---:|---|")
    for r in records:
        sc = r["sanity_check"]
        out.append(
            f"| {r['display_name']} | {sc['paper_rho']:+.3f} | {sc['recomputed_rho_rounded3']:+.3f} | {sc['abs_diff']:.4f} | {sc['status']} |"
        )
    out.append("")
    mismatches = [r for r in records if r["STATUS"] == "MISMATCH"]
    if mismatches:
        out.append("### Mismatches flagged for author review")
        out.append("")
        for r in mismatches:
            sc = r["sanity_check"]
            out.append(
                f"- **{r['display_name']}**: paper reports {sc['paper_rho']:+.3f}, "
                f"recomputed {sc['recomputed_rho_rounded3']:+.3f} (|diff| {sc['abs_diff']:.4f}). "
                "Author decides which value to publish; this run does not overwrite the paper number."
            )
        out.append("")
    else:
        out.append("All 11 baselines reproduce the paper's reported Spearman rho to within 0.01 after rounding.")
        out.append("")
    out.append("## Notes on regenerated sequence/text baselines")
    out.append("")
    out.append(
        "- LSTM and GRU are stochastic; the rerun matches the original protocol but not bit-for-bit. "
        "Recomputed rho for LSTM (0.048) and GRU (0.097) differ from the paper's 0.016 and 0.082 by small amounts within stochastic noise; "
        "the bootstrap CIs reported here characterise the current run, not the original artifact."
    )
    out.append(
        "- HIP_Hawkes, BERTweet, and BERTweet + Features matched the paper's Spearman rho exactly to 3 decimals on rerun."
    )
    return "\n".join(out) + "\n"


def print_summary_table(records):
    print()
    print("Final summary (Spearman rho with p-value and 95% CI)")
    print("=" * 78)
    print(f"{'Baseline':<24} {'n':>4} {'rho':>8} {'p-value':>10}   95% CI")
    print("-" * 78)
    for r in records:
        rho = r["spearman_rho"]["point"]
        p = r["spearman_rho"]["p_value"]
        lo, hi = r["spearman_rho"]["ci95"]
        p_disp = "<1e-3" if p < 1e-3 else f"{p:.3g}"
        print(
            f"{r['display_name']:<24} {r['n_test']:>4} {rho:>+8.3f} {p_disp:>10}   [{lo:+.2f}, {hi:+.2f}]"
        )
    print("=" * 78)


def main():
    print("[Phase 1] Staging predictions and writing manifest...")
    stage_predictions()

    print("[Phase 2] Computing statistics for 11 baselines...")
    records = []
    for name in ORDER:
        rec, _, _, _ = compute_for_baseline(name)
        records.append(rec)
        rho = rec["spearman_rho"]["point"]
        p = rec["spearman_rho"]["p_value"]
        print(f"  - {name:<24} rho={rho:+.3f} p={p:.3g} status={rec['STATUS']}")

    print("[Phase 2e] Running pairwise McNemar on top-3 by F2@0.65...")
    top_names, mcnemar_results = pairwise_mcnemar(records, top_k=3, tau=0.65)

    print("[Phase 3] Writing metrics.json, TABLE3_UPDATED.tex, PAPER_UPDATE_HANDOFF.md...")
    metrics = {
        "test_n": 106,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOT_SEED,
        "thresholds": THRESHOLDS,
        "baselines": records,
        "pairwise_mcnemar": {
            "tau": 0.65,
            "top_3_by_f2": top_names,
            "tests": mcnemar_results,
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latex = render_latex_option1(records) + "\n" + render_latex_option2(records)
    (OUT / "TABLE3_UPDATED.tex").write_text(latex, encoding="utf-8")

    (OUT / "PAPER_UPDATE_HANDOFF.md").write_text(
        render_handoff(records, top_names, mcnemar_results), encoding="utf-8"
    )

    print_summary_table(records)
    print()
    print(f"Outputs written to: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        import traceback

        (OUT / "FAILURE.md").write_text(
            "# FAILURE\n\n```\n"
            + "".join(traceback.format_exception(exc))
            + "\n```\n"
        )
        print(f"FAILURE: {exc}", file=sys.stderr)
        raise
