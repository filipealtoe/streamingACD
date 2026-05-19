# Clustering Threshold Ablation Investigation

Date: 2026-05-19
Scope: read-only metadata investigation across `C:\StreamingACD` and `C:\Explainable_ACD`.
Question: Do artifacts exist documenting the selection of the streaming clustering threshold τ=0.65 with enough fidelity to support a §3 ablation summary in the CIKM paper?

Short answer: yes — Tier 1 data exists, already packaged in this repo and cross-referenced in the paper claim audit.

---

## Section A — Search execution log

### Repositories searched

- `C:\StreamingACD` (primary)
- `C:\Explainable_ACD` (secondary; CLAUDE.md identifies this as the original local directory that spawned StreamingACD)

### Filename glob patterns and counts

| Pattern | StreamingACD hits | Explainable_ACD hits |
|---|---:|---:|
| `**/*tau*` | 0 | 0 |
| `**/*threshold*` | 0 | 7 |
| `**/*ablation*` | 2 (LLM ablation only) | 7 |
| `**/*sensitivity*` | 0 | 0 |
| `**/*sweep*` | 0 | 6 |
| `**/*clustering*` | 0 | 12 |
| `**/*0.65*` / `**/*0_65*` | 0 | 0 |

### Content searches

- Grep for `clustering` / `threshold_ablation` / `cluster_statistics` / `tau` / `0.65` in StreamingACD `.md` and `.json` files: 20 file hits, with the primary evidence at `reproducibility/source_artifacts/clustering/cluster_statistics.json` and the paper claim audit at `results/full_paper_claim_audit_2026-05-12.{md,json}`.
- Grep for `0.65` and `similarity_threshold` in `C:\Explainable_ACD\src`: confirms `clusterer.similarity_threshold` is the runtime config knob and that `src/streaming/run_pipeline_with_report.py:127` logs a default of 0.65 (the YAML default at `src/pipeline/config/default.yaml:48` reads 0.75; the canonical-run summary does not log the value).
- Grep for τ=0.65 in design logs: appears in `design_approaches_2026-01-17_Sergio.md`, but in the context of PSR binarization (F2 score), not clustering — different parameter, same numeric value.

### Notes and exceptions

- The current `C:\Explainable_ACD\docs\current_clustering_technology.md` indicates that the production clustering algorithm evolved to a stateful Louvain-on-kNN variant (`link_threshold=0.8`); the τ=0.65 online-cosine threshold corresponds to the earlier clustering family that the CIKM paper actually describes. This investigation is scoped to that earlier family because that is what §3 of the paper documents.
- No permission or encoding errors during the search.
- Investigation wall-clock: approximately 10 minutes.

---

## Section B — Findings classification

### TIER 1 — Direct ablation data

**Artifact:** `C:\StreamingACD\reproducibility\source_artifacts\clustering\cluster_statistics.json`

- 6 threshold values evaluated: τ ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
- Per-threshold metrics: `n_clusters`, `singleton_pct`, `cluster_yield_pct`, `mean_intra_sim` (cosine-to-centroid, sampled), `std_intra_sim`, `mean_size`, `median_size`, `max_size`, plus a size-bucket distribution.
- Provenance chain:
  - Generating script: `C:\Explainable_ACD\experiments\scripts\ablation_clustering_threshold.py` (default output path `experiments/results/threshold_ablation/cluster_statistics.json`; default thresholds list matches the artifact exactly).
  - SHA-256 `fe365886750959553c8e3e8ec7ed0003f66bc2f6afb27ec98858572c1c7abc03` registered in `reproducibility/source_artifacts/sha256sums.txt` and `reproducibility/artifact_checksums_2026-05-11.md` under "Clustering threshold ablation", with the original path `/Users/sergiopinto/explainableACD/experiments/results/threshold_ablation/cluster_statistics.json`.
  - Cited as REPRODUCED in `results/full_paper_claim_audit_2026-05-12.{md,json}` under claim id `clustering-threshold-ablation`, matching the paper text "tau=0.65 with 0.87 mean similarity and 87% grouped".
  - Committed to StreamingACD git on 2026-05-12 by Sérgio Pinto (`2ceac4e docs(reproducibility): add full paper claim audit`).

### TIER 2 — Indirect evidence

- `C:\Explainable_ACD\experiments\results\clustering_benchmark\threshold_tuning_results.csv` and `benchmark_results.csv`: BCubed F1 sweeps over a two-stage hybrid clustering pipeline at `stage1_threshold` ∈ [0.75, 0.91] and `stage2_threshold` ∈ [0.70, 0.86]. These do not cover the single-stage τ=0.65 regime but provide BCubed-F1 evidence on the same annotation corpus. Useful background only.
- `C:\Explainable_ACD\experiments\results\clustering_benchmark\ground_truth_stats.json`: 1600-tweet annotated ground truth used for the BCubed sweep; could be reused if a stricter ablation needed BCubed metrics.

### TIER 3 — Code that could regenerate the ablation

- `C:\Explainable_ACD\experiments\scripts\ablation_clustering_threshold.py` — already produced the Tier 1 artifact; can be rerun on a larger sample if the §3 summary needs to scale beyond 5,000 tweets.
- `C:\Explainable_ACD\experiments\scripts\analyze_clustering_thresholds.py`, `tune_clustering_thresholds.py`, `optimize_threshold.py` — adjacent threshold-tuning scripts (mostly for the two-stage hybrid). Not needed if Tier 1 is accepted.

### TIER 4 — Design-decision documentation

- `C:\Explainable_ACD\docs\current_clustering_technology.md` — narrates the post-paper evolution to Louvain-kNN; not relevant to the paper-time decision.
- No standalone qualitative justification of τ=0.65 was found.

---

## Section C — Recommended path forward

**Action A — Cite existing numbers.** The Tier 1 artifact provides six threshold values with multiple downstream metrics and a clean provenance chain through the reproducibility manifest into the paper claim audit. Recommended action: add a one-paragraph ablation summary to §3 citing the existing numbers.

Caveat to preserve in the wording: the ablation was run on a **5,000-tweet sample** (per `cluster_statistics.json` `total_tweets: 5000`, with `mean_intra_sim` estimated over up to 200 multi-member clusters per threshold). The full pipeline run used to populate the paper headline tables (`streaming_full/2026-01-17_03-56`) does not log its similarity threshold in the summary, so the §3 sentence should attribute the ablation to the sample and frame τ=0.65 as the selected operating point rather than the only point evaluated on the full corpus. The paper claim audit explicitly flags this caveat: "Keep wording clear that this is the 5,000-tweet ablation unless full-run evidence is added."

---

## Section D — Specific evidence

### Threshold-vs-metric table from `cluster_statistics.json`

| τ | n_clusters | singleton % | yield % | mean intra-sim | std intra-sim | n clusters measured |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 227   | 62.6 | 97.2 | 0.818 | 0.074 | 85  |
| 0.60 | 456   | 70.8 | 93.5 | 0.853 | 0.069 | 133 |
| **0.65** | **845**   | **77.5** | **86.9** | **0.869** | **0.063** | **190** |
| 0.70 | 1,616 | 84.8 | 72.6 | 0.895 | 0.054 | 200 |
| 0.75 | 2,745 | 89.5 | 50.8 | 0.924 | 0.045 | 200 |
| 0.80 | 3,881 | 93.1 | 27.7 | 0.948 | 0.034 | 200 |

Sample size: 5,000 tweets drawn from `data/pipeline_output/expose_3day_full/2026-01-19_10-57/tweets.parquet` and re-embedded with `paraphrase-multilingual-mpnet-base-v2` (per the ablation script defaults).

`mean_intra_sim` is the mean cosine similarity between member embeddings and the cluster centroid, averaged across up to 200 sampled multi-member clusters per threshold.

### Provenance details

- Generator: `experiments/scripts/ablation_clustering_threshold.py` in the Explainable_ACD checkout.
- Default thresholds in the script: `[0.55, 0.60, 0.65, 0.70, 0.75]`; the persisted artifact additionally contains 0.80, indicating the run was invoked with `--thresholds 0.55 0.60 0.65 0.70 0.75 0.80` or equivalent.
- SHA-256: `fe365886750959553c8e3e8ec7ed0003f66bc2f6afb27ec98858572c1c7abc03`, manifested at `reproducibility/source_artifacts/sha256sums.txt`.
- Git commit landing the artifact in StreamingACD: `2ceac4e` (2026-05-12).
- Paper-claim audit row: `clustering-threshold-ablation` in `results/full_paper_claim_audit_2026-05-12.{md,json}`; status REPRODUCED.

### Draft §3 LaTeX snippet

A version that surfaces the trade-off explicitly:

```latex
We selected the streaming clustering threshold $\tau=0.65$ from an
ablation over $\tau \in \{0.55, 0.60, 0.65, 0.70, 0.75, 0.80\}$ on a
5{,}000-tweet sample of the election-period corpus. Increasing $\tau$
monotonically tightens clusters (mean intra-cluster cosine similarity
rises from $0.82$ at $\tau{=}0.55$ to $0.95$ at $\tau{=}0.80$) but
fragments the stream (cluster yield, i.e.\ the share of tweets joining a
multi-member cluster, falls from $97.2\%$ to $27.7\%$). At $\tau=0.65$,
mean intra-cluster similarity is $0.87$ and yield is $86.9\%$,
preserving the bulk of the streamed posts while keeping clusters
topically coherent.
```

A tighter one-sentence variant:

```latex
The clustering threshold $\tau=0.65$ was selected from an ablation
over $\tau \in [0.55, 0.80]$ on a 5{,}000-tweet sample as the point
that retained $86.9\%$ of posts in multi-member clusters while keeping
mean intra-cluster cosine similarity at $0.87$, balancing coherence
against fragmentation at higher thresholds and over-merging at lower ones.
```

Underlying artifact (for §3 footnote or appendix): `reproducibility/source_artifacts/clustering/cluster_statistics.json` (SHA-256 `fe365886750959553c8e3e8ec7ed0003f66bc2f6afb27ec98858572c1c7abc03`), generated by `experiments/scripts/ablation_clustering_threshold.py` in the Explainable_ACD repo.

### Honest gaps

- No manual coherence annotation (Coherent/Mixed/Uncertain precision) was found, even though the ablation script generates an HTML annotation interface — neither `annotation_interface.html` nor any annotation export JSON is present in either repo. If §3 wants to claim *precision* rather than only mean intra-cluster similarity, that gap would need filling.
- The canonical full-corpus pipeline run `streaming_full/2026-01-17_03-56` does not record the clustering similarity threshold in its summary JSON. The runtime default (`run_pipeline_with_report.py:127`) is 0.65 and the YAML default (`pipeline/config/default.yaml:48`) is 0.75, so the actual value used for the headline run cannot be reconstructed from the summary alone. This is a separate provenance issue from the ablation question and does not block citing the ablation numbers.

---

## Final classification

**Tier 1 (Direct ablation data).** Recommended action: **A — Cite existing numbers.** Use the table and one of the draft sentences above; preserve the 5,000-tweet sample caveat.
