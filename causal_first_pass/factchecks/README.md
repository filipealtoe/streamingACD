---
tags:
  - fact-checking
  - evidence-extraction
  - politifact
  - benchmark
license: cc-by-nc-4.0
language:
  - en
pretty_name: PrimeFacts
dataset_name: primefacts
version: 1.0.0-alpha
size_categories:
  - 10K<n<100K
source_datasets:
  - PolitiFact
paper: "Premtim Sahitaj, Jawan Kolanowski, Ariana Sahitaj, Veronika Solopova, Max Upravitelev, Daniel Röder, Iffat Maab, Junichi Yamagishi, Sebastian Möller, Vera Schmitt, From Articles to Premises: Building PrimeFacts, an Extraction Methodology and Resource for Fact-Checking Evidence"
contact: sahitaj@tu-berlin.de
---

# PrimeFacts: A Resource for Fact-Checking Evidence

## Dataset Summary

PrimeFacts is a research corpus derived from PolitiFact (2007–2025) comprising 13,106 fact-checking articles enriched with structured annotations for claims, verdicts, authors, speakers, sources, and automatically extracted evidence spans. The dataset provides a foundation for studying automated evidence extraction, claim verification, and fact-checking automation.

Fact-checking articles encode rich supporting evidence and reasoning, yet this information remains largely inaccessible to automated systems due to unstructured presentation. PrimeFacts introduces a reproducible extraction pipeline using large language models (LLMs) to identify and rewrite cited evidence into context-independent premises. The resulting benchmark enables cross-article evidence retrieval, claim verification, and verdict classification tasks.

Only derived metadata and extracted evidence are included under the **CC BY-NC 4.0** license. Full article texts, author biographies, and speaker descriptions are not redistributed and are available upon request for research purposes only.

## Supported Tasks and Intended Use

PrimeFacts supports research in:

- Evidence extraction and decontextualization  
- Fact-checking and claim verification  
- Source attribution and hyperlink reasoning  
- Benchmarking retrieval and classification models on real-world fact-checks  

The dataset is intended for academic research in natural language processing, information retrieval, and automated verification.

## Dataset Structure

PrimeFacts consists of multiple JSONL files, each representing a distinct metadata layer of the fact-checking corpus.

**File:** `articles.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| url | string | Canonical PolitiFact URL |
| access | string (ISO 8601) | Access timestamp |
| label | string | Verdict label (e.g., true, false, barely-true) |
| tags | list[string] | Editorial topic tags |
| sources | list[object] | Cited source descriptions and URLs |
| statement | object | Original claim information (quote, date, source URL) |
| author | object | Author metadata (URL and date) |
| article.text | string | Excluded from release (text available only on request) |

**File:** `authors.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| url | string | Author profile URL |
| name | string | Full name |
| title | string | Editorial title or role |
| access | string (ISO 8601) | Access timestamp |
| summary | object | Excluded (bio text not redistributed) |

**File:** `speakers.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| url | string | Speaker profile URL |
| name | string | Display name |
| description | string | Excluded (description text not redistributed) |
| link | string | Reference link |
| access | string (ISO 8601) | Access timestamp |

**File:** `evidences.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| url | string | PolitiFact article URL |
| evidences | list[object] | Model-generated evidence statements |
| → start | int | Start index in article text |
| → end | int | End index in article text |
| → source | string | Source URL |

## License

This dataset is released under **Creative Commons Attribution – NonCommercial 4.0 International (CC BY-NC 4.0)**.

Only derived metadata and extracted evidence are covered by this license. Full article texts, author biographies, and speaker descriptions remain property of **PolitiFact** and are not redistributed. Access to full text is available on request for non-commercial academic research only.  

Users must attribute both *PolitiFact* as the original source and *PrimeFacts* as the derived dataset.

## Quality and Validation

A manual annotation study complemented automatic evaluation metrics:

- Sample size: 100 premises each for Decontextualization and Open Extraction  
- Annotators: Two independent fact-checking researchers  
- Metrics: Krippendorff’s α, observed agreement, macro-F1  

| Evaluation Aspect | Mode | Agreement | Krippendorff’s α | Macro-F1 |
|-------------------|------|------------|------------------|----------|
| Self-containedness | Decontextualization | 0.87 | 0.255 | — |
| Evidence type | Decontextualization | 0.58 | 0.441 | 0.859 |
| Self-containedness | Open Extraction | 0.835 | 0.474 | — |
| Evidence type | Open Extraction | 0.67 | 0.561 | 0.857 |


## Results Summary

Empirical evaluations demonstrate that:

- Decontextualized evidence (Decontextualization / Open Extraction) improves retrieval and verdict prediction.  
- Evidence rewritten into self-contained premises increases retrievability by ~30 % (MRR) and boosts verdict macro-F1 by 10–20 points.  
- Larger models yield more faithful, less redundant evidence extractions.  
- Cross-model consistency confirms robustness across verdict granularities and LLM architectures.  

These findings collectively indicate that fact-checking evidence can be systematically extracted and reused for automated verification.

## Citation

```bibtex
@inproceedings{sahitaj-2025,
  author    = {Premtim Sahitaj and Jawan Kolanowski and Ariana Sahitaj and Veronika Solopova and Max Upravitelev and Daniel Röder and Iffat Maab and Junichi Yamagishi and Sebastian Möller and Vera Schmitt},
  title     = {From Articles to Premises: Building PrimeFacts, an Extraction Methodology and Resource for Fact-Checking Evidence},
  booktitle = {Proceedings of the 2025 Joint International Conference on Language Resources and Evaluation},
  year      = {2025},
  pages     = {xx--yy},
  publisher = {ELRA},
  address   = {TBD},
  doi       = {TBD}
}
```