# Verify the CIKM 2026 paper values

- Author: Sérgio Pinto
- Timestamp: 2026-08-21 18:13 PDT (-0700)
- Reason: make every paper-facing numeric result traceable to executable evidence
  and expose unresolved cells without changing the manuscript.

## What changed

- Added a machine-readable manifest of the current camera-ready virality table.
- Added an executable verifier for all 11 rows and all 66 numeric cells.
- Freshly fits the six tabular baselines from the public feature matrix.
- Recomputes the sequence, Hawkes, and BERTweet rows from retained predictions.
- Reports retained tabular predictions as a diagnostic when a fresh fit differs.
- Added a text-free numerical feature bundle and fresh-fit command for the three
  PCA-64 + LLM + text-feature F1 cells.
- Sérgio Pinto, 2026-08-21 19:27 PDT: documented the retained per-benchmark
  threshold sweep and added a fixed CT24-development-threshold diagnostic so the
  reported reproduction and the stricter transfer check are both explicit.
- Sérgio Pinto, 2026-08-21 19:45 PDT: reconstructed the CT24 Fusion probability
  vector from its retained encoder and XGBoost components. The ClaimBuster and
  CT23 Fusion values remain linked to historical console output pending recovery
  of the January 17 prediction arrays or checkpoint bytes.
- Sérgio Pinto, 2026-08-21 19:58 PDT: recovered the Encoder Only CT24 and CT23
  probability vectors, added the original training/evaluation source snapshots,
  and rederived the ClaimBuster paper value from the unique confusion matrix
  consistent with the retained aggregate run evidence.
- Sérgio Pinto, 2026-08-21 20:10 PDT: corrected that ClaimBuster reconstruction
  to the 1,032-row public input and marked it unresolved. The arithmetic matches
  the author-recorded aggregate, but the historical prediction vector was not
  retained and is therefore not presented as direct reproduction.
- Sérgio Pinto, 2026-08-21 20:13 PDT: added a cell-by-cell audit of the paper's
  claim-normalization table. The retained 300-row metrics reproduce Approach 2 /
  Model A exactly; eight internal cells remain explicitly unresolved because
  their exact paper-run outputs were not recovered.
- Sérgio Pinto, 2026-08-21 20:18 PDT: added a Frontier-LLM aggregate audit.
  The reported confusion matrix, McNemar result, dimensional agreements, latency,
  and cost are arithmetically consistent; direct reproduction remains unresolved
  because the original 1,023 API responses were not recovered.
- Sérgio Pinto, 2026-08-21 20:20 PDT: made the main verifier fail explicitly
  for the missing Encoder Only ClaimBuster vector and the missing Fusion
  ClaimBuster/CT23 paper-run arrays or checkpoint, rather than only validating
  that their documentation boundary is internally consistent.
- Sérgio Pinto, 2026-08-21 20:23 PDT: recomputed the Table 3 exact McNemar tests
  from both the retained paper-run predictions and the fresh table-model fits.
  The retained vectors reproduce all three paper p-values, while the fresh
  RandomForest fit exposes one metric-cell and two paired-test inconsistencies.
- Sérgio Pinto, 2026-08-21 20:58 PDT: traced the RandomForest `0.530` Spearman
  cell to the retained commit vector and the remaining five cells to the fresh
  fit. The verifier now reproduces every numeric cell by its actual source but
  continues to fail the separate one-run consistency check. The historical
  vector manifest no longer claims an unrecovered generating command.
- Sérgio Pinto, 2026-08-21 20:29 PDT: extended the main verifier beyond table
  cells to the corpus, pipeline, claim/cluster mapping, and clustering-ablation
  values. The temporal scope is now stated exactly as 600 hourly windows over
  25 calendar dates, and the 87%/0.87 clustering result is tied to its 5,000-post
  sample.
- Sérgio Pinto, 2026-08-21 20:34 PDT: added an executable reconstruction of the
  cluster-72106 worked example. Eighteen fields reproduce from the public
  aggregate tables and four-head scores; the paper's 88-post detection count is
  the registry snapshot value, while the time series contains 80 posts through
  the stated detection timestamp.
- Sérgio Pinto, 2026-08-21 20:40 PDT: corrected the claim-normalization evidence
  boundary. The retained 300-row run numerically matches `0.5569`, but it uses
  `(10, 0.85, 0.5, 5, 0)` rather than the paper's
  `(10, 0.85, 0.65, 4, 2)`, so it is no longer labelled direct reproduction.
- Sérgio Pinto, 2026-08-21 20:40 PDT: added an executable audit of the reported
  86% shared CT24 errors. The retained vectors cannot be joined exactly because
  sentence IDs are missing from two of the three vector orders.
- Sérgio Pinto, 2026-08-21 20:44 PDT: extended verification from rounded table
  cells to protocol consistency. It now checks stream duration, canonical versus
  release units, EXPoSE parameters, detection-rate definitions, the actual
  pre-detection feature scope, split stratification, the MTL seed, and numerical
  side claims without retained evidence.
- Sérgio Pinto, 2026-08-21 20:51 PDT: added direct checks for the clustering
  endpoint values (`0.82`/`0.95` similarity and `97.2%`/`27.7%` yield) and made
  the unrecovered pure-retrieval `0.3937` and fine-tuned Model A `0.4427`
  claim-normalization scores explicit completeness failures.
- Sérgio Pinto, 2026-08-21 20:51 PDT: bound the four-head training counts
  (`22,402`, `17,003`, `5,399`, and `1,031`), exact class-weight ratio, and
  341-row test scope to the machine-readable parameter check.
- Sérgio Pinto, 2026-08-21 21:11 PDT: added a fresh single-item PSR latency
  benchmark over the released 42-feature matrix. It verifies the exact
  BayesianRidge and SVR fits by prediction hash and measures scaling plus
  prediction without batching; the separate four-head GPU latency trace remains
  independently classified.
- Sérgio Pinto, 2026-08-21 21:14 PDT: replaced the release verifier's fixed
  checksum-entry count with an explicit required-path set, so adding the PSR
  benchmark cannot produce a false failure while omission of any required
  evidence still does.
- Sérgio Pinto, 2026-08-21 21:14 PDT: separated the worked example's directly
  reproduced 88-post registry value from the temporal inconsistency that the
  time series contains 80 posts through the stated detection timestamp.
- Sérgio Pinto, 2026-08-21 21:20 PDT: verified the CT24 English dev-test file
  byte-for-byte against the official CheckThat! 2024 archive. The file has 318
  data rows, not 317; the retained four-head run selected on the separate
  1,031-row development split. Only hashes, schema, and counts are added.
- Sérgio Pinto, 2026-08-21 21:26 PDT: recovered the official CT24 test
  sentence-ID order and joined the LLM-feature, Encoder Only, and four-head
  vectors exactly. The claimed 86% does not reproduce under the explicit
  all-three, at-least-two, assignment-level, or pairwise overlap definitions.
- Sérgio Pinto, 2026-08-21 21:31 PDT: recomputed the CT24 development and test
  threshold sweeps. The development-selected threshold changes test F1 by
  `-0.0287`, directly supporting the approximate `0.03` magnitude; the audit
  also records that the reported `0.833` is the test-sweep optimum rather than
  the value at the development-selected threshold.
- Sérgio Pinto, 2026-08-21 21:31 PDT: parsed the retained four-head CUDA
  benchmark trace against its exact source. ClaimBuster and CT23 run at `6.09`
  and `4.56` ms/claim respectively as batch-amortized inference with batch size
  32, resolving the `<10 ms` claim within that measured boundary.
- Sérgio Pinto, 2026-08-21 21:36 PDT: traced the Table 4 published-baseline
  cells to primary sources. CT24 `0.802` is the FactFinders Llama2-7b result,
  ClaimBuster `0.920` is G2CW, and CT23 `0.898` is OpenFact GPT-3. The values
  are valid references, but the three-cell row is not one Llama2 model result.
- Sérgio Pinto, 2026-08-21 21:36 PDT: made the LLM-feature threshold protocol
  explicit. Its paper-facing values optimize F1 separately on each evaluation
  set; the packaged fixed-development-threshold diagnostic yields `0.687`,
  `0.881`, and `0.802` instead.
- Sérgio Pinto, 2026-08-21 21:40 PDT: added a portable reproduction of every
  Table 3 Spearman p-value, significance mark, and 95% bootstrap interval from
  the retained vectors. All 11 rows and all three exact McNemar tests pass.
- Sérgio Pinto, 2026-08-21 21:46 PDT: added one consolidated release-coverage
  audit. It verifies the aggregate scientific core and records that the public
  tree does not contain the manuscript's complete post-level, rationale,
  embedding, and 1,023-response bundle.
- Sérgio Pinto, 2026-08-21 21:46 PDT: added explicit checks for the retained MTL
  phase schedule and the historical cross-benchmark Fusion seed scope.
- Sérgio Pinto, 2026-08-21 21:53 PDT: replaced the root README's machine-local
  paths with the public global, Table 3, statistical, and release-coverage
  entry points.
- Sérgio Pinto, 2026-08-21 21:55 PDT: recovered the canonical 153.6 MB
  cluster-embedding matrix, recorded its 100,000 × 384 identity and checksum,
  and added optional external-asset verification. The file requires a GitHub
  release upload and is not committed as a normal Git blob.
- Sérgio Pinto, 2026-08-21 22:05 PDT: reran the public ClaimBuster input with
  the later locally retained seed-0 checkpoint. Its F1 is `0.9657` at threshold
  `0.65`, not the historical `0.9702`, so the audit records and rejects that
  candidate rather than substituting it for the missing paper-run vector.
- Sérgio Pinto, 2026-08-21 22:08 PDT: bound the full paper-value inventory to
  the exact 12-page camera-ready PDF using its byte size and SHA-256, with an
  optional verifier argument that does not redistribute the manuscript.
- Sérgio Pinto, 2026-08-21 22:18 PDT: corrected the Frontier-LLM report
  generator so it states that the unsupported temperature parameter is omitted,
  rather than describing the Opus request as deterministic, and bound the
  analysis script to the source checksum manifest.
- Sérgio Pinto, 2026-08-21 22:29 PDT: added an expected-asset manifest and a
  metadata-only GitHub release verifier. It authenticates the checkpoint and
  cluster embeddings by uploaded state, byte size, and SHA-256 without
  downloading either large binary.
- Sérgio Pinto, 2026-08-21 22:32 PDT: added an optional portable JSON report to
  the main verifier. Known failures are classified as missing historical
  evidence, manuscript-to-artifact mismatches, or release-scope mismatch;
  every other failure remains an unexpected artifact-integrity failure.
- Sérgio Pinto, 2026-08-21 23:43 PDT: recorded the verified publication of the
  canonical cluster embeddings and removed a machine-specific private archive
  path from the public claim-normalization run record.
- Sérgio Pinto, 2026-08-21 23:47 PDT: aligned the release-coverage audit with
  the verified public distribution. The embedding matrix remains outside Git
  while its GitHub Release tag, URL, byte size, and SHA-256 are recorded and
  checked as part of the public artifact.
- Sérgio Pinto, 2026-08-21 23:51 PDT: made the release-coverage and virality
  statistics commands accept their documented absolute `--output` paths, so
  reviewers can write generated reports outside a clean repository checkout.
- Sérgio Pinto, 2026-08-21 23:55 PDT: kept the virality prediction checksum
  manifest in strict `sha256sum` format by moving its authorship context to
  this change record; all retained prediction identities remain unchanged.
- Sérgio Pinto, 2026-08-21 23:59 PDT: marked the checksum-bound official CT23
  gold file as byte-preserved and recognized its CRLF terminators, preventing
  cross-platform checkout normalization without changing the source bytes.

## Verification

From the repository root:

```bash
uv run scripts/verify_cikm2026_virality_table.py
```
