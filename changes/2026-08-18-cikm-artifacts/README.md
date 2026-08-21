# CIKM 2026 Artifact Release Preparation

## Goal

Make the public 529-row virality dataset and paper-facing result artifacts discoverable and independently verifiable from a fresh
clone, while linking the repository to the accepted CIKM 2026 paper and conference.

## Scope

- add a focused CIKM 2026 entry point to the root README;
- document the dataset, schema, provenance, missingness, privacy boundary, and 529-versus-535 distinction;
- add release-facing checksums and citation metadata;
- add a public-only verifier with pinned inline dependencies;
- add a public-only fresh-fit reproducer for Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR;
- leave the underlying Parquet and result artifacts byte-for-byte unchanged.

Extended audit inputs, embeddings, model-specific reruns, and internal camera-ready decision documents remain local-only and are
protected from accidental staging.

## Non-goals

- no raw posts or user-level data;
- no conflation of the separate 535-claim registry with the 529-row virality dataset;
- no reconstructed Opus responses, participant responses, or omitted model checkpoint;
- no publication of temporal, anomaly, or embedding candidates without author privacy and source-rights approval;
- no repository license selection on behalf of the authors;
- no Overleaf edit, push, release, or repository-setting change as part of preparation.

## Goal proof

Normal entry point:

```bash
uv run scripts/verify_cikm2026_artifacts.py
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

Success oracle: both commands exit `0` and end with `VERDICT: PASS` using only files in the repository clone. The second command
must freshly fit the six numeric baselines and match the stored full-precision metrics within a declared `1e-12` tolerance.

Negative control: the same command with `--root` pointed at a copy containing a modified release artifact must exit non-zero and
print `VERDICT: FAIL`.

## Known limits

The proof establishes integrity and consistency of the released dataset, deterministic split, and stored paper-facing result
artifacts, plus fresh fits of Random, Ridge, BayesianRidge, RandomForest, LightGBM, and SVR. It does not establish that every other model can be freshly
trained from the public clone or that the complete streaming pipeline can be rerun without external inputs.

Extended local proof does not broaden the approved public boundary and is not required by either public entrypoint.

The ACM DOI is reserved in the camera-ready metadata but returned HTTP 404 during local preparation on 2026-08-18. The README
labels it as pre-publication rather than claiming that the paper page is already public. The official CIKM 2026 site returned
HTTP 200.

## Observed local proof (2026-08-18)

- focused public-artifact verifier: `18 passed, 0 failed`, `VERDICT: PASS`;
- fresh numeric fits: `6 reproduced, 0 mismatched`, `VERDICT: PASS`;
- corrupted result artifact: checksum and JSON checks failed, `VERDICT: FAIL`;
- zero-tolerance numerical control: BayesianRidge mismatch detected, `VERDICT: FAIL`;
- isolated copy with `EXPLAINABLE_ACD_ROOT` removed: both normal entrypoints passed;
- Ruff, Python byte-compilation, CFF 1.2 validation, Markdown local-link validation, and `git diff --check`: passed.
