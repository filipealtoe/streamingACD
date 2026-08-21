# Exclude User-Level Lookup Data

## Goal

Keep the public CIKM artifact focused on normalized claims, cluster-level scientific outputs, and aggregate features without
distributing a lookup table of account identifiers and handles.

## Scope

- remove the canonical run's `users.parquet` lookup table from the current repository tree;
- add an exact ignore rule to prevent accidental reintroduction;
- keep the normalized-claim registry, cluster tables, time series, and window-level outputs unchanged;
- align the CIKM data card and artifact guides with the two released units of analysis: 535 normalized claims and 529 virality
  cluster instances;
- make the public verifier reject a repository tree that contains the user-level lookup table or omits the retained scientific
  tables.

## Proof

The normal public entry points must pass from a clean copy of the candidate tree:

```bash
uv run scripts/verify_cikm2026_artifacts.py
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

The first command must fail when run against a repository root that still contains the excluded `users.parquet` path. Blob hashes
for `claims.parquet`, `clusters.parquet`, `cluster_timeseries.parquet`, `window_results.parquet`, and `registry/claims.json` must
remain unchanged.

## History boundary

This change removes the file from the current repository tree. It does not rewrite existing Git history.
