# Complete the CIKM 2026 reproducibility artifact

- Author: Sérgio Pinto
- Timestamp: 2026-08-21 00:13:44 PDT (-0700)
- Reason: connect the camera-ready paper to the public dataset, executable sources,
  exact prompts and parameters, retained results, and a verifiable model checkpoint.

## What changed

- Added the retained claim-normalization source snapshot and its CT25 result package.
- Added the original v4 check-worthiness prompt configuration and documented the
  exact soft-label extraction runner and the distinct prompt behavior of the
  frontier-LLM ablation.
- Added a paper-to-artifact index and one machine-readable parameter manifest.
- Added the four-head checkpoint release metadata, byte size, and SHA-256.
- Anchored the local check-worthiness output ignore rule at the repository root so
  the two public source snapshots are included in the artifact.
- Extended the public verifier to check the new source, prompt, parameter, result, and
  checkpoint manifests, with optional local hashing of the release asset.
- Updated the artifact guides to make each verification entry point easy to find.

## Verification

From the repository root:

```bash
uv run scripts/verify_cikm2026_artifacts.py
uv run scripts/verify_cikm2026_artifacts.py --checkpoint /path/to/best_model.pt
uv run scripts/reproduce_cikm2026_tabular_baselines.py
```

The source additions contain no raw posts, user identifiers, post identifiers, or
benchmark labels. The per-sample claim-normalization artifact contains only row index,
metric, route, latency, and token-count fields.
