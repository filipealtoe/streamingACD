# Reproducibility package

<!-- Sérgio Pinto, 2026-08-25 18:18 WEST — made the accepted CIKM 2026 artifact the public entry point and retained only verified reproduction guidance. -->
<!-- Sérgio Pinto, 2026-08-25 20:17 WEST — connected the public entry point to the complete RandomForest paper-row reproduction. -->
<!-- Sérgio Pinto, 2026-08-25 21:56 WEST — describes the RandomForest artifact as a version-pinned paper-cell reconstruction. -->

The public package for **“Predicting Narrative Virality on Social Media Streams
for Pre-Peak Misinformation Triage”** is documented in
[`cikm2026/README.md`](cikm2026/README.md).

Its contents are organized as follows:

- `cikm2026/`: artifact guide, schemas, parameters, asset identities, and checksum manifests;
- `runs/`: retained run configurations and prediction bundles;
- `source_artifacts/`: released numerical data, source snapshots, prompts, and text-free evaluation artifacts.

Run the package verifier from the repository root:

```bash
uv run --no-project scripts/verify_cikm2026_artifacts.py
```

Large checkpoint and embedding files are distributed through the GitHub release
and are cryptographically bound to this repository by byte size and SHA-256.
