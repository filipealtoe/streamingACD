# Claim-normalization table evidence

<!-- Sérgio Pinto, 2026-08-21 20:13 PDT — added a cell-by-cell recovery boundary for the CIKM claim-normalization table. -->
<!-- Sérgio Pinto, 2026-08-21 20:40 PDT — separated the retained numerical match from the different paper Model A protocol. -->

The CIKM table contains two external baseline values and nine values produced by
our three claim-normalization configurations across Models A, B, and C.

The retained text-free 300-row metric file recomputes METEOR
`0.5569184651538435`, which rounds to the paper value `0.5569`. Run:

```bash
uv run --no-project scripts/reproduce_cikm2026_claim_normalization_table.py
```

This is a numerical recovery match, not direct reproduction of the paper's
Model A protocol. The paper states `(10, 0.85, 0.65, 4, 2)` for topic clusters,
retrieval threshold, claim-verification threshold, positive examples, and
negative examples. The recovery run used `(10, 0.85, 0.5, 5, 0)`.

All nine internal cells are therefore listed as unresolved for their exact
paper-run protocols in `TABLE2_RECOVERY_AUDIT.json`; Model A additionally has
the checksum-bound numerical recovery match. Results from different 1,285-row
runs in the private source workspace were deliberately not substituted for the
missing paper evidence.

The released per-example evidence contains indices, numeric METEOR values,
route labels, latency, and token counts. It excludes benchmark text and model
generations.
