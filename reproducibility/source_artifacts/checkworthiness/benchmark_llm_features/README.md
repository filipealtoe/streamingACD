<!-- Sérgio Pinto, 2026-08-25 13:41 WEST — restored the exact numerical ClaimBuster and CT23 soft-label artifacts and documented the mandatory ID join. -->

# ClaimBuster and CT23 dimensional soft-label features

These files are byte-identical copies from Sérgio Pinto's commit
`093bb9accfa62b0864ce6a65f797ba3a23e81418` of May 14, 2026.

| Benchmark | Rows | SHA-256 |
|---|---:|---|
| ClaimBuster | 1,032 | `a26ac74990d876fe91e8cd9e34fe24134969a2c4bb997079b922d7cfc99e0a7a` |
| CheckThat! 2023 | 318 | `7db1dcdc780a0991b09af84288e1abc8d907ad1c1e57358953757fc86c7a53d6` |

Each row contains a public benchmark sentence ID and 60 numerical or boolean
LLM-derived features. The files contain no source text, Twitter/X post or user
identifier, username, profile, or location.

The stored order is not the benchmark order. Any evaluation or Fusion run must
join by `sentence_id` and validate one-to-one coverage. Positional alignment is
retained only as an explicitly labelled historical diagnostic.
