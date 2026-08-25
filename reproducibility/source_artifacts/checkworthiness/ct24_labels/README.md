<!-- Sérgio Pinto, 2026-08-25 14:05 WEST — added text-free CT24 sentence-ID and label order so recovered soft-label features can be joined one-to-one instead of combined by row position. -->

# CT24 label order

These three Parquet files contain only the public CheckThat! 2024 benchmark
`sentence_id` and binary `label` columns. They were mechanically derived from
the clean CT24 inputs in Sérgio Pinto's artifact commit
`093bb9accfa62b0864ce6a65f797ba3a23e81418`, preserving its original row order:

| Split | Rows | Positive labels | Source SHA-256 |
|---|---:|---:|---|
| train | 22,402 | 5,399 | `200d674d2187021c2376ad5ef10419cfda6421a9e325f731f0347fc76db026d4` |
| dev | 1,031 | 238 | `2ff8dbb80c86e79eb1d6fbe9d5f9e994eba6ccce1304154f4d8a551f04ffc07b` |
| test | 341 | 88 | `bfd7296b8718ed62950ebf7ffa88b1f7fb4ba890eb3088f6ed000296ca76d226` |

No benchmark text, Twitter/X post identifier, user identifier, username,
profile, or location is included. The public reproducer validates unique IDs,
identical ID sets, one-to-one joins, row counts, and file hashes before fitting
the corrected Fusion diagnostic.
