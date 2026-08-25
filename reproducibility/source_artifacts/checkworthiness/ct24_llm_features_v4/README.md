<!-- Sérgio Pinto, 2026-08-25 13:41 WEST — restored the exact numerical CT24 soft-label artifacts required to train the four-head model without republishing benchmark text. -->

# CT24 dimensional soft-label features

These three Parquet files are byte-identical copies from Sérgio Pinto's commit
`093bb9accfa62b0864ce6a65f797ba3a23e81418` of May 14, 2026.

| Split | Rows | SHA-256 |
|---|---:|---|
| train | 22,402 | `ec15929b5eb6d509d32d6b32b38e0d765bbfa15609a617b4f4a085741886f218` |
| development | 1,031 | `00d0ce6698dd9d7dc3a50f56f7c644abe22e4c9b3c6f10d97ed7ececfeb05af7` |
| test-gold | 341 | `60252ee38453d14a43b076fe8fa50e998a9dafebdb1d51f7e81d92470a9d0e1c` |

Each row contains a benchmark `sentence_id` and 60 numerical or boolean values
derived from the three checkability, verifiability, and harm prompts. There is
no source text, post or user identifier, username, profile, or location.

Always join these rows to the official CT24 data by `sentence_id`; their stored
row order intentionally differs from the corresponding benchmark split order.
