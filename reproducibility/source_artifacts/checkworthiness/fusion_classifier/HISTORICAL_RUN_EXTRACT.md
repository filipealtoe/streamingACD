<!-- Sérgio Pinto, 2026-08-21 21:48 PDT — preserved the non-sensitive cross-benchmark Fusion aggregate record without benchmark text or per-example identifiers. -->

# Fusion cross-benchmark historical aggregate

This is a non-sensitive extract from Sérgio Pinto's author design record
`design_approaches_2026-01-18_Sergio.md` (SHA-256
`b49cb060a1cd9e6af86486b9699695766946984aee30d4874dd87c95fa796584`,
filesystem modification time `2026-01-19T11:49:34-0800`). It makes the
historical aggregate record public and checksum-bound; it does not replace the
missing cross-benchmark per-example prediction vectors.

- Record date: 2026-01-18
- DeBERTa ensemble: seeds 0 and 456, temperature 0.7
- Fusion weight: 0.6 DeBERTa / 0.4 LLM XGBoost

| Benchmark | DeBERTa ensemble F1 | LLM XGBoost F1 | Fusion F1 |
|---|---:|---:|---:|
| ClaimBuster | 0.9702 | 0.4710 | 0.9605 |
| CT23 | 0.9327 | 0.5147 | 0.9151 |

The record states explicitly that the cross-benchmark ensemble used two seeds
because the seed-123 best-model checkpoint was unavailable. The CT24
temperature-scaled vector packaged separately is the retained three-seed
artifact used by the CT24 Fusion reproducer.

This extract contains no benchmark text, prediction vector, post/user
identifier, profile, or location.
