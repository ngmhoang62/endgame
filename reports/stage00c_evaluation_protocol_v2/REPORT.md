# Stage 00C v2 — ENDGAME Evaluation Protocol

**Status: PASS**

## Rebuilt populations

- RAW_TRAIN: **7000**
- PRIMARY_EVAL: **6991**
- NON_EVALUABLE: **9**
- RAW_CORPUS: **8532**
- EMPTY_DOCS excluded: **20**
- RETRIEVAL_CORPUS: **8512**

## Identity policy

- Empty passage docs excluded: **yes**
- Exact duplicate-content docs deduplicated: **no**
- Literal document IDs preserved: **yes**
- Alias/remap labels: **no**

## Empty-gold impact

- Removed empty-gold occurrences: **11**
- Queries still evaluable after removal: **2**
- Queries with no retrievable gold left: **9**

## Leakage grouping

- Rule: emb >= 0.95 OR (emb >= 0.94 AND lexical >= 0.80) + exact/accent duplicates
- Gold used for grouping: **no**
- Groups >1: **72**
- Queries in grouped components: **151**
- Post-hoc edge gold-overlap: **85.71%**

## Fold sizes

| Fold | N | Single | Multi | Empty-gold-affected | Duplicate-content literal gold | Near-dup members |
|---|---:|---:|---:|---:|---:|---:|
| fold_0 | 1399 | 1288 | 111 | 0 | 2 | 31 |
| fold_1 | 1397 | 1288 | 109 | 1 | 1 | 29 |
| fold_2 | 1399 | 1288 | 111 | 0 | 1 | 30 |
| fold_3 | 1398 | 1288 | 110 | 0 | 0 | 30 |
| fold_4 | 1398 | 1288 | 110 | 1 | 0 | 31 |

## Gates

- population_6991: **PASS**
- non_evaluable_9: **PASS**
- retrieval_docs_8512: **PASS**
- group_isolation: **PASS**
- fold_coverage: **PASS**
- size_balance: **PASS**
- answer_balance: **PASS**
- freq_balance: **PASS**
- length_balance: **PASS**
- density_balance: **PASS**

## Historical LegalIR comparison

- Same intended evaluable-query count: **6991**.
- Same empty-passage principle: **yes**.
- Different duplicate policy: ENDGAME keeps all non-empty duplicate-content IDs literal.
