# Stage 05A — Actually Scored PRIVATE Failure Map

**Label-free. No private labels/per-query Codabench outcomes used.**

- Population: **2080 - 103 = 1977**
- Current public: **R=0.941915361 / P=0.201922105**
- Rank-only → CE-LR set churn: **1133/1977 (57.3%)**

| Archetype | N | Fraction | Source disagreement@5 | CE margin5-6 | CE churn |
|---|---:|---:|---:|---:|---:|
| CONDITION | 271 | 13.7% | 0.560 | 0.7183 | 49.4% |
| DEFINITION | 117 | 5.9% | 0.633 | 0.5614 | 56.4% |
| DIRECT_REF | 92 | 4.7% | 0.603 | 0.6170 | 53.3% |
| GENERAL_SEMANTIC | 985 | 49.8% | 0.605 | 0.6206 | 57.6% |
| PROCEDURE | 403 | 20.4% | 0.577 | 0.6722 | 60.3% |
| SANCTION | 109 | 5.5% | 0.602 | 0.6430 | 67.9% |

Inspect first: `SCORED_1977.csv`, `CE_UNCERTAINTY_TOP200.csv`, `SOURCE_DISAGREEMENT_TOP200.csv`, `CE_CHURN_TOP200.csv`.

Any policy discovered here must beat frozen TRAIN OOF before promotion.
