# Learning-to-Rank on MSLR-WEB10K
*Learning-to-Rank is central to search and recommendation systems.
This project mirrors real Search/Relevance workflows: strong baselines, listwise ranking, offline metrics, per-query analysis, and cross-fold robustness checks*

---

## Project Overview

This project implements a full Learning-to-Rank (LTR) pipeline on the **MSLR-WEB10K** dataset. The goal is to simulate how modern search/relevance teams build and evaluate ranking models in production environments.

The pipeline includes:

- LETOR-format data parsing
- Query-level grouping
- Pointwise and Listwise ranking models
- Proper ranking metrics (NDCG, MRR, MAP)
- Per-query error analysis
- Feature importance inspection
- Feature ablation experiments
- 5-fold cross-validation
- Visualization of fold-level performance

The focus of this project is not just training a model, but understanding ranking behavior and validating improvements rigorously.

---
## Models Implemented

### 1) Pointwise Baseline — LightGBMRegressor

- Predicts the relevance label directly (regression objective)
- Documents are ranked by predicted score
- Serves as a strong boosted-tree baseline for tabular ranking features

---

### 2) Listwise Ranker — LightGBM LGBMRanker (LambdaMART)

- **Objective:** `lambdarank`
- Optimizes NDCG directly using query-level grouping
- Industry-standard learning-to-rank method widely used in search systems

---

## Dataset

**Dataset:** MSLR-WEB10K (Microsoft Learning to Rank dataset)

- 10,000 queries
- 136 engineered ranking features
- Relevance labels: 0–4
- Predefined 5-fold split
- LETOR / SVMrank format

Each fold contains `train.txt`, `vali.txt`, and `test.txt`.  
All splits are handled at the **query level** to avoid leakage.

---

## Problem Formulation

Given a query and a list of candidate documents with 136 engineered query-document features, we learn a scoring function:

```text
f(query, document) -> ranking score
```

Documents are ranked by decreasing score to maximize ranking quality.


## Evaluation Metrics

All metrics are computed **per query** and then averaged across queries:

- **NDCG@10** — Primary ranking metric (supports graded relevance)
- **MRR@10** — Rewards early retrieval of the first relevant document
- **MAP@10** — Precision averaged across relevant documents in top results

Metrics are evaluated on:
- Validation set (for early stopping)
- Test set (final reporting)
- All 5 folds (robust cross-validation)

---

## 5-Fold Cross-Validation Results

| Model | NDCG@10 (mean ± std) | MRR@10 (mean ± std) | MAP@10 (mean ± std) |
|--------|----------------------|----------------------|----------------------|
| Pointwise LGBMReg | 0.47291 ± 0.00730 | 0.83601 ± 0.00848 | 0.77621 ± 0.00530 |
| LambdaMART LGBMRank | **0.47520 ± 0.00500** | 0.83600 ± 0.00508 | 0.77431 ± 0.00331 |

---

### Observations

- LambdaMART consistently improved **NDCG@10** across folds.
- Average improvement: **+0.0023 NDCG@10**.
- LambdaMART showed **lower variance**, indicating improved stability and generalization.
- MRR is nearly identical, meaning both models retrieve relevant documents early.
- Improvements were strongest at top positions (NDCG@1 / @3 / @5).

## Feature Importance Analysis

- 130 out of 136 features had non-zero gain in the trained ranker.
- A small subset of features contributes a large portion of the total gain.
- However, many mid-tier features still play a meaningful role.
- Importance was evaluated using **gain importance**, which better reflects contribution to the ranking objective than simple split count.

---

## Feature Ablation Study (Top-K Features)

To measure the model’s dependence on feature subset quality, the ranker was retrained using only the top-K features ranked by gain importance.

| Features Used | NDCG@10 | Δ vs Full |
|---------------|----------|------------|
| 136 (Full) | 0.46746 | — |
| 95 | 0.46932 | +0.00186 |
| 85 | **0.47135** | **+0.00389** |
| 75 | 0.47098 | +0.00352 |
| 50 | 0.46943 | +0.00197 |
| 20 | 0.45904 | −0.00842 |
| 10 | 0.43176 | −0.03570 |

### Key Insight

Performance peaks around **75–85 features**, suggesting:

- Very low-importance features introduce noise and variance.
- Moderate feature pruning improves generalization.
- Aggressive pruning (top-10 or top-20 only) removes useful interaction signals and degrades ranking quality.

This indicates that ranking quality depends on a moderately diverse feature set rather than a very small set of dominant signals.

---

## Key Takeaways

- Query-level splitting and grouping are critical for ranking tasks (prevents leakage).
- Pointwise boosted trees provide a strong baseline for tabular ranking features.
- LambdaMART delivers consistent NDCG@10 improvements and lower variance across folds.
- Ranking gains are often concentrated at top positions (where search experience matters most).
- Feature ablation shows that reducing features (~136 → ~80) can maintain or slightly improve performance.