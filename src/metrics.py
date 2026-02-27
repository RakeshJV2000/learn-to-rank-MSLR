"""
Ranking metrics (NDCG/MRR/MAP).
"""
import numpy as np


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = (2 ** rels - 1)
    discounts = np.log2(np.arange(2, rels.size + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(y_true, y_score, group, k):
    """
    Returns per-query NDCG@k
    """
    results = []
    start = 0
    for g in group:
        end = start + g
        true = y_true[start:end]
        score = y_score[start:end]

        order = np.argsort(score)[::-1]
        ranked_true = true[order]

        dcg = dcg_at_k(ranked_true, k)

        ideal_order = np.argsort(true)[::-1]
        ideal_true = true[ideal_order]
        idcg = dcg_at_k(ideal_true, k)

        results.append(0.0 if idcg == 0 else dcg / idcg)
        start = end

    return np.array(results)


def mrr_at_k(y_true, y_score, group, k, rel_threshold=1):
    results = []
    start = 0
    for g in group:
        end = start + g
        true = y_true[start:end]
        score = y_score[start:end]

        order = np.argsort(score)[::-1]
        ranked_true = true[order][:k]

        hits = np.where(ranked_true >= rel_threshold)[0]
        if len(hits) == 0:
            results.append(0.0)
        else:
            results.append(1.0 / (hits[0] + 1))
        start = end

    return np.array(results)


def map_at_k(y_true, y_score, group, k, rel_threshold=1):
    results = []
    start = 0
    for g in group:
        end = start + g
        true = y_true[start:end]
        score = y_score[start:end]

        order = np.argsort(score)[::-1]
        ranked_true = true[order][:k]

        relevant = ranked_true >= rel_threshold
        if not np.any(relevant):
            results.append(0.0)
        else:
            precisions = []
            hits = 0
            for i, is_rel in enumerate(relevant):
                if is_rel:
                    hits += 1
                    precisions.append(hits / (i + 1))
            results.append(np.mean(precisions))
        start = end

    return np.array(results)