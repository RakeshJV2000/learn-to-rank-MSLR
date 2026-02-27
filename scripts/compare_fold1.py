#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np

from src.metrics import ndcg_at_k, mrr_at_k, map_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


# def eval_all(y_true, y_score, group):
#     return {
#         "NDCG@10": float(ndcg_at_k(y_true, y_score, group, 10).mean()),
#         "MRR@10":  float(mrr_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
#         "MAP@10":  float(map_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
#     }

def eval_all(y_true, y_score, group):
    return {
        "NDCG@1": float(ndcg_at_k(y_true, y_score, group, 1).mean()),
        "NDCG@3": float(ndcg_at_k(y_true, y_score, group, 3).mean()),
        "NDCG@5": float(ndcg_at_k(y_true, y_score, group, 5).mean()),
        "NDCG@10": float(ndcg_at_k(y_true, y_score, group, 10).mean()),
        "MRR@10":  float(mrr_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
        "MAP@10":  float(map_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = ap.parse_args()

    test_path = os.path.join(args.processed_dir, f"fold{args.fold}_test.npz")
    Xte, yte, qte, gte = load_npz(test_path)

    pointwise = joblib.load(os.path.join(args.artifacts_dir, f"pointwise_fold{args.fold}.joblib"))
    ranker = joblib.load(os.path.join(args.artifacts_dir, f"lambdamart_fold{args.fold}.joblib"))

    s_point = pointwise.predict(Xte)
    s_rank = ranker.predict(Xte)

    m_point = eval_all(yte, s_point, gte)
    m_rank = eval_all(yte, s_rank, gte)

    # print("\nModel\t\t\tNDCG@10\t\tMRR@10\t\tMAP@10")
    # print(f"Pointwise LGBMReg\t{m_point['NDCG@10']:.5f}\t\t{m_point['MRR@10']:.5f}\t\t{m_point['MAP@10']:.5f}")
    # print(f"LambdaMART LGBMRank\t{m_rank['NDCG@10']:.5f}\t\t{m_rank['MRR@10']:.5f}\t\t{m_rank['MAP@10']:.5f}")

    print("\nModel\t\t\tNDCG@1\tNDCG@3\tNDCG@5\tNDCG@10\tMRR@10\tMAP@10")

    print(f"Pointwise LGBMReg\t"
          f"{m_point['NDCG@1']:.5f}\t"
          f"{m_point['NDCG@3']:.5f}\t"
          f"{m_point['NDCG@5']:.5f}\t"
          f"{m_point['NDCG@10']:.5f}\t"
          f"{m_point['MRR@10']:.5f}\t"
          f"{m_point['MAP@10']:.5f}")

    print(f"LambdaMART LGBMRank\t"
          f"{m_rank['NDCG@1']:.5f}\t"
          f"{m_rank['NDCG@3']:.5f}\t"
          f"{m_rank['NDCG@5']:.5f}\t"
          f"{m_rank['NDCG@10']:.5f}\t"
          f"{m_rank['MRR@10']:.5f}\t"
          f"{m_rank['MAP@10']:.5f}")


if __name__ == "__main__":
    main()