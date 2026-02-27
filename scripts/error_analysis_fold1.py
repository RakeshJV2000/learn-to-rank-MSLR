#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.metrics import ndcg_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


def slice_by_group(arr: np.ndarray, group: List[int]) -> List[np.ndarray]:
    """Return list of per-query slices."""
    out = []
    start = 0
    for g in group:
        end = start + g
        out.append(arr[start:end])
        start = end
    return out


def topk_table(
    y_true_q: np.ndarray,
    score_a_q: np.ndarray,
    score_b_q: np.ndarray,
    k: int = 10,
) -> str:
    """
    Prints a small table for a single query:
    rank, label, score_pointwise, score_ranker
    shown in each model's ranking order.
    """
    # Order docs by each model
    ord_a = np.argsort(score_a_q)[::-1][:k]
    ord_b = np.argsort(score_b_q)[::-1][:k]

    lines = []
    lines.append("Pointwise top-k:")
    lines.append("rank\tlabel\tscore")
    for i, idx in enumerate(ord_a, start=1):
        lines.append(f"{i}\t{int(y_true_q[idx])}\t{float(score_a_q[idx]):.6f}")

    lines.append("\nLambdaMART top-k:")
    lines.append("rank\tlabel\tscore")
    for i, idx in enumerate(ord_b, start=1):
        lines.append(f"{i}\t{int(y_true_q[idx])}\t{float(score_b_q[idx]):.6f}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--show_examples", type=int, default=5, help="How many win/loss queries to print with top-k tables")
    args = ap.parse_args()

    test_path = os.path.join(args.processed_dir, f"fold{args.fold}_test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing processed test: {test_path}")

    point_path = os.path.join(args.artifacts_dir, f"pointwise_fold{args.fold}.joblib")
    rank_path = os.path.join(args.artifacts_dir, f"lambdamart_fold{args.fold}.joblib")
    if not os.path.exists(point_path) or not os.path.exists(rank_path):
        raise FileNotFoundError("Missing model artifacts. Train pointwise + lambdamart first.")

    Xte, yte, qid_te, group_te = load_npz(test_path)
    pointwise = joblib.load(point_path)
    ranker = joblib.load(rank_path)

    s_point = pointwise.predict(Xte)
    s_rank = ranker.predict(Xte)

    # Per-query NDCG arrays
    ndcg_point = ndcg_at_k(yte, s_point, group_te, args.k)
    ndcg_rank = ndcg_at_k(yte, s_rank, group_te, args.k)
    delta = ndcg_rank - ndcg_point

    print(f"Fold{args.fold} per-query NDCG@{args.k}")
    print(f"Pointwise mean: {ndcg_point.mean():.5f}")
    print(f"Ranker   mean: {ndcg_rank.mean():.5f}")
    print(f"Delta    mean: {delta.mean():.6f}")
    print(f"Delta  median: {np.median(delta):.6f}")
    print(f"Delta     p25: {np.quantile(delta, 0.25):.6f}")
    print(f"Delta     p75: {np.quantile(delta, 0.75):.6f}")

    # Identify best/worst queries
    win_idx = np.argsort(delta)[::-1]  # biggest positive first
    lose_idx = np.argsort(delta)       # most negative first

    print("\nTop 10 winning queries (largest +Δ):")
    for i in range(10):
        qi = int(win_idx[i])
        print(f"  q#{qi:4d}  Δ={delta[qi]:+.6f}  rank={ndcg_rank[qi]:.5f}  point={ndcg_point[qi]:.5f}")

    print("\nTop 10 losing queries (largest -Δ):")
    for i in range(10):
        qi = int(lose_idx[i])
        print(f"  q#{qi:4d}  Δ={delta[qi]:+.6f}  rank={ndcg_rank[qi]:.5f}  point={ndcg_point[qi]:.5f}")

    # Show a few detailed examples
    if args.show_examples > 0:
        y_slices = slice_by_group(yte, group_te)
        sp_slices = slice_by_group(s_point, group_te)
        sr_slices = slice_by_group(s_rank, group_te)

        print(f"\n\n=== Detailed examples: Top {args.show_examples} wins (showing top-{args.k}) ===")
        for j in range(args.show_examples):
            qi = int(win_idx[j])
            print(f"\n--- WIN example query_index={qi}  Δ={delta[qi]:+.6f} ---")
            print(topk_table(y_slices[qi], sp_slices[qi], sr_slices[qi], k=args.k))

        print(f"\n\n=== Detailed examples: Top {args.show_examples} losses (showing top-{args.k}) ===")
        for j in range(args.show_examples):
            qi = int(lose_idx[j])
            print(f"\n--- LOSS example query_index={qi}  Δ={delta[qi]:+.6f} ---")
            print(topk_table(y_slices[qi], sp_slices[qi], sr_slices[qi], k=args.k))


if __name__ == "__main__":
    main()