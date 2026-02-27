#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np

from src.metrics import ndcg_at_k, mrr_at_k, map_at_k


def load_npz(path: str):
    d = np.load(path)
    X = d["X"]
    y = d["y"]
    qid = d["qid"]
    group = d["group"].tolist()
    return X, y, qid, group


def summarize(name: str, arr: np.ndarray):
    return (
        f"{name}: mean={arr.mean():.5f}  "
        f"median={np.median(arr):.5f}  "
        f"p25={np.quantile(arr, 0.25):.5f}  "
        f"p75={np.quantile(arr, 0.75):.5f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    test_path = os.path.join(args.processed_dir, f"fold{args.fold}_test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing processed test file: {test_path}. Run build_fold first.")

    X, y, qid, group = load_npz(test_path)
    n = y.shape[0]
    print(f"Loaded fold{args.fold} test: X={X.shape}, y={y.shape}, queries={len(group):,}, docs={n:,}")

    # 1) Random scores baseline (should be low)
    rng = np.random.default_rng(42)
    random_scores = rng.random(n, dtype=np.float32)

    # 2) "Oracle-ish" scores = true label (upper-ish bound, not perfect but should be high)
    oracle_scores = y.astype(np.float32)

    # Evaluate both
    for tag, scores in [("RANDOM", random_scores), ("ORACLE_LABEL", oracle_scores)]:
        print(f"\n=== {tag} ===")
        nd1 = ndcg_at_k(y, scores, group, 1)
        nd3 = ndcg_at_k(y, scores, group, 3)
        nd5 = ndcg_at_k(y, scores, group, 5)
        nd10 = ndcg_at_k(y, scores, group, 10)
        mrr10 = mrr_at_k(y, scores, group, 10, rel_threshold=1)
        map10 = map_at_k(y, scores, group, 10, rel_threshold=1)

        print(summarize("NDCG@1 ", nd1))
        print(summarize("NDCG@3 ", nd3))
        print(summarize("NDCG@5 ", nd5))
        print(summarize("NDCG@10", nd10))
        print(summarize("MRR@10 ", mrr10))
        print(summarize("MAP@10 ", map10))

        # sanity expectation
        if tag == "RANDOM":
            print("Expect: low-ish values (roughly small).")
        else:
            print("Expect: much higher than RANDOM.")


if __name__ == "__main__":
    main()