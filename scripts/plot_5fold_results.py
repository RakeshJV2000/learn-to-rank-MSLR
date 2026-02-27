#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.metrics import ndcg_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


def main():
    processed_dir = "data/processed"
    artifacts_dir = "artifacts"
    out_path = "reports/ndcg_5fold.png"
    os.makedirs("reports", exist_ok=True)

    folds = np.arange(1, 6)

    point_ndcg = []
    rank_ndcg = []

    for fold in folds:
        test_path = os.path.join(processed_dir, f"fold{fold}_test.npz")
        Xte, yte, qid, gte = load_npz(test_path)

        point_model_path = os.path.join(artifacts_dir, f"pointwise_fold{fold}.joblib")
        rank_model_path = os.path.join(artifacts_dir, f"lambdamart_fold{fold}.joblib")

        if not os.path.exists(point_model_path) or not os.path.exists(rank_model_path):
            raise FileNotFoundError(
                f"Missing model for fold {fold}. Expected:\n"
                f"  {point_model_path}\n"
                f"  {rank_model_path}\n"
                f"Run your 5-fold training script first."
            )

        point = joblib.load(point_model_path)
        ranker = joblib.load(rank_model_path)

        sp = point.predict(Xte)
        sr = ranker.predict(Xte)

        p_nd = float(ndcg_at_k(yte, sp, gte, 10).mean())
        r_nd = float(ndcg_at_k(yte, sr, gte, 10).mean())

        point_ndcg.append(p_nd)
        rank_ndcg.append(r_nd)

        print(f"Fold {fold}: Pointwise NDCG@10={p_nd:.5f} | LambdaMART NDCG@10={r_nd:.5f}")

    point_ndcg = np.array(point_ndcg)
    rank_ndcg = np.array(rank_ndcg)

    # Plot
    plt.figure()
    plt.plot(folds, point_ndcg, marker="o", label="Pointwise (LGBMReg)")
    plt.plot(folds, rank_ndcg, marker="o", label="LambdaMART (LGBMRanker)")

    plt.xlabel("Fold")
    plt.ylabel("NDCG@10")
    plt.title("MSLR-WEB10K: 5-Fold NDCG@10 Comparison")
    plt.xticks(folds)
    plt.grid(True)
    plt.legend()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved plot to: {out_path}")

    # Optional: also print mean ± std for sanity
    print("\nSummary:")
    print(f"Pointwise mean±std: {point_ndcg.mean():.5f} ± {point_ndcg.std():.5f}")
    print(f"Ranker   mean±std: {rank_ndcg.mean():.5f} ± {rank_ndcg.std():.5f}")


if __name__ == "__main__":
    main()