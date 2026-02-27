#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import lightgbm as lgb

from src.metrics import ndcg_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


def eval_ndcg10(y_true, y_score, group):
    return float(ndcg_at_k(y_true, y_score, group, 10).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = ap.parse_args()

    print(f"\nRunning top-{args.topk} feature ablation (Fold{args.fold})")

    # Load trained full model
    model_path = os.path.join(args.artifacts_dir, f"lambdamart_fold{args.fold}.joblib")
    model = joblib.load(model_path)

    # Get top-k features by gain
    gain = model.booster_.feature_importance(importance_type="gain")
    topk_idx = np.argsort(gain)[::-1][:args.topk]

    print("Top feature indices:", topk_idx)

    # Load data
    train_path = os.path.join(args.processed_dir, f"fold{args.fold}_train.npz")
    vali_path = os.path.join(args.processed_dir, f"fold{args.fold}_vali.npz")
    test_path = os.path.join(args.processed_dir, f"fold{args.fold}_test.npz")

    Xtr, ytr, qtr, gtr = load_npz(train_path)
    Xva, yva, qva, gva = load_npz(vali_path)
    Xte, yte, qte, gte = load_npz(test_path)

    # Slice features
    Xtr_small = Xtr[:, topk_idx]
    Xva_small = Xva[:, topk_idx]
    Xte_small = Xte[:, topk_idx]

    # Train new ranker
    ranker_small = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )

    ranker_small.fit(
        Xtr_small, ytr,
        group=gtr,
        eval_set=[(Xva_small, yva)],
        eval_group=[gva],
        eval_at=[10],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # Evaluate
    full_scores = model.predict(Xte)
    small_scores = ranker_small.predict(Xte_small)

    full_ndcg = eval_ndcg10(yte, full_scores, gte)
    small_ndcg = eval_ndcg10(yte, small_scores, gte)

    print("\n=== Ablation Results ===")
    print(f"Full model NDCG@10 : {full_ndcg:.5f}")
    print(f"Top-{args.topk} NDCG@10 : {small_ndcg:.5f}")
    print(f"Difference : {small_ndcg - full_ndcg:+.6f}")


if __name__ == "__main__":
    main()