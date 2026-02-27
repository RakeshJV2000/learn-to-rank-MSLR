#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import lightgbm as lgb

from src.metrics import ndcg_at_k, mrr_at_k, map_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


def eval_split(y_true, y_score, group):
    nd10 = ndcg_at_k(y_true, y_score, group, 10)
    mrr10 = mrr_at_k(y_true, y_score, group, 10, rel_threshold=1)
    map10 = map_at_k(y_true, y_score, group, 10, rel_threshold=1)
    return float(nd10.mean()), float(mrr10.mean()), float(map10.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = os.path.join(args.processed_dir, f"fold{args.fold}_train.npz")
    vali_path = os.path.join(args.processed_dir, f"fold{args.fold}_vali.npz")
    test_path = os.path.join(args.processed_dir, f"fold{args.fold}_test.npz")

    for p in (train_path, vali_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run build_fold first.")

    Xtr, ytr, qtr, gtr = load_npz(train_path)
    Xva, yva, qva, gva = load_npz(vali_path)
    Xte, yte, qte, gte = load_npz(test_path)

    os.makedirs(args.out_dir, exist_ok=True)

    # Pointwise baseline: predict relevance label as a real-valued score
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    # Predict scores for ranking
    va_scores = model.predict(Xva)
    te_scores = model.predict(Xte)

    ndcg10_va, mrr10_va, map10_va = eval_split(yva, va_scores, gva)
    ndcg10_te, mrr10_te, map10_te = eval_split(yte, te_scores, gte)

    print("\n=== Pointwise LGBMRegressor Results ===")
    print(f"VAL  : NDCG@10={ndcg10_va:.5f}  MRR@10={mrr10_va:.5f}  MAP@10={map10_va:.5f}")
    print(f"TEST : NDCG@10={ndcg10_te:.5f}  MRR@10={mrr10_te:.5f}  MAP@10={map10_te:.5f}")

    out_path = os.path.join(args.out_dir, f"pointwise_fold{args.fold}.joblib")
    joblib.dump(model, out_path)
    print(f"\n✅ Saved model: {out_path}")


if __name__ == "__main__":
    main()