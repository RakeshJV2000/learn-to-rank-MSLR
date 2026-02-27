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
    nd1 = ndcg_at_k(y_true, y_score, group, 1).mean()
    nd3 = ndcg_at_k(y_true, y_score, group, 3).mean()
    nd5 = ndcg_at_k(y_true, y_score, group, 5).mean()
    nd10 = ndcg_at_k(y_true, y_score, group, 10).mean()
    mrr10 = mrr_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()
    map10 = map_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()
    return float(nd1), float(nd3), float(nd5), float(nd10), float(mrr10), float(map10)


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

    # ranker = lgb.LGBMRanker(
    #     objective="lambdarank",
    #     metric="ndcg",
    #     boosting_type="gbdt",
    #     n_estimators=5000,
    #     learning_rate=0.03,
    #     num_leaves=63,
    #     min_data_in_leaf=50,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=args.seed,
    #     n_jobs=-1,
    # )

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        boosting_type="gbdt",
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=127,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )

    ranker.fit(
        Xtr, ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        eval_at=[1, 3, 5, 10],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )

    va_scores = ranker.predict(Xva)
    te_scores = ranker.predict(Xte)

    va = eval_split(yva, va_scores, gva)
    te = eval_split(yte, te_scores, gte)

    print("\n=== LambdaMART (LGBMRanker) Results ===")
    print(f"VAL  : NDCG@1={va[0]:.5f} NDCG@3={va[1]:.5f} NDCG@5={va[2]:.5f} NDCG@10={va[3]:.5f}  "
          f"MRR@10={va[4]:.5f} MAP@10={va[5]:.5f}")
    print(f"TEST : NDCG@1={te[0]:.5f} NDCG@3={te[1]:.5f} NDCG@5={te[2]:.5f} NDCG@10={te[3]:.5f}  "
          f"MRR@10={te[4]:.5f} MAP@10={te[5]:.5f}")

    out_path = os.path.join(args.out_dir, f"lambdamart_fold{args.fold}.joblib")
    joblib.dump(ranker, out_path)
    print(f"\n✅ Saved model: {out_path}")


if __name__ == "__main__":
    main()