#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import joblib
import lightgbm as lgb
import pandas as pd

from src.metrics import ndcg_at_k, mrr_at_k, map_at_k


def load_npz(path: str):
    d = np.load(path)
    return d["X"], d["y"], d["qid"], d["group"].tolist()


def evaluate_all(y_true, y_score, group):
    return {
        "NDCG@10": float(ndcg_at_k(y_true, y_score, group, 10).mean()),
        "MRR@10":  float(mrr_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
        "MAP@10":  float(map_at_k(y_true, y_score, group, 10, rel_threshold=1).mean()),
    }


def train_pointwise(Xtr, ytr, Xva, yva, seed: int = 42):
    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model


def train_ranker(Xtr, ytr, gtr, Xva, yva, gva, seed: int = 42):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        Xtr, ytr,
        group=gtr,
        eval_set=[(Xva, yva)],
        eval_group=[gva],
        eval_at=[10],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model


def summarize(results, name: str):
    ndcg = np.array([r["NDCG@10"] for r in results])
    mrr  = np.array([r["MRR@10"] for r in results])
    mapv = np.array([r["MAP@10"] for r in results])

    print(f"\n{name}")
    print(f"NDCG@10 : {ndcg.mean():.5f} ± {ndcg.std():.5f}")
    print(f"MRR@10  : {mrr.mean():.5f} ± {mrr.std():.5f}")
    print(f"MAP@10  : {mapv.mean():.5f} ± {mapv.std():.5f}")


def main():
    processed_dir = "data/processed"
    artifacts_dir = "artifacts"
    reports_dir = "reports"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    pointwise_results = []
    ranker_results = []
    rows = []

    for fold in range(1, 6):
        print(f"\n========== Fold {fold} ==========")

        train_path = f"{processed_dir}/fold{fold}_train.npz"
        vali_path  = f"{processed_dir}/fold{fold}_vali.npz"
        test_path  = f"{processed_dir}/fold{fold}_test.npz"

        for p in (train_path, vali_path, test_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing processed file: {p}. Run build_fold for this fold.")

        Xtr, ytr, qtr, gtr = load_npz(train_path)
        Xva, yva, qva, gva = load_npz(vali_path)
        Xte, yte, qte, gte = load_npz(test_path)

        # ---- Pointwise ----
        print("Training pointwise...")
        point_model = train_pointwise(Xtr, ytr, Xva, yva, seed=42)
        sp = point_model.predict(Xte)
        metrics_p = evaluate_all(yte, sp, gte)
        print("Pointwise:", metrics_p)

        point_path = os.path.join(artifacts_dir, f"pointwise_fold{fold}.joblib")
        joblib.dump(point_model, point_path)

        # ---- Ranker ----
        print("Training LambdaMART...")
        rank_model = train_ranker(Xtr, ytr, gtr, Xva, yva, gva, seed=42)
        sr = rank_model.predict(Xte)
        metrics_r = evaluate_all(yte, sr, gte)
        print("LambdaMART:", metrics_r)

        rank_path = os.path.join(artifacts_dir, f"lambdamart_fold{fold}.joblib")
        joblib.dump(rank_model, rank_path)

        pointwise_results.append(metrics_p)
        ranker_results.append(metrics_r)

        rows.append({
            "fold": fold,
            "point_ndcg10": metrics_p["NDCG@10"],
            "point_mrr10": metrics_p["MRR@10"],
            "point_map10": metrics_p["MAP@10"],
            "rank_ndcg10": metrics_r["NDCG@10"],
            "rank_mrr10": metrics_r["MRR@10"],
            "rank_map10": metrics_r["MAP@10"],
            "point_model_path": point_path,
            "rank_model_path": rank_path,
        })

    # Save fold metrics CSV
    df = pd.DataFrame(rows)
    out_csv = os.path.join(reports_dir, "fivefold_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved fold metrics to: {out_csv}")

    print("\n\n===== FINAL 5-FOLD RESULTS =====")
    summarize(pointwise_results, "Pointwise LGBMReg")
    summarize(ranker_results, "LambdaMART LGBMRank")


if __name__ == "__main__":
    main()