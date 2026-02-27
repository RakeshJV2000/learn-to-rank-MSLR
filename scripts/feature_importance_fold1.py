#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    model_path = os.path.join(args.artifacts_dir, f"lambdamart_fold{args.fold}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("LambdaMART model not found.")

    model = joblib.load(model_path)

    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")

    feature_names = [f"f{i}" for i in range(len(gain_importance))]

    df = pd.DataFrame({
        "feature": feature_names,
        "gain": gain_importance,
        "split": split_importance
    })

    df = df.sort_values("gain", ascending=False)

    print("\nTop Features by Gain Importance\n")
    print(df.head(args.topk))

    print("\nSummary Statistics:")
    print(f"Total features: {len(df)}")
    print(f"Non-zero gain features: {(df['gain'] > 0).sum()}")

    # Optional: save to CSV
    out_path = os.path.join(args.artifacts_dir, f"fold{args.fold}_feature_importance.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full feature importance to: {out_path}")


if __name__ == "__main__":
    main()