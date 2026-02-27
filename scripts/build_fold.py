#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from src.data import parse_letor_file, sort_by_qid, sanity_check


def save_npz(out_path: str, X: np.ndarray, y: np.ndarray, qid: np.ndarray, group: np.ndarray) -> None:
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, qid=qid, group=group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw/MSLR-WEB10K", help="Path to MSLR-WEB10K root")
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--out_dir", type=str, default="data/processed")
    args = ap.parse_args()

    fold_dir = os.path.join(args.root, f"Fold{args.fold}")
    train_path = os.path.join(fold_dir, "train.txt")
    vali_path = os.path.join(fold_dir, "vali.txt")
    test_path = os.path.join(fold_dir, "test.txt")

    for p in (train_path, vali_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print(f"Building Fold{args.fold} from: {fold_dir}")
    print("Parsing train...")
    Xtr, ytr, qtr = parse_letor_file(train_path)
    train = sort_by_qid(Xtr, ytr, qtr)
    sanity_check(train, f"fold{args.fold}_train")

    print("Parsing vali...")
    Xva, yva, qva = parse_letor_file(vali_path)
    vali = sort_by_qid(Xva, yva, qva)
    sanity_check(vali, f"fold{args.fold}_vali")

    print("Parsing test...")
    Xte, yte, qte = parse_letor_file(test_path)
    test = sort_by_qid(Xte, yte, qte)
    sanity_check(test, f"fold{args.fold}_test")

    out_train = os.path.join(args.out_dir, f"fold{args.fold}_train.npz")
    out_vali = os.path.join(args.out_dir, f"fold{args.fold}_vali.npz")
    out_test = os.path.join(args.out_dir, f"fold{args.fold}_test.npz")

    save_npz(out_train, train.X, train.y, train.qid, np.asarray(train.group, dtype=np.int32))
    save_npz(out_vali, vali.X, vali.y, vali.qid, np.asarray(vali.group, dtype=np.int32))
    save_npz(out_test, test.X, test.y, test.qid, np.asarray(test.group, dtype=np.int32))

    print("\n✅ Saved:")
    print(f"  {out_train}")
    print(f"  {out_vali}")
    print(f"  {out_test}")


if __name__ == "__main__":
    main()