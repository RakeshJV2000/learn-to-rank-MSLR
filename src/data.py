"""
Data parsing + preprocessing for MSLR (LETOR/SVMrank format).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

_QID_RE = re.compile(r"qid:(\d+)")


@dataclass
class LetorSplit:
    X: np.ndarray        # [N, 136]
    y: np.ndarray        # [N]
    qid: np.ndarray      # [N]
    group: List[int]     # docs per query in the current row order


def parse_letor_file(
    path: str,
    num_features: int = 136,
    dtype=np.float32,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses MSLR LETOR/SVMrank file.
    Each line:
      <label> qid:<id> <fid>:<val> ... # comment
    Missing feature indices are treated as 0.
    Feature ids are 1-based [1..136].
    """
    X_rows: List[np.ndarray] = []
    y: List[int] = []
    qids: List[int] = []

    total_bytes = os.path.getsize(path)
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"Parsing {os.path.basename(path)}") if show_progress else None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if pbar is not None:
                pbar.update(len(line.encode("utf-8", errors="ignore")))

            line = line.strip()
            if not line:
                continue

            # remove comment
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            parts = line.split()
            if len(parts) < 2:
                continue

            label = int(float(parts[0]))

            m = _QID_RE.match(parts[1])
            if not m:
                raise ValueError(f"Could not parse qid from: {parts[1]}")
            qid = int(m.group(1))

            feats = np.zeros(num_features, dtype=dtype)

            for tok in parts[2:]:
                if ":" not in tok:
                    continue
                fid_s, val_s = tok.split(":", 1)
                fid = int(fid_s)
                if 1 <= fid <= num_features:
                    feats[fid - 1] = dtype(float(val_s))

            X_rows.append(feats)
            y.append(label)
            qids.append(qid)

    if pbar is not None:
        pbar.close()

    X = np.vstack(X_rows).astype(dtype, copy=False)
    y_arr = np.asarray(y, dtype=np.int32)
    qid_arr = np.asarray(qids, dtype=np.int32)
    return X, y_arr, qid_arr


def sort_by_qid(X: np.ndarray, y: np.ndarray, qid: np.ndarray) -> LetorSplit:
    """Sort rows by qid (stable), then build group sizes."""
    order = np.argsort(qid, kind="mergesort")  # stable
    Xs = X[order]
    ys = y[order]
    qs = qid[order]
    group = make_group(qs)
    return LetorSplit(X=Xs, y=ys, qid=qs, group=group)


def make_group(qid_sorted: np.ndarray) -> List[int]:
    """Given qid sorted array, returns docs-per-query list."""
    if qid_sorted.size == 0:
        return []

    group: List[int] = []
    cur = qid_sorted[0]
    cnt = 1
    for q in qid_sorted[1:]:
        if q == cur:
            cnt += 1
        else:
            group.append(cnt)
            cur = q
            cnt = 1
    group.append(cnt)
    return group


def sanity_check(split: LetorSplit, name: str) -> None:
    assert split.X.shape[0] == split.y.shape[0] == split.qid.shape[0], f"{name}: X/y/qid length mismatch"
    assert split.X.shape[1] == 136, f"{name}: expected 136 features, got {split.X.shape[1]}"
    assert sum(split.group) == split.X.shape[0], f"{name}: sum(group) != num_rows"

    # verify each group is a single qid
    start = 0
    for g in split.group:
        end = start + g
        q = split.qid[start]
        if not np.all(split.qid[start:end] == q):
            raise ValueError(f"{name}: grouping error around rows {start}:{end}")
        start = end

    uq = np.unique(split.qid).size
    print(f"[{name}] rows={split.X.shape[0]:,}  queries={uq:,}  labels={sorted(set(split.y.tolist()))}")