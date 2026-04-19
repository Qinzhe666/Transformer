#!/usr/bin/env python3
"""
Pre-process CSV data into per-year .pt files for fast loading.

Structure:
  pt_data/
    year_2015.pt  ->  {"X": Tensor(N,F), "Y1": Tensor(N,), "W": Tensor(N,), "day_lengths": Tensor(D,)}
    year_2016.pt
    ...

X is stored RAW (unnormalized). Normalization is applied at load time using
fold-specific mean/std from mean_std_cache.npz.
Y1 and W are cleaned (nan -> 0/1, clip).

Usage:
    python build_pt_data.py
"""

import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# ----------------------------
# Config (must match the run_pretrain_*/run_posttrain_* training scripts)
# ----------------------------
DATA_GLOB = "/home/ql84/Transformer/csv_data/**/*.csv"
PT_DATA_DIR = "./pt_data"

TARGET_COL = "Y1"
WGT_COL = "wgt"
EXCLUDE_COLS_EXTRA = ["Unnamed: 0", "Y0", "Y2", "Y3", WGT_COL]
CAT_FEATURE_COLS = ["X0", "X1", "X2", "X3"]
DROP_LAST_N = 0


def get_year_from_path(p: str) -> int:
    basename = os.path.basename(p)
    return int(basename[:4])


def build_year_pt(year: int, files: List[str], feature_cols: List[str]) -> Dict:
    """Read all CSVs for one year, return raw (unnormalized) tensors."""
    all_X, all_y, all_w = [], [], []
    day_lengths = []

    usecols = list(feature_cols) + [TARGET_COL, WGT_COL]
    for p in sorted(files):
        df = pd.read_csv(p, usecols=usecols)
        if DROP_LAST_N > 0:
            df = df.iloc[:-DROP_LAST_N]
        if len(df) == 0:
            continue

        X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)

        y = df[TARGET_COL].to_numpy(dtype=np.float32, copy=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        w = df[WGT_COL].to_numpy(dtype=np.float32, copy=True)
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)
        w = np.clip(w, 0.0, None)

        all_X.append(X)
        all_y.append(y)
        all_w.append(w)
        day_lengths.append(X.shape[0])

    if len(day_lengths) == 0:
        raise RuntimeError(f"No valid data for year {year}")

    return {
        "X": torch.from_numpy(np.concatenate(all_X, axis=0)),
        "Y1": torch.from_numpy(np.concatenate(all_y, axis=0)),
        "W": torch.from_numpy(np.concatenate(all_w, axis=0)),
        "day_lengths": torch.tensor(day_lengths, dtype=torch.int64),
    }


def main():
    all_files = sorted(glob.glob(DATA_GLOB, recursive=True))
    print(f"Total CSV files: {len(all_files)}")

    sample_df = pd.read_csv(all_files[0], nrows=1)
    all_cols = sample_df.columns.tolist()
    exclude = set([TARGET_COL] + EXCLUDE_COLS_EXTRA + CAT_FEATURE_COLS)
    feature_cols = [c for c in all_cols if c not in exclude]
    print(f"feature_cols: {len(feature_cols)} columns")

    year_to_files: Dict[int, List[str]] = {}
    for f in all_files:
        y = get_year_from_path(f)
        year_to_files.setdefault(y, []).append(f)

    years_sorted = sorted(year_to_files.keys())
    print(f"Years found: {years_sorted}")

    # --- Step 1: Build per-year .pt files ---
    Path(PT_DATA_DIR).mkdir(parents=True, exist_ok=True)

    for year in years_sorted:
        out_path = os.path.join(PT_DATA_DIR, f"year_{year}.pt")
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists")
            continue

        files = year_to_files[year]
        print(f"[BUILD] year={year}, {len(files)} files ...")
        data = build_year_pt(year, files, feature_cols)
        n_rows = data["X"].shape[0]
        n_days = len(data["day_lengths"])
        torch.save(data, out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  -> {out_path} ({n_rows} rows, {n_days} days, {size_mb:.1f} MB)")

    # --- Summary ---
    print("\n=== Summary ===")
    total_size = 0
    for f in sorted(os.listdir(PT_DATA_DIR)):
        fp = os.path.join(PT_DATA_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1024 / 1024
            total_size += sz
            print(f"  {f}: {sz:.1f} MB")
    print(f"  Total: {total_size:.1f} MB")
    print(f"\nFold mean/std already in mean_std_cache.npz")
    print("Done! Pre-train / post-train can now load from pt_data/.")


if __name__ == "__main__":
    main()
