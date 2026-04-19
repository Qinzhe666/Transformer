#!/usr/bin/env python3
# =========================
# SFT Post-train v3.1 — Final schedule: per-seqlen const phase + cosine finish
#
# Design (from MA-plateau analysis on v3+const data):
#   1. Primer: v3's sft_step=150k checkpoint (per (sl, fold, seed0, ssl=100k))
#   2. Constant-LR phase: head=1.5e-5, enc=5e-6, from 150k to SFT_COS_START_PER_SEQLEN
#   3. Cosine-decay phase: CosineAnnealingLR with T_max over the last portion,
#      eta_min=1e-6 (preserving useful refinement LR)
#
# Per-seqlen targets chosen with +30% margin over observed MA-plateau:
#   sl=3:  total 700k (plateau max ~550k on 2024)
#   sl=5:  total 700k (plateau max ~570k on 2021)
#   sl=8:  total 500k (plateau max ~400k on 2024)
#   sl=10: total 500k (plateau max ~470k on 2021)
#   sl=20: total 300k (v2 peak@150k, unclear — margin for safety)
#   sl=30: total 250k (clear early peak ~100k)
#
# Cosine phase = last 30% of training.
# =========================

import gc
import glob
import json
import os
import random
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# ----------------------------
# 0) CONFIG (edit here)
# ----------------------------
SEEDS_TO_RUN = [0]

SSL_MANIFEST_DIR = "./ssl_ckpts_v2"
DATA_GLOB = "/home/ql84/Transformer/csv_data/**/*.csv"
PT_DATA_DIR = "./pt_data"

# Prime from v3; write v3.1 (final) schedule outputs here.
SFT_PRIME_DIR = "./sft_posttrain_v3"
SFT_OUT_DIR = "./sft_posttrain_v3_1"

# SSL steps to use for SFT — per-seq_len mapping.
# (sl=3 and sl=5 already finished on previous machine; excluded here.)
SSL_STEPS_PER_SEQLEN = {
    8:  [100_000],
    10: [100_000],
    20: [100_000],
    30: [100_000],
}

# All 8 folds.
FOLD_FILTER = [
    "Fold_2015-2017_train_SSL",
    "Fold_2015-2018_train_SSL",
    "Fold_2015-2019_train_SSL",
    "Fold_2015-2020_train_SSL",
    "Fold_2015-2021_train_SSL",
    "Fold_2015-2022_train_SSL",
    "Fold_2015-2023_train_SSL",
    "Fold_2015-2024_train_SSL",
]
# GPU selection — defaults to 8 GPUs with 2 workers/GPU (= 16 workers total)
# for the continuation run on 8× RTX 4090. Override via env vars:
#   NUM_GPUS=4 N_WORKERS=8 python run_posttrain_v3_1_step_persl_ssl_v2.py
#   GPU_IDS=0,2,5 N_WORKERS=3 python run_posttrain_v3_1_step_persl_ssl_v2.py
_env_gpu_ids = os.environ.get("GPU_IDS")
_env_num_gpus = os.environ.get("NUM_GPUS")
if _env_gpu_ids:
    GPU_IDS = [int(x) for x in _env_gpu_ids.split(",") if x.strip()]
elif _env_num_gpus:
    GPU_IDS = list(range(int(_env_num_gpus)))
else:
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]   # default: 8 GPUs
# N_WORKERS: default = 2 × len(GPU_IDS) (2 workers per GPU on 24GB+ cards)
N_WORKERS = int(os.environ.get("N_WORKERS", 2 * len(GPU_IDS)))
CPU_THREADS_PER_WORKER = int(os.environ.get("CPU_THREADS_PER_WORKER", 2))

# Per-seqlen schedule (Option B: clean monotonic tiers).
# Rule: total_const_steps decreases with seqlen; cosine phase is fixed 50k.
# For seqlens where total_const ≤ 150k (primer max), start cos directly from
# the corresponding v2 step checkpoint (no const continuation needed).
#
# Final schedule:
#   sl=3:  const 150k→500k (+350k), cos 500k→550k (50k)  [continue from step150k]
#   sl=5:  const 150k→400k (+250k), cos 400k→450k (50k)
#   sl=8:  const 150k→300k (+150k), cos 300k→350k (50k)
#   sl=10: const 150k→250k (+100k), cos 250k→300k (50k)
#   sl=20: const 150k→150k (+  0k), cos 150k→200k (50k)  [primer = cos_start]
#   sl=30: const 100k→100k (+  0k), cos 100k→150k (50k)  [primer = cos_start from v2 step100k]
SFT_RESUME_STEP_PER_SEQLEN = {
    3:  150_000,
    5:  150_000,
    8:  150_000,
    10: 150_000,
    20: 150_000,
    30: 100_000,
}
SFT_TARGET_STEPS_PER_SEQLEN = {
    3:  550_000,
    5:  450_000,
    8:  350_000,
    10: 300_000,
    20: 200_000,
    30: 150_000,
}
SFT_COS_START_PER_SEQLEN = {
    3:  500_000,
    5:  400_000,
    8:  300_000,
    10: 250_000,
    20: 150_000,
    30: 100_000,
}

# Fallback constants (used if a seqlen is missing from the dicts).
SFT_RESUME_STEP = 150_000   # default primer
SFT_TARGET_STEPS = 500_000
SFT_COS_START_STEP = 350_000

SFT_EVAL_EVERY_STEPS = 10_000
SFT_BATCH_SIZE = 4096
SFT_LR_ENCODER = 5e-6
SFT_LR_HEAD = 1.5e-5
# eta_min for cosine phase — keep in useful refinement LR band.
SFT_LR_MIN = 1e-6
# When True, disable cosine entirely (constant LR throughout).  For the final
# schedule we want const → cos, so this is False.
SFT_CONSTANT_LR = False
SFT_WEIGHT_DECAY = 1e-2
SFT_GRAD_CLIP = 1.0
SFT_HUBER_BETA = 1.0


def _target_for(seq_len: int) -> int:
    return int(SFT_TARGET_STEPS_PER_SEQLEN.get(int(seq_len), SFT_TARGET_STEPS))


def _cos_start_for(seq_len: int) -> int:
    return int(SFT_COS_START_PER_SEQLEN.get(int(seq_len), SFT_COS_START_STEP))


def _resume_step_for(seq_len: int) -> int:
    return int(SFT_RESUME_STEP_PER_SEQLEN.get(int(seq_len), SFT_RESUME_STEP))

# data / features
TARGET_COLS = ["Y1"]
WGT_COL = "wgt"
EXCLUDE_COLS_EXTRA = ["Unnamed: 0", "Y0", "Y2", "Y3", WGT_COL]

# Categorical features to EXCLUDE from continuous features (but not used otherwise)
CAT_FEATURE_COLS = ["X0", "X1", "X2", "X3"]

_ALL_COLS = [
    "Unnamed: 0",
    "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9",
    "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19",
    "X20", "X21", "X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29",
    "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39",
    "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49",
    "X50", "X51", "X52", "X53",
    "Y0", "Y1", "Y2", "Y3", "wgt",
]

DROP_LAST_N = 0

# Include first 30 minutes (left-pad incomplete sequences with padding mask)
INCLUDE_OPEN_PERIOD = True

USE_TORCH_COMPILE = False

# AMP
SFT_USE_AMP = True
AMP_DTYPE = "bf16"

# RevIN config (must match SSL pretrain)
USE_REVIN = True
REVIN_EPS = 1e-5
REVIN_AFFINE = True

# Log-transform epsilon for numerical stability
LOG_EPS = 1e-6

# debug limits (optional)
LIMIT_MANIFESTS = None
LIMIT_TASKS = None
SMOKE_TEST_LIMIT_FILES = None

# Wait for checkpoint config
CKPT_WAIT_INTERVAL_SEC = 300
CKPT_MAX_WAIT_HOURS = 24

# Shared task queue config
QUEUE_RETRY_DELAY_SEC = 2
QUEUE_BACKOFF_THRESHOLD = 5

# ----------------------------
# 1) CUDA speed knobs
# ----------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# 2) Logging helpers
# ----------------------------
Path(SFT_OUT_DIR).mkdir(parents=True, exist_ok=True)
MASTER_LOG_PATH = os.path.join(
    SFT_OUT_DIR, f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def master_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_run_logger(path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


# ----------------------------
# 3) Random seed
# ----------------------------
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# 4) AMP context
# ----------------------------
from contextlib import nullcontext


def _autocast_ctx(enabled: bool, device: torch.device, amp_dtype: str):
    if enabled and device.type == "cuda":
        dt = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dt)
    return nullcontext()


def _need_grad_scaler(enabled: bool, device: torch.device, amp_dtype: str) -> bool:
    return bool(enabled and device.type == "cuda" and amp_dtype.lower() == "fp16")


# ----------------------------
# 5) File utils
# ----------------------------
def file_date_str(path: str) -> str:
    base = os.path.basename(path)
    stem = base.rsplit(".", 1)[0]
    m = re.search(r"(20\d{6})", stem)
    return m.group(1) if m else stem


def file_year(path: str) -> int:
    return int(file_date_str(path)[:4])


def date_range(files: List[str]) -> Tuple[str, str]:
    if not files:
        return ("", "")
    ds = [file_date_str(f) for f in files]
    return (min(ds), max(ds))


def split_files_by_years(files: List[str], years: List[int]) -> List[str]:
    ys = set(int(y) for y in years)
    return sorted([f for f in files if file_year(f) in ys])


def parse_end_year_from_fold_name(fold_name: str) -> int:
    m = re.search(r"Fold_2015-(\d{4})_train_SSL", fold_name)
    if not m:
        raise RuntimeError(f"Cannot parse end_year from fold_name={fold_name}")
    return int(m.group(1))


# ----------------------------
# 6) Load all data file list + infer feature columns
# ----------------------------
pt_year_files = sorted(glob.glob(os.path.join(PT_DATA_DIR, "year_*.pt")))
all_files = sorted(glob.glob(DATA_GLOB, recursive=True))

if pt_year_files:
    master_log(f"Using pt_data: {len(pt_year_files)} year files")
elif all_files:
    master_log(f"Using CSV data: {len(all_files)} files")
else:
    raise RuntimeError(f"No .pt files in {PT_DATA_DIR} and no csv under {DATA_GLOB}")

if all_files:
    sample_df = pd.read_csv(all_files[0], nrows=5)
    all_cols = sample_df.columns.tolist()
else:
    all_cols = list(_ALL_COLS)

exclude = set(TARGET_COLS + EXCLUDE_COLS_EXTRA + CAT_FEATURE_COLS)
feature_cols = [c for c in all_cols if c not in exclude]
feature_dim = len(feature_cols)
if feature_dim <= 0:
    raise RuntimeError("No feature columns detected.")
master_log(f"feature_dim={feature_dim} (excluding categorical: {CAT_FEATURE_COLS})")
master_log(f"Example features: {feature_cols[:10]}")

if all_files:
    available_years = sorted({file_year(f) for f in all_files})
else:
    available_years = sorted(
        int(os.path.basename(f).replace("year_", "").replace(".pt", ""))
        for f in pt_year_files
    )
master_log(
    f"available years: {available_years[0]}..{available_years[-1]} (n={len(available_years)})"
)


# ----------------------------
# 7) Mean/Std streaming (Welford) - only for continuous features
# ----------------------------
def compute_mean_std_stream(
    file_paths: Sequence[str], feature_cols: Sequence[str], drop_last_n: int = 0
):
    n = 0
    mean = None
    M2 = None
    for path in file_paths:
        df = pd.read_csv(path, usecols=list(feature_cols))
        if drop_last_n > 0:
            df = df.iloc[:-drop_last_n]
        if len(df) == 0:
            continue

        X = df.to_numpy(dtype=np.float64, copy=False)

        if mean is None:
            mean = np.zeros(X.shape[1], dtype=np.float64)
            M2 = np.zeros(X.shape[1], dtype=np.float64)

        batch_n = X.shape[0]
        batch_mean = X.mean(axis=0)
        batch_var = X.var(axis=0)

        if n == 0:
            mean[:] = batch_mean
            M2[:] = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            new_n = n + batch_n
            mean += delta * (batch_n / new_n)
            M2 += batch_var * batch_n + (delta * delta) * (n * batch_n / new_n)
            n = new_n

    if mean is None:
        raise RuntimeError("compute_mean_std_stream: no data")
    var = M2 / max(n, 1)
    std = np.sqrt(var).astype(np.float32)
    mean = mean.astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


# ----------------------------
# 8) Fast .pt loader + GPU-Resident Dataset
#    Stores flat tensors on GPU, vectorized batch construction.
# ----------------------------


def _load_data_for_years(
    years: List[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    file_paths: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    wgt_col: str = "wgt",
    drop_last_n: int = 0,
    pt_data_dir: str = PT_DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X, W, Y1, day_lengths for the given years.
    Tries .pt files first; falls back to CSV if any year is missing.
    Returns NORMALIZED X, cleaned W and Y1, and day_lengths (all numpy).
    """
    fm = feature_mean.astype(np.float32)
    fs = feature_std.astype(np.float32)
    fs = np.where(fs == 0, 1.0, fs).astype(np.float32)

    pt_paths = {y: os.path.join(pt_data_dir, f"year_{y}.pt") for y in years}
    use_pt = all(os.path.exists(p) for p in pt_paths.values())

    if use_pt:
        all_X_parts, all_w_parts, all_y_parts, all_dl_parts = [], [], [], []
        for y in sorted(years):
            data = torch.load(pt_paths[y], map_location="cpu", weights_only=True)
            all_X_parts.append(data["X"].numpy())
            all_w_parts.append(data["W"].numpy())
            all_y_parts.append(data["Y1"].numpy())
            all_dl_parts.append(data["day_lengths"].numpy())
            del data
        all_X_np = np.concatenate(all_X_parts, axis=0)
        all_y_np = np.concatenate(all_y_parts, axis=0)
        all_w_np = np.concatenate(all_w_parts, axis=0)
        day_lengths_np = np.concatenate(all_dl_parts, axis=0)
        del all_X_parts, all_w_parts, all_y_parts, all_dl_parts
        all_X_np = (all_X_np - fm) / fs
    else:
        if file_paths is None or feature_cols is None:
            raise RuntimeError("No .pt files found and no CSV file_paths/feature_cols provided")
        all_X_list, all_y_list, all_w_list = [], [], []
        day_lengths_list: List[int] = []
        usecols = list(feature_cols) + ["Y1", wgt_col]
        for p in file_paths:
            df = pd.read_csv(p, usecols=usecols)
            if drop_last_n > 0:
                df = df.iloc[:-drop_last_n]
            if len(df) == 0:
                continue
            w = df[wgt_col].to_numpy(dtype=np.float32, copy=True)
            w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)
            w = np.clip(w, 0.0, None)
            y = df["Y1"].to_numpy(dtype=np.float32, copy=True)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            X = (X - fm) / fs
            if X.shape[0] == 0:
                continue
            all_X_list.append(X)
            all_y_list.append(y)
            all_w_list.append(w)
            day_lengths_list.append(X.shape[0])
        if len(day_lengths_list) == 0:
            raise RuntimeError("No valid data found in CSV files")
        all_X_np = np.concatenate(all_X_list, axis=0)
        all_y_np = np.concatenate(all_y_list, axis=0)
        all_w_np = np.concatenate(all_w_list, axis=0)
        day_lengths_np = np.array(day_lengths_list, dtype=np.int64)
        del all_X_list, all_y_list, all_w_list

    return all_X_np, all_w_np, all_y_np, day_lengths_np


class GPUResidentDataset:
    """
    Load data from .pt files (fast) or CSVs (fallback), normalize,
    concatenate into flat GPU tensors.
    Batch construction is done entirely on GPU via vectorized indexing.

    Storage on GPU:
      all_X: (total_timesteps, F) float32
      all_y: (total_timesteps,) float32
      all_w: (total_timesteps,) float32
      day_offsets: (num_days+1,) int64
      index_flat: (N,) int64
      index_day:  (N,) int64
    """

    def __init__(
        self,
        data_years: List[int],
        seq_len: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: torch.device,
        file_paths: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        wgt_col: str = "wgt",
        drop_last_n: int = 0,
    ):
        self.seq_len = int(seq_len)
        self.device = device

        all_X_np, all_w_np, all_y_np, day_lengths_np = _load_data_for_years(
            years=data_years,
            feature_mean=feature_mean,
            feature_std=feature_std,
            file_paths=file_paths,
            feature_cols=feature_cols,
            wgt_col=wgt_col,
            drop_last_n=drop_last_n,
        )

        num_days = len(day_lengths_np)

        # Step 3: Build sample index arrays (on CPU first)
        # day_offsets: cumulative sum, shape (num_days+1,)
        day_offsets_np = np.zeros(num_days + 1, dtype=np.int64)
        np.cumsum(day_lengths_np, out=day_offsets_np[1:])

        # index_flat[i] = flat position in all_X for sample i
        # index_day[i] = which day sample i belongs to
        index_flat_list = []
        index_day_list = []
        for di in range(num_days):
            start = day_offsets_np[di]
            length = day_lengths_np[di]
            index_flat_list.append(np.arange(start, start + length, dtype=np.int64))
            index_day_list.append(np.full(length, di, dtype=np.int64))

        index_flat_np = np.concatenate(index_flat_list)
        index_day_np = np.concatenate(index_day_list)
        del index_flat_list, index_day_list

        self.num_samples = len(index_flat_np)
        self.num_days = num_days

        # Step 4: Move everything to GPU (all_X stored as float16 to save ~50% VRAM)
        self.all_X = torch.from_numpy(all_X_np).to(device=device, dtype=torch.float16)
        self.all_y = torch.from_numpy(all_y_np).to(device=device, dtype=torch.float16)
        self.all_w = torch.from_numpy(all_w_np).to(device=device, dtype=torch.float16)
        self.day_offsets = torch.from_numpy(day_offsets_np).to(device)
        self.index_flat = torch.from_numpy(index_flat_np).to(device)
        self.index_day = torch.from_numpy(index_day_np).to(device)

        # Pre-compute arange for window offsets
        self._offsets = torch.arange(self.seq_len, device=device, dtype=torch.int64)

        # Free numpy arrays
        del all_X_np, all_y_np, all_w_np, index_flat_np, index_day_np

        # Build day-to-sample-indices mapping for DayBatchSampler (CPU lists)
        self.day_to_sample_indices: Dict[int, List[int]] = {}
        cursor = 0
        for di in range(num_days):
            length = int(day_lengths_np[di])
            self.day_to_sample_indices[di] = list(range(cursor, cursor + length))
            cursor += length

    def __len__(self):
        return self.num_samples

    @torch.no_grad()
    def get_batch(
        self, sample_indices: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Fetch a batch directly on GPU using vectorized indexing.

        Args:
            sample_indices: 1D int64 tensor on GPU, shape (B,)

        Returns:
            x_batch: (B, seq_len, F) float32
            y_batch: (B,) float32
            w_batch: (B,) float32
            pad_mask: (B, seq_len) bool, True = padded
            day_idx: (B,) int64
            t_idx: (B,) int64  — position within day
        """
        B = sample_indices.shape[0]
        L = self.seq_len

        flat_positions = self.index_flat[sample_indices]  # (B,)
        day_ids = self.index_day[sample_indices]  # (B,)
        day_starts = self.day_offsets[day_ids]  # (B,)
        t_in_day = flat_positions - day_starts  # (B,)

        # Ideal window start (may be before day boundary for early timesteps)
        ideal_starts = flat_positions - L + 1  # (B,)

        # Build gather indices based on ideal (unclamped) positions: (B, L)
        # gather_idx[b, l] = ideal_starts[b] + l = flat_pos[b] - L + 1 + l
        gather_idx = ideal_starts.unsqueeze(1) + self._offsets.unsqueeze(0)  # (B, L)

        # Clamp to valid range: [day_starts, flat_positions]
        gather_idx = torch.clamp(
            gather_idx,
            min=day_starts.unsqueeze(1),
            max=flat_positions.unsqueeze(1),
        )

        # Fetch data from flat tensors (kept as float16; autocast handles promotion)
        x_batch = self.all_X[gather_idx]  # (B, L, F) float16
        y_batch = self.all_y[flat_positions]  # (B,)
        w_batch = self.all_w[flat_positions]  # (B,)

        # Compute padding mask: positions where ideal_start < day_start need padding
        # pad_len[b] = max(0, seq_len - 1 - t_in_day[b])
        pad_len = (L - 1 - t_in_day).clamp(min=0)  # (B,)

        # Vectorized mask: pad_mask[b, l] = (l < pad_len[b])
        pad_mask = self._offsets.unsqueeze(0) < pad_len.unsqueeze(1)  # (B, L)

        # Zero out padded positions in x_batch
        x_batch = x_batch * (~pad_mask).unsqueeze(-1)

        return x_batch, y_batch, w_batch, pad_mask, day_ids, t_in_day


# ----------------------------
# 9) RevIN (Reversible Instance Normalization) with Padding Mask Support
# ----------------------------
class RevIN(nn.Module):
    """
    Reversible Instance Normalization with padding mask support.
    When a padding_mask is provided, mean and std are computed only over
    non-padded (valid) positions.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "norm",
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if mode == "norm":
            return self._normalize(x, padding_mask)
        elif mode == "denorm":
            raise ValueError("Use denormalize() method with saved statistics")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _normalize(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)

            mean = (x * valid_mask).sum(dim=1, keepdim=True) / count
            diff = (x - mean) * valid_mask
            var = (diff * diff).sum(dim=1, keepdim=True) / count
            std = var.sqrt().clamp(min=self.eps)
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False)
            std = std.clamp(min=self.eps)

        x_norm = (x - mean) / std

        if padding_mask is not None:
            x_norm = x_norm * valid_mask

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
            if padding_mask is not None:
                x_norm = x_norm * valid_mask

        stats = {"mean": mean, "std": std}
        return x_norm, stats

    def denormalize(
        self, x: torch.Tensor, stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mean = stats["mean"]
        std = stats["std"]

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        x_denorm = x * std + mean
        return x_denorm


# ----------------------------
# 10) Model: TransformerSSLRegressorRevINGateLog (no categorical features)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]


class TransformerSSLRegressorRevINGateLog(nn.Module):
    """
    Transformer model with RevIN + Multiplicative Gating + Log(Std) for post-training.
    NO categorical features. Supports padding mask for incomplete sequences.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 2048,
        use_revin: bool = True,
        revin_eps: float = 1e-5,
        revin_affine: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(
                num_features=feature_dim,
                eps=revin_eps,
                affine=revin_affine,
            )
        else:
            self.revin = None

        self.in_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_norm_last = nn.LayerNorm(d_model)

        # scale_gate receives log(std) for multiplicative gating
        self.scale_gate = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.Sigmoid(),
        )

        self.y_head = nn.Linear(d_model, 1)

        # SSL compatibility
        self.out_norm_seq = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, feature_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def init_model_for_training(model: TransformerSSLRegressorRevINGateLog):
    model.apply(init_weights)
    nn.init.normal_(model.y_head.weight, std=1e-3)
    nn.init.zeros_(model.y_head.bias)
    nn.init.zeros_(model.mask_token)
    if hasattr(model, "scale_gate"):
        for m in model.scale_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    if model.use_revin and model.revin is not None and model.revin.affine:
        nn.init.ones_(model.revin.affine_weight)
        nn.init.zeros_(model.revin.affine_bias)


def model_forward_y_revin_gate_log(
    model: TransformerSSLRegressorRevINGateLog,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if model.use_revin and model.revin is not None:
        x_norm, stats = model.revin(x, mode="norm", padding_mask=padding_mask)
        instance_std = stats["std"].squeeze(1)
    else:
        x_norm = x
        instance_std = torch.ones(
            x.size(0), model.feature_dim, device=x.device, dtype=x.dtype
        )

    x1 = model.in_norm(x_norm)
    x1 = model.input_proj(x1)
    x1 = model.pos_encoder(x1)

    if padding_mask is not None:
        h = model.encoder(x1, src_key_padding_mask=padding_mask)
    else:
        h = model.encoder(x1)

    last = model.out_norm_last(h[:, -1, :])

    std_log = torch.log(instance_std + LOG_EPS)
    gate = model.scale_gate(std_log)
    gated_last = last * gate

    return model.y_head(gated_last)


# ----------------------------
# 11) Loss / Metrics
# ----------------------------
def smooth_l1_weighted(pred, target, w, beta: float):
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    w = w.view(-1).float().clamp_min(0.0)

    diff = pred - target
    ad = diff.abs()
    b = float(beta)
    loss = torch.where(ad < b, 0.5 * (diff * diff) / b, ad - 0.5 * b)
    return (loss * w).sum() / w.sum().clamp_min(1e-12)


def weighted_corr(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor) -> float:
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()
    w = w.view(-1).float().clamp_min(0.0)

    mask = torch.isfinite(y_true) & torch.isfinite(y_pred) & torch.isfinite(w)
    y_true, y_pred, w = y_true[mask], y_pred[mask], w[mask]
    if y_true.numel() < 2:
        return 0.0

    wsum = w.sum().clamp_min(1e-12)
    mx = (w * y_true).sum() / wsum
    my = (w * y_pred).sum() / wsum
    vx = (w * (y_true - mx) * (y_true - mx)).sum() / wsum
    vy = (w * (y_pred - my) * (y_pred - my)).sum() / wsum
    if vx.item() <= 0 or vy.item() <= 0:
        return 0.0
    cov = (w * (y_true - mx) * (y_pred - my)).sum() / wsum
    return float((cov / torch.sqrt(vx * vy)).item())


@torch.no_grad()
def evaluate_gpu(
    model,
    dataset: GPUResidentDataset,
    device,
    huber_beta: float,
    batch_size: int = 4096,
):
    """Evaluate using GPU-resident dataset with flat batching."""
    model.eval()
    loss_num = torch.zeros(1, device=device)
    loss_den = torch.zeros(1, device=device)
    ys, ps, ws = [], [], []

    all_indices = torch.arange(len(dataset), device=device)
    for indices in all_indices.split(batch_size):
        x, y, w, pad_mask, day_idx, t_idx = dataset.get_batch(indices)

        pred_raw = model_forward_y_revin_gate_log(model, x, padding_mask=pad_mask).view(-1)
        loss = smooth_l1_weighted(pred_raw, y, w, beta=huber_beta)

        ww = w.detach().float().clamp_min(0.0)
        ww_sum_batch = ww.sum()
        loss_num += loss.detach() * ww_sum_batch
        loss_den += ww_sum_batch

        ys.append(y.detach())
        ps.append(pred_raw.detach())
        ws.append(ww)

    y_all = torch.cat(ys, 0) if ys else torch.zeros(1, device=device)
    p_all = torch.cat(ps, 0) if ps else torch.zeros(1, device=device)
    w_all = torch.cat(ws, 0) if ws else torch.ones(1, device=device)

    return {
        "loss_huber_w": float((loss_num / loss_den.clamp_min(1e-12)).item()),
        "corr_overall_w": float(weighted_corr(y_all, p_all, w_all)),
    }


# ----------------------------
# 12) SSL ckpt loader (pos_encoder.pe compatibility)
# ----------------------------
def load_ssl_into_model_trainonly_revin_scale(
    model: TransformerSSLRegressorRevINGateLog, ckpt_path: str
):
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload.get("state_dict", None)
    if sd is None:
        raise RuntimeError(f"SSL ckpt missing state_dict: {ckpt_path}")

    filtered = {
        k: v
        for k, v in sd.items()
        if not k.startswith("y_head.") and not k.startswith("scale_gate.")
    }

    pe_key = "pos_encoder.pe"
    if pe_key in filtered:
        pe_src = filtered[pe_key]
        pe_tgt = model.state_dict()[pe_key]
        if pe_src.shape != pe_tgt.shape:
            L = min(pe_src.shape[1], pe_tgt.shape[1])
            new_pe = pe_tgt.clone()
            new_pe[:, :L, :] = pe_src[:, :L, :].to(new_pe.dtype)
            filtered[pe_key] = new_pe

    model.load_state_dict(filtered, strict=False)

    with torch.no_grad():
        nn.init.normal_(model.y_head.weight, std=1e-3)
        nn.init.zeros_(model.y_head.bias)

    with torch.no_grad():
        for m in model.scale_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)


# ----------------------------
# 13) Fold data builder
# ----------------------------
MEAN_STD_CACHE = "mean_std_cache.npz"


@dataclass
class FoldData:
    fold_name: str
    train_years: List[int]
    eval_years: List[int]
    train_files: List[str]
    eval_files: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_dim: int


def build_fold_data_from_foldname(fold_name: str) -> FoldData:
    end_year = parse_end_year_from_fold_name(fold_name)
    train_years = list(range(2015, end_year + 1))

    eval_years = []
    if (end_year + 1) in available_years:
        eval_years.append(end_year + 1)

    train_files = split_files_by_years(all_files, train_years) if all_files else []
    eval_files = split_files_by_years(all_files, eval_years) if all_files and eval_years else []

    if not eval_files:
        master_log(
            f"[WARN] {fold_name}: no eval data (year {end_year + 1} not available). Training only, no eval."
        )

    if SMOKE_TEST_LIMIT_FILES is not None:
        train_files = train_files[: int(SMOKE_TEST_LIMIT_FILES)]
        eval_files = eval_files[: int(SMOKE_TEST_LIMIT_FILES)]

    cache_key_mean = f"{fold_name}_mean"
    cache_key_std = f"{fold_name}_std"
    cached = None
    if os.path.exists(MEAN_STD_CACHE):
        try:
            cached = np.load(MEAN_STD_CACHE)
        except Exception:
            cached = None
    if cached is not None and cache_key_mean in cached and cache_key_std in cached:
        fm = cached[cache_key_mean]
        fs = cached[cache_key_std]
    else:
        fm, fs = compute_mean_std_stream(train_files, feature_cols, drop_last_n=DROP_LAST_N)
        existing = dict(np.load(MEAN_STD_CACHE)) if os.path.exists(MEAN_STD_CACHE) else {}
        existing[cache_key_mean] = fm
        existing[cache_key_std] = fs
        np.savez(MEAN_STD_CACHE, **existing)

    return FoldData(
        fold_name=fold_name,
        train_years=train_years,
        eval_years=eval_years,
        train_files=train_files,
        eval_files=eval_files,
        feature_mean=fm,
        feature_std=fs,
        feature_dim=feature_dim,
    )


# ----------------------------
# 14) Checkpointing / resume
# ----------------------------
def _get_orig_model(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def save_ckpt(path: str, model, opt, scaler, meta: Dict):
    payload = {
        "state_dict": _get_orig_model(model).state_dict(),
        "opt_state": opt.state_dict(),
        "scaler_state": scaler.state_dict()
        if (scaler is not None and scaler.is_enabled())
        else None,
        "meta": dict(meta),
        "use_revin": USE_REVIN,
        "scale_injection": True,
        "log_transform": True,
        "has_categorical": False,
    }
    torch.save(payload, path, _use_new_zipfile_serialization=False)


def load_last_ckpt(path: str, model, opt, scaler):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    opt.load_state_dict(payload["opt_state"])
    if (
        scaler is not None
        and scaler.is_enabled()
        and payload.get("scaler_state", None) is not None
    ):
        scaler.load_state_dict(payload["scaler_state"])
    meta = payload.get("meta", {})
    return meta


# ----------------------------
# 15) One task runner (GPU-resident data version)
# ----------------------------
def run_posttrain_one_task(
    gpu_id: int,
    fold_name: str,
    seed: int,
    ssl_step: int,
    ckpt_path: Optional[str],
    seq_len: int,
):
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    seed_everything(seed)

    run_dir = os.path.join(
        SFT_OUT_DIR,
        f"seqlen_{seq_len}",
        fold_name,
        f"seed{seed}",
        f"ssl_step{ssl_step:06d}",
    )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    _log = make_run_logger(os.path.join(run_dir, "run.log"))

    _log(
        f"[TASK START] gpu={gpu_id} fold={fold_name} seed={seed} ssl_step={ssl_step:06d} seq_len={seq_len}"
    )
    _log(
        f"device={device} torch={torch.__version__} cuda={torch.version.cuda if torch.cuda.is_available() else None}"
    )
    _log(f"ckpt_path={ckpt_path}")
    _log(
        f"[CONFIG] USE_REVIN={USE_REVIN} REVIN_AFFINE={REVIN_AFFINE} GATING=Multiplicative LOG_TRANSFORM=True NO_CAT=True PADDING_MASK=True SEQ_LEN={seq_len} GPU_RESIDENT=True"
    )

    # per-process fold cache (keyed by fold_name for mean/std)
    if not hasattr(run_posttrain_one_task, "_fold_cache"):
        run_posttrain_one_task._fold_cache = {}
    # dataset cache keyed by (fold_name, seq_len) since different SEQ_LENs need different datasets
    if not hasattr(run_posttrain_one_task, "_dataset_cache"):
        run_posttrain_one_task._dataset_cache = {}
    fold_cache: Dict[str, FoldData] = run_posttrain_one_task._fold_cache
    dataset_cache: Dict[
        str, Tuple[GPUResidentDataset, Optional[GPUResidentDataset]]
    ] = run_posttrain_one_task._dataset_cache

    if fold_name not in fold_cache:
        _log("[FOLD BUILD] building fold data (mean/std + file lists) ...")
        fd = build_fold_data_from_foldname(fold_name)
        fold_cache[fold_name] = fd
        _log(
            f"[FOLD BUILD DONE] train_days={len(fd.train_files)} range={date_range(fd.train_files)}"
        )
        _log(
            f"[FOLD BUILD DONE] eval_days={len(fd.eval_files)} range={date_range(fd.eval_files)}"
        )
    else:
        fd = fold_cache[fold_name]
        _log(
            f"[FOLD CACHE HIT] train_days={len(fd.train_files)} eval_days={len(fd.eval_files)}"
        )

    ds_cache_key = f"{fold_name}_sl{seq_len}"
    if ds_cache_key not in dataset_cache:
        # Evict stale cache entries to free GPU memory before loading new data
        for old_key in list(dataset_cache.keys()):
            if old_key != ds_cache_key:
                _log(f"[DATASET CACHE EVICT] {old_key}")
                del dataset_cache[old_key]
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _log(
            f"[DATASET BUILD] loading data → GPU tensors (train_years={fd.train_years}) ..."
        )
        train_ds = GPUResidentDataset(
            data_years=fd.train_years,
            seq_len=seq_len,
            feature_mean=fd.feature_mean,
            feature_std=fd.feature_std,
            device=device,
            file_paths=fd.train_files,
            feature_cols=feature_cols,
            wgt_col=WGT_COL,
            drop_last_n=DROP_LAST_N,
        )
        if fd.eval_years:
            eval_ds = GPUResidentDataset(
                data_years=fd.eval_years,
                seq_len=seq_len,
                feature_mean=fd.feature_mean,
                feature_std=fd.feature_std,
                device=device,
                file_paths=fd.eval_files,
                feature_cols=feature_cols,
                wgt_col=WGT_COL,
                drop_last_n=DROP_LAST_N,
            )
        else:
            eval_ds = None
        dataset_cache[ds_cache_key] = (train_ds, eval_ds)
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        _log(
            f"[DATASET BUILD DONE] train_samples={len(train_ds)} eval_samples={len(eval_ds) if eval_ds else 0} GPU_mem={gpu_mem_mb:.0f}MB"
        )
    else:
        train_ds, eval_ds = dataset_cache[ds_cache_key]
        _log(
            f"[DATASET CACHE HIT] train_samples={len(train_ds)} eval_samples={len(eval_ds) if eval_ds else 0}"
        )


    model = TransformerSSLRegressorRevINGateLog(
        feature_dim=int(fd.feature_dim),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=seq_len,
        use_revin=USE_REVIN,
        revin_eps=REVIN_EPS,
        revin_affine=REVIN_AFFINE,
    )
    init_model_for_training(model)

    # load SSL if ssl_step>0 (wait for checkpoint if not exists)
    if ssl_step > 0:
        if ckpt_path is None:
            raise RuntimeError("ssl_step>0 but ckpt_path is None")

        wait_start = time.time()
        max_wait_sec = CKPT_MAX_WAIT_HOURS * 3600
        actual_ckpt_path = ckpt_path

        while True:
            if "*" in ckpt_path:
                matches = sorted(glob.glob(ckpt_path))
                if matches:
                    actual_ckpt_path = matches[0]
                    break
            else:
                if os.path.exists(ckpt_path):
                    actual_ckpt_path = ckpt_path
                    break

            elapsed = time.time() - wait_start
            if elapsed > max_wait_sec:
                raise RuntimeError(
                    f"Timeout waiting for checkpoint after {CKPT_MAX_WAIT_HOURS}h: {ckpt_path}"
                )
            _log(
                f"[WAIT] Checkpoint not found, waiting {CKPT_WAIT_INTERVAL_SEC}s... "
                f"(elapsed: {elapsed / 60:.1f}min, pattern: {ckpt_path})"
            )
            time.sleep(CKPT_WAIT_INTERVAL_SEC)

        _log(f"[LOAD] Loading SSL checkpoint: {actual_ckpt_path}")
        load_ssl_into_model_trainonly_revin_scale(model, actual_ckpt_path)
    else:
        _log("[INFO] ssl_step=0 => training from scratch (no SSL load)")

    model = model.to(device)

    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            _log("[compile] torch.compile OK")
        except Exception as e:
            _log(f"[compile] FAILED -> eager. err={e}")

    # Differential learning rates. Cosine anneal from SFT_LR_{ENCODER,HEAD}
    # down to SFT_LR_MIN over the continuation window
    # (SFT_TARGET_STEPS - SFT_RESUME_STEP). We do NOT warmup — we already
    # resumed from a stable constant-LR run at these peak LRs.
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": SFT_LR_ENCODER},
            {"params": model.revin.parameters(), "lr": SFT_LR_ENCODER},
            {"params": model.y_head.parameters(), "lr": SFT_LR_HEAD},
            {"params": model.scale_gate.parameters(), "lr": SFT_LR_HEAD},
        ],
        weight_decay=SFT_WEIGHT_DECAY,
    )
    use_scaler = _need_grad_scaler(SFT_USE_AMP, device, AMP_DTYPE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    target_steps = _target_for(seq_len)
    cos_start = _cos_start_for(seq_len)
    resume_step = _resume_step_for(seq_len)
    # Cosine phase: from cos_start to target_steps.
    cos_t_max = int(target_steps - cos_start)
    if cos_t_max <= 0:
        raise RuntimeError(
            f"target_steps ({target_steps}) must be > cos_start ({cos_start})"
        )
    # Build scheduler — it will only be stepped when sft_step >= cos_start.
    scheduler = CosineAnnealingLR(opt, T_max=cos_t_max, eta_min=SFT_LR_MIN)

    n_batches_sft = (len(train_ds) + SFT_BATCH_SIZE - 1) // SFT_BATCH_SIZE
    if SFT_CONSTANT_LR:
        _log(
            f"SFT CONSTANT-LR continuation: resume_step={resume_step}, target={target_steps} "
            f"(no decay; head={SFT_LR_HEAD:.3e}, enc={SFT_LR_ENCODER:.3e}), "
            f"eval every {SFT_EVAL_EVERY_STEPS} steps ({n_batches_sft} batches/data-pass)"
        )
    else:
        _log(
            f"SFT const→cos schedule: resume_step={resume_step}, cos_start={cos_start}, "
            f"target={target_steps} (const window {cos_start - resume_step} steps, "
            f"cos T_max={cos_t_max}, eta_min={SFT_LR_MIN}), "
            f"eval every {SFT_EVAL_EVERY_STEPS} steps ({n_batches_sft} batches/data-pass)"
        )

    step_csv = os.path.join(run_dir, "sft_step_log.csv")
    # Snapshot of CSV at the boundary between const-LR and cosine-decay phases,
    # useful for per-phase diagnostics.
    const_only_csv = os.path.join(run_dir, "sft_step_log_const_only.csv")
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    # Prime run_dir from SFT_PRIME_DIR (one-time seed). We copy:
    #   - the primer step{resume_step:06d}.pt (and make it last.pt)
    #   - the CSV log up to sft_step == resume_step
    if not os.path.exists(last_path):
        prime_dir = os.path.join(
            SFT_PRIME_DIR, f"seqlen_{seq_len}", fold_name, f"seed{seed}",
            f"ssl_step{ssl_step:06d}",
        )
        primer_ckpt = os.path.join(prime_dir, f"step{resume_step:06d}.pt")
        if not os.path.exists(primer_ckpt):
            raise RuntimeError(
                f"Primer checkpoint missing: {primer_ckpt}. "
                f"Make sure step_v2 has finished sft_step={resume_step} for this task."
            )
        _log(f"[PRIME] seeding run_dir from {prime_dir}")
        _log(f"[PRIME] copying {primer_ckpt} -> {last_path}")
        shutil.copy2(primer_ckpt, last_path)

        # Copy CSV rows up to (and including) resume_step so the new CSV
        # contains a continuous trajectory from step=SFT_EVAL_EVERY_STEPS..target_steps.
        primer_csv = os.path.join(prime_dir, "sft_step_log.csv")
        if os.path.exists(primer_csv):
            try:
                with open(primer_csv, "r", encoding="utf-8") as fin:
                    header = fin.readline()
                    kept = [header]
                    for line in fin:
                        parts = line.rstrip("\n").split(",")
                        if len(parts) < 4:
                            continue
                        try:
                            s = int(float(parts[3]))
                        except Exception:
                            continue
                        if s <= resume_step:
                            kept.append(line)
                with open(step_csv, "w", encoding="utf-8") as fout:
                    fout.writelines(kept)
                _log(f"[PRIME] copied {len(kept)-1} CSV rows (sft_step <= {resume_step})")
            except Exception as e:
                _log(f"[PRIME][WARN] failed to seed CSV from {primer_csv}: {e}")
                with open(step_csv, "w", encoding="utf-8") as fout:
                    fout.write(
                        "fold,seed,ssl_step,sft_step,train_loss,eval_loss,eval_corr,gpu,seq_len\n"
                    )
        else:
            _log(f"[PRIME][WARN] no primer CSV at {primer_csv}; starting fresh CSV")
            with open(step_csv, "w", encoding="utf-8") as fout:
                fout.write(
                    "fold,seed,ssl_step,sft_step,train_loss,eval_loss,eval_corr,gpu,seq_len\n"
                )

    if not os.path.exists(step_csv):
        with open(step_csv, "w", encoding="utf-8") as f:
            f.write(
                "fold,seed,ssl_step,sft_step,train_loss,eval_loss,eval_corr,gpu,seq_len\n"
            )

    sft_step = 0
    best_corr = -2.0
    next_eval_step = SFT_EVAL_EVERY_STEPS
    data_epoch = 0

    # Always resume from last.pt (which was either present, or just primed above).
    resume_path = last_path
    if os.path.exists(last_path):
        _log(f"[RESUME] loading: {last_path}")
    else:
        step_files = sorted(glob.glob(os.path.join(run_dir, "step*.pt")))
        if step_files:
            resume_path = step_files[-1]
            _log(f"[RESUME] No last.pt, using latest step checkpoint: {resume_path}")

    if os.path.exists(resume_path):
        meta = load_last_ckpt(resume_path, _get_orig_model(model), opt, scaler)
        sft_step = int(meta.get("sft_step", meta.get("sft_global_step", 0)))
        best_corr = float(meta.get("best_corr", -2.0))
        data_epoch = int(meta.get("data_epoch", 0))
        next_eval_step = ((sft_step // SFT_EVAL_EVERY_STEPS) + 1) * SFT_EVAL_EVERY_STEPS
        _log(
            f"[RESUME] sft_step={sft_step}, best_corr={best_corr:.6f}, data_epoch={data_epoch}"
        )

        # On resume: if we're already past cos_start but const_only_csv was never
        # created (e.g. previous run died), reconstruct it by filtering the main CSV.
        if (not SFT_CONSTANT_LR) and (sft_step > cos_start) and (not os.path.exists(const_only_csv)):
            try:
                with open(step_csv, "r", encoding="utf-8") as fin:
                    header = fin.readline()
                    kept = [header]
                    for line in fin:
                        parts = line.rstrip("\n").split(",")
                        if len(parts) < 4:
                            continue
                        try:
                            s = int(float(parts[3]))
                        except Exception:
                            continue
                        if s <= cos_start:
                            kept.append(line)
                with open(const_only_csv, "w", encoding="utf-8") as fout:
                    fout.writelines(kept)
                _log(f"[RESUME-SNAPSHOT] reconstructed {const_only_csv} "
                     f"({len(kept)-1} rows with sft_step <= cos_start={cos_start})")
            except Exception as e:
                _log(f"[RESUME-SNAPSHOT][WARN] failed to reconstruct const-only CSV: {e}")

        # IMPORTANT: we just loaded opt.state_dict() which may carry whatever LR
        # was last saved (that would be the primer's constant LR). Override each
        # param_group with the cosine schedule's peak LR, then advance the
        # scheduler by (sft_step - SFT_RESUME_STEP) so the LR is at the
        # correct position in the decay curve.
        for group, peak_lr in zip(
            opt.param_groups,
            [SFT_LR_ENCODER, SFT_LR_ENCODER, SFT_LR_HEAD, SFT_LR_HEAD],
        ):
            group["lr"] = peak_lr
            group["initial_lr"] = peak_lr
        if SFT_CONSTANT_LR:
            _log(f"[RESUME] CONSTANT LR mode (no scheduler): "
                 f"enc={SFT_LR_ENCODER:.3e} head={SFT_LR_HEAD:.3e}")
        else:
            # Re-create scheduler on top of the (now-peaked) opt so base_lrs are right.
            scheduler = CosineAnnealingLR(opt, T_max=cos_t_max, eta_min=SFT_LR_MIN)
            # Advance scheduler only for the portion of past steps that fell in
            # the cosine phase (sft_step >= cos_start).
            past_cos_steps = max(0, int(sft_step) - int(cos_start))
            for _ in range(past_cos_steps):
                scheduler.step()
            cur_lrs = [g["lr"] for g in opt.param_groups]
            in_cos = "cos-phase" if sft_step >= cos_start else "const-phase"
            _log(f"[RESUME] {in_cos}; cosine advanced by {past_cos_steps} steps "
                 f"(of T_max={cos_t_max}); LRs: "
                 f"enc={cur_lrs[0]:.3e} head={cur_lrs[2]:.3e}")

    if sft_step >= target_steps:
        _log(f"[SKIP] Already finished (step={sft_step} >= {target_steps})")
        del model, opt, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    # Training loop — epochs are just a data-shuffling mechanism
    running_loss = torch.zeros(1, device=device)
    running_wsum = torch.zeros(1, device=device)

    while sft_step < target_steps:
        data_epoch += 1
        model.train()

        shuffle_gen = torch.Generator(device=device)
        shuffle_gen.manual_seed(seed * 7 + data_epoch)
        perm = torch.randperm(len(train_ds), device=device, generator=shuffle_gen)
        batch_indices_list = list(perm.split(SFT_BATCH_SIZE))
        del perm

        for indices in batch_indices_list:
            x, y, w, pad_mask, day_idx, t_idx = train_ds.get_batch(indices)

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(SFT_USE_AMP, device, AMP_DTYPE):
                pred_raw = model_forward_y_revin_gate_log(
                    model, x, padding_mask=pad_mask
                ).view(-1)
                loss = smooth_l1_weighted(pred_raw, y, w, beta=SFT_HUBER_BETA)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_GRAD_CLIP)
                opt.step()

            # Advance cosine decay only once we've entered the cos phase.
            # Note: sft_step has NOT been incremented yet (happens below).
            if (not SFT_CONSTANT_LR) and (sft_step >= cos_start):
                scheduler.step()

            ww = w.detach().float().clamp_min(0.0)
            running_loss += loss.detach() * ww.sum()
            running_wsum += ww.sum()
            sft_step += 1

            # Eval + checkpoint at step boundaries
            is_final = sft_step >= target_steps
            if sft_step >= next_eval_step or is_final:
                train_loss = float((running_loss / running_wsum.clamp_min(1e-12)).item())

                if eval_ds is not None:
                    model.eval()
                    met = evaluate_gpu(
                        model, eval_ds, device, huber_beta=SFT_HUBER_BETA,
                        batch_size=SFT_BATCH_SIZE,
                    )
                    eval_loss = met["loss_huber_w"]
                    eval_corr = met["corr_overall_w"]
                    model.train()
                else:
                    eval_loss = float("nan")
                    eval_corr = float("nan")

                cur_lr_head = opt.param_groups[2]["lr"]
                cur_lr_enc = opt.param_groups[0]["lr"]
                _log(
                    f"[EVAL] fold={fold_name} ssl_step={ssl_step:06d} step={sft_step}/{target_steps} gpu={gpu_id} "
                    f"TrL={train_loss:.6f} EvL={eval_loss:.6f} Corr={eval_corr:.6f} "
                    f"lr_head={cur_lr_head:.3e} lr_enc={cur_lr_enc:.3e}"
                )

                # Before writing the first eval that is in the cosine phase,
                # snapshot the CSV to `sft_step_log_const_only.csv` so we have
                # a clean record of the const-LR-only trajectory.
                # Criteria: sft_step > cos_start AND const-only CSV not yet created.
                if (not SFT_CONSTANT_LR) and (sft_step > cos_start) and (not os.path.exists(const_only_csv)):
                    try:
                        shutil.copy2(step_csv, const_only_csv)
                        _log(f"[SNAPSHOT] const-only trajectory saved to {const_only_csv} "
                             f"(sft_step <= cos_start={cos_start})")
                    except Exception as e:
                        _log(f"[SNAPSHOT][WARN] failed to copy {step_csv} -> {const_only_csv}: {e}")

                with open(step_csv, "a", encoding="utf-8") as f:
                    f.write(
                        f"{fold_name},{seed},{ssl_step},{sft_step},{train_loss},{eval_loss},{eval_corr},{gpu_id},{seq_len}\n"
                    )

                meta = {
                    "fold_name": fold_name,
                    "seed": seed,
                    "ssl_step": int(ssl_step),
                    "sft_step": sft_step,
                    "data_epoch": data_epoch,
                    "best_corr": float(best_corr),
                    "gpu_id": int(gpu_id),
                    "ckpt_path": ckpt_path if ckpt_path is not None else "",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "use_revin": USE_REVIN,
                    "scale_injection": True,
                    "log_transform": True,
                    "has_categorical": False,
                    "seq_len": seq_len,
                }
                save_ckpt(os.path.join(run_dir, f"step{sft_step:06d}.pt"), model, opt, scaler, meta)
                save_ckpt(last_path, model, opt, scaler, meta)

                if eval_ds is not None and eval_corr > best_corr:
                    best_corr = float(eval_corr)
                    meta["best_corr"] = float(best_corr)
                    save_ckpt(best_path, model, opt, scaler, meta)
                    _log(
                        f"[BEST] fold={fold_name} ssl_step={ssl_step:06d} best_corr={best_corr:.6f} @ step={sft_step}"
                    )

                next_eval_step = ((sft_step // SFT_EVAL_EVERY_STEPS) + 1) * SFT_EVAL_EVERY_STEPS
                running_loss = torch.zeros(1, device=device)
                running_wsum = torch.zeros(1, device=device)

            if is_final:
                break

        del batch_indices_list

    _log(
        f"[TASK DONE] fold={fold_name} ssl_step={ssl_step:06d} finished steps={sft_step}/{target_steps}"
    )

    del model, opt, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------
# 16) Discover manifests + build task list (support ssl_step=0)
# ----------------------------
@dataclass(frozen=True)
class Task:
    fold_name: str
    seed: int
    ssl_step: int
    ckpt_path: Optional[str]
    seq_len: int


def is_task_completed(
    fold_name: str, seed: int, ssl_step: int, seq_len: int
) -> Tuple[bool, int]:
    run_dir = os.path.join(
        SFT_OUT_DIR,
        f"seqlen_{seq_len}",
        fold_name,
        f"seed{seed}",
        f"ssl_step{ssl_step:06d}",
    )
    last_path = os.path.join(run_dir, "last.pt")

    if not os.path.exists(last_path):
        return False, 0

    try:
        payload = torch.load(last_path, map_location="cpu")
        meta = payload.get("meta", {})
        step = int(meta.get("sft_step", meta.get("sft_global_step", 0)))
        if step >= _target_for(seq_len):
            return True, step
        return False, step
    except Exception as e:
        master_log(f"[WARN] Failed to load {last_path}: {e}")
        return False, 0


def load_tasks(seeds: List[int]) -> List[Task]:
    all_tasks: List[Task] = []
    completed_tasks: List[Tuple[str, int, int, int, int]] = []
    pending_tasks: List[Task] = []
    in_progress_tasks: List[Tuple[str, int, int, int, int]] = []

    if FOLD_FILTER is not None:
        folds_to_process = FOLD_FILTER
    else:
        folds_to_process = [
            f"Fold_2015-{y}_train_SSL"
            for y in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        ]

    for seq_len in SSL_STEPS_PER_SEQLEN:
        ssl_ckpt_dir = os.path.join(SSL_MANIFEST_DIR, f"seqlen_{seq_len}")
        ssl_steps_for_sl = SSL_STEPS_PER_SEQLEN[seq_len]
        for seed in seeds:
            for fold_name in folds_to_process:
                for ssl_step in ssl_steps_for_sl:
                    ssl_step = int(ssl_step)

                    if ssl_step == 0:
                        task = Task(
                            fold_name=fold_name,
                            seed=seed,
                            ssl_step=0,
                            ckpt_path=None,
                            seq_len=seq_len,
                        )
                    else:
                        ckpt_pattern = (
                            f"ssl_{fold_name}_seed{seed}_step{ssl_step:06d}_train*.pt"
                        )
                        ckpt_matches = sorted(Path(ssl_ckpt_dir).glob(ckpt_pattern))

                        if ckpt_matches:
                            ckpt_path = str(ckpt_matches[0])
                        else:
                            ckpt_path = os.path.join(
                                ssl_ckpt_dir,
                                f"ssl_{fold_name}_seed{seed}_step{ssl_step:06d}_train*.pt",
                            )
                            master_log(
                                f"[INFO] seqlen={seq_len} seed={seed} fold={fold_name} ssl_step={ssl_step} ckpt not found yet, will wait"
                            )

                        task = Task(
                            fold_name=fold_name,
                            seed=seed,
                            ssl_step=ssl_step,
                            ckpt_path=ckpt_path,
                            seq_len=seq_len,
                        )

                    is_done, current_step = is_task_completed(
                        fold_name, seed, ssl_step, seq_len
                    )
                    if is_done:
                        completed_tasks.append(
                            (fold_name, seed, ssl_step, current_step, seq_len)
                        )
                    elif current_step > 0:
                        in_progress_tasks.append(
                            (fold_name, seed, ssl_step, current_step, seq_len)
                        )
                        pending_tasks.append(task)
                    else:
                        pending_tasks.append(task)
                    all_tasks.append(task)

    master_log("=" * 70)
    master_log(
        f"Task Status Summary (seeds={seeds}, seq_lens={list(SSL_STEPS_PER_SEQLEN)}) [SEQ_LEN Ablation Post-train v3 GPU-Resident]"
    )
    master_log("=" * 70)
    master_log(f"Total tasks discovered: {len(all_tasks)}")
    master_log(f"Already completed: {len(completed_tasks)}")
    master_log(f"In progress (will resume): {len(in_progress_tasks)}")
    master_log(f"Pending (not started): {len(pending_tasks) - len(in_progress_tasks)}")
    master_log(f"Tasks to run: {len(pending_tasks)}")

    if completed_tasks:
        master_log("\n[COMPLETED TASKS]")
        for fold_name, seed, ssl_step, step, seq_len in completed_tasks[:10]:
            master_log(
                f"  done: seqlen={seq_len} {fold_name} seed={seed} ssl_step={ssl_step:06d} (finished at step {step})"
            )
        if len(completed_tasks) > 10:
            master_log(f"  ... and {len(completed_tasks) - 10} more")

    if in_progress_tasks:
        master_log("\n[IN PROGRESS TASKS - will resume]")
        for fold_name, seed, ssl_step, current_step, seq_len in in_progress_tasks:
            master_log(
                f"  resuming: seqlen={seq_len} {fold_name} seed={seed} ssl_step={ssl_step:06d} (at step {current_step})"
            )

    master_log("=" * 70)

    if LIMIT_TASKS is not None:
        pending_tasks = pending_tasks[: int(LIMIT_TASKS)]
    return pending_tasks


# ----------------------------
# 17) Shared task queue scheduler
# ----------------------------
def is_ckpt_available(ckpt_path: Optional[str]) -> Tuple[bool, Optional[str]]:
    if ckpt_path is None:
        return True, None
    if "*" in ckpt_path:
        matches = sorted(glob.glob(ckpt_path))
        if matches:
            return True, matches[0]
        return False, None
    else:
        if os.path.exists(ckpt_path):
            return True, ckpt_path
        return False, None


def _mp_worker_fn(
    rank: int,
    task_queue,
    tasks_done,
    total_count: int,
    gpu_ids: list,
):
    import queue as queue_mod

    gpu_id = gpu_ids[rank % len(gpu_ids)] if gpu_ids else -1
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    try:
        os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
        os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
        torch.set_num_threads(CPU_THREADS_PER_WORKER)
    except Exception:
        pass

    master_log(f"[WORKER {rank}] started on gpu={gpu_id}")

    tasks_completed_locally = 0
    consecutive_unavailable = 0
    worker_start = time.time()

    while True:
        if tasks_done.value >= total_count:
            break

        if time.time() - worker_start > CKPT_MAX_WAIT_HOURS * 3600:
            master_log(
                f"[WORKER {rank}] global timeout after {CKPT_MAX_WAIT_HOURS}h, exiting"
            )
            break

        try:
            task_tuple = task_queue.get(timeout=30)
        except queue_mod.Empty:
            if tasks_done.value >= total_count:
                break
            continue

        task = Task(
            fold_name=task_tuple[0],
            seed=task_tuple[1],
            ssl_step=task_tuple[2],
            ckpt_path=task_tuple[3],
            seq_len=task_tuple[4],
        )

        resolved_ckpt_path = task.ckpt_path
        if task.ssl_step > 0 and task.ckpt_path is not None:
            available, resolved = is_ckpt_available(task.ckpt_path)
            if not available:
                task_queue.put(task_tuple)
                consecutive_unavailable += 1
                if consecutive_unavailable >= QUEUE_BACKOFF_THRESHOLD:
                    master_log(
                        f"[WORKER {rank}] {consecutive_unavailable} consecutive unavailable, "
                        f"sleeping {CKPT_WAIT_INTERVAL_SEC}s..."
                    )
                    time.sleep(CKPT_WAIT_INTERVAL_SEC)
                    consecutive_unavailable = 0
                else:
                    time.sleep(QUEUE_RETRY_DELAY_SEC)
                continue
            resolved_ckpt_path = resolved

        consecutive_unavailable = 0

        try:
            master_log(
                f"[WORKER {rank}] starting: seqlen={task.seq_len} fold={task.fold_name} seed={task.seed} "
                f"ssl_step={task.ssl_step} (done={tasks_done.value}/{total_count})"
            )
            run_posttrain_one_task(
                gpu_id=gpu_id,
                fold_name=task.fold_name,
                seed=task.seed,
                ssl_step=task.ssl_step,
                ckpt_path=resolved_ckpt_path,
                seq_len=task.seq_len,
            )
            tasks_completed_locally += 1
        except Exception as e:
            try:
                master_log(
                    f"[MP-ERROR] rank={rank} gpu={gpu_id} seqlen={task.seq_len} fold={task.fold_name} "
                    f"seed={task.seed} ssl_step={task.ssl_step}: {e}"
                )
                master_log(traceback.format_exc())
            except Exception:
                print(
                    f"[MP-ERROR] rank={rank} gpu={gpu_id} seqlen={task.seq_len} fold={task.fold_name} "
                    f"seed={task.seed} ssl_step={task.ssl_step}: {e}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)

        with tasks_done.get_lock():
            tasks_done.value += 1

    master_log(
        f"[WORKER {rank}] exiting. completed {tasks_completed_locally} tasks locally"
    )


def run_all_tasks_multiprocess_pool(tasks: List[Task]):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    tasks_done = ctx.Value("i", 0)

    for t in tasks:
        task_queue.put((t.fold_name, t.seed, t.ssl_step, t.ckpt_path, t.seq_len))

    total_count = len(tasks)

    master_log(f"[MP] Launching {N_WORKERS} workers with shared task queue")
    master_log(f"[MP] Total tasks: {total_count}, GPUs: {GPU_IDS}")

    processes = []
    for rank in range(N_WORKERS):
        p = ctx.Process(
            target=_mp_worker_fn,
            args=(rank, task_queue, tasks_done, total_count, GPU_IDS),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    master_log(
        f"[MP] All workers finished. tasks_done={tasks_done.value}/{total_count}"
    )


# ----------------------------
# 18) MAIN
# ----------------------------
def main():
    tasks = load_tasks(SEEDS_TO_RUN)
    master_log(f"Discovered tasks: {len(tasks)} (seeds={SEEDS_TO_RUN})")
    master_log(f"SSL_STEPS_PER_SEQLEN={SSL_STEPS_PER_SEQLEN}")
    master_log(f"Example tasks: {tasks[:3]}")

    master_log(
        f"RUN: seeds={SEEDS_TO_RUN} seq_lens={list(SSL_STEPS_PER_SEQLEN)} GPUs={GPU_IDS} workers={N_WORKERS} cpu_threads/worker={CPU_THREADS_PER_WORKER}"
    )
    master_log(
        f"SFT_RESUME_STEP_PER_SEQLEN={SFT_RESUME_STEP_PER_SEQLEN} SFT_TARGET_STEPS_PER_SEQLEN={SFT_TARGET_STEPS_PER_SEQLEN} "
        f"eval_every={SFT_EVAL_EVERY_STEPS} SFT_BS={SFT_BATCH_SIZE} "
        f"peak_lr_enc={SFT_LR_ENCODER} peak_lr_head={SFT_LR_HEAD} lr_min={SFT_LR_MIN} "
        f"AMP={SFT_USE_AMP}/{AMP_DTYPE} compile={USE_TORCH_COMPILE}"
    )
    master_log(
        f"PRIME from {SFT_PRIME_DIR} (per-seqlen step)  ->  OUT {SFT_OUT_DIR}"
    )
    master_log(
        f"USE_REVIN={USE_REVIN} REVIN_AFFINE={REVIN_AFFINE} GATING=Multiplicative LOG_TRANSFORM=True NO_CAT=True PADDING_MASK=True"
    )
    master_log("BIG Transformer: d_model=128, nhead=4, num_layers=3, dim_ff=256")
    if SFT_CONSTANT_LR:
        master_log(
            f"CONSTANT LR mode: enc={SFT_LR_ENCODER}, head={SFT_LR_HEAD}, "
            f"WD={SFT_WEIGHT_DECAY}, per-seqlen targets={SFT_TARGET_STEPS_PER_SEQLEN}"
        )
    else:
        master_log(
            f"Const→Cos schedule: peak enc={SFT_LR_ENCODER}, peak head={SFT_LR_HEAD}, "
            f"eta_min={SFT_LR_MIN}, WD={SFT_WEIGHT_DECAY}"
        )
        master_log(f"  per-seqlen targets:     {SFT_TARGET_STEPS_PER_SEQLEN}")
        master_log(f"  per-seqlen cos starts:  {SFT_COS_START_PER_SEQLEN}")
    master_log("GPU-RESIDENT DATA: all data on GPU, vectorized batch construction")
    master_log(f"Outputs -> {os.path.abspath(SFT_OUT_DIR)}")
    master_log(f"Master log -> {os.path.abspath(MASTER_LOG_PATH)}")

    t0 = time.time()
    run_all_tasks_multiprocess_pool(tasks)
    master_log(f"ALL DONE. elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
