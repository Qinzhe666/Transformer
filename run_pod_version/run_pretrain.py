#!/usr/bin/env python3
# =========================
# SSL Pre-train with RevIN (BIG MODEL, FINAL)
#   - RevIN (Reversible Instance Normalization) for handling distribution shift
#   - NO categorical features (X0-X3 ignored)
#   - LEFT-PADDING + PADDING MASK for first 30 minutes (incomplete sequences)
#     → All sequences are fixed length SEQ_LEN=30
#     → For early timesteps, left-pad with 0 and pass padding_mask to Transformer
#     → RevIN computes stats only on non-padded positions
#   - Bigger model: d_model=128, num_layers=3, dim_feedforward=256, dropout=0.2
#   - Seeds: 7
#   - Folds: 2015-2023 only
#   - 200 epochs per (fold, seed) task
#   - 4 parallel processes, one per GPU
#   - X recon + ΔX diff loss, weighted by wgt (only on non-padded positions)
#   - Save checkpoint every 5 epochs
# =========================

import gc
import glob
import json
import math
import os
import random
import re
import traceback
from concurrent.futures import as_completed, ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------
# 0) CONFIG
# ----------------------------
# Train Fold_2015-2025 only
SEEDS = [1, 2, 3]
GPU_IDS = [0, 1, 2]
N_WORKERS = 3

FOLD_FILTER = [
    "Fold_2015-2021_train_SSL",
    "Fold_2015-2022_train_SSL",
    "Fold_2015-2023_train_SSL",
    "Fold_2015-2024_train_SSL",
]

DATA_GLOB = "/workspace/csv_data/**/*.csv"
SSL_SAVE_DIR_BASE = "./ssl_ckpts_seqlen_ablation"

# SEQ_LEN ablation: train with different sequence lengths
SEQ_LENS_TO_RUN = [3, 5, 8, 10]

# SSL hyperparams
SSL_MAX_EPOCHS = 100
SSL_LR = 8e-5
SSL_LR_MIN = 1e-6
SSL_COSINE_START_EPOCH = 101
SSL_WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

SSL_MASK_RATIO = 0.30
SSL_SPAN_MIN = 2
SSL_SPAN_MAX = 6
SSL_DELTA_LAMBDA = 0.2

SSL_BATCH_SIZE = 4096

# RevIN config
USE_REVIN = True
REVIN_EPS = 1e-5
REVIN_AFFINE = True

# Data params
TARGET_COLS = ["Y1"]
WGT_COL = "wgt"
EXCLUDE_COLS_EXTRA = ["Unnamed: 0", "Y0", "Y2", "Y3", WGT_COL]
SEQ_LEN = 30  # default, overridden per-task by SEQ_LENS_TO_RUN
DROP_LAST_N = 0

# Categorical features to EXCLUDE from continuous features (but not used otherwise)
CAT_FEATURE_COLS = ["X0", "X1", "X2", "X3"]

# Include first 30 minutes (left-pad incomplete sequences)
INCLUDE_OPEN_PERIOD = True

SSL_USE_AMP = True
AMP_DTYPE = "bf16"

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
# 2) Logging
# ----------------------------
Path(SSL_SAVE_DIR_BASE).mkdir(parents=True, exist_ok=True)
MASTER_LOG_PATH = os.path.join(
    SSL_SAVE_DIR_BASE, f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
def _autocast_ctx(enabled: bool, device: torch.device, amp_dtype: str):
    if enabled and device.type == "cuda":
        dt = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dt)
    return nullcontext()


def _need_grad_scaler(enabled: bool, device: torch.device, amp_dtype: str) -> bool:
    return bool(enabled and device.type == "cuda" and amp_dtype.lower() == "fp16")


# ----------------------------
# 5) File utilities
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


def split_train_files_by_years(files: List[str], train_years: List[int]) -> List[str]:
    train_set = set(train_years)
    return sorted([f for f in files if file_year(f) in train_set])


# ----------------------------
# 6) Mean/Std streaming (Welford)
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
# 7) GPU-Resident SSL Dataset + Day Batch Sampler
# ----------------------------
class GPUResidentSSLDataset:
    """
    Load per-day CSVs, normalize, concatenate into flat GPU tensors.
    Each epoch iterates through ALL data. Batch construction via GPU indexing.

    Storage on GPU:
      all_X: (total_timesteps, F) float32
      all_w: (total_timesteps,) float32
      day_offsets: (num_days+1,) int64
    """

    def __init__(
        self,
        file_paths: List[str],
        seq_len: int,
        drop_last_n: int,
        feature_cols: List[str],
        wgt_col: str,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: torch.device,
        include_open_period: bool = True,
    ):
        self.seq_len = int(seq_len)
        self.device = device
        self.n_features = len(feature_cols)

        fm = feature_mean.astype(np.float32)
        fs = feature_std.astype(np.float32)
        fs = np.where(fs == 0, 1.0, fs).astype(np.float32)

        all_X_list: List[np.ndarray] = []
        all_w_list: List[np.ndarray] = []
        day_lengths: List[int] = []

        usecols = list(feature_cols) + [wgt_col]
        for p in file_paths:
            df = pd.read_csv(p, usecols=usecols)
            if drop_last_n > 0:
                df = df.iloc[:-drop_last_n]
            if len(df) == 0:
                continue

            w = df[wgt_col].to_numpy(dtype=np.float32, copy=True)
            w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)
            w = np.clip(w, 0.0, None)

            X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            X = (X - fm) / fs

            T = X.shape[0]
            if T == 0:
                continue

            all_X_list.append(X)
            all_w_list.append(w)
            day_lengths.append(T)

        if len(day_lengths) == 0:
            raise RuntimeError("GPUResidentSSLDataset: no valid data.")

        all_X_np = np.concatenate(all_X_list, axis=0)
        all_w_np = np.concatenate(all_w_list, axis=0)
        day_lengths_np = np.array(day_lengths, dtype=np.int64)
        del all_X_list, all_w_list

        num_days = len(day_lengths_np)
        day_offsets_np = np.zeros(num_days + 1, dtype=np.int64)
        np.cumsum(day_lengths_np, out=day_offsets_np[1:])

        min_start = 0 if include_open_period else (seq_len - 1)

        index_flat_list: List[np.ndarray] = []
        index_day_list: List[np.ndarray] = []
        for di in range(num_days):
            start = int(day_offsets_np[di])
            length = int(day_lengths_np[di])
            if length <= min_start:
                continue
            sample_start = start + min_start
            sample_end = start + length
            index_flat_list.append(np.arange(sample_start, sample_end, dtype=np.int64))
            index_day_list.append(
                np.full(sample_end - sample_start, di, dtype=np.int64)
            )

        index_flat_np = np.concatenate(index_flat_list)
        index_day_np = np.concatenate(index_day_list)
        del index_flat_list, index_day_list

        self.num_samples = len(index_flat_np)
        self.num_days = num_days

        self.all_X = torch.from_numpy(all_X_np).to(device)
        self.all_w = torch.from_numpy(all_w_np).to(device)
        self.day_offsets = torch.from_numpy(day_offsets_np).to(device)
        self.index_flat = torch.from_numpy(index_flat_np).to(device)
        self.index_day = torch.from_numpy(index_day_np).to(device)

        self._offsets = torch.arange(seq_len, device=device, dtype=torch.int64)

        del all_X_np, all_w_np, index_flat_np, index_day_np

        self.day_to_sample_indices: Dict[int, List[int]] = {}
        cursor = 0
        for di in range(num_days):
            length = int(day_lengths_np[di])
            valid_length = max(0, length - min_start)
            if valid_length > 0:
                self.day_to_sample_indices[di] = list(
                    range(cursor, cursor + valid_length)
                )
                cursor += valid_length

        # Pre-compute pad_lens on CPU (numpy) — no GPU→CPU sync needed in training loop
        day_starts_repeated = np.repeat(
            day_offsets_np[:-1], np.maximum(0, day_lengths_np - min_start)
        )
        t_in_day_np = self.index_flat.cpu().numpy() - day_starts_repeated
        self.pad_lens_np = np.maximum(0, seq_len - 1 - t_in_day_np).astype(np.int32)
        del day_starts_repeated, t_in_day_np

    def __len__(self):
        return self.num_samples

    @torch.no_grad()
    def get_batch(
        self, sample_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetch a batch on GPU via vectorized indexing.

        Returns:
            x_batch: (B, seq_len, F) float32, left-padded with 0
            w_batch: (B, seq_len) float32, left-padded with 0
            pad_mask: (B, seq_len) bool, True = padded position
        """
        B = sample_indices.shape[0]
        L = self.seq_len

        flat_positions = self.index_flat[sample_indices]
        day_ids = self.index_day[sample_indices]
        day_starts = self.day_offsets[day_ids]
        t_in_day = flat_positions - day_starts

        ideal_starts = flat_positions - L + 1
        gather_idx = ideal_starts.unsqueeze(1) + self._offsets.unsqueeze(0)

        gather_idx = torch.clamp(
            gather_idx,
            min=day_starts.unsqueeze(1),
            max=flat_positions.unsqueeze(1),
        )

        x_batch = self.all_X[gather_idx]
        w_batch = self.all_w[gather_idx]

        pad_len = (L - 1 - t_in_day).clamp(min=0)
        pad_mask = self._offsets.unsqueeze(0) < pad_len.unsqueeze(1)

        x_batch = x_batch * (~pad_mask).unsqueeze(-1).float()
        w_batch = w_batch * (~pad_mask).float()

        return x_batch, w_batch, pad_mask


class DayBatchSampler:
    """Yield sample indices grouped by day, time-ordered within each day."""

    def __init__(
        self,
        dataset: GPUResidentSSLDataset,
        batch_size: int,
        shuffle_days: bool,
        seed: int,
    ):
        self.batch_size = int(batch_size)
        self.shuffle_days = bool(shuffle_days)
        self.seed = int(seed)
        self.epoch = 0
        self.days = sorted(dataset.day_to_sample_indices.keys())
        self.day_to_indices = dataset.day_to_sample_indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        days = self.days[:]
        if self.shuffle_days:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(days)
        for day_i in days:
            idxs = self.day_to_indices[day_i]
            for j in range(0, len(idxs), self.batch_size):
                yield idxs[j : j + self.batch_size]

    def __len__(self):
        n = 0
        for day_i in self.days:
            n += (
                len(self.day_to_indices[day_i]) + self.batch_size - 1
            ) // self.batch_size
        return n


# ----------------------------
# 9) RevIN (Reversible Instance Normalization) with Padding Mask Support
# ----------------------------
class RevIN(nn.Module):
    """
    Reversible Instance Normalization with padding mask support.

    When a padding_mask is provided, mean and std are computed only over
    non-padded (valid) positions, preventing zero-padding from biasing statistics.
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
        """
        Args:
            x: Input tensor of shape (B, L, F)
            mode: "norm" for normalization
            padding_mask: (B, L) bool tensor, True = padded position (optional)
        """
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
        """
        Normalize input to zero mean and unit variance per instance.
        If padding_mask is provided, compute stats only on valid (non-padded) positions.

        Args:
            x: (B, L, F) input tensor
            padding_mask: (B, L) bool tensor, True = padded position

        Returns:
            x_norm: Normalized tensor
            stats: Dict with 'mean' and 'std' for denormalization
        """
        if padding_mask is not None:
            # valid_mask: (B, L, 1), 1.0 for valid positions, 0.0 for padded
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)

            # Masked mean: (B, 1, F)
            mean = (x * valid_mask).sum(dim=1, keepdim=True) / count

            # Masked variance: (B, 1, F)
            diff = (x - mean) * valid_mask
            var = (diff * diff).sum(dim=1, keepdim=True) / count
            std = var.sqrt().clamp(min=self.eps)
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False)
            std = std.clamp(min=self.eps)

        x_norm = (x - mean) / std

        # Zero out padded positions after normalization
        if padding_mask is not None:
            valid_mask_f = (~padding_mask).unsqueeze(-1).float()
            x_norm = x_norm * valid_mask_f

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
            # Zero out padded positions again after affine (affine_bias would leak)
            if padding_mask is not None:
                x_norm = x_norm * valid_mask_f

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
# 10) Model with RevIN (no categorical features)
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


class TransformerSSLRegressorRevIN(nn.Module):
    """
    Transformer model with RevIN for SSL pre-training.
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
        self.use_revin = use_revin
        self.d_model = d_model

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

        self.out_norm_seq = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, feature_dim)

        self.out_norm_last = nn.LayerNorm(d_model)
        self.y_head = nn.Linear(d_model, 1)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def init_model_for_training(model: TransformerSSLRegressorRevIN):
    model.apply(init_weights)
    nn.init.normal_(model.y_head.weight, std=1e-3)
    nn.init.zeros_(model.y_head.bias)
    nn.init.zeros_(model.mask_token)
    if model.use_revin and model.revin is not None and model.revin.affine:
        nn.init.ones_(model.revin.affine_weight)
        nn.init.zeros_(model.revin.affine_bias)


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = bool(flag)


def ssl_forward_recon_revin(
    model: TransformerSSLRegressorRevIN,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Forward pass for SSL reconstruction with RevIN and padding mask support.

    Args:
        model: The transformer model with RevIN
        x: Input tensor (B, L, F), already globally normalized
        padding_mask: (B, L) bool tensor, True = padded position
        time_mask: (B, L) bool tensor, True = SSL-masked position

    Returns:
        recon: Reconstructed tensor in original (globally normalized) scale
        revin_stats: RevIN statistics
    """
    revin_stats = None

    # Apply RevIN normalization (instance-level), respecting padding mask
    if model.use_revin:
        x_norm, revin_stats = model.revin(x, mode="norm", padding_mask=padding_mask)
    else:
        x_norm = x

    # Transformer forward
    x1 = model.in_norm(x_norm)
    x1 = model.input_proj(x1)

    # Apply SSL masking in embedding space
    if time_mask is not None:
        mt = time_mask.unsqueeze(-1).to(x1.dtype)
        x1 = x1 * (1.0 - mt) + model.mask_token * mt

    x1 = model.pos_encoder(x1)

    # Pass padding_mask as src_key_padding_mask to Transformer encoder
    # Transformer expects True = ignore this position
    h = model.encoder(x1, src_key_padding_mask=padding_mask)

    h_seq = model.out_norm_seq(h)
    recon_norm = model.recon_head(h_seq)

    # Denormalize back to original scale
    if model.use_revin:
        recon = model.revin.denormalize(recon_norm, revin_stats)
    else:
        recon = recon_norm

    return recon, revin_stats


# ----------------------------
# 11) SSL mask + loss (respects padding)
# ----------------------------
def make_time_span_mask_fast(
    B: int,
    L: int,
    mask_ratio: float,
    span_min: int,
    span_max: int,
    rng: np.random.RandomState,
    pad_lens: np.ndarray,
) -> np.ndarray:
    """
    Fast numpy span-mask generation using vectorized broadcasting.
    Accepts pre-computed pad_lens (no GPU→CPU sync needed).

    Args:
        pad_lens: (B,) int32 array — number of leading padded positions per sample.
    Returns:
        mask: (B, L) bool array, True = masked position.
    """
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    mask = np.zeros((B, L), dtype=np.bool_)
    positions = np.arange(L, dtype=np.int32)

    valid_starts = pad_lens.astype(np.int32)
    valid_lens = np.maximum(0, L - valid_starts)
    target_k = np.maximum(1, np.round(valid_lens * mask_ratio).astype(np.int32))
    active = valid_lens >= 2

    avg_span = (span_min + span_max) / 2.0
    n_rounds = int(math.ceil(L * mask_ratio / avg_span)) + 4

    for _ in range(n_rounds):
        covered = mask.sum(axis=1)
        need_more = active & (covered < target_k)
        if not need_more.any():
            break

        n_act = int(need_more.sum())
        vs = valid_starts[need_more]
        vl = valid_lens[need_more]
        sl = rng.randint(span_min, span_max + 1, size=n_act)
        max_s = np.maximum(vs, vs + vl - sl)
        rs = np.maximum(1, max_s - vs + 1)
        starts = vs + rng.randint(0, 2**31, size=n_act) % rs
        ends = np.minimum(starts + sl, vs + vl)

        span_mask = (positions[None, :] >= starts[:, None]) & (
            positions[None, :] < ends[:, None]
        )
        mask[need_more] |= span_mask

    all_valid_masked = active.copy()
    for i in np.where(all_valid_masked)[0]:
        vs_i, vl_i = int(valid_starts[i]), int(valid_lens[i])
        if not mask[i, vs_i : vs_i + vl_i].all():
            continue
        mask[i, vs_i + rng.randint(0, vl_i)] = False

    return mask


def masked_smooth_l1_loss_weighted_fast(
    recon: torch.Tensor,
    target: torch.Tensor,
    time_mask: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-12,
    padding_mask: Optional[torch.Tensor] = None,
):
    """
    Compute Huber loss ONLY on masked AND non-padded positions.
    time_mask: (B, L) bool tensor, True = SSL-masked position
    padding_mask: (B, L) bool tensor, True = padded position (to exclude)
    """
    w = weights.to(dtype=recon.dtype)

    # m is non-zero ONLY where time_mask is True AND padding_mask is False
    m = time_mask.to(dtype=recon.dtype) * w
    if padding_mask is not None:
        valid = (~padding_mask).to(dtype=recon.dtype)
        m = m * valid
    m = m.unsqueeze(-1)

    diff = recon - target
    abs_diff = diff.abs()
    beta_t = torch.tensor(beta, device=recon.device, dtype=recon.dtype)

    loss = torch.where(
        abs_diff < beta_t,
        0.5 * (diff * diff) / beta_t,
        abs_diff - 0.5 * beta_t,
    )
    loss = loss * m

    denom = (m.sum() * target.size(-1)).clamp_min(1.0)
    return loss.sum() / (denom + eps)


def ssl_loss_huber_with_delta_weighted(
    recon_x: torch.Tensor,
    x_true: torch.Tensor,
    time_mask: torch.Tensor,
    w: torch.Tensor,
    delta_lambda: float,
    padding_mask: Optional[torch.Tensor] = None,
):
    """
    SSL loss with X reconstruction and delta-X smoothness.
    Computed ONLY on masked AND non-padded regions.
    """
    loss_x = masked_smooth_l1_loss_weighted_fast(
        recon_x,
        x_true,
        time_mask,
        w,
        beta=1.0,
        padding_mask=padding_mask,
    )

    # Delta-X loss
    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dx_hat = recon_x[:, 1:, :] - recon_x[:, :-1, :]
    time_mask_dx = time_mask[:, 1:]
    w_dx = w[:, 1:]

    # For delta loss, valid only if BOTH current and previous are non-padded
    if padding_mask is not None:
        padding_mask_dx = padding_mask[:, 1:] | padding_mask[:, :-1]
    else:
        padding_mask_dx = None

    loss_dx = masked_smooth_l1_loss_weighted_fast(
        dx_hat,
        dx_true,
        time_mask_dx,
        w_dx,
        beta=1.0,
        padding_mask=padding_mask_dx,
    )
    return loss_x + float(delta_lambda) * loss_dx


# ----------------------------
# 12) Checkpoint save/load
# ----------------------------
def ssl_save_ckpt(
    fold_name: str,
    seed: int,
    epoch: int,
    train_loss: float,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler,
    run_dir: str,
    ssl_cfg: Dict,
    feature_dim: int,
) -> str:
    loss_str = f"{train_loss:.6f}".replace(".", "p")
    ckpt_name = f"ssl_{fold_name}_seed{seed}_ep{epoch:03d}_train{loss_str}.pt"
    ckpt_path = os.path.join(run_dir, ckpt_name)

    state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {
        "fold_name": fold_name,
        "seed": seed,
        "epoch": epoch,
        "train_loss": float(train_loss),
        "feature_dim": feature_dim,
        "state_dict": state_cpu,
        "opt_state": opt.state_dict(),
        "scaler_state": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "ssl_cfg": ssl_cfg,
        "use_revin": USE_REVIN,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def ssl_load_ckpt(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer, scaler):
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    if "opt_state" in payload and payload["opt_state"] is not None:
        opt.load_state_dict(payload["opt_state"])
    if scaler and scaler.is_enabled() and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    return payload.get("epoch", 0), payload.get("train_loss", float("inf"))


# ----------------------------
# 13) Worker function (one fold+seed)
# ----------------------------
def ssl_pretrain_worker(task: Dict) -> Dict:
    fold_name = task["fold_name"]
    seed = task["seed"]
    gpu_id = task["gpu_id"]
    train_files = task["train_files"]
    feature_cols = task["feature_cols"]
    feature_dim = task["feature_dim"]
    seq_len = task["seq_len"]
    ssl_save_dir = task["ssl_save_dir"]

    torch.set_num_threads(2)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    seed_everything(seed)

    run_dir = ssl_save_dir
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    _log = make_run_logger(os.path.join(run_dir, f"ssl_{fold_name}_seed{seed}.log"))

    _log(
        f"[SSL START] fold={fold_name} seed={seed} gpu={gpu_id} device={device} seq_len={seq_len}"
    )
    _log(f"[CONFIG] USE_REVIN={USE_REVIN} REVIN_AFFINE={REVIN_AFFINE}")
    _log(f"[CONFIG] INCLUDE_OPEN_PERIOD={INCLUDE_OPEN_PERIOD}")
    _log(f"[CONFIG] No categorical features (X0-X3 excluded)")

    # Compute mean/std
    _log(f"Computing mean/std for {len(train_files)} files...")
    feature_mean, feature_std = compute_mean_std_stream(
        train_files, feature_cols, drop_last_n=DROP_LAST_N
    )
    _log(f"mean/std ready: shape={feature_mean.shape}")

    # Build GPU-resident dataset (all data on GPU)
    _log(f"Building GPU-resident SSL dataset from {len(train_files)} files...")
    train_ds = GPUResidentSSLDataset(
        file_paths=train_files,
        seq_len=seq_len,
        drop_last_n=DROP_LAST_N,
        feature_cols=feature_cols,
        wgt_col=WGT_COL,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
        include_open_period=INCLUDE_OPEN_PERIOD,
    )
    train_sampler = DayBatchSampler(
        train_ds, SSL_BATCH_SIZE, shuffle_days=True, seed=seed * 1000003 + 17,
    )
    gpu_mem_mb = (
        torch.cuda.memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0
    )
    n_batches = (len(train_ds) + SSL_BATCH_SIZE - 1) // SSL_BATCH_SIZE
    _log(
        f"GPU-resident dataset: {len(train_ds)} samples, {train_ds.num_days} days, "
        f"~{n_batches} batches/epoch (bs={SSL_BATCH_SIZE}), GPU_mem={gpu_mem_mb:.0f}MB"
    )

    # Build model - BIG MODEL, no categorical features
    model = TransformerSSLRegressorRevIN(
        feature_dim=feature_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=seq_len,
        use_revin=USE_REVIN,
        revin_eps=REVIN_EPS,
        revin_affine=REVIN_AFFINE,
    ).to(device)
    init_model_for_training(model)
    set_requires_grad(model.y_head, False)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=SSL_LR, weight_decay=SSL_WEIGHT_DECAY)

    def get_cosine_lr(epoch: int) -> float:
        if epoch < SSL_COSINE_START_EPOCH:
            return SSL_LR
        progress = (epoch - SSL_COSINE_START_EPOCH) / (
            SSL_MAX_EPOCHS - SSL_COSINE_START_EPOCH
        )
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return SSL_LR_MIN + (SSL_LR - SSL_LR_MIN) * cosine_factor

    use_scaler = _need_grad_scaler(SSL_USE_AMP, device, AMP_DTYPE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    ssl_cfg = {
        "SEQ_LEN": seq_len,
        "SSL_MAX_EPOCHS": SSL_MAX_EPOCHS,
        "SSL_LR": SSL_LR,
        "SSL_WEIGHT_DECAY": SSL_WEIGHT_DECAY,
        "SSL_MASK_RATIO": SSL_MASK_RATIO,
        "SSL_SPAN_MIN": SSL_SPAN_MIN,
        "SSL_SPAN_MAX": SSL_SPAN_MAX,
        "SSL_DELTA_LAMBDA": SSL_DELTA_LAMBDA,
        "SSL_BATCH_SIZE": SSL_BATCH_SIZE,
        "FULL_DATA_PER_EPOCH": True,
        "USE_REVIN": USE_REVIN,
        "REVIN_EPS": REVIN_EPS,
        "REVIN_AFFINE": REVIN_AFFINE,
        "INCLUDE_OPEN_PERIOD": INCLUDE_OPEN_PERIOD,
    }

    # Resume from last checkpoint if exists
    start_ep = 1
    best_loss = float("inf")
    saved = []

    ckpt_pattern = os.path.join(run_dir, f"ssl_{fold_name}_seed{seed}_ep*.pt")
    existing_ckpts = sorted(glob.glob(ckpt_pattern))
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        _log(f"[RESUME] Found {len(existing_ckpts)} checkpoints, loading {last_ckpt}")
        start_ep, best_loss = ssl_load_ckpt(last_ckpt, model, opt, scaler)
        start_ep += 1
        _log(f"[RESUME] Resuming from epoch {start_ep}")

        manifest_path = os.path.join(run_dir, f"manifest_{fold_name}_seed{seed}.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
                saved = list(existing_manifest.get("saved", []))
                manifest_best = existing_manifest.get("best_loss", float("inf"))
                if manifest_best < best_loss:
                    best_loss = manifest_best
                _log(f"[RESUME] Loaded {len(saved)} epochs from existing manifest")
            except Exception as e:
                _log(f"[WARN] Failed to load manifest, rebuilding from ckpts: {e}")
                ep_pattern = re.compile(r"_ep(\d+)_train(\dp\d+)\.pt$")
                for ckpt in existing_ckpts:
                    match = ep_pattern.search(ckpt)
                    if match:
                        ep_num = int(match.group(1))
                        loss_str = match.group(2).replace("p", ".")
                        saved.append(
                            {
                                "epoch": ep_num,
                                "train_loss": float(loss_str),
                                "path": ckpt,
                            }
                        )
                saved = sorted(saved, key=lambda d: d["epoch"])
                _log(f"[RESUME] Rebuilt {len(saved)} epochs from ckpt files")

    if start_ep > SSL_MAX_EPOCHS:
        _log(f"[SKIP] Already finished (start_ep={start_ep} > {SSL_MAX_EPOCHS})")
        return {"fold_name": fold_name, "seed": seed, "status": "skipped"}

    # Training loop — each epoch iterates through ALL data
    for ep in range(start_ep, SSL_MAX_EPOCHS + 1):
        current_lr = get_cosine_lr(ep)
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr
        if ep >= SSL_COSINE_START_EPOCH:
            _log(f"[LR DECAY] epoch={ep:03d} lr={current_lr:.2e}")

        model.train()

        # Flat shuffle: generate perm on GPU, keep CPU copy for mask gen
        shuffle_gen = torch.Generator(device=device)
        shuffle_gen.manual_seed(seed * 7 + ep)
        perm = torch.randperm(len(train_ds), device=device, generator=shuffle_gen)
        perm_np = perm.cpu().numpy()
        batch_indices_list = list(perm.split(SSL_BATCH_SIZE))
        del perm
        torch.cuda.empty_cache()

        rng = np.random.RandomState(seed + ep * 1000003)

        # Accumulate loss on GPU to avoid per-batch GPU→CPU sync
        running_loss = torch.zeros(1, device=device)
        nb = 0

        for batch_i, indices in enumerate(batch_indices_list):
            x, w, pad_mask = train_ds.get_batch(indices)
            B, L, _F = x.shape

            # Per-batch mask: tiny CPU arrays, no GPU→CPU sync
            start = batch_i * SSL_BATCH_SIZE
            batch_pad_lens = train_ds.pad_lens_np[perm_np[start : start + B]]
            time_mask_np = make_time_span_mask_fast(
                B=B, L=L,
                mask_ratio=SSL_MASK_RATIO,
                span_min=SSL_SPAN_MIN,
                span_max=SSL_SPAN_MAX,
                rng=rng,
                pad_lens=batch_pad_lens,
            )
            time_mask = torch.from_numpy(time_mask_np).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_scaler:
                with _autocast_ctx(True, device, AMP_DTYPE):
                    recon, _ = ssl_forward_recon_revin(
                        model, x, padding_mask=pad_mask, time_mask=time_mask
                    )
                    loss_total = ssl_loss_huber_with_delta_weighted(
                        recon_x=recon,
                        x_true=x,
                        time_mask=time_mask,
                        w=w,
                        delta_lambda=SSL_DELTA_LAMBDA,
                        padding_mask=pad_mask,
                    )
                scaler.scale(loss_total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                with _autocast_ctx(SSL_USE_AMP, device, AMP_DTYPE):
                    recon, _ = ssl_forward_recon_revin(
                        model, x, padding_mask=pad_mask, time_mask=time_mask
                    )
                    loss_total = ssl_loss_huber_with_delta_weighted(
                        recon_x=recon,
                        x_true=x,
                        time_mask=time_mask,
                        w=w,
                        delta_lambda=SSL_DELTA_LAMBDA,
                        padding_mask=pad_mask,
                    )
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            running_loss += loss_total.detach()
            nb += 1

        del batch_indices_list, perm_np
        train_loss = float(running_loss.item() / max(1, nb))

        _log(f"[{fold_name}][seed {seed}] ep={ep:03d} | loss={train_loss:.6f}")

        # Save checkpoint every 5 epochs
        if ep % 5 == 0 or ep == SSL_MAX_EPOCHS:
            ckpt_path = ssl_save_ckpt(
                fold_name=fold_name,
                seed=seed,
                epoch=ep,
                train_loss=train_loss,
                model=model,
                opt=opt,
                scaler=scaler,
                run_dir=run_dir,
                ssl_cfg=ssl_cfg,
                feature_dim=feature_dim,
            )
            saved.append({"epoch": ep, "train_loss": train_loss, "path": ckpt_path})

        if train_loss < best_loss:
            best_loss = train_loss

    # Save manifest
    manifest = {
        "fold_name": fold_name,
        "seed": seed,
        "feature_dim": feature_dim,
        "ssl_cfg": ssl_cfg,
        "saved": sorted(saved, key=lambda d: d["epoch"]),
        "best_loss": float(best_loss),
        "use_revin": USE_REVIN,
    }
    manifest_path = os.path.join(run_dir, f"manifest_{fold_name}_seed{seed}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _log(
        f"[{fold_name}][seed {seed}] DONE. saved={len(saved)} best_loss={best_loss:.6f}"
    )

    # Cleanup
    del model, opt, scaler, train_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold_name": fold_name,
        "seed": seed,
        "status": "done",
        "best_loss": best_loss,
    }


# ----------------------------
# 14) Main: build tasks and run parallel
# ----------------------------
def main():
    master_log("=" * 80)
    master_log(
        "SSL Pre-train SEQ_LEN Ablation: RevIN + Left-Padding (No Categorical Features)"
    )
    master_log(f"Seeds: {SEEDS}")
    master_log(f"GPUs: {GPU_IDS}")
    master_log(f"Workers: {N_WORKERS}")
    master_log(f"SEQ_LENS: {SEQ_LENS_TO_RUN}")
    master_log(f"USE_REVIN: {USE_REVIN}")
    master_log(f"REVIN_AFFINE: {REVIN_AFFINE}")
    master_log(f"INCLUDE_OPEN_PERIOD: {INCLUDE_OPEN_PERIOD}")
    master_log(f"No categorical features (X0-X3 excluded from model)")
    master_log("=" * 80)

    # Load all files and infer feature columns
    all_files = sorted(glob.glob(DATA_GLOB, recursive=True))
    if len(all_files) == 0:
        raise RuntimeError(f"No csv found under {DATA_GLOB}")
    master_log(f"Total files: {len(all_files)}")

    sample_df = pd.read_csv(all_files[0], nrows=5)
    all_cols = sample_df.columns.tolist()
    exclude = set(TARGET_COLS + EXCLUDE_COLS_EXTRA + CAT_FEATURE_COLS)
    feature_cols = [c for c in all_cols if c not in exclude]
    feature_dim = len(feature_cols)
    master_log(f"feature_dim={feature_dim} (excluding categorical: {CAT_FEATURE_COLS})")

    # Build folds
    def _make_fold(end_year: int) -> Dict:
        train_years = list(range(2015, end_year + 1))
        name = f"Fold_2015-{end_year}_train_SSL"
        train_files = split_train_files_by_years(all_files, train_years)
        return {"name": name, "train_years": train_years, "train_files": train_files}

    all_folds = [
        _make_fold(y) for y in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    ]
    folds = [f for f in all_folds if f["name"] in FOLD_FILTER]

    # Build task list
    tasks = []
    gpu_idx = 0
    for fold in folds:
        for sl in SEQ_LENS_TO_RUN:
            ssl_save_dir = os.path.join(SSL_SAVE_DIR_BASE, f"seqlen_{sl}")
            for seed in SEEDS:
                task = {
                    "fold_name": fold["name"],
                    "seed": seed,
                    "gpu_id": GPU_IDS[gpu_idx % len(GPU_IDS)],
                    "train_files": fold["train_files"],
                    "feature_cols": feature_cols,
                    "feature_dim": feature_dim,
                    "seq_len": sl,
                    "ssl_save_dir": ssl_save_dir,
                }
                tasks.append(task)
                gpu_idx += 1

    master_log(f"Total tasks: {len(tasks)}")
    for t in tasks[:10]:
        master_log(f"  {t['fold_name']} seed={t['seed']} gpu={t['gpu_id']}")
    if len(tasks) > 10:
        master_log(f"  ... and {len(tasks) - 10} more")

    # Run with ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_task = {executor.submit(ssl_pretrain_worker, t): t for t in tasks}

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                master_log(
                    f"[DONE] {result['fold_name']} seed={result['seed']} "
                    f"status={result['status']}"
                )
            except Exception as e:
                master_log(f"[ERROR] {task['fold_name']} seed={task['seed']}: {e}")
                traceback.print_exc()

    master_log("=" * 80)
    master_log(f"All tasks completed: {len(results)}/{len(tasks)}")
    master_log("=" * 80)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
