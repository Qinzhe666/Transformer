#!/usr/bin/env python3
"""Seqlen-ensemble evaluation pipeline.

    1. Given --dev-folds, pick per-seqlen deploy sft_step via mean-regret
       plateau on the DEV folds' sft_step_log.csv files.
    2. Run inference on DEV folds, rank all seqlen subsets by avg IC.
    3. Run inference on --holdout-folds and score the top-K DEV combos.

Usage:
    python eval_ensemble.py \
        --dev-folds Fold_2015-2017_train_SSL ... Fold_2015-2022_train_SSL \
        --holdout-folds Fold_2015-2023_train_SSL Fold_2015-2024_train_SSL
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Constants ────────────────────────────────────────────────────────────────
PT_DATA_DIR = "./pt_data"
MEAN_STD_CACHE = "mean_std_cache.npz"
DATA_GLOB = "/home/ql84/Transformer/csv_data/**/*.csv"
BATCH_SIZE = 4096
LOG_EPS = 1e-6
SSL_STEP = 100_000
SEED = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step-selection: read sft_step_log.csv from v3_1 (with v3 primer) or v3.
SFT_V3 = "./sft_posttrain_v3"
SFT_V3_1 = "./sft_posttrain_v3_1"
PRIMER_STEP = 150_000   # step ≤ PRIMER → v3 dir, else v3_1
MA_WINDOW = 5

DEFAULT_DEV_FOLDS = [f"Fold_2015-{y}_train_SSL" for y in range(2017, 2023)]
DEFAULT_HOLDOUT_FOLDS = [f"Fold_2015-{y}_train_SSL" for y in (2023, 2024)]
DEFAULT_SEQLENS = [3, 5, 8, 10, 20, 30]


# ── Small utilities ──────────────────────────────────────────────────────────
def fold_to_eval_year(fold: str) -> int:
    m = re.search(r"Fold_\d{4}-(\d{4})_", fold)
    if not m:
        raise ValueError(f"Cannot parse eval year from {fold!r}")
    return int(m.group(1)) + 1


def discover_feature_dim() -> int:
    df = pd.read_csv(sorted(glob.glob(DATA_GLOB, recursive=True))[0], nrows=2)
    exclude = {"Y1", "Unnamed: 0", "Y0", "Y2", "Y3", "wgt", "X0", "X1", "X2", "X3"}
    return sum(c not in exclude for c in df.columns)


def weighted_corr(y_true, y_pred, w) -> float:
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred) & torch.isfinite(w)
    y_true, y_pred, w = y_true[mask], y_pred[mask], w[mask]
    if y_true.numel() < 2:
        return 0.0
    wsum = w.sum().clamp_min(1e-12)
    mx = (w * y_true).sum() / wsum
    my = (w * y_pred).sum() / wsum
    vx = (w * (y_true - mx) ** 2).sum() / wsum
    vy = (w * (y_pred - my) ** 2).sum() / wsum
    if vx.item() <= 0 or vy.item() <= 0:
        return 0.0
    cov = (w * (y_true - mx) * (y_pred - my)).sum() / wsum
    return float((cov / torch.sqrt(vx * vy)).item())


# ── Model (must match training) ──────────────────────────────────────────────
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode="norm", padding_mask=None):
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * valid).sum(dim=1, keepdim=True) / count
            diff = (x - mean) * valid
            var = (diff * diff).sum(dim=1, keepdim=True) / count
            std = var.sqrt().clamp(min=self.eps)
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=self.eps)
        x_norm = (x - mean) / std
        if padding_mask is not None:
            x_norm = x_norm * valid
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
            if padding_mask is not None:
                x_norm = x_norm * valid
        return x_norm, {"mean": mean, "std": std}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerSSLRegressorRevINGateLog(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.0, max_len=2048,
                 use_revin=True, revin_eps=1e-5, revin_affine=True):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.use_revin = use_revin
        self.revin = RevIN(feature_dim, eps=revin_eps, affine=revin_affine) if use_revin else None
        self.in_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm_last = nn.LayerNorm(d_model)
        self.scale_gate = nn.Sequential(nn.Linear(feature_dim, d_model), nn.Sigmoid())
        self.y_head = nn.Linear(d_model, 1)
        self.out_norm_seq = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, feature_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))


def model_forward(model, x, padding_mask=None):
    if model.use_revin and model.revin is not None:
        x_norm, stats = model.revin(x, mode="norm", padding_mask=padding_mask)
        instance_std = stats["std"].squeeze(1)
    else:
        x_norm = x
        instance_std = torch.ones(x.size(0), model.feature_dim, device=x.device, dtype=x.dtype)
    h = model.input_proj(model.in_norm(x_norm))
    h = model.pos_encoder(h)
    h = model.encoder(h, src_key_padding_mask=padding_mask) if padding_mask is not None else model.encoder(h)
    last = model.out_norm_last(h[:, -1, :])
    gate = model.scale_gate(torch.log(instance_std + LOG_EPS))
    return model.y_head(last * gate)


def load_model(path: str, feature_dim: int):
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    pe_len = payload["state_dict"]["pos_encoder.pe"].shape[1]
    model = TransformerSSLRegressorRevINGateLog(
        feature_dim=feature_dim, d_model=128, nhead=4, num_layers=3,
        dim_feedforward=256, dropout=0.0, max_len=pe_len,
        use_revin=True, revin_eps=1e-5, revin_affine=True,
    ).to(DEVICE)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


# ── Step 1: pick per-seqlen deploy sft_step (plateau_hi of mean regret) ──────
def _load_traj(sl: int, fold: str) -> pd.DataFrame:
    for base in (SFT_V3_1, SFT_V3):
        p = os.path.join(base, f"seqlen_{sl}", fold, f"seed{SEED}",
                         f"ssl_step{SSL_STEP:06d}", "sft_step_log.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)[["sft_step", "eval_corr"]]
            df = df.drop_duplicates("sft_step").sort_values("sft_step").reset_index(drop=True)
            df["ma5"] = df["eval_corr"].rolling(MA_WINDOW, min_periods=1).mean()
            return df
    raise FileNotFoundError(f"No sft_step_log.csv for sl={sl} fold={fold}")


def select_deploy(seqlens: Sequence[int], dev_folds: Sequence[str],
                  tol_bps: float) -> Dict[int, Tuple[str, int]]:
    """For each sl: take plateau_hi — largest step whose mean regret (across
    dev folds) is within `tol_bps` bps of the min."""
    tol = tol_bps * 1e-4
    out: Dict[int, Tuple[str, int]] = {}
    for sl in seqlens:
        regs = [_load_traj(sl, f).assign(regret=lambda d: d["ma5"].max() - d["ma5"])
                                 [["sft_step", "regret"]] for f in dev_folds]
        common = sorted(set.intersection(*[set(r["sft_step"]) for r in regs]))
        mr = pd.DataFrame({"sft_step": common})
        mr["mean_regret"] = [
            float(np.mean([r.loc[r["sft_step"] == s, "regret"].iloc[0] for r in regs]))
            for s in common
        ]
        mn = mr["mean_regret"].min()
        step = int(mr.loc[mr["mean_regret"] <= mn + tol, "sft_step"].max())
        base = SFT_V3_1 if step > PRIMER_STEP else SFT_V3
        out[sl] = (base, step)
        print(f"  sl={sl:>2}  n_steps={len(mr):>3}  chosen={step:>6d} "
              f"(min_regret={mn*100:.4f}%  tol={tol_bps}bps)  base={base}")
    return out


def ckpt_path(deploy: Dict[int, Tuple[str, int]], sl: int, fold: str) -> str:
    base, step = deploy[sl]
    return os.path.join(base, f"seqlen_{sl}", fold, f"seed{SEED}",
                        f"ssl_step{SSL_STEP:06d}", f"step{step:06d}.pt")


# ── Per-year data ────────────────────────────────────────────────────────────
class YearData:
    def __init__(self, eval_year: int, device):
        data = torch.load(os.path.join(PT_DATA_DIR, f"year_{eval_year}.pt"),
                          map_location="cpu", weights_only=True)
        dl = data["day_lengths"].numpy()
        day_offsets = np.zeros(len(dl) + 1, dtype=np.int64)
        np.cumsum(dl, out=day_offsets[1:])
        index_flat = np.concatenate([
            np.arange(day_offsets[i], day_offsets[i] + dl[i], dtype=np.int64)
            for i in range(len(dl))
        ])
        index_day = np.concatenate([np.full(dl[i], i, dtype=np.int64) for i in range(len(dl))])
        self.device = device
        self.X_raw = data["X"].to(device)
        self.Y = data["Y1"].to(device)
        self.W = data["W"].to(device)
        self.day_offsets = torch.from_numpy(day_offsets).to(device)
        self.index_flat = torch.from_numpy(index_flat).to(device)
        self.index_day = torch.from_numpy(index_day).to(device)
        self.num_samples = len(index_flat)


def normalize_X(year_data: YearData, fold_mean: np.ndarray, fold_std: np.ndarray):
    fm = torch.from_numpy(fold_mean.astype(np.float32)).to(year_data.device)
    fs = torch.from_numpy(np.where(fold_std == 0, 1.0, fold_std).astype(np.float32)).to(year_data.device)
    return (year_data.X_raw - fm) / fs


@torch.no_grad()
def get_batch(yd: YearData, X_norm: torch.Tensor, idx: torch.Tensor,
              seq_len: int, offsets: torch.Tensor):
    flat = yd.index_flat[idx]
    day_starts = yd.day_offsets[yd.index_day[idx]]
    gather_idx = torch.clamp(
        (flat - seq_len + 1).unsqueeze(1) + offsets.unsqueeze(0),
        min=day_starts.unsqueeze(1), max=flat.unsqueeze(1),
    )
    pad_len = (seq_len - 1 - (flat - day_starts)).clamp(min=0)
    pad_mask = offsets.unsqueeze(0) < pad_len.unsqueeze(1)
    x = X_norm[gather_idx] * (~pad_mask).unsqueeze(-1).float()
    return x, pad_mask


# ── Inference: build per-(fold, seqlen) predictions ──────────────────────────
def build_predictions(deploy, folds, eval_years, seqlens, feature_dim):
    cache = np.load(MEAN_STD_CACHE)
    preds: Dict[int, Dict[int, torch.Tensor]] = {}
    ys: Dict[int, torch.Tensor] = {}
    ws: Dict[int, torch.Tensor] = {}

    for fold, year in zip(folds, eval_years):
        print(f"  [year {year}]  fold={fold}")
        yd = YearData(year, DEVICE)
        X_norm = normalize_X(yd, cache[f"{fold}_mean"], cache[f"{fold}_std"])
        ys[year], ws[year], preds[year] = yd.Y.clone(), yd.W.clone(), {}

        for sl in seqlens:
            cp = ckpt_path(deploy, sl, fold)
            if not os.path.exists(cp):
                print(f"    sl={sl:>2}  MISSING: {cp}")
                continue
            model = load_model(cp, feature_dim)
            offsets = torch.arange(sl, device=DEVICE, dtype=torch.int64)
            buf = torch.empty(yd.num_samples, device=DEVICE, dtype=torch.float32)
            pos = 0
            for chunk in torch.arange(yd.num_samples, device=DEVICE).split(BATCH_SIZE):
                x, pad = get_batch(yd, X_norm, chunk, sl, offsets)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pr = model_forward(model, x, padding_mask=pad).view(-1)
                buf[pos:pos + pr.numel()] = pr.float()
                pos += pr.numel()
            preds[year][sl] = buf
            ic = weighted_corr(ys[year], buf, ws[year])
            print(f"    sl={sl:>2}  n={yd.num_samples:>9d}  IC={ic*100:+.4f}%")
            del model
            torch.cuda.empty_cache()

        del X_norm, yd
        torch.cuda.empty_cache()
    return preds, ys, ws


# ── Ensemble scoring ─────────────────────────────────────────────────────────
def ensemble_ics(combo, preds, ys, ws, years) -> List[float]:
    out = []
    for y in years:
        pr = [preds[y][s] for s in combo if s in preds[y]]
        if not pr:
            out.append(float("nan"))
            continue
        ens = torch.stack(pr, dim=0).mean(dim=0)
        out.append(weighted_corr(ys[y], ens, ws[y]))
    return out


def combo_label(combo) -> str:
    return "+".join(f"sl{s}" for s in combo)


def print_table(years, rows):
    """rows: list of (label, [ic_per_year])."""
    print(f"  {'Rank':>4s}  {'Combination':>33s}", end="")
    for y in years:
        print(f"  {y:>8d}", end="")
    print(f"  {'AVG':>8s}  {'MIN':>8s}")
    for i, (label, yics) in enumerate(rows):
        print(f"  {i+1:>4d}  {label:>33s}", end="")
        for ic in yics:
            print(f"  {ic*100:>8.4f}", end="")
        print(f"  {np.nanmean(yics)*100:>8.4f}  {np.nanmin(yics)*100:>8.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-folds", nargs="+", default=DEFAULT_DEV_FOLDS)
    ap.add_argument("--holdout-folds", nargs="+", default=DEFAULT_HOLDOUT_FOLDS)
    ap.add_argument("--seqlens", type=int, nargs="+", default=DEFAULT_SEQLENS)
    ap.add_argument("--tol-bps", type=float, default=5.0,
                    help="Step-selection plateau tolerance (bps of IC).")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of DEV-ranked ensembles to validate on HOLDOUT.")
    cfg = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    feature_dim = discover_feature_dim()
    dev_years = [fold_to_eval_year(f) for f in cfg.dev_folds]
    ho_years = [fold_to_eval_year(f) for f in cfg.holdout_folds]

    print(f"Device: {DEVICE}  feature_dim: {feature_dim}  seqlens: {cfg.seqlens}")
    print(f"DEV     ({len(cfg.dev_folds)}): {cfg.dev_folds} → {dev_years}")
    print(f"HOLDOUT ({len(cfg.holdout_folds)}): {cfg.holdout_folds} → {ho_years}")

    # Step 1 ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Picking per-seqlen deploy step (tol={cfg.tol_bps}bps, plateau_hi)")
    deploy = select_deploy(cfg.seqlens, cfg.dev_folds, cfg.tol_bps)

    # Step 2 ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 2] DEV inference + ensemble search")
    dev_preds, dev_ys, dev_ws = build_predictions(
        deploy, cfg.dev_folds, dev_years, cfg.seqlens, feature_dim)
    combos = [c for sz in range(1, len(cfg.seqlens) + 1)
              for c in combinations(cfg.seqlens, sz)]
    dev_results = sorted(
        ((c, ensemble_ics(c, dev_preds, dev_ys, dev_ws, dev_years)) for c in combos),
        key=lambda r: float(np.nanmean(r[1])), reverse=True,
    )
    print(f"\n  Top {cfg.top_k} DEV ensembles by avg IC:")
    print_table(dev_years, [(combo_label(c), y) for c, y in dev_results[:cfg.top_k]])

    top_combos = [c for c, _ in dev_results[:cfg.top_k]]
    dev_lookup = dict(dev_results)
    del dev_preds, dev_ys, dev_ws
    torch.cuda.empty_cache()

    # Step 3 ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 3] HOLDOUT inference + scoring top-{cfg.top_k} DEV combos")
    ho_preds, ho_ys, ho_ws = build_predictions(
        deploy, cfg.holdout_folds, ho_years, cfg.seqlens, feature_dim)

    print(f"\n  DEV-selected ensembles on HOLDOUT (DEV rank order):")
    print(f"  {'Rank':>4s}  {'Combination':>33s}  {'DEV_avg':>9s}  {'DEV_min':>9s}  {'HO_avg':>9s}  {'HO_min':>9s}")
    for i, combo in enumerate(top_combos):
        d = dev_lookup[combo]
        h = ensemble_ics(combo, ho_preds, ho_ys, ho_ws, ho_years)
        print(f"  {i+1:>4d}  {combo_label(combo):>33s}  "
              f"{np.nanmean(d)*100:>9.4f}  {np.nanmin(d)*100:>9.4f}  "
              f"{np.nanmean(h)*100:>9.4f}  {np.nanmin(h)*100:>9.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
