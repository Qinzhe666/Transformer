#!/usr/bin/env python3
"""Gather supporting statistics for paper_v2.md.

Prints to stdout; redirect to logs/paper_stats.log.

1. SSL-step ablation: for each (sl, fold) in v3, compare best MA5 eval_corr
   at ssl_step=20K vs 50K vs 100K. Shows diminishing returns.

2. Cosine-vs-const comparison: on same (sl, fold), compare best MA5 in v3
   (const) vs best MA5 in v2 (cos continuation) using rows with sft_step in
   the overlap [150K, 200K]. v2 only covers sl=[3,5,8,10], seed 0.

3. Per-fold best MA5 step distribution in v3_1 (+v3 primer):
   for each sl, print min/median/max best step across the 8 folds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
MA = 5
FOLDS = [f"Fold_2015-{y}_train_SSL" for y in range(2017, 2025)]
SEQLENS = [3, 5, 8, 10, 20, 30]


EXPECTED_COLS = ["fold", "seed", "ssl_step", "sft_step", "train_loss",
                 "eval_loss", "eval_corr", "gpu", "seq_len"]


def load_csv(base: str, sl: int, fold: str, ssl_step: int = 100000):
    p = Path(base) / f"seqlen_{sl}" / fold / f"seed{SEED}" / f"ssl_step{ssl_step:06d}" / "sft_step_log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "eval_corr" not in df.columns:
        df = pd.read_csv(p, header=None, names=EXPECTED_COLS)
    df = df[["sft_step", "eval_corr"]].drop_duplicates("sft_step").sort_values("sft_step").reset_index(drop=True)
    df["ma5"] = df["eval_corr"].rolling(MA, min_periods=1).mean()
    return df


def section(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ── (1) SSL step ablation ──────────────────────────────────────────────────
section("(1) SSL STEP ABLATION on v3 (constant-LR SFT, seed 0)")
print("best MA5 eval_corr% within sft_step<=150K, averaged across 8 folds")
rows = []
for sl in SEQLENS:
    row = {"sl": sl}
    for ssl in (20000, 50000, 100000):
        vals = []
        for f in FOLDS:
            df = load_csv("./sft_posttrain_v3", sl, f, ssl_step=ssl)
            if df is not None:
                vals.append(df["ma5"].max() * 100)
        row[f"ssl{ssl//1000}K"] = np.mean(vals) if vals else float("nan")
        row[f"ssl{ssl//1000}K_n"] = len(vals)
    row["delta_50_20"] = row["ssl50K"] - row["ssl20K"]
    row["delta_100_50"] = row["ssl100K"] - row["ssl50K"]
    rows.append(row)
d = pd.DataFrame(rows)
print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── (2) Cosine vs constant SFT ─────────────────────────────────────────────
section("(2) COSINE vs CONSTANT SFT (ssl=100K, seed 0)")
print("How much does cosine decay help vs constant LR *at equal budget*?")
print()
print("2a) LAST checkpoint at sft_step=150K (const in v3) vs v2 cos-decay at sft_step=200K")
print("    — v2 used 50K more gradient steps AND a cos-decay, so ∆ captures both")
rows = []
for sl in [3, 5, 8, 10]:
    for f in FOLDS:
        v2 = load_csv("./sft_posttrain_v2", sl, f)
        v3 = load_csv("./sft_posttrain_v3", sl, f)
        if v2 is None or v3 is None:
            continue
        v3_last = v3[v3["sft_step"] == 150000]["eval_corr"].iloc[-1] * 100 if (v3["sft_step"] == 150000).any() else float("nan")
        v2_end  = v2[v2["sft_step"] == v2["sft_step"].max()]["eval_corr"].iloc[-1] * 100
        # Best-MA5 within cosine window (150K,200K] captures "true" cos-decay benefit
        cos_tail = v2[v2["sft_step"] > 150000]
        v2_best_ma = cos_tail["ma5"].max() * 100 if len(cos_tail) else float("nan")
        v3_150_ma = v3.loc[v3["sft_step"] <= 150000, "ma5"].max() * 100
        rows.append({
            "sl": sl, "fold": f[5:14],
            "v3_const_last150": v3_last,
            "v2_cos_last200": v2_end,
            "delta_last_bps": (v2_end - v3_last) * 100,
            "v3_const_peakMA<=150": v3_150_ma,
            "v2_cos_peakMA_150to200": v2_best_ma,
            "delta_peakMA_bps": (v2_best_ma - v3_150_ma) * 100,
        })
d2 = pd.DataFrame(rows)
if not d2.empty:
    agg = d2.groupby("sl").agg(
        mean_delta_last=("delta_last_bps", "mean"),
        min_delta_last=("delta_last_bps", "min"),
        max_delta_last=("delta_last_bps", "max"),
        mean_delta_peakMA=("delta_peakMA_bps", "mean"),
        min_delta_peakMA=("delta_peakMA_bps", "min"),
        max_delta_peakMA=("delta_peakMA_bps", "max"),
        n_folds=("fold", "count"),
    ).reset_index()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
else:
    print("No v2 data found.")

# ── (3) Best MA5 step distribution in v3_1 (+v3 primer) ───────────────────
section("(3) BEST-MA5 STEP DISTRIBUTION across 8 folds")
print("Where does each sl's MA5 eval_corr peak? Lower sl peaks later, higher sl peaks earlier.")
rows = []
for sl in SEQLENS:
    best_steps = []
    for f in FOLDS:
        df = load_csv("./sft_posttrain_v3_1", sl, f)
        if df is None:
            df = load_csv("./sft_posttrain_v3", sl, f)
        if df is None:
            continue
        idx = df["ma5"].idxmax()
        best_steps.append(int(df.loc[idx, "sft_step"]))
    if best_steps:
        rows.append({
            "sl": sl,
            "min_K": min(best_steps) // 1000,
            "med_K": int(np.median(best_steps)) // 1000,
            "max_K": max(best_steps) // 1000,
            "mean_K": int(np.mean(best_steps)) // 1000,
            "std_K": int(np.std(best_steps)) // 1000,
            "folds": len(best_steps),
        })
d3 = pd.DataFrame(rows)
print(d3.to_string(index=False))

# ── (4) Sanity: peak MA5 across sls shows monotone single-model ranking ──
section("(4) Single-model peak MA5 eval_corr% (v3_1+v3 primer), per (sl, fold)")
rows = []
for sl in SEQLENS:
    row = {"sl": sl}
    for f in FOLDS:
        df = load_csv("./sft_posttrain_v3_1", sl, f)
        if df is None:
            df = load_csv("./sft_posttrain_v3", sl, f)
        if df is None:
            row[f[5:14]] = float("nan")
        else:
            row[f[5:14]] = df["ma5"].max() * 100
    rows.append(row)
d4 = pd.DataFrame(rows)
print(d4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
