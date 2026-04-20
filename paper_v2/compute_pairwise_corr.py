#!/usr/bin/env python3
"""Compute per-fold pairwise correlation between per-seqlen predictions.

Uses the *same* PER_SL_DEPLOY checkpoints as eval_ensemble.py (read from a
JSON file). For each of the 8 folds we:
  1. Run inference with all 6 seqlens to get a (N,) prediction vector per sl.
  2. Compute pearson corr(pred_sl_i, pred_sl_j) for every pair.
  3. Also record the weighted IC of each single model (sanity check).

Output:
  logs/pairwise_corr.csv   per-(fold, sl_i, sl_j) rows
  logs/pairwise_corr.md    per-fold pretty matrix + 8-fold average

We use a single `persl_deploy_7dev.json` (dev=all 7 early folds, test=2025),
since pairwise correlation of model outputs only needs a reasonable deploy
step — the LOO-specific choices move these by <1 bp of corr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ensemble as ee  # noqa: E402  reuse model + inference


def weighted_corr_np(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    a, b, w = a[mask], b[mask], w[mask]
    if a.size < 2:
        return float("nan")
    ws = w.sum()
    if ws <= 0:
        return float("nan")
    ma = (w * a).sum() / ws
    mb = (w * b).sum() / ws
    va = (w * (a - ma) ** 2).sum() / ws
    vb = (w * (b - mb) ** 2).sum() / ws
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = (w * (a - ma) * (b - mb)).sum() / ws
    return float(cov / np.sqrt(va * vb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[3, 5, 8, 10, 20, 30])
    ap.add_argument("--tol-bps", type=float, default=5.0)
    ap.add_argument("--out-csv", default="logs/pairwise_corr.csv")
    ap.add_argument("--out-md", default="logs/pairwise_corr.md")
    args = ap.parse_args()

    # Pick deploy steps using 7 early folds (same as the test=2025 LOO run) —
    # pairwise correlation is insensitive to exact step choice (<1bp drift).
    dev_folds = [f"Fold_2015-{y}_train_SSL" for y in range(2017, 2024)]
    deploy = ee.select_deploy(args.seqlens, dev_folds, args.tol_bps)
    for sl in args.seqlens:
        assert sl in deploy, f"select_deploy missing sl={sl}"

    feature_dim = ee.discover_feature_dim()
    all_folds = [f"Fold_2015-{y}_train_SSL" for y in range(2017, 2025)]
    eval_years = [ee.fold_to_eval_year(f) for f in all_folds]

    rows = []
    per_fold_matrices: Dict[str, pd.DataFrame] = {}
    per_fold_ic: Dict[str, Dict[int, float]] = {}

    preds, ys, ws = ee.build_predictions(deploy, all_folds, eval_years, args.seqlens, feature_dim)

    for fold, year in zip(all_folds, eval_years):
        w_np = ws[year].cpu().numpy()
        y_np = ys[year].cpu().numpy()
        # weighted corrs
        mat = pd.DataFrame(index=args.seqlens, columns=args.seqlens, dtype=float)
        for sl in args.seqlens:
            if sl not in preds[year]:
                continue
            pi = preds[year][sl].cpu().numpy()
            ic = weighted_corr_np(y_np, pi, w_np)
            per_fold_ic.setdefault(fold, {})[sl] = ic
            for sl2 in args.seqlens:
                if sl2 not in preds[year]:
                    continue
                pj = preds[year][sl2].cpu().numpy()
                c = weighted_corr_np(pi, pj, w_np)
                mat.loc[sl, sl2] = c
                if sl < sl2:
                    rows.append({"fold": fold, "year": year, "sl_i": sl, "sl_j": sl2,
                                 "corr": c, "IC_i": per_fold_ic[fold][sl],
                                 "IC_j": weighted_corr_np(y_np, pj, w_np)})
        per_fold_matrices[fold] = mat

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # Aggregate: mean corr across folds per pair
    agg = df.groupby(["sl_i", "sl_j"]).agg(mean_corr=("corr", "mean"),
                                            min_corr=("corr", "min"),
                                            max_corr=("corr", "max")).reset_index()

    # Write markdown
    lines = []
    lines.append("# Per-fold pairwise prediction correlation\n")
    lines.append("Correlations computed on weighted predictions (weights = W).  \n")
    lines.append(f"Seqlens: {args.seqlens}\n")
    lines.append("Deploy steps per sl (base, step):\n")
    for sl in args.seqlens:
        b, s = deploy[sl]
        lines.append(f"  - sl{sl}: ({b}, {s})")
    lines.append("")

    lines.append("## Per-fold correlation matrices\n")
    for fold in all_folds:
        lines.append(f"### {fold}  (single-model IC%: "
                     + ", ".join(f"sl{sl}={per_fold_ic[fold][sl]*100:+.3f}"
                                 for sl in args.seqlens if sl in per_fold_ic.get(fold, {}))
                     + ")\n")
        mat = per_fold_matrices[fold]
        header = "| sl |" + "|".join(f" sl{sl} " for sl in args.seqlens) + "|"
        sep = "|---|" + "|".join("---" for _ in args.seqlens) + "|"
        lines.append(header)
        lines.append(sep)
        for sl in args.seqlens:
            cells = []
            for sl2 in args.seqlens:
                v = mat.loc[sl, sl2]
                cells.append(f"{v:.3f}" if pd.notna(v) else " - ")
            lines.append(f"| **sl{sl}** | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Mean pairwise correlation across 8 folds\n")
    lines.append("| sl_i \\ sl_j |" + "|".join(f" sl{sl} " for sl in args.seqlens) + "|")
    lines.append("|---|" + "|".join("---" for _ in args.seqlens) + "|")
    for sl in args.seqlens:
        cells = []
        for sl2 in args.seqlens:
            if sl == sl2:
                cells.append("1.000")
                continue
            lo, hi = (sl, sl2) if sl < sl2 else (sl2, sl)
            m = agg[(agg["sl_i"] == lo) & (agg["sl_j"] == hi)]["mean_corr"]
            cells.append(f"{m.iloc[0]:.3f}" if len(m) else " - ")
        lines.append(f"| **sl{sl}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Pair summary (off-diagonal)\n")
    lines.append("| sl_i | sl_j | mean_corr | min_corr | max_corr |")
    lines.append("|---|---|---|---|---|")
    for _, r in agg.iterrows():
        lines.append(f"| sl{int(r['sl_i'])} | sl{int(r['sl_j'])} | "
                     f"{r['mean_corr']:.3f} | {r['min_corr']:.3f} | {r['max_corr']:.3f} |")

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out_csv} ({len(df)} rows) and {args.out_md}")


if __name__ == "__main__":
    main()
