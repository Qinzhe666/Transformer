#!/usr/bin/env python3
"""Aggregate the 8 LOO eval_ensemble logs into summary tables.

Reads logs/loo_test{YEAR}.log for YEAR in 2018..2025 and prints:
  (a) Chosen deploy step per (test_year, sl)
  (b) Individual sl ICs on each test year
  (c) DEV rank + test IC for a chosen "focus" combo across the 8 runs
  (d) Per test year, the top-1 DEV ensemble and its test IC, plus the
      test-year best combo (and where it sits on DEV rank)
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

LOG_DIR = "logs"
YEARS = list(range(2018, 2026))
FOCUS_COMBO = "sl3+sl5+sl10+sl20+sl30"
FULL_COMBO = "sl3+sl5+sl8+sl10+sl20+sl30"


def parse_log(year: int) -> dict:
    """Return dict with keys: deploy, singles (sl -> ic), dev_top (combo -> (rank, avg)),
    ho (combo -> ho_avg)."""
    path = os.path.join(LOG_DIR, f"loo_test{year}.log")
    with open(path) as f:
        text = f.read()

    deploy: Dict[int, Tuple[int, str]] = {}
    for m in re.finditer(r"sl=\s*(\d+)\s+n_steps=\s*\d+\s+chosen=\s*(\d+).*?base=(\S+)", text):
        sl, step, base = int(m.group(1)), int(m.group(2)), m.group(3)
        deploy[sl] = (step, base)

    # Singles on the TEST fold come from the [STEP 3] HOLDOUT section.
    step3_idx = text.find("[STEP 3]")
    step3 = text[step3_idx:]
    # The single-sl block in STEP 3 looks like:
    #   [year 2018]  fold=...
    #     sl= 3  n=...  IC=+3.0463%
    singles: Dict[int, float] = {}
    for m in re.finditer(r"sl=\s*(\d+)\s+n=\s*\d+\s+IC=([+-][\d.]+)%", step3):
        sl, ic = int(m.group(1)), float(m.group(2))
        if sl not in singles:   # first occurrence == test year
            singles[sl] = ic

    # DEV-ranked table (with HO columns) at end of STEP 3.
    dev_top: Dict[str, Tuple[int, float]] = {}
    ho_map: Dict[str, float] = {}
    m = re.search(r"DEV-selected ensembles.*?\n(.*?)\nDONE", text, re.S)
    assert m, f"no final table in {path}"
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith(("Rank", "-")):
            continue
        parts = line.split()
        # parts: rank, combo, dev_avg, dev_min, ho_avg, ho_min
        if len(parts) < 6:
            continue
        try:
            rank = int(parts[0])
            combo = parts[1]
            dev_avg = float(parts[2])
            ho_avg = float(parts[4])
        except ValueError:
            continue
        dev_top[combo] = (rank, dev_avg)
        ho_map[combo] = ho_avg
    return {"deploy": deploy, "singles": singles, "dev_top": dev_top, "ho": ho_map}


def main():
    per_year = {y: parse_log(y) for y in YEARS}

    seqlens = [3, 5, 8, 10, 20, 30]

    # (a) Deploy steps picked by DEV for each test year
    print("=" * 120)
    print("(a) Deploy step chosen by DEV (7 folds excluding the test year)")
    print("=" * 120)
    hdr = f"  test_year  " + "  ".join(f"sl{sl:>2}" for sl in seqlens)
    print(hdr)
    for y in YEARS:
        row = f"     {y}    " + "  ".join(
            f"{per_year[y]['deploy'][sl][0]:>5d}" for sl in seqlens
        )
        print(row)

    # (b) Single sl ICs on the test year
    print("\n" + "=" * 120)
    print("(b) Single-seqlen IC (%) on each test year")
    print("=" * 120)
    print("  test_year  " + "  ".join(f"sl{sl:>2}     " for sl in seqlens))
    for y in YEARS:
        s = per_year[y]["singles"]
        print(f"     {y}    " + "  ".join(f"{s[sl]:+7.4f}" for sl in seqlens))

    # (c) Focus combos: DEV rank + test IC across 8 years
    print("\n" + "=" * 120)
    print(f"(c) FOCUS combo tracker across 8 LOO runs")
    print("=" * 120)
    print(f"  {'combo':<35s}  " + "  ".join(f"{y}_rnk  {y}_IC " for y in YEARS))
    for combo in [FOCUS_COMBO, FULL_COMBO]:
        cells = []
        for y in YEARS:
            dt = per_year[y]["dev_top"].get(combo)
            ho = per_year[y]["ho"].get(combo)
            if dt is None or ho is None:
                cells.append(f"  --   --    ")
            else:
                cells.append(f" #{dt[0]:>2d}  {ho:+6.3f}")
        print(f"  {combo:<35s}  " + "  ".join(cells))

    # (d) Per year: DEV top-1 vs test-best
    print("\n" + "=" * 120)
    print("(d) DEV top-1 vs test-best per year")
    print("=" * 120)
    print(f"  {'year':>4s}  {'DEV top-1':<33s} {'test_IC':>8s}   {'test-best (DEV rank)':<33s} {'test_IC':>8s}  {'gap(bps)':>8s}")
    for y in YEARS:
        # DEV top-1
        dev_sorted = sorted(per_year[y]["dev_top"].items(), key=lambda kv: kv[1][0])
        dev_top1_combo = dev_sorted[0][0]
        dev_top1_ho = per_year[y]["ho"][dev_top1_combo]
        # test-best (largest ho_avg)
        test_best_combo, test_best_ho = max(per_year[y]["ho"].items(), key=lambda kv: kv[1])
        test_best_rank = per_year[y]["dev_top"][test_best_combo][0]
        gap = (test_best_ho - dev_top1_ho) * 100  # bps
        print(f"  {y}  {dev_top1_combo:<33s} {dev_top1_ho:+7.4f}  "
              f"  {test_best_combo+f' (#{test_best_rank})':<33s} {test_best_ho:+7.4f}  {gap:>7.2f}")

    # (e) Average test IC of focus combo vs all-6 vs DEV top-1 across 8 years
    print("\n" + "=" * 120)
    print("(e) Mean test IC across 8 LOO folds (lower bound of a zero-tuning deploy)")
    print("=" * 120)
    summaries = []
    for combo in [FOCUS_COMBO, FULL_COMBO]:
        ics = [per_year[y]["ho"].get(combo) for y in YEARS]
        ics_valid = [x for x in ics if x is not None]
        mean = sum(ics_valid) / len(ics_valid)
        ranks = [per_year[y]["dev_top"].get(combo, (None,))[0] for y in YEARS]
        summaries.append((combo, mean, ics, ranks))
    # Also: follow DEV top-1 each year (dynamic choice, allowed because DEV only)
    dev_top1_ics = []
    for y in YEARS:
        dev_sorted = sorted(per_year[y]["dev_top"].items(), key=lambda kv: kv[1][0])
        dev_top1_ics.append(per_year[y]["ho"][dev_sorted[0][0]])
    summaries.append(("DEV top-1 (per-year dynamic)", sum(dev_top1_ics) / len(dev_top1_ics), dev_top1_ics, []))

    print(f"  {'combo':<40s} {'mean_test_IC':>12s}  {'per-year IC':<60s}")
    for combo, mean, ics, _ in summaries:
        per = "  ".join(f"{(x if x is not None else float('nan')):+6.3f}" for x in ics)
        print(f"  {combo:<40s}  {mean:+8.4f}%    {per}")


if __name__ == "__main__":
    main()
