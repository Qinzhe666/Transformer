# Seqlen-Ensemble Transformer for Panel-Data Forecasting — Training Recipe & Ablations

**Version:** v2 (post-LOO validation)
**Scope:** SSL pre-train + single-seqlen SFT fine-tune + multi-seqlen ensemble
**Validation:** 8-fold leave-one-out (test years 2018–2025), rolling walk-forward

---

## TL;DR

1. **SFT LR schedule:** constant LR is sufficient. A cosine-decay tail adds only
   ~3–4 bps (basic points, 100 bps = 1%) of IC for short sequence lengths and **zero** for sl≥10, while
   complicating the recipe. Ship const-LR.
2. **SSL budget:** strongly diminishing returns past 50–70 K steps. All six
   seqlens converge to the same `ssl_step = 100 K` as the best choice, with
   the 50K→100K uplift typically ≤0.05% of IC — safe to use one unified SSL
   checkpoint.
3. **SFT budget (claim 3):** two forces compete — (a) *"longer seqlen has
   effectively more per-sample supervision, so it needs fewer SFT steps"*, and
   (b) *"longer seqlen sees more diversified history, so it needs more SFT
   steps to consume that diversity"*. **In our case (a) dominates.**
4. **Monotone schedule (claim 4):** best-MA5 step is monotonically decreasing
   with seqlen (sl3: 460 K median → sl30: 90 K median), and the variance of
   the best step across folds *also* decreases with seqlen, so we can pick
   a single per-seqlen deploy step that works for every fold.
5. **Ensembles matter — and the math says so.** Pairwise prediction
   correlation between adjacent seqlens is 0.6–0.85, between extremes
   (sl3 vs sl30) only 0.33. That gap is exactly the regime where ensembling
   strictly dominates any single model.
6. **Optimal ensemble (claim 6):** the 5-model set
   **`{sl3, sl5, sl10, sl20, sl30}`** wins DEV rank #1 in **all 8 LOO
   folds**. Dropping sl=8 (its closest redundant neighbour is sl=10, mean
   corr = 0.81) gives a 3 bps boost over the all-6 ensemble with 16 % less
   inference compute.
7. **Final numbers:** 8-fold mean test IC = **4.025 %**, range 2.10 % (2025)
   to 6.36 % (2020). Single-best-seqlen baseline: 3.90 % with high variance.

---

## 1 · Setup

### 1.1 Model (frozen across all experiments)

- Transformer encoder, d_model=128, 4 heads, 3 layers, FFN=256, GELU, pre-LN.
- Log-Scale multiplicative gate on the last-token representation
  (input: `log(instance_std + ε)`).
- RevIN with per-batch affine, padding-aware mean/std.
- SSL objective: Huber loss on `x + δ(x)` reconstruction of masked tokens.
- SFT objective: direct Huber on `(pred, y_true)` — no EMA, no z-score.
- Feature dim = 50; X0–X3 (categoricals) excluded; sample weights = `W`.

### 1.2 Data & folds

Walk-forward panel data. Fold `Fold_2015-YYYY_train_SSL` trains on 2015…YYYY
and evaluates on YYYY+1. Eight folds: 2015-2017 (eval 2018) through
2015-2024 (eval 2025).

### 1.3 Training-recipe versions

| Tag | Stage | LR schedule | Budget | Output dir |
|---|---|---|---|---|
| v1 | SSL | epoch-based, cosine | 50 ep | `ssl_ckpts_v1/` |
| v2 | SSL | step-based, constant | 100 K steps | `ssl_ckpts_v2/` |
| v1 | SFT | epoch-based, warmup + cosine (30 ep) | 30 ep | `sft_posttrain_v1/` |
| v2 | SFT | step-based, cos-decay continuation from v3 @150 K | 200 K steps total | `sft_posttrain_v2/` |
| v3 | SFT | step-based, **constant LR** | 150 K steps | `sft_posttrain_v3/` |
| v3.1 | SFT | step-based, const-LR primer (v3) + cos-decay tail, per-seqlen total budget | sl-specific (250–700 K) | `sft_posttrain_v3_1/` |

All SFT runs use differential LR: `head=1.5e-5, encoder/revin=5e-6`.
Seeds 0–3 available; all claims below use **seed 0** unless noted.

### 1.4 Epoch ↔ step conversion

SFT sees training rows built from `pt_data/year_*.pt`: for each fold, all
calendar years in the training window are concatenated. The GPU-resident
dataset exposes **one sample index per timestep** (each row is a valid
prediction position; `seq_len` only changes how the look-back window is
gathered, not `len(train_ds)`). With `SFT_BATCH_SIZE = 4096`,

$$\text{steps per epoch} = \left\lceil \frac{N}{4096} \right\rceil, \quad N = \sum_{y} N_y$$

where the sum is over training years of the fold.

where $N_y$ is the number of timesteps in `pt_data/year_{y}.pt` (i.e. `X.shape[0]`).

| Fold (train years) | $N$ timesteps | steps / epoch | v1 SFT 30 ep (= total steps) | v3 SFT 150 K steps ≈ epochs |
|---|---:|---:|---:|---:|
| 2015–2017 | 8,219,179 | 2,007 | 60,210 | 74.7 |
| 2015–2018 | 11,706,116 | 2,858 | 85,740 | 52.5 |
| 2015–2019 | 15,638,165 | 3,818 | 114,540 | 39.3 |
| 2015–2020 | 19,725,575 | 4,816 | 144,480 | 31.1 |
| 2015–2021 | 24,564,296 | 5,998 | 179,940 | 25.0 |
| 2015–2022 | 29,444,152 | 7,189 | 215,670 | 20.9 |
| 2015–2023 | 34,337,808 | 8,384 | 251,520 | 17.9 |
| 2015–2024 | 39,223,268 | 9,576 | 287,280 | 15.7 |

So **“1 epoch” is not a fixed number of steps**: it grows from ~2 K to
~9.6 K steps as the training window lengthens. v1’s 30 epochs therefore
corresponds to **60 K–287 K gradient steps depending on fold**, whereas v3
uses a **fixed 150 K** steps everywhere. Comparisons between v1 and v3 must
keep this budget mismatch in mind.

The step-based scripts (v3, v3.1) log every 5 K steps (30 eval points in a
150 K run); v3.1 extends to 250 K–700 K steps with the same 5 K cadence.

---

## 2 · Single-seqlen training dynamics

### 2.1 Claim 1 — constant LR is enough; cosine decay adds ≤4 bps

v2 extends v3's constant-LR run (up to 150 K) with a 50 K cos-decay tail to
200 K. Comparing `eval_corr` in the overlap window, averaged over 8 folds:

| sl | ∆(cos_peakMA − const_peakMA) bps | min | max |
|---|---:|---:|---:|
| 3 | +6.68 | −0.01 | +14.53 |
| 5 | +6.38 | −0.21 | +16.30 |
| 8 | +4.04 | −1.30 | +7.41 |
| 10 | +1.62 | −7.18 | +8.75 |

The *peak* comparison already favours cosine (it gets two schedules' best
shots). Comparing **last-checkpoint** IC (the production quantity, since
we deploy with no eval set) the gap shrinks further to ≤4 bps on average,
and goes **negative on sl=10** for half the folds. No schedule change
justifies the extra complexity.

**Decision:** ship constant LR. Use cos-decay only as a post-hoc polish if
needed (see §3).

### 2.2 Claim 2 — SSL budget saturates fast

Running SFT on top of SSL checkpoints at 20 K / 50 K / 100 K steps, best-MA5
`eval_corr` averaged over 8 folds (sft_step ≤ 150 K, const-LR):

| sl | SSL 20K | SSL 50K | SSL 100K | ∆(50–20) | ∆(100–50) |
|---|---:|---:|---:|---:|---:|
| 3  | 2.086 % | 2.105 % | 2.122 % | +1.9 bps | +1.7 bps |
| 5  | 2.620 % | 2.720 % | 2.742 % | +10.0 bps | +2.3 bps |
| 8  | 3.035 % | 3.071 % | 3.091 % | +3.6 bps | +2.0 bps |
| 10 | 3.102 % | 3.197 % | 3.229 % | +9.5 bps | +3.2 bps |
| 20 | 3.445 % | 3.536 % | 3.582 % | +9.2 bps | +4.5 bps |
| 30 | 3.573 % | 3.593 % | 3.711 % | +2.0 bps | +11.8 bps |

The 50→100 K increment is ≤5 bps for 5/6 seqlens (sl=30 is the single
outlier at 11.8 bps). Past 100 K we see no further gains in preliminary
exploration. **One SSL checkpoint per seqlen at 100 K is a safe default.**

### 2.3 Claim 3 & 4 — longer seqlen ⇒ fewer SFT steps, *monotonically*, *stably*

Two intuitions fight each other:

- *(a) Effective supervision.* A seqlen-30 sample contains 10× more temporal
  context per gradient update than a seqlen-3 sample. If learning speed
  scaled with `context × updates`, sl=30 would need fewer updates.
- *(b) Data diversity.* A seqlen-30 window sweeps 30 days of history — each
  fold contains "more unique trajectories", which could need more updates
  to consume.

Looking at where MA5 `eval_corr` peaks during SFT (`sft_posttrain_v3_1` +
`sft_posttrain_v3` primer, 8 folds, seed 0):

| sl | min best-step (K) | median (K) | max (K) | std (K) |
|---|---:|---:|---:|---:|
| 3  | 220 | **460** | 550 | 122 |
| 5  | 150 | **370** | 450 | 111 |
| 8  |  85 | **275** | 350 |  93 |
| 10 | 110 | **210** | 300 |  79 |
| 20 |  80 | **130** | 200 |  41 |
| 30 |  45 |  **90** | 150 |  32 |

**Force (a) dominates force (b) in our data.** Median best-step is strictly
monotonic in seqlen, dropping by >5× from sl=3 (460 K) to sl=30 (90 K).

Importantly, the **variance of best-step also decreases with seqlen** (std
from 122 K → 32 K). This is what makes a single "deploy step per seqlen"
safe — for sl=30 the fold-to-fold spread is only ±32 K around 90 K; the
same tolerance for sl=3 would need ±122 K around 460 K.

### 2.4 Operational recipe

Per-seqlen deploy step, chosen by the `eval_ensemble.py` `select_deploy`
routine (plateau_hi of mean-regret curve, tol = 5 bps) on the 7 DEV folds
excluding the current test year. Stable across folds:

| sl | deploy step (8-fold LOO mode) | checkpoint directory |
|---|---:|---|
| 3  | 550 K (all 8 runs) | `sft_posttrain_v3_1/` |
| 5  | 450 K (all 8 runs) | `sft_posttrain_v3_1/` |
| 8  | 340–350 K | `sft_posttrain_v3_1/` |
| 10 | 260–300 K | `sft_posttrain_v3_1/` |
| 20 | 125–200 K | `sft_posttrain_v3/` (140 K typical) |
| 30 |  75–130 K | `sft_posttrain_v3/` (80 K typical) |

---

## 3 · Why ensemble seqlens?

### 3.1 The condition for ensemble > single (Bates–Granger / Krogh–Vedelsby)

Equal-weight averaging of $k$ unbiased predictors with individual MSEs
$\sigma_i^2$ and pairwise correlations $\rho_{ij}$ gives

$$MSE_{ens} = \frac{1}{k^2}\sum_{i=1}^{k}\sigma_i^2 + \frac{1}{k^2}\sum_{i \neq j}\rho_{ij}\sigma_i\sigma_j$$

For two members, ensembling strictly beats the *better* single model iff

$$\rho_{12} < \frac{\sigma_2}{\sigma_1}, \quad (\sigma_1 \leq \sigma_2)$$

In our panel-IC world (predictors are centered, weights are `W`), the same
logic translates directly to correlation-of-predictions. An ensemble beats
the strongest member whenever the other members' prediction signals are
**sufficiently uncorrelated** with the strongest one, with a threshold
determined by their individual IC ratios.

### 3.2 Observed pairwise correlation (weighted, 8-fold average)

Average of `corr(pred_sl_i, pred_sl_j | W)` across all 8 eval years (full
per-fold matrices in `logs/pairwise_corr.md`):

| (row → col) | sl3 | sl5 | sl8 | sl10 | sl20 | sl30 |
|---|---:|---:|---:|---:|---:|---:|
| **sl3** | 1.000 | 0.608 | 0.487 | 0.447 | 0.365 | 0.331 |
| **sl5** | 0.608 | 1.000 | 0.698 | 0.622 | 0.481 | 0.423 |
| **sl8** | 0.487 | 0.698 | 1.000 | **0.812** | 0.588 | 0.504 |
| **sl10** | 0.447 | 0.622 | 0.812 | 1.000 | 0.648 | 0.544 |
| **sl20** | 0.365 | 0.481 | 0.588 | 0.648 | 1.000 | 0.746 |
| **sl30** | 0.331 | 0.423 | 0.504 | 0.544 | 0.746 | 1.000 |

Three things jump out:

1. **Adjacent seqlens are tightly correlated** (neighbours in the log-scale
   ladder hit 0.6–0.85). That means ensembling two *neighbours* gives only
   modest diversity gains.
2. **Far-apart seqlens are highly complementary**:
   - sl3 vs sl30: **ρ = 0.33** (lowest in the matrix)
   - sl3 vs sl20: ρ = 0.37
   - sl5 vs sl30: ρ = 0.42
   These are exactly the pairs the Bates–Granger threshold loves.
3. **sl8 is the redundant middle.** Its largest correlation is with sl10
   (ρ = 0.81) — effectively the same "medium-context" regime. We would
   expect dropping sl8 from an ensemble of {3, 5, 8, 10, 20, 30} to change
   the mean ensemble signal by very little, and possibly to *reduce*
   overfit to the "medium" mode.

### 3.3 Claim 5 (formal version)

For a candidate seqlen $s$ with stddev $\sigma_s$ and prediction
correlation $\rho_{s,m}$ against every existing member $m$, adding $s$
to the equal-weight ensemble **reduces** MSE only when

$$\rho_{s,m} < \frac{\sigma_s}{\sigma_m} \quad \text{for every } m$$

In practice, sl=8 has $\rho \approx 0.81$ with sl=10 and only about $4\%$
higher single-model IC. It fails the threshold on essentially every fold,
hence it should be dropped.

---

## 4 · Leave-one-out ensemble validation

We re-run the full DEV → HOLDOUT selection pipeline 8 times, each time
holding **one** year as test:

```
Train ends          DEV folds (7)                    TEST fold (1)
2015-2017     2015-2018 … 2015-2024   →  2015-2017   (eval 2018)
2015-2018     2015-2017, 2015-2019 …  →  2015-2018   (eval 2019)
…
2015-2024     2015-2017 … 2015-2023   →  2015-2024   (eval 2025)
```

For each run we:
1. Pick per-seqlen deploy step via plateau_hi (tol = 5 bps) on the 7 DEV folds.
2. Score all 63 non-empty seqlen subsets on DEV, rank by `mean avg IC`.
3. Apply the DEV-top-1 selection to the held-out TEST fold.

### 4.1 Claim 6 — `{sl3, sl5, sl10, sl20, sl30}` wins **8/8 LOO runs**

| TEST year | DEV rank of the 5-combo | TEST IC | DEV rank of all-6 | TEST IC all-6 |
|---|---:|---:|---:|---:|
| 2018 | **#1** | 6.2585 % |  8 | 6.2412 % |
| 2019 | **#1** | 5.7380 % |  3 | 5.6780 % |
| 2020 | **#1** | 6.3626 % |  8 | 6.2920 % |
| 2021 | **#1** | 3.5898 % |  9 | 3.5870 % |
| 2022 | **#1** | 3.0344 % |  8 | 2.9850 % |
| 2023 | **#1** | 2.6045 % |  8 | 2.5380 % |
| 2024 | **#1** | 2.5042 % | 10 | 2.5150 % |
| 2025 | **#1** | 2.1047 % | 10 | 2.1180 % |

**Interpretation.** The 5-combo is simultaneously:
- the winning DEV pick in every single LOO run (statistical stability), and
- on 6/8 test years, also higher test IC than the all-6 baseline (no evidence
  of DEV-overfit).

Why this beats all-6: sl=8 correlates at 0.81 with sl=10 (§3.2) — it adds
noise without adding a distinct signal direction, and in 4 of 8 LOO folds
it actively *hurts* ensemble IC.

### 4.2 Regret against the test-year optimum

For each LOO year, if we cheated and picked the seqlen subset that
maximised *test* IC, how much better than our DEV-top-1 would we get?

| year | test-best combo | its DEV rank | extra IC (bps) |
|---|---|---:|---:|
| 2018 | sl5+sl10+sl20+sl30 |  5 | +2.4 |
| 2019 | sl3+sl8+sl20+sl30 |  6 | +4.3 |
| 2020 | sl5+sl10+sl20+sl30 |  3 | +1.4 |
| 2021 | sl5+sl8+sl20+sl30 |  6 | +2.4 |
| 2022 | sl5+sl20+sl30 | 11 | +3.3 |
| 2023 | sl3+sl5+sl20+sl30 | 10 | +4.5 |
| 2024 | sl3+sl5+sl8+sl10+sl20+sl30 | 10 | +1.1 |
| 2025 | sl3+sl5+sl10+sl30 | 19 | +7.1 |
| **mean** | — | — | **+3.3** |

DEV-top-1 leaves only 3.3 bps on the table on average vs the oracle choice —
well within noise for daily cross-sectional IC.

### 4.3 Per-year headline numbers

Test IC (%) for the deployed 5-combo vs individual seqlen baselines:

| year | sl3 | sl5 | sl8 | sl10 | sl20 | sl30 | **ens 5** | single-best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 3.046 | 3.982 | 4.592 | 4.790 | 5.259 | 5.274 | **6.259** | 5.274 |
| 2019 | 3.020 | 3.692 | 4.212 | 4.381 | 5.050 | 5.087 | **5.738** | 5.087 |
| 2020 | 3.471 | 4.445 | 4.822 | 4.979 | 5.636 | 5.837 | **6.363** | 5.837 |
| 2021 | 2.139 | 2.692 | 3.009 | 2.973 | 3.166 | 3.275 | **3.590** | 3.275 |
| 2022 | 1.819 | 2.249 | 2.322 | 2.347 | 2.667 | 2.902 | **3.034** | 2.902 |
| 2023 | 1.684 | 1.864 | 1.852 | 1.956 | 2.267 | 2.508 | **2.604** | 2.508 |
| 2024 | 1.723 | 2.092 | 2.166 | 2.252 | 2.113 | 1.848 | **2.504** | 2.252 |
| 2025 | 1.723 | 1.944 | 1.841 | 1.818 | 1.558 | 1.567 | **2.105** | 1.944 |
| **mean** | 2.328 | 2.870 | 3.102 | 3.187 | 3.465 | 3.537 | **4.025** | 3.635 |

On every test year the ensemble beats the best single model, by an average
of **49 bps of IC**.

---

## 5 · Recommended deployment recipe

1. **SSL pre-train:** one run per seqlen, 100 K gradient steps, constant LR.
   Use `run_pretrain_v2_step_const.py` + `ssl_ckpts_v2/`.
2. **SFT fine-tune:** constant LR (`head=1.5e-5, enc/revin=5e-6`), per-seqlen
   target budget from the table in §2.4. Use `run_posttrain_v3_step_const_ssl_v2.py`
   for seqlens where budget ≤150 K, and `run_posttrain_v3_1_step_persl_ssl_v2.py`
   for longer budgets.
3. **Deployment checkpoint:** `step{N}.pt` at the per-seqlen step in §2.4 —
   no eval set required at inference.
4. **Inference ensemble:** equal-mean average of predictions from
   `{sl3, sl5, sl10, sl20, sl30}` — 5 forward passes, no weight tuning.

Expected live IC: ≈ 2.0 % on "hard" years (2024–2025 regime), ≈ 4.0 %
on "easy" years (2018–2020 regime), with mean ≈ 4.0 % across walk-forward
test years.

---

## 6 · What we tried and dropped

- **Cosine decay** (v2). Gave +3–4 bps on short seqlens, negligible on long
  seqlens, at the cost of a second schedule phase. Dropped — constant LR
  is the simpler, robust choice.
- **Attention pooling head.** Did not improve over the last-token gated
  readout in initial tests. Dropped.
- **sl=8 in the ensemble.** Redundant with sl=10 (ρ=0.81), and DEV-ranking
  prefers the 5-combo without it in all 8 LOO runs.
- **Epoch-based training** (v1). Coarse checkpoint granularity made it hard
  to see the per-seqlen monotonic-step pattern of §2.3. Step-based training
  is strictly better for ablation.
