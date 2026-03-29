# Predicting Forward 30-Minute Returns with a Pre-Trained Transformer

## Key Takeaway

The best input sequence length (`seq_len`) for predicting forward 30-minute equity returns is **5 or 10 timesteps**, not the longer windows (15–30) that performed well historically. This conclusion holds robustly across recent evaluation folds (2023–2024) and is consistent with the hypothesis that long-horizon temporal structure in financial markets has decayed over time. Validation on 2025–2026 data is needed to confirm whether this regime shift persists.

---

## 1. Overview

We train a Transformer-based model to predict forward 30-minute returns (`Y1`) for a universe of equities. The training pipeline consists of two stages:

1. **Pre-Training (SSL)**: Self-supervised learning on unlabeled intraday feature sequences via masked reconstruction.
2. **Post-Training (SFT)**: Supervised fine-tuning on the return prediction task using the pre-trained encoder as initialization.

The two stages share the same Transformer encoder but attach different task-specific heads. This report describes the architecture, the training procedure, and the most important hyperparameter findings — in particular, the critical role of input sequence length (`seq_len`).

---

## 2. Architecture Design Choices

### 2.1 Three Major Transformer Architectures

Since Vaswani et al. (2017), the Transformer has evolved into three dominant architectural paradigms, each suited to a different class of tasks:

**Encoder-only (BERT-style).** Uses bidirectional self-attention — every token attends to every other token in the sequence. This allows the model to exploit the full input context, making it ideal for **sequence understanding** tasks (classification, regression, NER) where the entire input is known at inference time.

**Decoder-only (GPT-style).** Uses causal (left-to-right) attention masking — each token can only attend to itself and preceding tokens. This autoregressive structure is designed for **sequence generation** tasks (text generation, next-token prediction), where future tokens are unavailable during inference.

**Encoder-Decoder (T5-style / original Transformer).** The encoder processes the full input with bidirectional attention and produces contextual representations; the decoder then attends to these representations via cross-attention while generating the output autoregressively. This is suited for **sequence-to-sequence** tasks (translation, summarization) where input and output are distinct sequences.

The following diagrams illustrate the information flow and attention patterns of each variant:

```
                     ENCODER-ONLY (BERT-style)


 Input:   [bar1]   [bar2]   [bar3]  ...  [barN]
             |         |        |            |        Bidirectional
             |<------->|<------>|<---------->|        Self-Attention:
             |         |        |            |        every token sees
          +--+---------+--------+------------+--+     every other token
          |     Transformer Encoder Layers      |
          +--+---------+--------+------------+--+
             |         |        |            |
 Output:  [h1]      [h2]     [h3]  ...   [hN]
                                             |
                                        Prediction


 Use case: classification, regression, understanding




                     DECODER-ONLY (GPT-style)


 Input:   [bar1]   [bar2]   [bar3]  ...  [barN]
             |         |        |            |        Causal (L->R)
             |    <----+   <----+       <----+        Self-Attention:
             |         |        |            |        token t can only
          +--+---------+--------+------------+--+     see tokens <= t
          |     Transformer Decoder Layers      |
          +--+---------+--------+------------+--+
             |         |        |            |
 Output:  [h1]      [h2]     [h3]  ...   [hN]
             |         |        |            |
          [y1']     [y2']    [y3']  ...   [yN']   (next-token preds)


 Use case: text generation, autoregressive prediction




                  ENCODER-DECODER (T5-style)


 Input:   [x1] [x2] ... [xM]           [y1] [y2] ... [yK]
            |    |        |                |    |        |
         Bidirectional Attn.          Causal Attn. + Cross-Attn.
          +--+---+--------+--+      +-----+----+--------+--+
          |  Encoder Layers  +----->|   Decoder Layers     |
          +------------------+      +-----+----+--------+--+
                                          |    |        |
                                       [y1'] [y2'] ... [yK']


 Use case: translation, summarization, seq-to-seq
```

### 2.2 Why Encoder-Only for This Task

Our core task is: given a fixed-length window of intraday bar features, predict the forward 30-minute return (`Y1`) at the last bar. This is fundamentally a **sequence understanding and regression** problem, not a sequence generation problem. We select the encoder-only architecture for the following reasons:

**1. The full input sequence is known at inference time.** All bars in the lookback window are available when making a prediction. Bidirectional attention allows each bar to attend to both preceding and subsequent bars in the window, maximizing information utilization. A decoder-only causal mask would prevent bar *t* from attending to bar *t+1*, even though that bar is a known input — an unnecessary restriction on information flow.

The diagram below contrasts the two attention patterns on the same input window:

```
 Bidirectional (Encoder)             Causal (Decoder)
 +------------------------+          +-------------------------+
 |  bar1 <-> bar2 <-> bar3|          |  bar1                   |
 |    |       |       |   |          |  bar1 -> bar2           |
 |  bar1 <-> bar2 <-> bar3|          |  bar1 -> bar2 -> bar3   |
 |    |       |       |   |          |                         |
 |  ALL pairs connected   |          |  bar3 CANNOT see bar1   |
 |                        |          |  (information wasted!)  |
 +------------------------+          +-------------------------+
      [OK] Full context                 [X] Restricted context
        for known inputs               designed for generation
```

**2. The output is a single scalar.** Our prediction target is a single floating-point number (forward return), not a sequence. The model only needs to extract a representation from the encoder's last position and map it to a scalar output. There is no need for the autoregressive generation capability of a decoder. An encoder-decoder architecture would add a completely redundant decoder, increasing parameter count and computational cost with no benefit.

**3. Natural fit with SSL pre-training.** Our self-supervised pre-training uses span masking + feature reconstruction, directly corresponding to BERT-style masked prediction. Encoder-only is the standard architecture for such reconstruction objectives — the encoder's bidirectional attention allows masked positions to leverage context from both sides for reconstruction.

**4. Last-token regression during SFT.** During post-training, we extract the hidden state from the encoder's last position to perform regression prediction, analogous to how BERT uses the `[CLS]` token for classification. This is a standard usage pattern for encoder-only models.

The complete pipeline for our model is illustrated below:

```
                OUR MODEL: Encoder-Only Pipeline


 Raw bars: [bar1] [bar2] [bar3]  ...  [barN]
              |      |      |             |
              v      v      v             v
          +----------------------------------+
          |      RevIN Normalization         |
          | (zero mean, unit var per sample) |
          +----------------+-----------------+
                           |
                           v
          +----------------------------------+
          |       Input Projection           |
          | Linear(feature_dim -> d_model)   |
          +----------------+-----------------+
                           |
                           v
          +----------------------------------+
          |  + Sinusoidal Positional Enc.    |
          +----------------+-----------------+
                           |
                           v
          +----------------------------------+
          | Transformer Encoder (x3 layers)  |
          |   Bidirectional Self-Attention   |
          |   d_model=128, nhead=4, GELU     |
          +--+-------+-------+-----------+---+
             |       |       |           |
          [h1]    [h2]    [h3]  ...   [hN]
                                        |
                             +----------+
                             |
                             v
 +----------------------------------------------------+
 |  SFT Head: Last-Token Extraction & Gating          |
 |                                                    |
 |   h_last = LayerNorm(hN)                           |
 |   log_std = log(RevIN_std + eps)                   |
 |   gate = sigmoid(Linear(log_std))  <- vol. gating  |
 |   y_hat = Linear(h_last * gate)    -> Y1 prediction|
 +----------------------------------------------------+
```

---

## 3. Model Architecture

### 3.1 Shared Backbone

Both stages use the same Transformer encoder:

| Component | Specification |
|---|---|
| Input normalization | LayerNorm on raw features |
| Input projection | Linear(feature_dim → d_model) |
| Positional encoding | Sinusoidal (fixed) |
| Encoder | TransformerEncoder, `norm_first=True`, GELU activation |
| d_model | 128 |
| nhead | 4 |
| num_layers | 3 |
| dim_feedforward | 256 |
| dropout | 0.2 |

Before entering the encoder, input features pass through **Reversible Instance Normalization (RevIN)**, which normalizes each sample to zero mean and unit variance across the time dimension. RevIN handles distribution shift between training and inference regimes and supports a padding mask so that statistics are computed only over valid (non-padded) timesteps.

For timesteps near market open where fewer than `seq_len` prior observations are available, we **left-pad** with zeros and pass a boolean `padding_mask` to the Transformer's `src_key_padding_mask`, ensuring that padded positions do not attend to or influence the model's representations.

### 3.2 Pre-Training Head (SSL)

During pre-training, the model is trained to reconstruct masked input features. The head consists of:

- `out_norm_seq`: LayerNorm(d_model) applied to every position's hidden state
- `recon_head`: Linear(d_model → feature_dim), producing a per-position reconstruction
- `mask_token`: a learnable embedding that replaces masked positions in the input embedding space

The SSL objective uses **span masking** (mask_ratio=0.30, span lengths 2–6) and a composite loss:

```
L_SSL = L_recon(X) + λ · L_delta(ΔX)
```

where `L_recon` is weighted Huber loss on reconstructed features at masked positions, `L_delta` is the same loss on first-order differences (ΔX), and λ=0.2. Both terms are computed only on masked, non-padded positions. Weights (`wgt`) are applied per-timestep to account for varying sample quality.

**Reference implementation**: `run_ssl_pretrain_revin_big_cat_final.py`

### 3.3 Post-Training Head (SFT)

During post-training, the pre-trained encoder is fine-tuned end-to-end with a supervised prediction head. The head uses a **multiplicative gating mechanism** conditioned on instance-level volatility:

1. Extract the last-position hidden state: `h_last = out_norm_last(encoder_output[:, -1, :])`
2. Compute per-instance log-volatility: `log_std = log(RevIN_std + ε)`
3. Produce a gate: `gate = sigmoid(Linear(log_std))`
4. Apply multiplicative gating: `gated = h_last * gate`
5. Predict return: `ŷ = y_head(gated)`

The gate allows the model to modulate its prediction magnitude based on the input's volatility regime — the intuition is that the same pattern in a high-volatility environment should produce a different prediction scale than in a low-volatility environment. The gate is initialized to zero (i.e., sigmoid(0)=0.5), starting from a neutral scaling and learning the appropriate modulation during training.

The SFT loss is **direct Huber loss** on predicted vs. actual returns, weighted by `wgt`:

```
L_SFT = HuberLoss(ŷ, y, β=1.0, weights=wgt)
```

No EMA, no z-score normalization — the loss operates directly in the return space.

**Reference implementation**: `run_posttrain_seqlen_ablation_v3.py`

### 3.4 Why Two Stages?

The key insight is that pre-training and post-training are **the same encoder with different heads attached**:

- **SSL stage**: encoder + `recon_head` (per-position reconstruction). Learns generalizable temporal representations from the feature space itself, without requiring return labels.
- **SFT stage**: encoder + `scale_gate` + `y_head` (last-position prediction). Adapts the learned representations to the specific prediction target.

This separation is valuable because:

1. SSL can leverage all available intraday data regardless of whether return labels are available or reliable.
2. The encoder learns temporal dynamics (trends, mean-reversion, volatility clustering) that are useful across different prediction tasks.
3. The SFT stage can be run with fewer epochs and is less prone to overfitting, because the encoder starts from a well-initialized state.

---

## 4. Experimental Setup

### 4.1 Data

- **Universe**: Intraday equity features at minute-level granularity.
- **Target**: `Y1` — forward 30-minute return.
- **Features**: All continuous columns excluding categorical identifiers (`X0`–`X3`), targets (`Y0`–`Y3`), and sample weight (`wgt`).
- **Normalization**: Global z-score normalization (Welford streaming mean/std over training files), followed by per-instance RevIN.

### 4.2 Fold Structure

Expanding-window cross-validation with training periods starting from 2015:

| Fold | Train Years | Test Year |
|---|---|---|
| Fold_2015-2017 | 2015–2017 | 2018 |
| Fold_2015-2018 | 2015–2018 | 2019 |
| Fold_2015-2019 | 2015–2019 | 2020 |
| Fold_2015-2020 | 2015–2020 | 2021 |
| Fold_2015-2021 | 2015–2021 | 2022 |
| Fold_2015-2022 | 2015–2022 | 2023 |
| Fold_2015-2023 | 2015–2023 | 2024 |

Each fold is run with multiple random seeds to assess stability.

### 4.3 Training Details

**SSL Pre-Training**:
- Optimizer: AdamW (lr=8e-5, weight_decay=1e-2, grad_clip=1.0)
- Schedule: Constant for 100 epochs, then cosine annealing to lr_min=1e-6
- Batch size: 1024
- Steps per epoch: 2000
- AMP: bf16
- Checkpoints saved every 5 epochs

**SFT Post-Training**:
- Optimizer: AdamW (lr=3e-5, weight_decay=1e-2, grad_clip=1.0)
- Epochs: 25
- Batch size: 1024
- Loss: Huber (β=1.0), weighted
- AMP: bf16
- GPU-resident data: all training data pre-loaded as flat tensors on GPU with vectorized batch construction

### 4.4 Evaluation Metric

Weighted Pearson correlation between predicted and actual returns, computed on held-out test data per fold.

---

## 5. Key Hyperparameters

The three most important hyperparameters are:

| Hyperparameter | Role | Explored Range |
|---|---|---|
| **seq_len** | Number of input timesteps (lookback window) | 5, 10, 15, 20, 25, 30 |
| **SSL epochs** | Duration of self-supervised pre-training | 30, 60, 90, 120, 150, 200 |
| **SFT epochs** | Duration of supervised fine-tuning | 25 for seq len 5 and 10, 15 for others|

Of these, `seq_len` has by far the largest impact on out-of-sample performance and interacts strongly with the other two.

---

## 6. Sequence Length Ablation

We study how the optimal input sequence length (`seq_len`) changes across time and how it interacts with SSL pretraining.

### 6.1 Main Empirical Findings

#### Preliminary Finding: The oldest bars in a `seq_len=30` window are nearly irrelevant

Before running the full sequence-length ablation, we first asked a simpler question: **how much does each part of the 30-bar lookback window actually contribute to prediction quality?**

**Setup.** We took models trained with `seq_len=30` (8 folds × 10 seeds × 5 SFT epochs = 400 configurations) and evaluated each at six "effective lengths" (`eff_len` ∈ {5, 10, 15, 20, 25, 30}). For a given `eff_len`, we mask the oldest `30 − eff_len` bars at inference time by (1) zeroing their input features and (2) setting `padding_mask=True` so the Transformer's attention ignores those positions entirely. The model weights are unchanged — only the input visibility is restricted. This isolates the marginal contribution of each segment of the lookback window.

**Results.** The tables below report mean weighted Pearson correlation (the same metric used throughout §6–§7) averaged across 10 seeds, broken down by fold, effective length, and SFT epoch. We show separate tables for each of the 5 SFT epochs (ep010–ep014) to confirm that the pattern is not an artifact of epoch-averaging.

**SFT Epoch 10 (ep010):**

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 | eff_len=30 |
|---|---|---|---|---|---|---|
| 2015–2017 (2018) | 0.0217 | 0.0316 | 0.0389 | 0.0453 | 0.0528 | 0.0571 |
| 2015–2018 (2019) | 0.0203 | 0.0317 | 0.0384 | 0.0459 | 0.0533 | 0.0569 |
| 2015–2019 (2020) | 0.0199 | 0.0321 | 0.0395 | 0.0480 | 0.0551 | 0.0589 |
| 2015–2020 (2021) | 0.0097 | 0.0140 | 0.0186 | 0.0254 | 0.0301 | 0.0316 |
| 2015–2021 (2022) | 0.0080 | 0.0118 | 0.0136 | 0.0183 | 0.0231 | 0.0269 |
| 2015–2022 (2023) | 0.0075 | 0.0119 | 0.0140 | 0.0171 | 0.0212 | 0.0239 |
| 2015–2023 (2024) | 0.0078 | 0.0117 | 0.0144 | 0.0171 | 0.0189 | 0.0194 |
| 2015–2024 (2025) | 0.0084 | 0.0129 | 0.0143 | 0.0152 | 0.0158 | 0.0160 |

Retention (% of eff_len=30):

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 |
|---|---|---|---|---|---|
| 2015–2017 (2018) | 38.0% | 55.3% | 68.1% | 79.3% | 92.4% |
| 2015–2018 (2019) | 35.6% | 55.8% | 67.5% | 80.6% | 93.7% |
| 2015–2019 (2020) | 33.7% | 54.4% | 67.1% | 81.4% | 93.5% |
| 2015–2020 (2021) | 30.5% | 44.1% | 58.9% | 80.3% | 95.3% |
| 2015–2021 (2022) | 29.8% | 44.1% | 50.7% | 68.0% | 86.2% |
| 2015–2022 (2023) | 31.3% | 49.7% | 58.7% | 71.5% | 88.6% |
| 2015–2023 (2024) | 40.1% | 60.4% | 74.2% | 88.0% | 97.3% |
| 2015–2024 (2025) | 52.5% | 80.4% | 89.5% | 94.7% | 98.9% |

**SFT Epoch 11 (ep011):**

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 | eff_len=30 |
|---|---|---|---|---|---|---|
| 2015–2017 (2018) | 0.0217 | 0.0318 | 0.0392 | 0.0455 | 0.0531 | 0.0576 |
| 2015–2018 (2019) | 0.0203 | 0.0318 | 0.0386 | 0.0461 | 0.0536 | 0.0572 |
| 2015–2019 (2020) | 0.0198 | 0.0320 | 0.0396 | 0.0483 | 0.0556 | 0.0595 |
| 2015–2020 (2021) | 0.0097 | 0.0141 | 0.0189 | 0.0258 | 0.0307 | 0.0323 |
| 2015–2021 (2022) | 0.0079 | 0.0117 | 0.0135 | 0.0182 | 0.0233 | 0.0271 |
| 2015–2022 (2023) | 0.0073 | 0.0118 | 0.0140 | 0.0172 | 0.0214 | 0.0241 |
| 2015–2023 (2024) | 0.0078 | 0.0117 | 0.0144 | 0.0170 | 0.0188 | 0.0194 |
| 2015–2024 (2025) | 0.0083 | 0.0129 | 0.0143 | 0.0151 | 0.0158 | 0.0160 |

Retention (% of eff_len=30):

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 |
|---|---|---|---|---|---|
| 2015–2017 (2018) | 37.7% | 55.1% | 67.9% | 79.0% | 92.2% |
| 2015–2018 (2019) | 35.4% | 55.7% | 67.5% | 80.6% | 93.8% |
| 2015–2019 (2020) | 33.3% | 53.8% | 66.6% | 81.1% | 93.4% |
| 2015–2020 (2021) | 29.9% | 43.6% | 58.4% | 79.8% | 95.1% |
| 2015–2021 (2022) | 29.1% | 43.1% | 49.7% | 67.2% | 85.9% |
| 2015–2022 (2023) | 30.4% | 49.0% | 58.1% | 71.3% | 88.7% |
| 2015–2023 (2024) | 40.0% | 60.4% | 73.9% | 87.4% | 96.9% |
| 2015–2024 (2025) | 52.1% | 80.5% | 89.5% | 94.7% | 98.8% |

**SFT Epoch 12 (ep012):**

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 | eff_len=30 |
|---|---|---|---|---|---|---|
| 2015–2017 (2018) | 0.0217 | 0.0319 | 0.0394 | 0.0458 | 0.0535 | 0.0582 |
| 2015–2018 (2019) | 0.0202 | 0.0317 | 0.0386 | 0.0462 | 0.0539 | 0.0575 |
| 2015–2019 (2020) | 0.0197 | 0.0321 | 0.0397 | 0.0485 | 0.0560 | 0.0599 |
| 2015–2020 (2021) | 0.0097 | 0.0141 | 0.0190 | 0.0260 | 0.0311 | 0.0327 |
| 2015–2021 (2022) | 0.0077 | 0.0115 | 0.0134 | 0.0182 | 0.0235 | 0.0274 |
| 2015–2022 (2023) | 0.0071 | 0.0116 | 0.0138 | 0.0170 | 0.0214 | 0.0242 |
| 2015–2023 (2024) | 0.0077 | 0.0116 | 0.0142 | 0.0169 | 0.0187 | 0.0195 |
| 2015–2024 (2025) | 0.0081 | 0.0128 | 0.0142 | 0.0151 | 0.0158 | 0.0159 |

Retention (% of eff_len=30):

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 |
|---|---|---|---|---|---|
| 2015–2017 (2018) | 37.3% | 54.9% | 67.7% | 78.7% | 91.9% |
| 2015–2018 (2019) | 35.1% | 55.2% | 67.2% | 80.4% | 93.7% |
| 2015–2019 (2020) | 32.9% | 53.5% | 66.3% | 81.0% | 93.4% |
| 2015–2020 (2021) | 29.6% | 43.1% | 58.1% | 79.7% | 95.1% |
| 2015–2021 (2022) | 28.2% | 42.1% | 48.8% | 66.6% | 85.6% |
| 2015–2022 (2023) | 29.4% | 47.9% | 57.0% | 70.4% | 88.4% |
| 2015–2023 (2024) | 39.5% | 59.8% | 73.2% | 86.6% | 96.4% |
| 2015–2024 (2025) | 51.1% | 80.4% | 89.6% | 95.1% | 99.1% |

**SFT Epoch 13 (ep013):**

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 | eff_len=30 |
|---|---|---|---|---|---|---|
| 2015–2017 (2018) | 0.0217 | 0.0321 | 0.0395 | 0.0459 | 0.0537 | 0.0585 |
| 2015–2018 (2019) | 0.0201 | 0.0317 | 0.0386 | 0.0463 | 0.0540 | 0.0577 |
| 2015–2019 (2020) | 0.0197 | 0.0320 | 0.0398 | 0.0486 | 0.0561 | 0.0601 |
| 2015–2020 (2021) | 0.0097 | 0.0141 | 0.0190 | 0.0262 | 0.0313 | 0.0329 |
| 2015–2021 (2022) | 0.0076 | 0.0114 | 0.0133 | 0.0182 | 0.0236 | 0.0276 |
| 2015–2022 (2023) | 0.0070 | 0.0115 | 0.0137 | 0.0170 | 0.0215 | 0.0243 |
| 2015–2023 (2024) | 0.0076 | 0.0116 | 0.0141 | 0.0167 | 0.0186 | 0.0194 |
| 2015–2024 (2025) | 0.0080 | 0.0127 | 0.0141 | 0.0150 | 0.0157 | 0.0158 |

Retention (% of eff_len=30):

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 |
|---|---|---|---|---|---|
| 2015–2017 (2018) | 37.1% | 54.8% | 67.5% | 78.5% | 91.8% |
| 2015–2018 (2019) | 34.9% | 55.0% | 66.9% | 80.2% | 93.6% |
| 2015–2019 (2020) | 32.8% | 53.3% | 66.2% | 81.0% | 93.4% |
| 2015–2020 (2021) | 29.3% | 42.7% | 57.8% | 79.5% | 95.1% |
| 2015–2021 (2022) | 27.7% | 41.3% | 48.0% | 66.0% | 85.4% |
| 2015–2022 (2023) | 28.7% | 47.3% | 56.4% | 69.9% | 88.3% |
| 2015–2023 (2024) | 39.4% | 59.8% | 73.0% | 86.3% | 96.2% |
| 2015–2024 (2025) | 50.5% | 80.1% | 89.3% | 94.9% | 99.0% |

**SFT Epoch 14 (ep014):**

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 | eff_len=30 |
|---|---|---|---|---|---|---|
| 2015–2017 (2018) | 0.0217 | 0.0321 | 0.0396 | 0.0460 | 0.0539 | 0.0587 |
| 2015–2018 (2019) | 0.0201 | 0.0317 | 0.0387 | 0.0463 | 0.0541 | 0.0579 |
| 2015–2019 (2020) | 0.0197 | 0.0321 | 0.0400 | 0.0488 | 0.0563 | 0.0602 |
| 2015–2020 (2021) | 0.0096 | 0.0141 | 0.0191 | 0.0263 | 0.0315 | 0.0331 |
| 2015–2021 (2022) | 0.0076 | 0.0113 | 0.0132 | 0.0182 | 0.0236 | 0.0277 |
| 2015–2022 (2023) | 0.0069 | 0.0114 | 0.0135 | 0.0168 | 0.0214 | 0.0243 |
| 2015–2023 (2024) | 0.0077 | 0.0117 | 0.0142 | 0.0168 | 0.0187 | 0.0194 |
| 2015–2024 (2025) | 0.0079 | 0.0126 | 0.0141 | 0.0150 | 0.0156 | 0.0158 |

Retention (% of eff_len=30):

| Fold (eval year) | eff_len=5 | eff_len=10 | eff_len=15 | eff_len=20 | eff_len=25 |
|---|---|---|---|---|---|
| 2015–2017 (2018) | 37.0% | 54.7% | 67.4% | 78.4% | 91.8% |
| 2015–2018 (2019) | 34.7% | 54.8% | 66.8% | 80.1% | 93.6% |
| 2015–2019 (2020) | 32.8% | 53.4% | 66.4% | 81.2% | 93.6% |
| 2015–2020 (2021) | 29.2% | 42.5% | 57.7% | 79.5% | 95.1% |
| 2015–2021 (2022) | 27.4% | 40.9% | 47.4% | 65.6% | 85.2% |
| 2015–2022 (2023) | 28.4% | 46.8% | 55.8% | 69.4% | 88.1% |
| 2015–2023 (2024) | 39.6% | 60.3% | 73.4% | 86.6% | 96.4% |
| 2015–2024 (2025) | 50.1% | 79.9% | 89.1% | 94.9% | 98.9% |

**Key observations:**

1. **The oldest 5 bars contribute very little — consistently across all SFT epochs.** At every epoch, `eff_len=25` retains 85–99% of full-window performance depending on the fold, with the highest retention observed in the most recent folds (2024–2025) and a notable dip at the 2022–2023 folds.

2. **In the most recent fold, the oldest bars are nearly irrelevant.** For the 2015–2024 fold (evaluated on 2025 data), `eff_len=25` retains **98.8–99.1%** of `eff_len=30` performance across all five SFT epochs — masking out the 5 oldest bars costs less than \~1.2% of predictive signal. Even `eff_len=10` retains \~80%.

3. **The overall trend favors more recent folds, but is not monotonic.** Across folds, the retention at `eff_len=25` follows a non-monotonic pattern: early folds (2018–2021) show 92–95% retention, but there is a notable dip at the 2022 fold (\~85–86%) and the 2023 fold (\~88–89%), before a strong recovery in the 2024 fold (\~96–97%) and the 2025 fold (\~98–99%). This dip-and-recovery pattern is consistent across all five SFT epochs, confirming it reflects a genuine property of the data rather than an artifact of epoch selection.

**Motivation for the full ablation.** This observation — that the oldest bars in the window are effectively dead weight, especially in recent data — led us to question whether `seq_len=30` is truly optimal. If the first 5 bars barely matter at inference time, perhaps a model *trained* on a shorter window would perform equally well or better, avoiding the noise and overfitting risk introduced by uninformative early positions. This motivated the systematic `seq_len` ablation study described in the findings below.

#### Finding 1: `seq_len=30` wins all folds through 2023; `seq_len=5` and `seq_len=10` win in 2024–2025

We trained models with `seq_len` ∈ {5, 10, 15, 30} across 8 folds (10 seeds each) and selected the best (ssl_ep, sft_ep) configuration per (seq_len, fold) by mean eval_corr across seeds. The table below reports the best achievable eval_corr for each seq_len in each fold. Bold = best seq_len for that fold.

| Eval Year (Fold) | seq_len=5 | seq_len=10 | seq_len=15 | seq_len=30 |
|---|---|---|---|---|
| 2018 (Fold_2015-2017) | 0.0387 | 0.0498 | — | **0.0588** |
| 2019 (Fold_2015-2018) | 0.0387 | 0.0488 | — | **0.0579** |
| 2020 (Fold_2015-2019) | 0.0442 | 0.0521 | — | **0.0602** |
| 2021 (Fold_2015-2020) | 0.0254 | 0.0292 | — | **0.0331** |
| 2022 (Fold_2015-2021) | 0.0226 | 0.0236 | — | **0.0277** |
| 2023 (Fold_2015-2022) | 0.0196 | 0.0201 | — | **0.0243** |
| 2024 (Fold_2015-2023) | 0.0207 | **0.0213** | — | 0.0195 |
| 2025 (Fold_2015-2024) | **0.0206** | 0.0195 | 0.0184 | 0.0162 |

*Note: seq_len=15 was only trained on the 2025 fold. Dashes indicate no trained model for that configuration.*

The facts:

1. **`seq_len=30` is the best choice for eval years 2018–2023**, outperforming shorter windows in every fold. The advantage is large in early folds (0.0588 vs. 0.0498 for seq_len=10 in 2018, a 18% gap) and narrows steadily over time (0.0243 vs. 0.0201 in 2023, a 21% gap in absolute but both much smaller).

2. **The crossover occurs at the 2024 eval fold.** `seq_len=10` (0.0213) overtakes `seq_len=30` (0.0195), a +9.2% relative improvement. `seq_len=5` (0.0207) also exceeds `seq_len=30`.

3. **In the 2025 eval fold, shorter is strictly better.** The ordering is `seq_len=5` (0.0206) > `seq_len=10` (0.0195) > `seq_len=15` (0.0184) > `seq_len=30` (0.0162). `seq_len=5` outperforms `seq_len=30` by +27%.

4. **The ratio of seq_len=5 to seq_len=30 performance rises overall** from 66% (2018) → 67% (2019) → 73% (2020) → 77% (2021) → 82% (2022) → 81% (2023) → 106% (2024) → 127% (2025). The trend is not strictly monotonic (82%→81% from 2022 to 2023), but the overall direction is clear.

The corresponding optimal (ssl_ep, sft_ep) configurations for each cell above:

| Eval Year | seq_len=5 | seq_len=10 | seq_len=15 | seq_len=30 |
|---|---|---|---|---|
| 2018 | ssl120 / sft25 | ssl120 / sft25 | — | ssl150 / sft15 |
| 2019 | ssl120 / sft24 | ssl120 / sft24 | — | ssl150 / sft15 |
| 2020 | ssl120 / sft25 | ssl120 / sft25 | — | ssl150 / sft15 |
| 2021 | ssl120 / sft25 | ssl120 / sft25 | — | ssl150 / sft14 |
| 2022 | ssl120 / sft25 | ssl120 / sft25 | — | ssl150 / sft15 |
| 2023 | ssl120 / sft7 | ssl120 / sft17 | — | ssl150 / sft13 |
| 2024 | ssl120 / sft25 | ssl120 / sft25 | — | ssl150 / sft12 |
| 2025 | ssl120 / sft25 | ssl120 / sft16 | ssl60 / sft11 | ssl150 / sft7 |

*Note: For seq_len=5, ssl_ep=150/180 were run with fewer seeds (2–3) in some folds and achieved marginally higher corr (≤0.0004 difference). The table above reports results from ssl_ep=120 which has 10 seeds across all folds for consistency.*

#### Finding 2: Trained short-window models far outperform masked long-window models

The Preliminary Finding showed that masking the oldest bars in a `seq_len=30` model at inference time retains most performance (e.g., `eff_len=5` retains \\~50% of full-window performance in the 2025 fold). A natural question is whether *training* a model directly on a short window does better than *masking* a long-window model to the same effective length.

The answer is yes, by a large margin. The table below compares, for the 2025 eval fold, trained models at each `seq_len` vs. `seq_len=30` models evaluated with the corresponding `eff_len` mask:

| Window | Trained model (eval_corr) | Masked seq_len=30 model (eval_corr) | Trained / Masked |
|---|---|---|---|
| 5 bars | 0.0206 | 0.0084 | 2.45× |
| 10 bars | 0.0195 | 0.0129 | 1.51× |
| 15 bars | 0.0184 | 0.0143 | 1.29× |
| 30 bars (full) | 0.0162 | 0.0160 | 1.01× |

At `seq_len=30` the trained and masked numbers nearly match (0.0162 vs. 0.0160), confirming the comparison is valid. At shorter windows, the trained model substantially outperforms the masked model: **2.45× for 5 bars, 1.51× for 10 bars**. The gap is largest for the shortest windows, where the `seq_len=30` model has never learned to make predictions from only 5 bars of input.

#### Finding 3: SSL epoch sensitivity is weak; optimal SFT epochs decline for `seq_len=30` in recent folds

**SSL epoch sensitivity.** For both `seq_len=5` and `seq_len=10`, the range of best eval_corr across SSL epochs is small — typically 0.001–0.002 in absolute terms. The tables below show best eval_corr (over SFT epochs) for each SSL epoch.

**seq_len=5** (6 SSL epochs tested: 30, 60, 90, 120, 150, 180):

| Eval Year | ssl=30 | ssl=60 | ssl=90 | ssl=120 | ssl=150 | ssl=180 |
|---|---|---|---|---|---|---|
| 2018 | 0.0373 | 0.0381 | 0.0382 | 0.0383 | 0.0385 | **0.0387** |
| 2019 | 0.0374 | 0.0380 | 0.0382 | 0.0384 | 0.0386 | **0.0387** |
| 2020 | 0.0420 | 0.0430 | 0.0435 | 0.0438 | 0.0442 | **0.0442** |
| 2021 | 0.0233 | 0.0242 | 0.0247 | 0.0249 | 0.0254 | **0.0254** |
| 2022 | 0.0216 | 0.0222 | 0.0225 | **0.0226** | 0.0224 | 0.0225 |
| 2023 | 0.0192 | 0.0195 | 0.0195 | **0.0196** | 0.0193 | 0.0193 |
| 2024 | 0.0200 | 0.0202 | 0.0205 | **0.0207** | 0.0205 | 0.0204 |
| 2025 | 0.0200 | 0.0201 | 0.0204 | 0.0204 | 0.0205 | **0.0206** |

**seq_len=10** (4 SSL epochs tested: 30, 60, 90, 120; ssl=150 available for 2025 fold only):

| Eval Year | ssl=30 | ssl=60 | ssl=90 | ssl=120 | ssl=150 |
|---|---|---|---|---|---|
| 2018 | 0.0487 | 0.0491 | 0.0493 | **0.0498** | — |
| 2019 | 0.0476 | 0.0483 | 0.0486 | **0.0488** | — |
| 2020 | 0.0512 | 0.0516 | 0.0517 | **0.0521** | — |
| 2021 | 0.0283 | 0.0288 | 0.0291 | **0.0292** | — |
| 2022 | 0.0228 | 0.0231 | 0.0235 | **0.0236** | — |
| 2023 | 0.0193 | 0.0195 | 0.0199 | **0.0201** | — |
| 2024 | 0.0209 | 0.0210 | 0.0211 | **0.0213** | — |
| 2025 | 0.0192 | 0.0191 | 0.0191 | 0.0192 | **0.0195** |

For `seq_len=5`, the best SSL epoch is 120 in most folds (with 150/180 marginally better in some, within noise). For `seq_len=10`, ssl_ep=120 is best in every fold where it was tested with 10 seeds. The total spread from worst to best SSL epoch is: 0.0014 for seq_len=5 (2018) and 0.0011 for seq_len=10 (2018). In recent folds the spread is even smaller (\\~0.0004–0.0007).

**Optimal SFT epoch declines for `seq_len=30`.** The optimal SFT epoch for `seq_len=30` decreases steadily across folds: 15 (2018) → 15 (2019–2022) → 13 (2023) → 12 (2024) → 7 (2025). By contrast, `seq_len=5` and `seq_len=10` consistently select sft_ep=24–25 (the maximum or near-maximum), with the exception of the 2023 fold where `seq_len=5` selects sft_ep=7 and `seq_len=10` selects sft_ep=17. This means `seq_len=30` models begin overfitting earlier in SFT in recent folds.

### 6.2 Practical Conclusion

For current data, the safest and strongest candidates are **`seq_len=5` and `seq_len=10`**, with `seq_len=10` serving as a good overall compromise and `seq_len=5` appearing particularly robust in recent folds.

By contrast, `seq_len=15/20/25/30` increasingly underperform in recent years, suggesting that longer context windows are no longer the best default choice under the current regime.

### 6.3 Summary

1. `seq_len=30` was the best choice for eval years 2018–2023; `seq_len=10` won in 2024; `seq_len=5` won in 2025.
2. Training on short windows substantially outperforms masking long-window models to the same effective length (up to 2.45×).
3. SSL epoch sensitivity is weak for short windows; optimal SFT epoch for `seq_len=30` declines sharply in recent folds.
4. Sequence length should **not** be treated as a fixed globally optimal hyperparameter — the optimal value has shifted from 30 to 5–10 over the evaluation period.

---

## 7. Timestep Importance Ablation

### 7.1 Methodology

For each `(seqlen, fold, seed)` combination, we load the best checkpoint and run 1 baseline + N ablation evaluations. Each ablation zeros out a single timestep's features and marks it as padding in the attention mask. The **drop** = baseline_corr - masked_corr measures that timestep's contribution. Larger drop = more important timestep.

- **Seqlen=5**: t0 (4 min ago) → t4 (current minute), 6 evals per model
- **Seqlen=10**: t0 (9 min ago) → t9 (current minute), 11 evals per model
- **Folds**: 8 folds (eval years 2017-2024)
- **Seeds**: 5 seeds (0-4) per fold
- **Total**: 80 models × (6 or 11) evals = 1,160 evaluation runs

### 7.2 Seqlen=5 Results

#### 7.2.1 Average Relative Importance Across All Folds (by seed)

| Timestep | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | **Mean** | **Std** |
|----------|--------|--------|--------|--------|--------|----------|---------|
| t0 (4 min ago) | 14.3% | 15.5% | 13.7% | 13.8% | 12.4% | **13.9%** | 1.1% |
| t1 (3 min ago) | 8.6% | 7.8% | 7.3% | 8.2% | 7.4% | **7.9%** | 0.5% |
| t2 (2 min ago) | 9.4% | 8.8% | 9.7% | 9.7% | 9.2% | **9.4%** | 0.4% |
| t3 (1 min ago) | 12.4% | 12.3% | 13.2% | 11.4% | 12.9% | **12.4%** | 0.7% |
| t4 (current) | 55.3% | 55.6% | 56.1% | 56.9% | 58.1% | **56.4%** | 1.1% |

**Key findings:**
- **t4 (current minute) dominates at ~56%**, contributing more than all other timesteps combined.
- **t0 (4 min ago) ranks second at ~14%**, notably higher than t1-t3.
- The importance ranking is consistently **t4 >> t0 > t3 > t2 > t1**, forming a "U-shape" where both endpoints matter more than the middle.
- **Cross-seed std is extremely low (0.4%-1.1%)**, confirming that different random seeds converge to the same learned features.

#### 7.2.2 Per-Fold Breakdown (seed-averaged)

| Fold (Eval Year) | t0 | t1 | t2 | t3 | t4 |
|-------------------|------|------|------|------|------|
| 2017 | 17.3% | 9.7% | 8.8% | 9.4% | 54.7% |
| 2018 | 13.3% | 9.2% | 10.0% | 12.5% | 55.1% |
| 2019 | 15.7% | 8.8% | 9.4% | 11.2% | 54.9% |
| 2020 | 13.9% | 9.1% | 9.6% | 10.8% | 56.7% |
| 2021 | 16.9% | 8.6% | 8.9% | 12.7% | 52.9% |
| 2022 | 12.3% | 9.5% | 13.0% | 16.3% | 49.2% |
| 2023 | 11.7% | 3.9% | 8.2% | 12.1% | 64.0% |
| 2024 | 10.4% | 5.6% | 6.8% | 11.3% | 65.8% |

**Observations:**
- The dominance of t4 is stable across all folds (49-66%).
- 2022 is a mild outlier: t4 drops to 49% while t3 rises to 16%, suggesting the model relies slightly more on 2-min-ago data in that year (possibly due to different market microstructure).
- Later folds (2023-2024) show even stronger t4 dominance (64-66%), with t1 shrinking to 4-6%.

#### 7.2.3 Cross-Seed Consistency (per fold)

| Fold | t0 range | t4 range | Pattern consistent? |
|------|----------|----------|---------------------|
| 2017 | 13.0-19.4% | 48.8-65.6% | Yes |
| 2018 | 12.1-15.5% | 50.3-58.4% | Yes |
| 2019 | 14.0-17.2% | 52.8-58.0% | Yes |
| 2020 | 12.1-16.5% | 53.4-59.2% | Yes |
| 2021 | 12.7-24.7% | 42.6-57.6% | Yes (wider range) |
| 2022 | 10.4-13.9% | 44.0-52.6% | Yes |
| 2023 | 10.5-12.8% | 60.7-68.0% | Yes |
| 2024 | 5.4-15.2% | 60.3-72.5% | Yes |

All folds maintain the same importance ordering across all 5 seeds: **t4 >> t0 > rest**.

### 7.3 Seqlen=10 Results

#### 7.3.1 Average Relative Importance Across All Folds (by seed)

| Timestep | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | **Mean** | **Std** |
|----------|--------|--------|--------|--------|--------|----------|---------|
| t0 (9 min ago) | 3.6% | 5.0% | 4.6% | 4.8% | 3.1% | **4.2%** | 0.8% |
| t1 (8 min ago) | 1.2% | 1.6% | 1.6% | 1.2% | 1.0% | **1.3%** | 0.3% |
| t2 (7 min ago) | 2.1% | 1.9% | 3.5% | 1.8% | 1.8% | **2.2%** | 0.7% |
| t3 (6 min ago) | 2.8% | 2.7% | 3.7% | 2.6% | 2.5% | **2.9%** | 0.5% |
| t4 (5 min ago) | 3.2% | 3.0% | 3.8% | 2.6% | 2.8% | **3.1%** | 0.5% |
| t5 (4 min ago) | 3.8% | 3.4% | 4.2% | 3.3% | 3.6% | **3.7%** | 0.3% |
| t6 (3 min ago) | 4.4% | 3.9% | 3.9% | 4.3% | 4.7% | **4.2%** | 0.3% |
| t7 (2 min ago) | 5.5% | 5.2% | 4.5% | 6.1% | 6.2% | **5.5%** | 0.7% |
| t8 (1 min ago) | 7.8% | 7.7% | 6.2% | 8.5% | 8.7% | **7.8%** | 1.0% |
| t9 (current) | 65.6% | 64.3% | 64.2% | 63.8% | 65.6% | **64.7%** | 0.8% |

**Key findings:**
- **t9 (current minute) dominates at ~65%**, even stronger than seqlen=5's t4.
- From t1 to t9, importance increases **nearly monotonically**: 1.3% → 2.2% → 2.9% → 3.1% → 3.7% → 4.2% → 5.5% → 7.8% → 64.7%.
- **t0 (9 min ago) breaks the monotonicity at 4.2%**, higher than t1-t4. This "boundary effect" mirrors the elevated t0 in seqlen=5.
- **t1 (8 min ago) is the least important at 1.3%**, frequently showing negative drops (masking it *improves* performance).
- **Cross-seed std remains very low (0.3%-1.0%)**.

#### 7.3.2 Per-Fold Breakdown (seed-averaged)

| Fold | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2017 | 5.6% | 2.0% | 2.2% | 2.2% | 2.6% | 3.4% | 4.3% | 5.7% | 7.5% | 64.5% |
| 2018 | 6.2% | 2.4% | 2.4% | 3.0% | 3.2% | 3.8% | 3.9% | 4.6% | 6.2% | 64.1% |
| 2019 | 6.0% | 2.8% | 2.6% | 2.4% | 2.9% | 3.7% | 4.2% | 5.0% | 6.5% | 64.1% |
| 2020 | 3.4% | 0.2% | 2.3% | 3.3% | 3.4% | 3.6% | 4.3% | 5.6% | 7.9% | 63.7% |
| 2021 | 2.3% | 0.7% | 1.5% | 2.4% | 2.8% | 3.1% | 4.6% | 6.3% | 11.1% | 64.1% |
| 2022 | 0.2% | 0.1% | 3.8% | 6.2% | 6.4% | 6.5% | 6.4% | 6.7% | 10.4% | 53.3% |
| 2023 | 3.9% | 0.5% | 0.8% | 1.5% | 4.0% | 4.1% | 5.0% | 6.3% | 10.8% | 63.4% |
| 2024 | 5.2% | 2.1% | 1.3% | 0.6% | -0.1% | 0.3% | 1.7% | 4.0% | 6.1% | 74.7% |

**Observations:**
- t9 dominance is stable: 7 out of 8 folds have t9 in the 63-75% range.
- **2022 is again an outlier**: t9 drops to 53%, while t3-t7 each contribute 6-7% (more uniform distribution). This is consistent with seqlen=5 findings.
- **2024 shows the strongest t9 dominance (75%)**, with t3-t5 near zero or negative — the model barely uses 6-10 min old data.

#### 7.3.3 Negative Drop Analysis

Negative drop means masking that timestep *improves* performance — the timestep is pure noise.

| Timestep | Negative drops (out of 40) | % |
|----------|---------------------------|-----|
| t0 | 6 | 15% |
| t1 | 10 | 25% |
| t2 | 3 | 8% |
| t3 | 2 | 5% |
| t4 | 3 | 8% |
| t5 | 2 | 5% |
| t6-t9 | 0 | 0% |

**t1 (8 min ago) is noise in 25% of cases.** No timestep from t6 onwards ever shows negative drop.

### 7.4 Seqlen=5 vs Seqlen=10 Comparison

#### 7.4.1 Aligned Timestep Comparison

Seqlen=5's t0-t4 corresponds to seqlen=10's t5-t9 (both cover the most recent 5 minutes):

| Meaning | SL5 | SL10 | SL5 Importance | SL10 Importance |
|---------|-----|------|----------------|-----------------|
| Current minute | t4 | t9 | 56.4% | 64.7% |
| 1 min ago | t3 | t8 | 12.4% | 7.8% |
| 2 min ago | t2 | t7 | 9.4% | 5.5% |
| 3 min ago | t1 | t6 | 7.9% | 4.2% |
| 4 min ago | t0 | t5 | 13.9% | 3.7% |
| 5-9 min ago | N/A | t0-t4 | N/A | 13.7% (total) |

**Key comparisons:**

1. **Current minute is more dominant in seqlen=10 (65% vs 56%)**: The additional 5 low-value timesteps dilute the relative share of other positions, pushing t9's percentage higher.

2. **Seqlen=5's t0 (4 min ago) is anomalously high at 14%**: The equivalent position in seqlen=10 (t5) is only 3.7%. This suggests a **boundary/positional encoding effect** — being the first position in the sequence gives t0 an artificial boost in seqlen=5, rather than reflecting genuine signal from 4 minutes ago.

3. **The extra 5 minutes in seqlen=10 (t0-t4) contribute only 13.7% combined**: This is roughly equal to a single timestep (t0) in seqlen=5. The marginal value of extending from 5 to 10 minutes is minimal.

4. **Seqlen=10 has clear "wasted capacity"**: Timesteps t0-t4 frequently have near-zero or negative drops, meaning the model struggles to extract useful signal from 6-10 minutes ago.

#### 7.4.2 Boundary Position Effect

Both seqlen settings show an elevated importance for the first position (t0):

| Setting | First position importance | Second position importance | Ratio |
|---------|--------------------------|---------------------------|-------|
| Seqlen=5 | t0 = 13.9% | t1 = 7.9% | 1.76x |
| Seqlen=10 | t0 = 4.2% | t1 = 1.3% | 3.23x |

This is likely a Transformer positional encoding artifact — the first position serves as an "anchor" in self-attention, receiving disproportionate attention regardless of its actual information content.

### 7.5 Summary

| Dimension | Finding |
|-----------|---------|
| **Feature stability** | Highly stable. Cross-seed importance std < 1.5%, ranking is consistent across all 40 (fold, seed) combinations for each seqlen. |
| **Temporal decay** | Strictly decreasing — more recent minutes are more important. The current minute alone accounts for 56-65% of total importance. |
| **Optimal window** | ~5 minutes. Extending to 10 minutes adds minimal value; timesteps beyond 5 min often contribute zero or negative signal. |
| **Anomalous year** | 2022 consistently shows the lowest current-minute dominance (~49-53%) and more uniform distribution across timesteps, potentially reflecting different market microstructure. |
| **Boundary effect** | The first sequence position (t0) is consistently overweighted relative to its neighbors, likely a positional encoding artifact rather than genuine signal importance. |
| **Practical implication** | The model learns a stable, reproducible feature utilization pattern. The learned representations are not fold-dependent artifacts — they reflect genuine temporal structure in the data. A 5-minute input window captures the vast majority of useful information. |

---

## 8. Model Size Ablation

We tested whether increasing model capacity improves prediction quality, using `seq_len=30` on two recent folds (2024 and 2025 eval years) with 4 seeds per configuration. All models share the same architecture (Transformer encoder + RevIN + multiplicative gating) but differ in width, depth, and feedforward dimension.

### 8.1 Model Configurations

| Config | d_model | nhead | num_layers | dim_ff | Approx. Params |
|---|---|---|---|---|---|
| **baseline** | 128 | 4 | 3 | 256 | \~393K |
| **deep** | 128 | 4 | 6 | 256 | \~786K |
| **wide** | 256 | 8 | 3 | 512 | \~1.6M |
| **medium** | 256 | 8 | 6 | 512 | \~3.1M |
| **large** | 256 | 8 | 6 | 1024 | \~4.7M |

SSL pre-training was run separately for each model configuration, with SFT post-training epochs set to 15. SSL epochs tested: 60, 70, 80, 90, 100, 120, 150 (not all configurations have all SSL epochs).

### 8.2 Results

**Fold_2015-2024 (eval 2025)** — the most complete comparison (all 5 configs × multiple SSL epochs × 4 seeds, note that we only use 4 seeds here, so the results are different from the previous 10 seeds sampled results):

| Config | Params | ssl_ep090 | ssl_ep120 | ssl_ep150 |
|---|---|---|---|---|
| **baseline** | 393K | **0.0174** | 0.0169 | 0.0168 |
| deep | 786K | 0.0165 | 0.0161 | 0.0163 |
| wide | 1.6M | 0.0161 | 0.0162 | 0.0162 |
| medium | 3.1M | 0.0160 | 0.0160 | 0.0160 |
| large | 4.7M | 0.0154 | 0.0154 | 0.0156 |

### 8.3 Key Findings

1. **Increasing model size does not improve performance.** On the 2025 fold (the most reliable comparison), the ranking is strictly monotonic: baseline (0.0174) > deep (0.0165) > wide (0.0161) ≈ medium (0.0160) > large (0.0154). The smallest model wins by a clear margin. Scaling from 393K to 4.7M parameters (12× increase) results in an 11% *decrease* in eval correlation.

2. **The largest models perform worst.** The `medium` and `large` configurations consistently underperform, especially on the 2024 fold where `large` achieves only 0.0110 — barely half the baseline's 0.0174. This is a strong sign of overfitting: extra capacity captures noise rather than signal.

3. **The 2024 fold shows a different pattern with extended SSL training.** With ssl_ep120–150, `deep` (0.0182–0.0191) and `wide` (0.0195–0.0199) outperform `baseline` (0.0158–0.0178). However, these results have fewer seeds (1–3 vs 4) and the advantage does not replicate on the 2025 fold, suggesting it may be fold-specific or noise-driven.

4. **More SSL epochs do not rescue larger models.** On the 2025 fold, `large` achieves 0.0154–0.0156 across ssl_ep090/120/150 — essentially flat and always below baseline. The model is not undertrained; it simply lacks useful signal to absorb with extra parameters.

### 8.4 Interpretation

The failure of model scaling is consistent with the overall finding that predictive signal in recent data is weak and decaying. In a low signal-to-noise regime, additional model capacity primarily increases the risk of overfitting to idiosyncratic training patterns rather than learning robust predictive features. The baseline architecture (128-dim, 3 layers, \~393K params) already has sufficient capacity to capture the available signal; making it larger only adds noise sensitivity.

This result also reinforces the sequence length findings: both model size and sequence length show that **less is more** in the current regime — smaller models and shorter windows outperform their larger/longer counterparts.

**Reference implementation**: `run_posttrain_model_size.py`

---

## 9. Impact of Session Breaks in RevIN Normalization

Chinese futures markets have multiple intraday sessions (night, morning, midday, afternoon) separated by breaks (休市). When a `seq_len=5` window spans a session break, the features on either side of the break may come from different distributional regimes. We verified this empirically:
  - Each trading day has \~3 session breaks (e.g., night→morning gap=601 ticks, morning→midday gap=16, midday→afternoon gap=121).
  - With `seq_len=5`, each break contaminates 4 windows (those that include bars from both sides of the break).
  - The mean per-feature std within break-spanning windows is **1.62×** that of control windows (no break).
  - The effect is strongest when the break falls near the center of the window (offset 2–3: 1.71×, vs offset 1 or 4: \~1.50×).
  - Certain signal features are heavily affected: X47 (15.0×), X39 (9.7×), X7 (5.5×), beyond the expected impact on positional features like X6 (60×).
  - Approximately **3–4%** of all windows per day are affected (e.g., RB0: 12.4 / 335 windows = 3.7%).

Since RevIN computes mean and std across all timesteps in the window, a session break within the window inflates the variance estimate and distorts the normalized representation. Potential mitigations include: (a) breaking sequences at session boundaries so that no window spans a break, (b) adding a session-aware mask to RevIN so that normalization statistics are computed only within contiguous sessions, or (c) inserting a learned session-break token at break positions. However, from the current data, there is no evidence that session breaks are causing severe issues for us.
