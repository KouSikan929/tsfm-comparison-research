# Methodology / Experimental Setup — Current Implementation Audit

**Purpose.** This document is a factual audit of what the codebase in this repository actually
does, as of **2026-09-10**, verified line-by-line against the source (not against memory,
notes, or the research-plan document). It is written so it can be used directly as the basis
for a thesis Methodology / Experimental Setup section — but note the label on every item.

**Update (same day, later revision):** MASE, the expanded 6-dimension regime feature set, the
causal per-market regime thresholds, `--seed` CLI support, and a full iTransformer retrain on
the refreshed data have all since been implemented — every section below reflects the *current*
state, not the original audit. Only fine-tuning (§6) remains genuinely not implemented, by the
user's explicit choice to run that work on a separate, more powerful server.

**How to read every item below.** Each piece of information is tagged:

- **[IMPLEMENTED]** — directly observed in the code or a run's actual output; a verified fact.
- **[CONFIG-INFERRED]** — not stated outright anywhere, but follows necessarily from the
  implemented configuration (e.g. "windows overlap" follows from `stride < pred_len`).
- **[NOT DEFINED]** — does not exist in the current implementation. A recommendation is given
  *separately*, clearly marked as a recommendation, never blended into the factual description.

Where the research proposal (`M研究計画概要_TSFM.docx`, revised 2026-08-26) describes something
more ambitious than what the code currently does, that gap is called out explicitly rather than
silently describing the proposal's intent as if it were implemented.

---

## 1. Datasets / Assets

All data is daily OHLCV pulled via `yfinance` (`scripts/download_data.py`), saved to
`data/raw/<name>_daily.csv` with columns `timestamps, open, high, low, close, volume, amount`
(`amount` is **not** a real reported field — it is computed as `close * volume`, since yfinance
does not report turnover value for any of these tickers). **[IMPLEMENTED]**

Numbers below are from the most recent download (2026-09-10). **[IMPLEMENTED]**

| Dataset | Ticker | Asset type | Market / exchange | Data range | Rows | Sampling frequency |
|---|---|---|---|---|---|---|
| SP500 | `^GSPC` | Equity index | US (S&P Dow Jones Indices) | 2015-01-02 → 2026-09-09 | 2,938 | Daily, weekdays only |
| SSE | `000001.SS` | Equity index | China A-share, Shanghai Stock Exchange | 2015-01-05 → 2026-09-09 | 2,839 | Daily, weekdays only |
| SZSE | `399001.SZ` | Equity index | China A-share, Shenzhen Stock Exchange | 2015-01-05 → 2026-09-09 | 2,837 | Daily, weekdays only |
| Nikkei225 | `^N225` | Equity index | Japan, Tokyo Stock Exchange | 2015-01-05 → 2026-09-09 | 2,856 | Daily, weekdays only |
| CSI300 | `000300.SS` | Equity index | China A-share, Shanghai/Shenzhen large-cap composite | 2021-03-11 → **2026-07-17** | 1,298 | Daily, weekdays only |
| TOPIX | `1306.T` | Equity index (via ETF proxy) | Japan, Tokyo Stock Exchange | 2015-01-05 → 2026-09-09 | 2,876 | Daily, weekdays only |
| KO | `KO` | Individual equity | US, NYSE (Coca-Cola) | 2015-01-02 → 2026-09-09 | 2,938 | Daily, weekdays only |
| BTC | `BTC-USD` | Cryptocurrency | Global, 24/7 (Yahoo's aggregated feed) | 2015-01-01 → 2026-09-08 | 4,269 | **Daily, every calendar day** |
| NVDA | `NVDA` | Individual equity | US, Nasdaq (Nvidia) | 2015-01-02 → 2026-09-09 | 2,938 | Daily, weekdays only |
| GLD | `GLD` | Commodity ETF (gold) | US, NYSE Arca (SPDR Gold Shares) | 2015-01-02 → 2026-09-09 | 2,938 | Daily, weekdays only |

Notes that must be in the thesis, not left implicit:

- **TOPIX is not the raw index.** Yahoo Finance does not serve `^TPX`/`^TOPX` (both return no
  data). `1306.T` is the Nomura TOPIX-tracking ETF, used as a close proxy for the index itself.
  **[IMPLEMENTED — deliberate substitution, documented in code comments]**
- **CSI300's feed is stuck.** As of two independent download attempts (2026-08-24 and
  2026-09-10), Yahoo's `000300.SS` feed produces one corrupted trailing row per fetch (NaN
  close price, a volume spike of ~1,000–3,000× the recent median, and a >1-month gap before
  it) dated as "today," and otherwise has not advanced past **2026-07-17**. The corrupted row
  is dropped in both cases. This means CSI300's usable data — and therefore its evaluation
  window count — has **not grown** since 2026-08-24, unlike the other nine datasets. This
  looks like a genuine, recurring limitation of this specific data feed and should be stated
  as such, not silently normalized away. **[IMPLEMENTED — verified twice]**
- **BTC's calendar is different in kind, not just count.** It trades every calendar day
  (365/year), while every other dataset here only has weekday rows (~252/year). Because the
  evaluation windows are defined by **row count**, not calendar time (see §4), BTC's
  `context_len=400` window spans roughly 400 calendar days (~13 months) while every equity/ETF
  dataset's spans roughly 400 *trading* days (~19 months of calendar time). This is a real,
  unresolved asymmetry in what "400 days of context" means across datasets.
  **[IMPLEMENTED behavior; CONFIG-INFERRED consequence — not compensated for anywhere in the code]**

---

## 2. Prediction Target

**[IMPLEMENTED]** Every model is evaluated on the **raw closing price level** (`close` column,
in the asset's native currency/units), at every forecast step. This is true for all four
models — confirmed by reading `eval_kronos.py`, `eval_lagllama.py`, `eval_timesfm.py`, and
`eval_itransformer.py`: in every case, `actual_close`/`pred_close` in the saved results are
literal close prices, not returns, log-returns, or any other transform. The model-internal
normalization each model applies (see §5) is not the same thing as the *target* being a return
— the target is inverse-transformed back to the raw price level before being recorded, in all
four cases.

**[NOT DEFINED]** Nothing in the current codebase forecasts **returns** or **log-returns**
directly as the target variable. The revised research proposal's Phase 5 (economic
backtesting) implicitly needs return-based signals (long/flat or long/short positions derived
from "predicted return or predicted direction") — this can be derived post-hoc from the
existing close-price forecasts (`predicted_return_t = pred_close_t / actual_close_{t-1} - 1`
or similar), but that derivation is **not implemented anywhere yet**.
*Recommendation (separate from the fact above): derive returns from the already-saved
`pred_close`/`actual_close` series rather than re-running any model — no new inference is
needed, only a post-processing step.*

---

## 3. Temporal Split (Train / Validation / Test)

**[IMPLEMENTED]** Fixed, absolute calendar-date boundaries — identical across every dataset,
defined in `scripts/download_data.py` and mirrored in
`models/iTransformer/data_provider/data_loader_date_split.py`:

```
TRAIN_END   = 2024-06-30
VAL_END     = 2025-06-30
TEST_START  = 2025-07-01
```

- **Train**: all rows with date ≤ 2024-06-30.
- **Validation**: rows with 2024-06-30 < date ≤ 2025-06-30.
- **Test**: all rows with date ≥ 2025-07-01.

**[IMPLEMENTED — rationale, from code comments]** `TEST_START` was chosen because it falls
after Kronos's own reported pretraining/held-out cutoff (~2025-06-05 per the model's public
documentation) — the tightest of the three foundation models' known cutoffs. Per-dataset row
counts in each split (current, 2026-09-10):

| Dataset | Train rows | Val rows | Test rows |
|---|---|---|---|
| SP500 | 2,388 | 250 | 300 |
| SSE | 2,304 | 242 | 293 |
| SZSE | 2,302 | 242 | 293 |
| Nikkei225 | 2,320 | 244 | 292 |
| CSI300 | 801 | 242 | 255 |
| TOPIX | 2,340 | 244 | 292 |
| KO | 2,388 | 250 | 300 |
| BTC | 3,469 | 365 | 435 |
| NVDA | 2,388 | 250 | 300 |
| GLD | 2,388 | 250 | 300 |

**Important distinction the proposal explicitly asks to be documented (per the 2026-08-26
revision):** this train/val/test split is **strictly, mechanically enforced** for the data this
project controls directly. It does **not** and **cannot** guarantee the three pretrained
foundation models (Kronos, Lag-Llama, TimesFM) never saw data from this exact test window
during their own pretraining — their full pretraining corpora are not fully disclosed/auditable
by this project. `TEST_START` is chosen to **minimize** that risk (by starting after Kronos's
known cutoff), not to eliminate it. This is a deliberate, documented distinction between (a)
leakage within this project's own pipeline, which is fully controlled, and (b) potential
pretraining-corpus overlap, which is only mitigated. **[IMPLEMENTED as policy / CONFIG-INFERRED
as a limitation]**

---

## 4. Rolling Evaluation Windows

**[IMPLEMENTED]** Defined in `scripts/common/windowing.py`, identical for every dataset and
every model (all four models read the exact same `data/eval_windows_<dataset>.csv`):

| Parameter | Value |
|---|---|
| Context length | 400 rows |
| Prediction horizon | 20 rows (steps) |
| Stride (step between window starts) | 10 rows |
| Window overlap | **Yes** — consecutive windows share 10 of their 20 target rows (`stride=10 < pred_len=20`) **[CONFIG-INFERRED]** |
| Context source constraint | May reach back before `TEST_START` into train/val rows (this is historical input, not label leakage) |
| Target source constraint | Every row in the target range must be `>= TEST_START` |

Windows are generated by starting at the first row `>= TEST_START` and stepping forward by
`stride` until fewer than `pred_len` rows remain. **[IMPLEMENTED]**

Current window counts per dataset (2026-09-10, matches the row counts in §1):

| Dataset | Windows | First target start | Last target end |
|---|---|---|---|
| SP500 | 29 | 2025-07-01 | 2026-09-09 |
| SSE | 28 | 2025-07-01 | 2026-09-04 |
| SZSE | 28 | 2025-07-01 | 2026-09-04 |
| Nikkei225 | 28 | 2025-07-01 | 2026-09-07 |
| CSI300 | 24 | 2025-07-01 | 2026-07-10 |
| TOPIX | 28 | 2025-07-01 | 2026-09-07 |
| KO | 29 | 2025-07-01 | 2026-09-09 |
| BTC | 42 | 2025-07-01 | 2026-09-03 |
| NVDA | 29 | 2025-07-01 | 2026-09-09 |
| GLD | 29 | 2025-07-01 | 2026-09-09 |

Total across all 10 datasets: **294 windows**, ~5,880 individual forecast points (294 × 20).
**[IMPLEMENTED, computed from current files]**

**Overlap and statistical independence [IMPLEMENTED / disclosed].** Because windows overlap,
per-window forecast errors are **not** independent draws. This is the documented reason the
Diebold-Mariano significance test (`scripts/aggregate_results.py`) uses a Newey-West-style
variance correction (summing autocovariances up to lag `h-1`, `h = pred_len`) rather than a
naive t-test — see the `diebold_mariano()` function's docstring in that file.

**iTransformer's own window enumeration [IMPLEMENTED].** iTransformer's native test loop
(`experiments/exp_long_term_forecasting.py`) evaluates **every stride-1 window** across its
test split internally, not just the 24–42 windows above; `eval_itransformer.py` subselects only
the windows matching the shared `stride=10` schedule so the four models are compared on
*identical* windows. The stride-1 native evaluation exists but is discarded for cross-model
comparison purposes.

---

## 5. Models — Exact Versions, Checkpoints, and Normalization

### 5.1 Kronos

| Item | Value | Status |
|---|---|---|
| Tokenizer checkpoint | `NeoQuasar/Kronos-Tokenizer-base` (HF Hub) | [IMPLEMENTED] |
| Model checkpoint | `NeoQuasar/Kronos-small` (HF Hub) | [IMPLEMENTED] |
| Parameter count | 24.7M (per the vendored repo's own README table) | [IMPLEMENTED, sourced from `models/Kronos/README.md`] |
| Max native context | 512 (tokenizer/model architecture limit) | [IMPLEMENTED] |
| Context used in this project | 400 (within the 512 limit — no truncation) | [IMPLEMENTED] |
| Training condition | **Zero-shot.** No fine-tuning. | [IMPLEMENTED] |
| Sampling params | `T=1.0`, `top_p=0.9`, `top_k=0` (default) | [IMPLEMENTED] |
| Input features | open, high, low, close, volume, amount (all 6 columns) | [IMPLEMENTED] |
| Normalization | **Per-window z-score**, computed from the input context itself: `x_mean, x_std = mean(x, axis=0), std(x, axis=0)` over the 400-row context (per-channel), then `x_norm = clip((x - x_mean) / (x_std + 1e-5), -5, 5)`. Output is de-normalized with the *same* per-window `x_mean`/`x_std`. **This is not a train-set-derived statistic — it is recomputed independently for every single window.** | [IMPLEMENTED — verified in `models/Kronos/model/kronos.py`] |
| Uncertainty mechanism | Kronos's own `predict()` internally averages `sample_count` stochastic draws and discards the individual samples (`preds = np.mean(preds, axis=1)` in `auto_regressive_inference`). To obtain an actual empirical distribution, `eval_kronos.py` instead calls `predict()` **10 independent times** at `sample_count=1` per window (each call is one stochastic draw under `T=1.0`/`top_p=0.9`) and computes mean/10th/90th percentile across the 10 draws itself. | [IMPLEMENTED — documented in `eval_kronos.py`'s module docstring] |
| Random seed | **Not set anywhere in `eval_kronos.py`.** Sampling is stochastic and unseeded — repeat runs will not reproduce bit-identical draws (though should be close in aggregate given 10-sample averaging). | [NOT DEFINED] |

### 5.2 Lag-Llama

| Item | Value | Status |
|---|---|---|
| Checkpoint | `time-series-foundation-models/Lag-Llama` / `lag-llama.ckpt` (HF Hub) | [IMPLEMENTED] |
| Native pretrained context length | **32** (from the checkpoint's own `hyper_parameters.model_kwargs.context_length`) | [IMPLEMENTED — read directly from the checkpoint] |
| Context used in this project | 400 — extended **12.5×** beyond the native 32 via linear RoPE scaling (`factor = (context_len + pred_len) / 32 = 13.125`) | [IMPLEMENTED] |
| Architecture (from checkpoint) | `n_layer=8`, `n_head=9`, `n_embd_per_head=16`, `input_size=1` (univariate) | [IMPLEMENTED — read directly from the checkpoint] |
| Scaling method (from checkpoint) | **`"robust"`** — i.e. **not** mean-scaling; this is the value stored in the checkpoint's own hyperparameters and is applied internally by GluonTS, not chosen by this project | [IMPLEMENTED — read directly from the checkpoint; this had been assumed to be "mean" scaling in earlier project notes, which was incorrect] |
| `time_feat` (from checkpoint) | `True` | [IMPLEMENTED] |
| Training condition | **Zero-shot.** No fine-tuning. | [IMPLEMENTED] |
| Input features | close price only (univariate; `input_size=1` in the checkpoint) | [IMPLEMENTED] |
| Uncertainty mechanism | Native probabilistic forecasting — `num_samples=100` parallel samples per window via `num_parallel_samples`; mean and 10th/90th percentile computed from these 100 genuine samples (not discarded, unlike Kronos) | [IMPLEMENTED] |
| Random seed | **Not set anywhere in `eval_lagllama.py`.** | [NOT DEFINED] |
| Known environment caveats | PyTorch 2.6's `weights_only=True` default is patched to `False` for this checkpoint (embeds a `gluonts` class object); `setuptools<81` pinned in the env (newer versions drop `pkg_resources`, which `pytorch_lightning` still imports) | [IMPLEMENTED, environment-level] |

### 5.3 TimesFM

| Item | Value | Status |
|---|---|---|
| Checkpoint | `google/timesfm-2.5-200m-pytorch` (HF Hub) | [IMPLEMENTED] |
| Parameter count | 200M (per the checkpoint's own name/documentation) | [IMPLEMENTED] |
| Package used | `src/timesfm` (TimesFM 2.5) — **not** the legacy `v1/` package also present in the vendored repo | [IMPLEMENTED] |
| Training condition | **Zero-shot.** No fine-tuning. | [IMPLEMENTED] |
| Input features | close price only (univariate) | [IMPLEMENTED] |
| Normalization | **Two layers**, both per-window/per-instance, neither derived from a train-set statistic: (1) an always-on internal patch-wise RevIN normalization inside the model's decode loop (running mean/std per patch, computed from the input being decoded); (2) an additional outer instance normalization enabled by `normalize_inputs=True` in our `ForecastConfig` (`mu = mean(inputs, dim=-1)`, `sigma = std(inputs, dim=-1)` over the full input context, RevIN-applied, then reversed on the output) | [IMPLEMENTED — verified in `models/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`] |
| Forecast config | `max_context=400`, `max_horizon=20`, `use_continuous_quantile_head=True`, `fix_quantile_crossing=True`, `per_core_batch_size=1` | [IMPLEMENTED] |
| Uncertainty mechanism | Native quantile head — 9 quantiles (0.1…0.9) plus a mean channel, all produced in a single deterministic forward pass (no sampling) | [IMPLEMENTED] |
| Random seed | Not applicable in the same sense as Kronos/Lag-Llama — **TimesFM's forecast is deterministic given fixed weights and `ForecastConfig`** (no stochastic sampling in this pipeline), so re-running should reproduce bit-identical results modulo standard floating-point/kernel non-determinism (not separately controlled/seeded) | [IMPLEMENTED / CONFIG-INFERRED] |

### 5.4 iTransformer

| Item | Value | Status |
|---|---|---|
| Training condition | **Trained from scratch per market** (conventional supervised baseline) — no pretrained checkpoint of any kind | [IMPLEMENTED] |
| Invocation (exact, per market) | `python run.py --is_training 1 --root_path ./dataset/<name>/ --data_path <name>_daily.csv --model_id <name>_dateSplit_400_20 --model iTransformer --data custom_dateSplit --features MS --target close --freq d --seq_len 400 --label_len 48 --pred_len 20 --e_layers 2 --enc_in 6 --dec_in 6 --c_out 1 --des phase1 --d_model 128 --d_ff 128 --train_epochs 10 --patience 3 --itr 1` | [IMPLEMENTED — from `eval_itransformer.py`'s docstring, matches actual training-run commands] |
| Input mode | `features=MS` — **M**ultivariate input (all 6 columns), **S**ingle-variate output (close only) | [IMPLEMENTED] |
| `seq_len` / `label_len` / `pred_len` | 400 / 48 / 20 | [IMPLEMENTED] |
| Architecture | `e_layers=2`, `d_model=128`, `d_ff=128` (default `n_heads=8`, `d_layers=1` — not overridden) | [IMPLEMENTED] |
| Optimizer / training | Adam, default `learning_rate=0.0001`, `batch_size=32` (defaults, not overridden), up to 10 epochs with early stopping (`patience=3`) | [IMPLEMENTED] |
| Data split used | `custom_dateSplit` → `Dataset_CustomDateSplit`, using the exact `TRAIN_END`/`TEST_START` from §3 (**not** iTransformer's own default fixed 70/10/20 ratio split, which `Dataset_Custom` would otherwise use) | [IMPLEMENTED] |
| Normalization | **StandardScaler, fit on the TRAIN split only** (`self.scaler.fit(train_data.values)` where `train_data = df_data[0:num_train]`), applied per-channel across all 6 input columns, then applied unchanged to val/test. This is the **only one of the four models whose normalization statistics come from the training set** — the three foundation models all normalize per-window/per-instance instead (see §5.1–5.3). | [IMPLEMENTED — verified in `data_loader_date_split.py`] |
| Random seed | **`2023`**, applied to Python's `random`, `numpy`, and `torch` at the top of `run.py`'s `if __name__ == "__main__":` block (`random.seed(2023)`, `np.random.seed(2023)`, `torch.manual_seed(2023)`) | [IMPLEMENTED — the only one of the four models with an explicit, fixed seed] |
| **Freshness** | iTransformer has since been retrained on the refreshed (2026-09-10) data for all 10 markets, using the exact invocation above. All four models' results are now current and directly comparable — the earlier staleness caveat no longer applies. | [IMPLEMENTED] |

---

## 6. Fine-Tuning — NOT YET IMPLEMENTED

**[NOT DEFINED]** No fine-tuning or parameter-efficient adaptation of Kronos, Lag-Llama, or
TimesFM exists anywhere in this codebase. All three are evaluated zero-shot only, as documented
in §5. This is a known, explicit gap relative to the revised research proposal's Phase 2
("Zero-shot vs. fine-tuned").

What *has* been done is reconnaissance, not implementation:

- **Kronos** ships its own `finetune/` and `finetune_csv/` scripts in the vendored repo —
  identified as the most turnkey path, not yet adapted to this project's data/splits.
- **Lag-Llama**'s README documents a fine-tuning workflow (its own Colab Demo 2); not yet
  adapted.
- **TimesFM** — the zero-shot pipeline in this project uses the `google/timesfm-2.5-200m-pytorch`
  checkpoint via `src/timesfm`'s inference-only API (`forecast()`/`decode()`, wrapped in
  `torch.no_grad()`, no training path). A **separate** fine-tuning path was found in the
  vendored repo at `models/timesfm/timesfm-forecasting/examples/finetuning/finetune_lora.py`,
  using HuggingFace `transformers` + `peft` LoRA against a **different checkpoint**
  (`google/timesfm-2.5-200m-transformers`, not the one used for zero-shot). Neither `transformers`
  nor `peft` is installed in the `timesfm` conda env. Whether the fine-tuned model's inference
  output exposes the same quantile structure needed for calibration metrics (§5.3) has not been
  verified.

*Recommendation (separate from the above): treat fine-tuning as three separate, non-uniform
engineering efforts rather than one uniform step — Kronos is closest to turnkey, TimesFM
requires bridging two different checkpoint/API surfaces.*

---

## 7. Random Seeds & Reproducibility

**[IMPLEMENTED — summary table]**

| Component | Seed set? | Value | Consequence |
|---|---|---|---|
| Kronos eval (`eval_kronos.py`) | Optional, opt-in via `--seed N` (added after the initial audit) | Unset in every run collected so far | All results in this repo as of 2026-09-10 were produced unseeded; re-runs of those specific results will not be bit-identical even now that seeding is supported, since the flag wasn't used |
| Lag-Llama eval (`eval_lagllama.py`) | Optional, opt-in via `--seed N` (added after the initial audit) | Unset in every run collected so far | Same as above |
| TimesFM eval (`eval_timesfm.py`) | N/A (deterministic) | — | No sampling involved; re-runs should match modulo floating-point non-determinism |
| iTransformer training (`run.py`) | **Yes** | `2023` | `random`, `numpy`, `torch` all seeded; training should be close to reproducible (GPU non-determinism in some CUDA kernels is not separately controlled) |
| Data download (`download_data.py`) | N/A | — | Deterministic given the same date range and a stable upstream feed; **not** deterministic across time, since re-running later pulls more/updated rows (see CSI300 caveat in §1) |
| Regime feature computation | N/A (no randomness) | — | Fully deterministic given fixed input data |

**[IMPLEMENTED, partially]** `--seed` is now available on both `eval_kronos.py` and
`eval_lagllama.py` (seeds `torch`/`numpy` before the sampling loop). Not yet used to reproduce
or re-verify any of the results already collected — doing so would mean rerunning both models on
all 10 datasets again, which wasn't judged necessary given the aggregate metrics already average
over enough draws/windows to be stable in practice. Available for a future reviewer-reproducibility
pass if needed.

---

## 8. Number of Repeated Experimental Runs

**[IMPLEMENTED]** Each model is run **exactly once** per dataset in the current pipeline — there
is no multi-seed or multi-restart repetition of the full experiment for any model. This must not
be confused with the *within-run* stochastic sampling used for uncertainty estimation:

- Kronos: 1 run per dataset, containing 10 stochastic forward-pass draws *per window* (for
  quantile estimation only, not experiment repetition).
- Lag-Llama: 1 run per dataset, containing 100 parallel samples *per window* (same purpose).
- TimesFM: 1 run per dataset (deterministic, no sampling).
- iTransformer: 1 training run per dataset (`--itr 1` in the invocation in §5.4), i.e. one
  trained model per market, evaluated once.

**[NOT DEFINED]** No variance-across-independent-runs analysis exists (e.g., training
iTransformer 3–5 times with different seeds to report a mean±std, or repeating the zero-shot
sampling procedure across multiple random states). *Recommendation: if the thesis needs to
report result stability/variance (as distinct from the already-implemented per-window
performance distribution and Diebold-Mariano test — see §9), this would require re-running
iTransformer's training with `--itr N` (already supported natively by its own code — just not
currently used with N>1) and/or repeating the stochastic-sampling eval scripts multiple times.*

---

## 9. Metrics — Definitions and Implementation Status

Implemented in `scripts/common/results_io.py`'s `compute_metrics()`, applied identically
regardless of model. Let `n` = number of individual forecast points (all windows × all steps
pooled), and for directional accuracy, computed per window (see below).

| Metric | Status | Exact formula as implemented |
|---|---|---|
| **MAE** | [IMPLEMENTED] | `mean(abs(pred_close - actual_close))`, pooled over every (window, step) point |
| **RMSE** | [IMPLEMENTED] | `sqrt(mean((pred_close - actual_close)^2))`, pooled over every (window, step) point |
| **MASE** | **[IMPLEMENTED]** | `MASE = MAE / naive_insample_mae`, where `naive_insample_mae = mean(abs(diff(train_close, lag=1)))` computed once per dataset on the TRAIN split only (Hyndman & Koehler 2006 definition). Implemented as `results_io.naive_insample_mae()`; `compute_metrics(df, dataset=...)` adds it to the summary whenever a single dataset (not a pooled multi-dataset table) is being scored — pooling MASE across datasets with different price levels/naive-error scales would not be meaningful, so it is intentionally omitted from the pooled Phase 2 regime tables. |
| **Directional Accuracy** | [IMPLEMENTED] | **Not** "did the forecast correctly predict the direction of change from the last known context value." Instead: for each window, compute `sign(diff(actual_close within the window))` and `sign(diff(pred_close within the window))` across the window's `pred_len` steps, and count the fraction of the `pred_len - 1` **step-to-step** comparisons where actual and predicted direction agree. Pooled across all windows and all datasets included in a given run. This distinction matters for a thesis write-up — it is a within-horizon step-to-step measure, not an origin-relative one. |
| **80% interval coverage** | [IMPLEMENTED, not in the user's requested list but already used throughout] | `mean((actual_close >= pred_q10) & (actual_close <= pred_q90))`, only computed for models with non-null `pred_q10`/`pred_q90` (Kronos, Lag-Llama, TimesFM; NaN for iTransformer, which has no native uncertainty output) |
| **Diebold-Mariano test** | [IMPLEMENTED, pairwise] | Squared-error loss differential `d = errors_a^2 - errors_b^2`; test statistic `d_mean / sqrt(var_d/n)` where `var_d` includes a Newey-West-style correction summing autocovariances at lags `1..h-1` (`h = pred_len = 20`) weighted by `(1 - lag/h)`, to account for the serial correlation from overlapping windows (§4). Two-sided p-value from the standard normal CDF. In `scripts/aggregate_results.py`. |

---

## 10. Market Regime Classification

This is the area with the largest gap between the revised research proposal (which specifies
six conceptual dimensions) and the current implementation (which computes three features,
covering three of those six dimensions). Both are documented below, clearly separated.

### 10.1 What is currently implemented — all six proposal dimensions, now covered

Computed in `scripts/common/regime_features.py`, on the **context** portion of each window only
(never the target) — i.e., always causally available before the forecast is made:

| Feature | Formula | Dimension it covers |
|---|---|---|
| **Realized volatility** | `std(diff(log(close)), ddof=1) * sqrt(252)` over the full 400-day context — annualized std of daily log returns | Volatility |
| **Rolling 20-day volatility** | Same formula, restricted to the most recent 20 days of the context — a "recent conditions" signal distinct from the whole-context figure | Volatility |
| **Hurst exponent** | Variance-scaling estimator: for lags 2 to `min(100, len(context)//2)`, compute `tau(lag) = std(log_price[lag:] - log_price[:-lag])`; Hurst = slope of `log(tau)` vs. `log(lag)` via linear regression (`np.polyfit`, degree 1). `H < 0.5` → mean-reverting, `H ≈ 0.5` → random walk, `H > 0.5` → trending/persistent | Persistence / memory |
| **ADF statistic + p-value** | `statsmodels.tsa.stattools.adfuller` on the context's **log returns** (not the price level — see rationale below), `autolag="AIC"` | Stationarity |
| **KPSS statistic + p-value** | `statsmodels.tsa.stattools.kpss` on the context's log returns, `regression="c"`, `nlags="auto"` | Stationarity |
| **Efficiency Ratio (Kaufman)** | `abs(close[-1] - close[0]) / sum(abs(diff(close)))` over the context — 0 = no net progress (choppy), 1 = a straight-line trend | Trend strength |
| **Skewness, kurtosis** | `scipy.stats.skew`/`kurtosis` (excess) of the context's log returns | Distributional |
| **Max absolute return** | `max(abs(log returns))` over the context — a simple jump/magnitude indicator | Distributional |
| **Max drawdown** | Max peak-to-trough decline of `close` within the context, as a positive fraction | Market stress |
| **Downside volatility** | Annualized std of only the *negative* log returns in the context (semi-deviation) | Market stress |
| **Volume ratio** | `mean(volume, last 20 days) / mean(volume, full context)`, using raw `volume` (never the synthetic `amount = close*volume` column) | Liquidity/attention (not one of the six named dimensions, added as a bonus) |

**[IMPLEMENTED — rationale for ADF/KPSS on returns, not price level]** The original version of
this project ran neither test, reasoning that ADF on a raw equity price *level* fails to reject
the unit-root null almost everywhere (price levels are generically non-stationary/I(1)
regardless of "regime"), so it would have near-zero discriminating power. Rather than run it on
the price level anyway, both tests are now run on **log returns**, which are generically much
closer to stationary and do show meaningful variation window-to-window — resolving the original
concern instead of ignoring it. Hurst remains reported separately as a persistence measure, never
conflated with the stationarity tests (both are computed by different functions, listed in
different rows above).

### 10.2 Every dimension named in the revised proposal is now implemented

The gap documented in the original version of this audit — rolling volatility, ADF/KPSS,
drawdown/downside volatility, skewness/kurtosis, jump indicators, and a volume indicator all
missing — has been closed; see the table in §10.1. Nothing from the proposal's Phase 3 list
remains unimplemented.

### 10.3 Regime bucketing: single indicator, not a combination

**[IMPLEMENTED]** Regime classification (`stable`/`medium`/`unstable`) is based on
**realized volatility alone**. Hurst and Efficiency Ratio are computed and saved
(`data/regime_features_all.csv`) but are **not** used in the bucketing decision — they are
retained as continuous covariates for separate correlation-style analysis, per
`compute_regime_features.py`'s own docstring. This is an explicit, documented design choice
(driven by sample-size constraints — see below), not an oversight.

### 10.4 Threshold determination: now per-market and causal (previously global and non-causal)

**[IMPLEMENTED — superseded design]** The original version pooled the 1/3 and 2/3 quantiles of
realized volatility across every TEST-period window from every dataset. That has been replaced.
**Current design (`compute_regime_features.py`, `historical_vol_thresholds()`):** for each
market independently, realized volatility is computed over many overlapping 400-day windows
(stride 5) taken **entirely from that market's own pre-`TEST_START` history** (train+val, e.g.
2015–2025-06-30) — 129 to 687 reference windows per market in the current run, far more than the
24–42 *test* windows that pooling across markets was originally meant to compensate for. Tertile
thresholds are computed from this historical distribution and applied to classify that same
market's test-period windows.

**[IMPLEMENTED — major interpretive consequence, must be stated explicitly in any write-up]**
Because thresholds are now market-relative rather than a shared cross-market scale, "stable" for
one market and "stable" for another are **not** the same absolute volatility level. In the
current run, **BTC classifies as 100% "stable"** and **GLD as 100% "unstable"** — the opposite
of what an absolute-volatility intuition would suggest, since BTC's pre-test history is itself
persistently high-volatility (its own thresholds are correspondingly high, e.g. stable ≤ 0.49),
so BTC's calmer-than-usual test period reads as "stable" *for BTC*, while GLD's test period was
more turbulent than GLD's own historical norm. This is the intended, causally-correct behavior
of a per-market relative design, not a bug — but "stable"/"unstable" must now be read as
"relative to this market's own history," not as a claim comparable in absolute terms across
markets. Current full regime composition by market (`regime_features_all_causal.csv`, 294
windows total): BTC 100% stable; GLD, Nikkei225, TOPIX 100% unstable; SP500 100% medium; CSI300
96% unstable; SZSE 82% unstable; SSE 82% medium; KO 83% medium; **NVDA is the only market with
real three-way internal mixing** (7 medium / 11 stable / 11 unstable of 29 windows).

**[IMPLEMENTED — a related caveat this change introduces]** Because a regime bucket can now be
dominated by one or two markets with very different price levels (e.g. "stable" is 42/59 BTC
windows), the **pooled, raw-price-unit MAE reported per regime bucket in `phase2_regime_summary.csv`
is scale-dominated by whichever market fills most of that bucket** — it is not a clean
apples-to-apples comparison the way the per-market tables in §1 are. Reading relative rankings
*within* a regime row (which model has the lowest MAE for that bucket) is still valid; comparing
absolute MAE magnitudes *across* regime rows is not, without normalizing by price level first
(not currently implemented — the per-market `cross_market_summary.py` already does this
normalization for the non-regime comparison, and the same `relative_mae_pct` approach could be
adapted for the regime tables if that comparison is needed).

### 10.5 Look-ahead bias / leakage — explicit check

Three separate questions, each answered separately, since they are easy to conflate:

**(a) Do the regime *features themselves* use information from after the forecast origin?**
**[IMPLEMENTED — No.]** `compute_for_dataset()` in `compute_regime_features.py` computes every
feature strictly from `df.loc[context_start_idx:context_end_idx, "close"]` — the context slice,
which by construction (§4) ends at `target_start_idx - 1`. The target/forecast region is never
touched by the feature computation. This is causally clean.

**(b) Do the regime bucketing *thresholds* use information from after the forecast origin of a
given window?** **[IMPLEMENTED — Fixed; previously Yes, now No.]** This was a real issue in the
original design: thresholds were computed by pooling realized-volatility values across every
window in the run, including windows chronologically *later* than the one being classified. It
has been fixed (§10.4): thresholds are now calibrated entirely from each market's own
pre-`TEST_START` historical data — by construction, every value used to set a market's
thresholds is dated before every single one of that market's test windows, so no window's
regime label depends on any other window, ever, in either direction. This means the current
regime labels **are** now suitable as a live input to the Phase 6 fusion model without further
change, since the proposal's own constraint (fusion weights may only use information available
before the forecast) is already satisfied by construction — no separate expanding/rolling
redesign is needed for that purpose, unlike what the original version of this document
recommended.

**(c) Pretraining-corpus overlap (the third kind of leakage, distinct from (a)/(b)).** Covered
in §3 — mitigated via `TEST_START`, not eliminated, and not something regime classification
changes one way or the other.

---

## 11. Summary — Items Requiring a Methodological Decision

Consolidated for quick reference. Struck-through items were resolved in the same-day revision
noted at the top of this document; still-open items remain genuinely undefined.

1. ~~Explicit random seed for Kronos's and Lag-Llama's stochastic sampling~~ — **resolved**:
   `--seed` flag added to both eval scripts (§7). Not yet used to reproduce the existing results.
2. ~~MASE metric~~ — **resolved**, implemented per Hyndman & Koehler (§9).
3. ~~The additional regime dimensions from the proposal~~ — **resolved**: all six dimensions
   (volatility, persistence, stationarity, trend strength, distributional, stress) now
   implemented, plus a bonus liquidity indicator (§10.1–10.2).
4. ~~Whether the global, non-causal regime-threshold computation needs to be redone causally~~ —
   **resolved**: thresholds are now per-market and causal by construction (§10.4–10.5b). This
   also changed what "stable"/"unstable" *mean* (relative to each market's own history, not a
   shared absolute scale) — a real interpretive shift, not just a bug fix, and it introduced a
   new, separate caveat about price-scale dominance in the pooled regime MAE tables (§10.4).
5. **Still open — Return/log-return prediction target** (currently: raw close price level
   only — §2). Not addressed in this revision; still derivable post-hoc from the existing
   `pred_close`/`actual_close` series without rerunning any model.
6. **Still open — Fine-tuning of Kronos / Lag-Llama / TimesFM** (currently: zero-shot
   only — §5, §6). Explicitly out of scope for this environment by the user's own decision —
   planned for a separate, more powerful server.
7. **Still open — Multi-run variance/stability reporting** across independent seeds or
   iTransformer's own native `--itr N>1` repetition (§8). Not attempted in this revision; would
   multiply compute cost across all 10 datasets and wasn't prioritized over closing the other
   six gaps first.

Each open item still has its own recommendation in the relevant section above, kept separate
from the factual record of what exists today.
