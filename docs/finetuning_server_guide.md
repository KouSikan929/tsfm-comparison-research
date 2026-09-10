# Fine-Tuning: Server Execution Guide (A40)

**Status as of 2026-09-10:** all three fine-tuning pipelines (Kronos, Lag-Llama, TimesFM) have
been implemented and validated with local 1-epoch/single-asset smoke tests on an RTX 4060
(8GB). They are **not** full experiments — see §9 for exactly what to change before running
real training on the A40. iTransformer is **not** included here: it has no pretrained checkpoint
to fine-tune (it's trained from scratch already, per the existing Phase 1 pipeline) and needs no
new code.

Every command below is copy-paste-ready for a server console, assuming the repo has been
`git pull`ed onto the server (see `SERVER_SETUP.md` for the GitHub sync workflow) at, say,
`~/tsfmds`. Replace that path if yours differs.

---

## 1. Environments

Same 3 conda environments as the zero-shot pipeline (`kronos`, `lag-llama`, `timesfm`), each
Python 3.10, plus one new dependency set for TimesFM fine-tuning specifically. If these envs
don't already exist on the server, see `SERVER_SETUP.md` section 4 for the base creation
commands (torch install, etc.) — this guide only covers what's *additionally* needed for
fine-tuning.

```bash
# One-time, if not already installed on the server for zero-shot eval:
conda activate kronos
pip install pyyaml   # only new dependency Kronos fine-tuning needs beyond the zero-shot env

conda activate lag-llama
# no new dependencies -- fine-tuning reuses exactly what zero-shot eval already needs

conda activate timesfm
pip install transformers peft accelerate   # NEW -- the zero-shot env doesn't have these
```

**Verify PyTorch sees the A40** (run once per environment you'll actually train in):

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0)"
```

Expected output on the A40: `CUDA available: True`, `Device: NVIDIA A40`, `VRAM (GB): ~48`.

---

## 2. Expected paths

All relative to the repo root (`~/tsfmds` below):

| What | Path |
|---|---|
| Raw OHLCV data | `data/raw/<dataset>_daily.csv` (already present if the repo was pulled with data committed) |
| Leakage-safe fine-tuning data | `data/finetune_data/<dataset>_finetune.csv` — **generated automatically** by any `finetune_*.py` script if missing (calls `scripts/prep_finetune_data.py`); rows with `timestamps <= 2025-06-30` only, i.e. the train+val period, never touching the `2025-07-01+` test period |
| Zero-shot foundation model checkpoints | HuggingFace cache (`~/.cache/huggingface`), downloaded automatically on first eval run, same as the existing zero-shot pipeline |
| Lag-Llama pretrained checkpoint (also used to warm-start fine-tuning) | `data/checkpoints/lag-llama/lag-llama.ckpt`, downloaded automatically |
| **Fine-tuned checkpoints (output)** | `data/finetuned_checkpoints/<model>/<dataset>/` — see §4 for the exact sub-path per model |
| Fine-tuning configs (Kronos only) | `data/finetuned_checkpoints/kronos/<dataset>/config.yaml` (auto-generated per run) |
| Evaluation results | `data/results/<dataset>/<model_name>_results.csv` — fine-tuned variants use model names `kronos_finetuned`, `lag-llama_finetuned`, `timesfm_finetuned` (see §6) so they never overwrite the zero-shot `kronos`/`lag-llama`/`timesfm` results |
| Figures | `data/results/<dataset>/phase1_metrics_comparison.png`, `phase1_example_window.png`, plus `data/results/cross_market_summary.png` and `data/results/phase2_regime_*.png` at the top level |

---

## 3. `cd` and activate

```bash
cd ~/tsfmds
git pull   # make sure you have the latest code + data before starting
```

Every command below assumes you're in `~/tsfmds/scripts` unless stated otherwise:

```bash
cd ~/tsfmds/scripts
```

---

## 4. Fine-tuning commands, per model

Every parameter you're likely to want to change (epochs, batch size, LR, device, dataset,
context/prediction length, seed, output dir) is a CLI flag — run any script with `--help` for
the full list. The commands below show the **A40 production** values (see §9 for exactly how
these differ from the local smoke-test defaults).

### 4.1 Kronos

```bash
conda activate kronos
cd ~/tsfmds/scripts

python finetune_kronos.py \
  --dataset SP500 \
  --epochs 20 \
  --batch_size 64 \
  --lr 4e-5 \
  --device cuda --device_id 0 \
  --seed 42 \
  --context_len 400 --pred_len 20 \
  --train_ratio 0.8 --val_ratio 0.2 \
  --num_workers 4
```

Output checkpoint: `data/finetuned_checkpoints/kronos/SP500/basemodel/best_model` (an HF-format
directory, loadable with `Kronos.from_pretrained(<path>)`). A generated `config.yaml` sits
alongside it in `data/finetuned_checkpoints/kronos/SP500/config.yaml` if you want to inspect or
hand-edit it directly instead of going through the CLI wrapper.

**Change dataset / GPU / hyperparameters:**

```bash
python finetune_kronos.py --dataset NVDA --epochs 20 --batch_size 64 --lr 4e-5 --device cuda --device_id 1
```

**CSI300 needs an override** (smallest dataset, ~1043 pre-test rows — see §8):

```bash
python finetune_kronos.py --dataset CSI300 --epochs 20 --batch_size 64 --lr 4e-5 \
  --context_len 256 --val_ratio 0.3
```

### 4.2 Lag-Llama

```bash
conda activate lag-llama
cd ~/tsfmds/scripts

python finetune_lagllama.py \
  --dataset SP500 \
  --epochs 20 \
  --batch_size 32 \
  --num_batches_per_epoch 50 \
  --lr 1e-5 \
  --device cuda \
  --seed 42 \
  --context_len 400 --pred_len 20 \
  --val_ratio 0.2
```

Output: a serialized GluonTS predictor directory at
`data/finetuned_checkpoints/lag-llama/SP500/predictor/` (loadable with
`gluonts.torch.model.predictor.PyTorchPredictor.deserialize(Path(...))`). Lightning's own
per-epoch checkpoints (for the resume workflow in §7) live under
`data/finetuned_checkpoints/lag-llama/SP500/lightning_logs/`.

**Change dataset / hyperparameters:**

```bash
python finetune_lagllama.py --dataset BTC --epochs 20 --batch_size 32 --num_batches_per_epoch 50 --lr 1e-5
```

**CSI300 override:**

```bash
python finetune_lagllama.py --dataset CSI300 --epochs 20 --batch_size 32 --lr 1e-5 \
  --context_len 256 --val_ratio 0.3
```

Lag-Llama doesn't have a `--device_id`/multi-GPU selector exposed in this wrapper (it uses
whichever GPU `torch.cuda.is_available()` picks up as device 0) — set `CUDA_VISIBLE_DEVICES`
before the command if you need to target a specific GPU on a multi-GPU box:

```bash
CUDA_VISIBLE_DEVICES=1 python finetune_lagllama.py --dataset SP500 --epochs 20 ...
```

### 4.3 TimesFM (LoRA)

```bash
conda activate timesfm
cd ~/tsfmds/scripts

python finetune_timesfm.py \
  --dataset SP500 \
  --epochs 20 \
  --batch_size 32 \
  --num_samples 3000 \
  --lr 1e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --device cuda \
  --seed 42 \
  --context_len 400 --pred_len 20 \
  --val_ratio 0.2
```

Output: a PEFT LoRA adapter directory at `data/finetuned_checkpoints/timesfm/SP500/` (loadable
with `peft.PeftModel.from_pretrained(base_model, <path>)` — see §6). Note this fine-tunes
`google/timesfm-2.5-200m-transformers`, a **different checkpoint repo** than the zero-shot
pipeline's `google/timesfm-2.5-200m-pytorch` — both are Google's own official releases of the
same underlying model, repackaged for different APIs (the zero-shot repo is inference-only; the
`-transformers` one is trainable via HuggingFace + PEFT). This is intentional, not a mismatch —
see `docs/methodology_experimental_setup.md` §6 for the full explanation.

**Change dataset / GPU / hyperparameters:**

```bash
CUDA_VISIBLE_DEVICES=1 python finetune_timesfm.py --dataset GLD --epochs 20 --batch_size 32 --num_samples 3000 --lr 1e-4
```

**CSI300 override:**

```bash
python finetune_timesfm.py --dataset CSI300 --epochs 20 --batch_size 32 --num_samples 3000 --lr 1e-4 \
  --context_len 256 --val_ratio 0.3
```

(TimesFM specifically requires `context_len` to be a multiple of 32 internally — the script
rounds up automatically and logs when it does, so any value you pass is safe, but picking an
already-aligned value like 256 or 416 avoids the log-noise and a slightly-larger-than-requested
effective context.)

### 4.4 Run all 10 assets for one model (loop)

```bash
conda activate kronos   # or lag-llama / timesfm
cd ~/tsfmds/scripts
for ds in SP500 SSE SZSE Nikkei225 TOPIX KO BTC NVDA GLD; do
  echo "=== $ds ==="
  python finetune_kronos.py --dataset $ds --epochs 20 --batch_size 64 --lr 4e-5
done
# CSI300 separately, with its override (see above):
python finetune_kronos.py --dataset CSI300 --epochs 20 --batch_size 64 --lr 4e-5 --context_len 256 --val_ratio 0.3
```

Swap `finetune_kronos.py`/its flags for `finetune_lagllama.py`/`finetune_timesfm.py` and their
respective flags to do the same for the other two models. Each dataset's run is fully
independent (own output directory), so these loops are safe to interrupt and resume dataset-by-
dataset (rerunning a dataset just overwrites its own output dir, doesn't touch the others).

---

## 5. Recommended A40 production parameters — rationale

| Parameter | Local smoke test | A40 production | Why |
|---|---|---|---|
| Kronos `--epochs` | 1 | 15-30 | Enough to see real convergence; the repo's own example config used 20 |
| Kronos `--batch_size` | 8 | 64-160 | A40's 48GB VRAM comfortably fits Kronos-small (24.7M params) at much larger batches than the 4060's 8GB; larger batches = faster wall-clock per epoch |
| Kronos `--lr` | 1e-6 | 4e-5 | The vendored repo's own example config's `predictor_learning_rate`; smoke test used a much lower LR mostly to avoid needing to think about it for 1 epoch, not for any principled reason |
| Lag-Llama `--epochs` | 1 | 15-30 | Same reasoning as Kronos |
| Lag-Llama `--num_batches_per_epoch` | 10 | 50-100 | GluonTS's own estimator default is 50; smoke test used 10 purely for speed |
| Lag-Llama `--batch_size` | 8 | 32-64 | A40 VRAM headroom |
| TimesFM `--num_samples` | 40 | 2000-5000 | This stands in for "epoch size" for TimesFM's random-window sampler (§ design note in `finetune_timesfm.py`) -- 40 was chosen purely to make the smoke test finish in seconds |
| TimesFM `--batch_size` | 4 | 16-32 | A40 VRAM headroom (LoRA fine-tuning is memory-light even for the 200M base model) |
| All three `--val_ratio`/`--train_ratio` | 0.2/0.8 | same | Not a smoke-test-only choice -- this split ratio is a structural requirement (validation split must contain more rows than context_len+pred_len+1), not a speed shortcut; keep as-is on the server too |
| All three `--seed` | 42 | 42 (or your choice) | Fine to keep constant -- but see §9's note on reproducibility |

None of these are hard requirements — they're a reasonable starting point. If you have wall-clock
budget, increasing epochs further (with early-stopping-style manual monitoring of the printed
val loss) is more likely to matter than any other single change.

---

## 6. Evaluating fine-tuned checkpoints

Reuses the exact same shared window definitions (`data/eval_windows_<dataset>.csv`) and results
schema (`scripts/common/results_io.py`) as the zero-shot pipeline — the fine-tuned results are
directly comparable to the zero-shot ones already in `data/results/`.

### 6.1 Kronos

```bash
conda activate kronos
cd ~/tsfmds/scripts
python eval_kronos.py --dataset SP500 \
  --checkpoint "../data/finetuned_checkpoints/kronos/SP500/basemodel/best_model" \
  --output_model_name kronos_finetuned
```

### 6.2 Lag-Llama

```bash
conda activate lag-llama
cd ~/tsfmds/scripts
python eval_lagllama.py --dataset SP500 \
  --checkpoint "../data/finetuned_checkpoints/lag-llama/SP500/predictor" \
  --output_model_name lag-llama_finetuned
```

### 6.3 TimesFM

```bash
conda activate timesfm
cd ~/tsfmds/scripts
python eval_timesfm_finetuned.py --dataset SP500 \
  --checkpoint "../data/finetuned_checkpoints/timesfm/SP500" \
  --output_model_name timesfm_finetuned
```

(Note: this is a **separate script** from `eval_timesfm.py`, not a `--checkpoint` flag on it —
the fine-tuned checkpoint uses a different loading stack, see §4.3. `eval_timesfm.py` remains
the zero-shot-only script, unchanged.)

### 6.4 Run evaluation across all 10 assets (loop)

```bash
conda activate kronos
cd ~/tsfmds/scripts
for ds in SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD; do
  python eval_kronos.py --dataset $ds \
    --checkpoint "../data/finetuned_checkpoints/kronos/$ds/basemodel/best_model" \
    --output_model_name kronos_finetuned
done
```

Same pattern for `eval_lagllama.py` (env `lag-llama`, checkpoint path
`../data/finetuned_checkpoints/lag-llama/$ds/predictor`) and `eval_timesfm_finetuned.py` (env
`timesfm`, checkpoint path `../data/finetuned_checkpoints/timesfm/$ds`).

**Known quirk to expect, not a bug:** TimesFM's fine-tuned evaluation (via the
`transformers`-based HF checkpoint) can produce a noticeably flatter point forecast than the
zero-shot `src/timesfm` path under default settings — verified during the local smoke test by
checking the *base* (non-fine-tuned) `google/timesfm-2.5-200m-transformers` checkpoint through
the same code path, which showed the same flatness. This mechanically depresses directional
accuracy specifically (a flat forecast has zero step-to-step direction, which never matches a
nonzero actual direction) without necessarily meaning MAE/RMSE are bad. Don't be alarmed by a
low `dir_acc` for `timesfm_finetuned` specifically; do sanity-check MAE/RMSE against the
zero-shot numbers for the same dataset.

---

## 7. Resuming from an interrupted run

**Kronos:** the training script saves a new `best_model` checkpoint (via HF's `save_pretrained`)
every time validation loss improves, but does **not** natively support resuming a partially-
completed epoch or continuing from a specific step. If a run is interrupted, the simplest
correct option is to point `--pretrained_predictor` at the last saved `best_model` directory and
restart (effectively continuing fine-tuning from wherever it left off, at the cost of restarting
the current epoch):

```bash
python finetune_kronos.py --dataset SP500 --epochs 10 \
  --pretrained_predictor "../data/finetuned_checkpoints/kronos/SP500/basemodel/best_model"
```

**Lag-Llama:** genuinely resumable — PyTorch Lightning checkpoints every epoch under
`data/finetuned_checkpoints/lag-llama/<dataset>/lightning_logs/lightning_logs/version_N/checkpoints/`.
`finetune_lagllama.py` doesn't currently expose a `--resume_from` flag, but you can resume
manually by passing that `.ckpt` path as `ckpt_path` to `estimator.train()` in the script (GluonTS
supports this natively — see the `.train()` signature in the script's own comments) or, more
simply, by treating the saved Lightning checkpoint the same way as Kronos above: point
`LagLlamaEstimator`'s own `ckpt_path` (used for warm-starting) at that `.ckpt` file instead of the
original pretrained release and rerun.

**TimesFM (LoRA):** the script saves the adapter (`model.save_pretrained(output_dir)`) every
time validation loss improves, same pattern as Kronos. To continue from an interrupted run, load
the existing adapter instead of applying a fresh `LoraConfig` — this isn't wired up as a flag
yet; the straightforward manual fix is to replace the `get_peft_model(model, lora_config)` call
in `finetune_timesfm.py` with `PeftModel.from_pretrained(model, <existing adapter dir>, is_trainable=True)`
when you want to continue from a saved adapter, then rerun.

None of the three have a fully automatic "pick up exactly where it left off, mid-epoch" resume —
all three resume at the granularity of "restart from the last saved checkpoint/adapter," which
is the standard level of resumability for this kind of short fine-tuning run and should be
sufficient in practice.

---

## 8. The CSI300 constraint (read before looping over all 10 datasets)

CSI300 is the smallest dataset by a wide margin (~1043 rows in the leakage-safe fine-tuning
period, versus ~2100-4200 for every other dataset — see `docs/methodology_experimental_setup.md`
§1). With the project's default `context_len=400, pred_len=20`, the validation split needs at
least 421 rows; a 0.2 `val_ratio` on CSI300's 1043 rows only gives ~209 — **every fine-tuning
script will warn and may crash or produce an empty validation set on CSI300 with default flags.**

Use the override shown in each model's section above (`--context_len 256 --val_ratio 0.3`) for
CSI300 specifically. This was hit and fixed for real during the local smoke test (see the
`--train_ratio`/`--val_ratio` docstrings in `finetune_kronos.py` and `finetune_lagllama.py`,
and the analogous warning in `finetune_timesfm.py`) — it's a real, reproducible constraint of
this dataset's size, not a hypothetical edge case.

---

## 9. Local smoke test vs. A40 production — full parameter diff

| Parameter | Local smoke test (this environment) | A40 production (recommended) |
|---|---|---|
| Dataset | SP500 only | All 10 (loop, §4.4) |
| Epochs | 1 | 15-30 |
| Kronos batch_size | 8 | 64-160 |
| Lag-Llama batch_size | 8 | 32-64 |
| Lag-Llama num_batches_per_epoch | 10 | 50-100 |
| TimesFM batch_size | 4 | 16-32 |
| TimesFM num_samples | 40 | 2000-5000 |
| Kronos lr | 1e-6 | 4e-5 |
| Lag-Llama lr | 1e-5 | 1e-5 (unchanged -- already a reasonable value) |
| TimesFM lr | 1e-4 | 1e-4 (unchanged) |
| GPU | RTX 4060 Laptop, 8GB | A40, 48GB |
| Wall-clock per dataset (approx) | Kronos ~1 min, Lag-Llama ~5s, TimesFM ~10s | Kronos: tens of minutes; Lag-Llama: several minutes; TimesFM: a few minutes (all rough estimates -- actual A40 throughput not yet measured) |
| Purpose | Verify the pipeline runs end-to-end | Produce real, comparison-worthy fine-tuned models |
| Result quality | Explicitly not meaningful (1 epoch, tiny sample) | Meaningful, reported in the thesis |

Everything else (CLI structure, output paths, eval scripts, results schema) is **identical**
between local and server runs — only the values above need to change, and all of them are CLI
flags, not code edits.

---

## 10. After evaluation: comparing zero-shot, fine-tuned, and iTransformer

All three already-existing comparison scripts accept a `--models` flag (added specifically to
support this) — pass whichever subset of the 7 possible model names you have results for:
`kronos`, `lag-llama`, `timesfm`, `itransformer`, `kronos_finetuned`, `lag-llama_finetuned`,
`timesfm_finetuned`.

### 10.1 Per-market comparison (statistical metrics + Diebold-Mariano + figures)

```bash
conda activate base   # or any env with pandas/numpy/scipy/matplotlib/statsmodels -- these 3 scripts don't need any model-specific packages
cd ~/tsfmds/scripts

python aggregate_results.py --dataset SP500 \
  --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned

python plot_phase1_summary.py --dataset SP500 \
  --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned
```

Produces (in `data/results/SP500/`): `phase1_summary.csv` (MAE/RMSE/MASE/DirAcc/coverage per
model), `phase1_dm_test.csv` (every pairwise Diebold-Mariano comparison, including
zero-shot-vs-its-own-fine-tuned-version), `phase1_metrics_comparison.png`,
`phase1_example_window.png`.

**Loop over all 10 datasets:**

```bash
for ds in SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD; do
  python aggregate_results.py --dataset $ds --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned
  python plot_phase1_summary.py --dataset $ds --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned
done
```

**Specifically isolating the zero-shot-vs-fine-tuned improvement per model** (the proposal's
Phase 2 question): read the `mae`/`rmse`/`mase` rows for `kronos` vs `kronos_finetuned` (etc.)
directly out of `phase1_summary.csv`, and the corresponding row in `phase1_dm_test.csv`
(`model_a=kronos, model_b=kronos_finetuned` or vice versa) for whether the improvement (or
regression) is statistically significant.

### 10.2 Cross-market summary

```bash
python plot_cross_market_summary.py --datasets SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD \
  --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned
```

Produces `data/results/cross_market_summary.{csv,png}` — relative-MAE-% comparison across every
market and every model variant.

### 10.3 Regime-based evaluation with fine-tuned models

Regime features don't need to be recomputed (they're a property of the input data, not of which
model is being scored) — reuse the existing causal per-market regime file:

```bash
python analyze_phase2_regimes.py --datasets SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD \
  --models kronos lag-llama timesfm itransformer kronos_finetuned lag-llama_finetuned timesfm_finetuned
```

Produces `data/results/phase2_regime_summary.csv`, `phase2_regime_dm_test.csv`,
`phase2_regime_comparison.png`, `phase2_regime_market_confound.png` — same causal-threshold
methodology already used for the zero-shot regime analysis (see
`docs/methodology_experimental_setup.md` §10), now broken out per regime for every model
including the fine-tuned variants. If `--datasets` here doesn't match every dataset that was
included when `compute_regime_features.py` last ran, rerun that first:

```bash
python compute_regime_features.py --datasets SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD
```

---

## 11. Verifying all expected experiments have completed

A quick completeness check — run from `~/tsfmds`:

```bash
cd ~/tsfmds
for ds in SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD; do
  for model in kronos_finetuned lag-llama_finetuned timesfm_finetuned; do
    f="data/results/$ds/${model}_results.csv"
    if [ -f "$f" ]; then
      n=$(wc -l < "$f")
      echo "OK   $ds / $model  ($((n-1)) rows)"
    else
      echo "MISSING   $ds / $model"
    fi
  done
done
```

Expect **30 "OK" lines** (10 datasets × 3 fine-tuned models) once everything is done, each with
`n_windows * pred_len` rows (e.g. 540 for a 27-window/20-step dataset — cross-check the exact
window count for each dataset against `data/eval_windows_<dataset>.csv`'s row count, since it
varies per market — see `docs/methodology_experimental_setup.md` §4 for the current per-dataset
window counts).

Also check the checkpoint directories exist and are non-empty (a `MISSING` result above is often
caused by the checkpoint never having been produced, not an eval-script problem):

```bash
for ds in SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD; do
  ls -d data/finetuned_checkpoints/kronos/$ds/basemodel/best_model 2>/dev/null || echo "MISSING kronos checkpoint: $ds"
  ls -d data/finetuned_checkpoints/lag-llama/$ds/predictor 2>/dev/null || echo "MISSING lag-llama checkpoint: $ds"
  ls -d data/finetuned_checkpoints/timesfm/$ds 2>/dev/null || echo "MISSING timesfm checkpoint: $ds"
done
```

Finally, confirm the comparison outputs exist for every dataset:

```bash
for ds in SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD; do
  ls data/results/$ds/phase1_summary.csv >/dev/null 2>&1 && echo "OK summary: $ds" || echo "MISSING summary: $ds"
done
ls data/results/cross_market_summary.png >/dev/null 2>&1 && echo "OK cross-market summary" || echo "MISSING cross-market summary"
ls data/results/phase2_regime_summary.csv >/dev/null 2>&1 && echo "OK regime summary" || echo "MISSING regime summary"
```

---

## 12. What was fixed while validating this locally (context for the server run)

Two real bugs in vendored code were found and fixed during the local smoke tests — both fixes
are already committed, nothing further to do, but worth knowing about if something looks
unfamiliar in the vendored repos' diffs:

1. **`models/Kronos/finetune_csv/finetune_base_model.py`** had a Python scoping bug: local
   `import json, os` statements inside `else:` branches that never execute (since this project
   always uses `pre_trained_tokenizer=True`/`pre_trained_predictor=True`) still made `os` local
   to the entire `main()` function per Python's scoping rules, breaking an earlier
   `os.makedirs()` call with `UnboundLocalError`. Fixed by removing the redundant local imports
   (both names are already imported at module level).
2. **`finetune_lagllama.py`** (this project's own new script): `predictor.serialize(path)`
   requires the output directory to already exist — GluonTS doesn't create it. Fixed with a
   `path.mkdir(parents=True, exist_ok=True)` before the call.

One more thing that is **not a bug**, just a real constraint discovered during validation and
already reflected in the recommended commands above: TimesFM's `transformers`-based forward pass
requires `context_len` to be a multiple of 32 (the model's patch size) and does not auto-round
this the way the zero-shot `src/timesfm` path does — `finetune_timesfm.py` and
`eval_timesfm_finetuned.py` both round up automatically and log when they do, so this doesn't
need manual handling, but don't be surprised to see `context_len=400 is not a multiple of the
patch size (32); rounding up to 416` in the logs.
