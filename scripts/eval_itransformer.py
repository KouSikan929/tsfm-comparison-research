"""
Phase 1 rolling-window evaluation for iTransformer on SP500 data.

Unlike the other three models, iTransformer has no pretrained checkpoint --
it's trained from scratch first (see the `python run.py ...` invocation this
script's docstring mirrors, using --data custom_dateSplit so the split
boundaries match TRAIN_END/VAL_END/TEST_START exactly, same as the other
three models -- see models/iTransformer/data_provider/data_loader_date_split.py).

iTransformer's own test() loop (in experiments/exp_long_term_forecasting.py)
already evaluates EVERY stride-1 window across the test period and saves
predictions to results/<setting>/pred.npy + true.npy (shape: n_windows,
pred_len, 1 channel since features=MS). This script:
  1. Loads those normalized-space predictions.
  2. Inverse-transforms the target ('close') channel back to raw price units,
     using the *same* StandardScaler stats Dataset_CustomDateSplit fit
     internally (recomputed here rather than reusing test()'s built-in
     --inverse path, which has a known shape-mismatch bug for MS mode: it
     calls scaler.inverse_transform() on a single-channel array against a
     scaler fit on all 6 channels).
  3. Subselects only the windows whose target_start matches our shared
     data/eval_windows.csv (stride=10 among iTransformer's native stride=1
     windows), so iTransformer is compared on *exactly* the same windows as
     Kronos/Lag-Llama/TimesFM.
  4. Writes the shared long-format results CSV via scripts/common/results_io.py.

Prerequisite (run first, from the itransformer conda env, cwd =
models/iTransformer -- also run scripts/prep_itransformer_data.py --dataset
<name> first to create dataset/<name>/<name>_daily.csv):
    python run.py --is_training 1 --root_path ./dataset/<name>/ \
        --data_path <name>_daily.csv --model_id <name>_dateSplit_400_20 \
        --model iTransformer --data custom_dateSplit --features MS \
        --target close --freq d --seq_len 400 --label_len 48 --pred_len 20 \
        --e_layers 2 --enc_in 6 --dec_in 6 --c_out 1 --des phase1 \
        --d_model 128 --d_ff 128 --train_epochs 10 --patience 3 --itr 1

Usage (from the itransformer conda env, cwd = scripts/):
    python eval_itransformer.py --dataset SP500
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ITRANSFORMER_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "iTransformer")
sys.path.append(os.path.abspath(ITRANSFORMER_ROOT))

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from results_io import save_results, compute_metrics  # noqa: E402

SEQ_LEN = 400
PRED_LEN = 20
STRIDE = 10  # must match scripts/common/windowing.py's STRIDE


def find_setting_dir(dataset):
    results_glob = os.path.join(ITRANSFORMER_ROOT, "results", f"{dataset}_dateSplit_400_20_*", "")
    matches = glob.glob(results_glob)
    if not matches:
        raise FileNotFoundError(
            f"No results/ directory found for {dataset}_dateSplit_400_20_*. "
            "Run the training command in this script's docstring first."
        )
    return sorted(matches)[-1]  # most recent if multiple itr runs exist


def get_close_scaler_stats(dataset):
    """Recompute the same StandardScaler stats Dataset_CustomDateSplit fits
    internally on the train split, so we can inverse-transform the (single-
    channel) predictions ourselves -- avoids a known bug in this codebase's
    built-in --inverse path for MS-mode (single-channel array against a
    scaler fit on all 6 channels)."""
    from data_provider.data_loader_date_split import Dataset_CustomDateSplit

    ds = Dataset_CustomDateSplit(
        root_path=os.path.join(ITRANSFORMER_ROOT, "dataset", dataset),
        data_path=f"{dataset}_daily.csv",
        flag="train",
        size=[SEQ_LEN, 48, PRED_LEN],
        features="MS",
        target="close",
        timeenc=1,
        freq="d",
    )
    # df_data columns are [open, high, low, volume, amount, close] (date dropped,
    # close last) -- see data_loader_date_split.py's column reordering.
    close_mean = ds.scaler.mean_[-1]
    close_scale = ds.scaler.scale_[-1]
    return close_mean, close_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    args = parser.parse_args()
    dataset = args.dataset

    print(f"1. Locating trained iTransformer results for {dataset}...")
    setting_dir = find_setting_dir(dataset)
    print(f"   Using {setting_dir}")
    preds = np.load(os.path.join(setting_dir, "pred.npy"))  # (269, pred_len, 1), normalized
    trues = np.load(os.path.join(setting_dir, "true.npy"))
    print(f"   preds shape: {preds.shape}")

    print("2. Recomputing close-price scaler stats for manual inverse-transform...")
    close_mean, close_scale = get_close_scaler_stats(dataset)
    preds_raw = preds[..., 0] * close_scale + close_mean
    trues_raw = trues[..., 0] * close_scale + close_mean

    print("3. Loading shared eval windows + subselecting matching windows...")
    windows = load_windows(dataset)
    df = pd.read_csv(data_path(dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    all_rows = []
    for _, w in windows.iterrows():
        # iTransformer's native test set walks stride-1 from the same
        # first-test-target row our windows.csv starts from, so window_id*STRIDE
        # is exactly the matching native-index offset (verified: both use the
        # same first_test_idx / seq_len construction).
        native_idx = int(w.window_id) * STRIDE
        target_slice = df.loc[w.target_start_idx:w.target_end_idx]

        # Sanity check: the actual close values in pred.npy's paired true.npy
        # should match our own ground truth exactly (same underlying data).
        mismatch = np.abs(trues_raw[native_idx] - target_slice["close"].values).max()
        if mismatch > 1e-2:
            raise RuntimeError(
                f"Window alignment mismatch at window_id={w.window_id} "
                f"(native_idx={native_idx}): max diff {mismatch:.4f}. "
                "iTransformer's native window index no longer matches windowing.py's construction."
            )

        for step, (_, row) in enumerate(target_slice.iterrows(), start=1):
            all_rows.append({
                "model": "itransformer",
                "window_id": w.window_id,
                "target_date": row["timestamps"].date().isoformat(),
                "step_ahead": step,
                "actual_close": row["close"],
                "pred_close": preds_raw[native_idx, step - 1],
                "pred_q10": np.nan,  # point-forecast model, no native uncertainty estimate
                "pred_q90": np.nan,
            })

    out_path = save_results(all_rows, "itransformer", dataset)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    metrics = compute_metrics(pd.DataFrame(all_rows))
    print("\nAggregate metrics (all windows):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
