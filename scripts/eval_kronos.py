"""
Phase 1 rolling-window evaluation for Kronos on SP500 data.

Runs Kronos-small over every window in data/eval_windows.csv (shared across
all four models -- see scripts/common/windowing.py), and writes a long-format
results CSV via scripts/common/results_io.py.

Uncertainty: Kronos's own predict() averages internally over sample_count and
throws away the per-sample draws (see model/kronos.py's
auto_regressive_inference, which does `preds = np.mean(preds, axis=1)` before
returning). To get an actual predictive distribution -- comparable to
Lag-Llama's/TimesFM's quantile outputs -- we instead call predict() N_SAMPLES
times independently at sample_count=1 (each call is one stochastic draw, since
T=1.0 top_p=0.9 sampling is non-deterministic) and compute empirical
mean/q10/q90 across draws ourselves.

Usage (from the kronos conda env):
    python eval_kronos.py --dataset SP500
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

KRONOS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "Kronos")
sys.path.append(os.path.abspath(KRONOS_ROOT))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from results_io import save_results, compute_metrics  # noqa: E402

N_SAMPLES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional seed for the stochastic sampling (T=1.0/top_p=0.9 draws). "
                              "Unset by default -- prior runs in this project were unseeded, so results "
                              "won't be bit-identical to earlier runs even with a seed now.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print("1. Loading Kronos-small model + tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512, device="cuda")

    print(f"2. Loading {args.dataset} data + shared windows...")
    df = pd.read_csv(data_path(args.dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(args.dataset)
    print(f"   {len(windows)} windows to evaluate, {N_SAMPLES} stochastic samples each")

    all_rows = []
    t0 = time.time()
    for i, w in windows.iterrows():
        x_df = df.loc[w.context_start_idx:w.context_end_idx, ["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df.loc[w.context_start_idx:w.context_end_idx, "timestamps"]
        target_slice = df.loc[w.target_start_idx:w.target_end_idx]
        y_timestamp = target_slice["timestamps"]
        pred_len = len(y_timestamp)

        draws = np.zeros((N_SAMPLES, pred_len))
        for s in range(N_SAMPLES):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
            draws[s] = pred_df["close"].values

        pred_mean = draws.mean(axis=0)
        pred_q10 = np.quantile(draws, 0.1, axis=0)
        pred_q90 = np.quantile(draws, 0.9, axis=0)

        for step, (_, row) in enumerate(target_slice.iterrows(), start=1):
            all_rows.append({
                "model": "kronos",
                "window_id": w.window_id,
                "target_date": row["timestamps"].date().isoformat(),
                "step_ahead": step,
                "actual_close": row["close"],
                "pred_close": pred_mean[step - 1],
                "pred_q10": pred_q10[step - 1],
                "pred_q90": pred_q90[step - 1],
            })

        elapsed = time.time() - t0
        print(f"   window {int(w.window_id) + 1}/{len(windows)} done ({elapsed:.1f}s elapsed)")

    out_path = save_results(all_rows, "kronos", args.dataset)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    metrics = compute_metrics(pd.DataFrame(all_rows))
    print("\nAggregate metrics (all windows):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
