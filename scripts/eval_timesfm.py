"""
Phase 1 rolling-window evaluation for TimesFM 2.5 on SP500 data.

Runs TimesFM over every window in data/eval_windows.csv (shared across all
four models), using its native quantile head for the 80% interval, and
writes a long-format results CSV via scripts/common/results_io.py.

The model is compiled once (context_len/pred_len are fixed across all windows
by construction -- see scripts/common/windowing.py) and reused for every window.

Usage (from the timesfm conda env):
    python eval_timesfm.py --dataset SP500
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

TIMESFM_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "timesfm", "src")
sys.path.append(os.path.abspath(TIMESFM_ROOT))
import timesfm  # noqa: E402

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from results_io import save_results, compute_metrics  # noqa: E402

# quantile_forecast[..., 0] = mean, [..., 1..9] = quantiles 0.1..0.9 (see
# TimesFM_2p5_200M_Definition.quantiles / decode_index in timesfm_2p5_base.py).
Q10_IDX = 1
Q90_IDX = 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    args = parser.parse_args()

    print(f"1. Loading {args.dataset} data + shared windows...")
    df = pd.read_csv(data_path(args.dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(args.dataset)
    context_len = int(windows.iloc[0].context_end_idx - windows.iloc[0].context_start_idx + 1)
    pred_len = int(windows.iloc[0].target_end_idx - windows.iloc[0].target_start_idx + 1)
    print(f"   {len(windows)} windows to evaluate, context_len={context_len}, pred_len={pred_len}")

    print("2. Loading + compiling TimesFM 2.5 (200M, torch)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=pred_len,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            per_core_batch_size=1,
        )
    )

    all_rows = []
    t0 = time.time()
    for i, w in windows.iterrows():
        context_slice = df.loc[w.context_start_idx:w.context_end_idx]
        target_slice = df.loc[w.target_start_idx:w.target_end_idx]
        context_values = context_slice["close"].to_numpy(dtype=np.float32)

        point_forecast, quantile_forecast = model.forecast(horizon=pred_len, inputs=[context_values])
        pred_mean = point_forecast[0]
        q10 = quantile_forecast[0][:, Q10_IDX]
        q90 = quantile_forecast[0][:, Q90_IDX]

        for step, (_, row) in enumerate(target_slice.iterrows(), start=1):
            all_rows.append({
                "model": "timesfm",
                "window_id": w.window_id,
                "target_date": row["timestamps"].date().isoformat(),
                "step_ahead": step,
                "actual_close": row["close"],
                "pred_close": pred_mean[step - 1],
                "pred_q10": q10[step - 1],
                "pred_q90": q90[step - 1],
            })

        elapsed = time.time() - t0
        print(f"   window {int(w.window_id) + 1}/{len(windows)} done ({elapsed:.1f}s elapsed)")

    out_path = save_results(all_rows, "timesfm", args.dataset)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    metrics = compute_metrics(pd.DataFrame(all_rows))
    print("\nAggregate metrics (all windows):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
