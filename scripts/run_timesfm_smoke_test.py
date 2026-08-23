"""
End-to-end smoke test for TimesFM 2.5 on our SP500 data.

Loads the pretrained TimesFM_2p5_200M_torch checkpoint from Hugging Face,
runs one forecast on the same SP500 window used in the Kronos and Lag-Llama
smoke tests, and saves a plot with an 80% quantile interval plus
MAE/RMSE/directional-accuracy/coverage metrics.

Usage:
    python run_timesfm_smoke_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIMESFM_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "timesfm", "src")
sys.path.append(os.path.abspath(TIMESFM_ROOT))
import timesfm  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "SP500_daily.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke_test")

CONTEXT_LENGTH = 400
PRED_LEN = 120

# quantile_forecast[..., 0] = mean, [..., 1..9] = quantiles 0.1..0.9 (see
# TimesFM_2p5_200M_Definition.quantiles / decode_index in timesfm_2p5_base.py).
Q10_IDX = 1
Q90_IDX = 9


def main():
    print("1. Loading TimesFM 2.5 (200M, torch) from Hugging Face (will download on first run)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LENGTH,
            max_horizon=PRED_LEN,
            normalize_inputs=True,  # SP500 close is in the thousands; normalize for stability
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            per_core_batch_size=1,
        )
    )

    print("2. Loading SP500 data slice (same window as the other smoke tests)...")
    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    window = df.iloc[-(CONTEXT_LENGTH + PRED_LEN):].reset_index(drop=True)
    context_df = window.iloc[:CONTEXT_LENGTH]
    target_df = window.iloc[CONTEXT_LENGTH:].reset_index(drop=True)

    print(f"   Context: {context_df['timestamps'].iloc[0].date()} -> {context_df['timestamps'].iloc[-1].date()} ({CONTEXT_LENGTH} rows)")
    print(f"   Forecast target: {target_df['timestamps'].iloc[0].date()} -> {target_df['timestamps'].iloc[-1].date()} ({PRED_LEN} rows)")

    context_values = context_df["close"].to_numpy(dtype=np.float32)

    print("3. Running prediction...")
    point_forecast, quantile_forecast = model.forecast(horizon=PRED_LEN, inputs=[context_values])
    pred_mean = point_forecast[0]  # median (decode_index), shape (PRED_LEN,)
    q10 = quantile_forecast[0][:, Q10_IDX]
    q90 = quantile_forecast[0][:, Q90_IDX]
    actual = target_df["close"].values

    mae = np.mean(np.abs(pred_mean - actual))
    rmse = np.sqrt(np.mean((pred_mean - actual) ** 2))
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(pred_mean))
    dir_acc = np.mean(actual_dir == pred_dir)
    coverage = np.mean((actual >= q10) & (actual <= q90))

    print(f"\nMAE (close, median forecast):  {mae:.4f}")
    print(f"RMSE (close, median forecast): {rmse:.4f}")
    print(f"Directional accuracy: {dir_acc:.2%}")
    print(f"80% interval coverage (well-calibrated if ~80%): {coverage:.2%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    dates = target_df["timestamps"].values
    ax.plot(dates, actual, label="Ground Truth", color="blue", linewidth=1.5)
    ax.plot(dates, pred_mean, label="Prediction (median)", color="red", linewidth=1.5)
    ax.fill_between(dates, q10, q90, color="red", alpha=0.2, label="80% interval")
    ax.set_ylabel("Close Price")
    ax.set_title(f"TimesFM 2.5 on S&P 500 | MAE={mae:.2f} RMSE={rmse:.2f} DirAcc={dir_acc:.1%} Cov80={coverage:.1%}")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "timesfm_sp500_smoke_test.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
