"""
End-to-end smoke test for Kronos on our SP500 data.

Loads the pretrained Kronos-small model + tokenizer from Hugging Face,
runs one forecast on a slice of data/raw/SP500_daily.csv where we already
know the ground-truth future (so we can plot predicted vs. actual), and
saves the plot + metrics instead of calling plt.show() (headless run).

Usage:
    python run_kronos_smoke_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KRONOS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "Kronos")
sys.path.append(os.path.abspath(KRONOS_ROOT))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "SP500_daily.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke_test")

LOOKBACK = 400
PRED_LEN = 120


def main():
    print("1. Loading Kronos-small model + tokenizer from Hugging Face (will download on first run)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512, device="cuda" if _cuda_available() else "cpu")

    print("2. Loading SP500 data slice...")
    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    window = df.iloc[-(LOOKBACK + PRED_LEN):].reset_index(drop=True)
    x_df = window.loc[: LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = window.loc[: LOOKBACK - 1, "timestamps"]
    y_timestamp = window.loc[LOOKBACK : LOOKBACK + PRED_LEN - 1, "timestamps"]
    y_actual = window.loc[LOOKBACK : LOOKBACK + PRED_LEN - 1, ["close", "volume"]].reset_index(drop=True)

    print(f"   Context: {x_timestamp.iloc[0].date()} -> {x_timestamp.iloc[-1].date()} ({LOOKBACK} rows)")
    print(f"   Forecast target: {y_timestamp.iloc[0].date()} -> {y_timestamp.iloc[-1].date()} ({PRED_LEN} rows)")

    print("3. Running prediction...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("\nForecasted data head:")
    print(pred_df.head())

    mae = np.mean(np.abs(pred_df["close"].values - y_actual["close"].values))
    rmse = np.sqrt(np.mean((pred_df["close"].values - y_actual["close"].values) ** 2))
    actual_dir = np.sign(np.diff(y_actual["close"].values))
    pred_dir = np.sign(np.diff(pred_df["close"].values))
    dir_acc = np.mean(actual_dir == pred_dir)

    print(f"\nMAE (close):  {mae:.4f}")
    print(f"RMSE (close): {rmse:.4f}")
    print(f"Directional accuracy: {dir_acc:.2%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(y_timestamp.values, y_actual["close"].values, label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(y_timestamp.values, pred_df["close"].values, label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price")
    ax1.set_title(f"Kronos-small on S&P 500 | MAE={mae:.2f} RMSE={rmse:.2f} DirAcc={dir_acc:.1%}")
    ax1.legend(loc="best")
    ax1.grid(True)

    ax2.plot(y_timestamp.values, y_actual["volume"].values, label="Ground Truth", color="blue", linewidth=1.5)
    ax2.plot(y_timestamp.values, pred_df["volume"].values, label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume")
    ax2.legend(loc="best")
    ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "kronos_sp500_smoke_test.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def _cuda_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
