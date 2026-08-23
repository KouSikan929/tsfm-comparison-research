"""
End-to-end smoke test for Lag-Llama on our SP500 data.

Loads the pretrained Lag-Llama checkpoint from Hugging Face, runs one
probabilistic forecast (via gluonts) on the same SP500 window used in the
Kronos smoke test, and saves a plot with an 80% prediction interval plus
MAE/RMSE/directional-accuracy/coverage metrics.

Note: trading days are treated as consecutive daily steps (calendar gaps
from weekends/holidays are ignored) -- fine for this smoke test, but worth
revisiting for the real Phase 1/2/3 evaluation if day-of-week time features
matter.

Usage:
    python run_lagllama_smoke_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

# PyTorch 2.6 defaults torch.load to weights_only=True, which rejects this checkpoint
# (official Lag-Llama release, https://huggingface.co/time-series-foundation-models/Lag-Llama)
# because it embeds a gluonts distribution-output class alongside the tensor weights.
# The library's own load_from_checkpoint() calls torch.load internally without the
# override, so we patch the default here rather than chase every call site. Trusted
# source, so this is safe for this specific checkpoint.
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions

LAG_LLAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "lag-llama")
sys.path.append(os.path.abspath(LAG_LLAMA_ROOT))
from lag_llama.gluon.estimator import LagLlamaEstimator  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "SP500_daily.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smoke_test")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints", "lag-llama")

CONTEXT_LENGTH = 400
PRED_LEN = 120
NUM_SAMPLES = 100


def build_predictor(ckpt_path, prediction_length, context_length, device, num_samples):
    # weights_only=False: this checkpoint (official Lag-Llama release on Hugging Face,
    # https://huggingface.co/time-series-foundation-models/Lag-Llama) embeds a gluonts
    # distribution-output class alongside the tensor weights, which PyTorch 2.6's default
    # weights_only=True safe-unpickler rejects. Trusted source, so this is safe here.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("1. Downloading Lag-Llama checkpoint from Hugging Face (first run only)...")
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
        local_dir=CKPT_DIR,
    )

    print("2. Loading SP500 data slice (same window as the Kronos smoke test)...")
    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    window = df.iloc[-(CONTEXT_LENGTH + PRED_LEN):].reset_index(drop=True)
    target_df = window.iloc[CONTEXT_LENGTH:].reset_index(drop=True)

    print(f"   Context: {window['timestamps'].iloc[0].date()} -> {window['timestamps'].iloc[CONTEXT_LENGTH - 1].date()} ({CONTEXT_LENGTH} rows)")
    print(f"   Forecast target: {target_df['timestamps'].iloc[0].date()} -> {target_df['timestamps'].iloc[-1].date()} ({PRED_LEN} rows)")

    # gluonts requires a uniformly-spaced index; real calendar dates have weekend/holiday
    # gaps that break that. We treat trading days as consecutive daily steps (same
    # simplification noted in the module docstring), so use a synthetic uniform index
    # here and keep window["timestamps"] separately for plotting.
    full_series = window["close"].astype("float32").reset_index(drop=True)
    full_series.index = pd.period_range(start="2000-01-01", periods=len(full_series), freq="D")
    ds = PandasDataset(full_series.to_frame(name="close"), target="close")

    print("3. Building predictor from checkpoint...")
    predictor = build_predictor(
        ckpt_path=ckpt_path,
        prediction_length=PRED_LEN,
        context_length=CONTEXT_LENGTH,
        device=device,
        num_samples=NUM_SAMPLES,
    )

    print("4. Running prediction...")
    forecast_it, ts_it = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=NUM_SAMPLES)
    forecasts = list(forecast_it)

    forecast = forecasts[0]
    pred_mean = forecast.mean
    actual = target_df["close"].values

    mae = np.mean(np.abs(pred_mean - actual))
    rmse = np.sqrt(np.mean((pred_mean - actual) ** 2))
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(pred_mean))
    dir_acc = np.mean(actual_dir == pred_dir)

    q10 = np.quantile(forecast.samples, 0.1, axis=0)
    q90 = np.quantile(forecast.samples, 0.9, axis=0)
    coverage = np.mean((actual >= q10) & (actual <= q90))

    print(f"\nMAE (close, mean forecast):  {mae:.4f}")
    print(f"RMSE (close, mean forecast): {rmse:.4f}")
    print(f"Directional accuracy: {dir_acc:.2%}")
    print(f"80% interval coverage (well-calibrated if ~80%): {coverage:.2%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    dates = target_df["timestamps"].values
    ax.plot(dates, actual, label="Ground Truth", color="blue", linewidth=1.5)
    ax.plot(dates, pred_mean, label="Prediction (mean)", color="red", linewidth=1.5)
    ax.fill_between(dates, q10, q90, color="red", alpha=0.2, label="80% interval")
    ax.set_ylabel("Close Price")
    ax.set_title(f"Lag-Llama on S&P 500 | MAE={mae:.2f} RMSE={rmse:.2f} DirAcc={dir_acc:.1%} Cov80={coverage:.1%}")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lagllama_sp500_smoke_test.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
