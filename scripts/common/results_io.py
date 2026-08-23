"""
Shared long-format results schema + metrics for the Phase 1 rolling-window
evaluation. Every model's eval script writes one CSV in this exact shape to
data/results/<dataset>/<model>_results.csv, so the aggregation step can
compare all four models -- across any number of datasets -- without any
model-specific logic.

Columns:
    model          -- e.g. "kronos", "lag-llama", "timesfm", "itransformer"
    window_id      -- matches data/eval_windows_<dataset>.csv
    target_date    -- ISO date of this forecast step
    step_ahead     -- 1-indexed step within the window's pred_len
    actual_close   -- ground truth close price
    pred_close     -- point forecast (mean for Kronos/Lag-Llama, median for TimesFM,
                       inverse-transformed prediction for iTransformer)
    pred_q10       -- 10th percentile forecast (NaN if the model has no uncertainty estimate)
    pred_q90       -- 90th percentile forecast (NaN if unavailable)

Only depends on pandas/numpy, so it works unmodified in every model's conda env.
"""
import os

import numpy as np
import pandas as pd

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "results")

RESULT_COLUMNS = [
    "model", "window_id", "target_date", "step_ahead",
    "actual_close", "pred_close", "pred_q10", "pred_q90",
]


def results_dir(dataset: str) -> str:
    return os.path.join(RESULTS_ROOT, dataset)


def save_results(rows: list[dict], model_name: str, dataset: str) -> str:
    """rows: list of dicts with keys matching RESULT_COLUMNS (pred_q10/pred_q90
    may be omitted/None if the model has no uncertainty estimate)."""
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[RESULT_COLUMNS]

    out_dir = results_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_results.csv")
    df.to_csv(out_path, index=False)
    return out_path


def load_results(model_name: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(results_dir(dataset), f"{model_name}_results.csv")
    df = pd.read_csv(path)
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Aggregate metrics over an entire long-format results table (all windows).

    If `df` pools rows from multiple datasets/markets (has a "dataset"
    column), directional accuracy groups by (dataset, window_id) rather than
    window_id alone -- window_id restarts at 0 for every dataset, so
    grouping by window_id alone would silently splice together unrelated
    windows from different markets when computing step-to-step direction."""
    mae = np.mean(np.abs(df["pred_close"] - df["actual_close"]))
    rmse = np.sqrt(np.mean((df["pred_close"] - df["actual_close"]) ** 2))

    # Directional accuracy: computed per-window (direction of change from the
    # last context value isn't available here, so we use step-to-step direction
    # within each window's forecast, matching actual step-to-step direction).
    group_keys = ["dataset", "window_id"] if "dataset" in df.columns else ["window_id"]
    dir_correct, dir_total = 0, 0
    for _, g in df.sort_values(group_keys + ["step_ahead"]).groupby(group_keys):
        if len(g) < 2:
            continue
        actual_dir = np.sign(np.diff(g["actual_close"].values))
        pred_dir = np.sign(np.diff(g["pred_close"].values))
        dir_correct += np.sum(actual_dir == pred_dir)
        dir_total += len(actual_dir)
    dir_acc = dir_correct / dir_total if dir_total > 0 else np.nan

    n_windows = df[group_keys].drop_duplicates().shape[0]
    metrics = {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "n_points": len(df), "n_windows": n_windows}

    if df["pred_q10"].notna().any() and df["pred_q90"].notna().any():
        covered = (df["actual_close"] >= df["pred_q10"]) & (df["actual_close"] <= df["pred_q90"])
        metrics["coverage_80"] = covered.mean()
    else:
        metrics["coverage_80"] = np.nan

    return metrics
