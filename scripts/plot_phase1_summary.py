"""
Phase 1 summary visualization: MAE/RMSE/DirAcc/Coverage bar charts, plus one
example window's forecast overlay across all 4 models.

Usage:
    python plot_phase1_summary.py --dataset SP500
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from results_io import load_results, results_dir  # noqa: E402

MODELS = ["kronos", "lag-llama", "timesfm", "itransformer"]
COLORS = {"kronos": "#d62728", "lag-llama": "#9467bd", "timesfm": "#2ca02c", "itransformer": "#1f77b4"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--models", nargs="+", default=MODELS,
                         help="Subset of models to plot (default: all 4) -- must match what "
                              "aggregate_results.py was run with for this dataset.")
    args = parser.parse_args()
    dataset = args.dataset
    models = args.models
    out_dir = results_dir(dataset)

    summary = pd.read_csv(os.path.join(out_dir, "phase1_summary.csv"), index_col=0)
    per_model = {m: load_results(m, dataset) for m in models}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("mae", "MAE (lower better)"), ("rmse", "RMSE (lower better)"),
               ("dir_acc", "Directional Accuracy (higher better)"), ("coverage_80", "80% Interval Coverage (target: 0.80)")]

    for ax, (col, title) in zip(axes.flat, metrics):
        vals = summary[col]
        bars = ax.bar(vals.index, vals.values, color=[COLORS[m] for m in vals.index])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if col == "dir_acc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (50%)")
            ax.legend(fontsize=8)
        if col == "coverage_80":
            ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="target (80%)")
            ax.legend(fontsize=8)
        for b, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"Phase 1: leakage-safe rolling windows, {dataset}, context=400/pred=20", fontsize=12)
    plt.tight_layout()
    out1 = os.path.join(out_dir, "phase1_metrics_comparison.png")
    plt.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # Example window overlay: pick a middle window for all included models
    example_window_id = 13
    fig2, ax = plt.subplots(figsize=(10, 5))
    actual_plotted = False
    for m in models:
        df = per_model[m]
        w = df[df["window_id"] == example_window_id].sort_values("step_ahead")
        if not actual_plotted:
            ax.plot(w["target_date"], w["actual_close"], label="Ground Truth", color="black", linewidth=2, marker="o", markersize=3)
            actual_plotted = True
        ax.plot(w["target_date"], w["pred_close"], label=m, color=COLORS[m], linewidth=1.5, alpha=0.85)
        if w["pred_q10"].notna().any():
            ax.fill_between(w["target_date"], w["pred_q10"], w["pred_q90"], color=COLORS[m], alpha=0.1)

    ax.set_ylabel("Close Price")
    ax.set_title(f"Example window (window_id={example_window_id}): all 4 models vs. ground truth")
    ax.legend(loc="best")
    ax.grid(True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    out2 = os.path.join(out_dir, "phase1_example_window.png")
    plt.savefig(out2, dpi=150)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
