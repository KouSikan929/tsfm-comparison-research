"""
Cross-market Phase 1 summary: combines all datasets' phase1_summary.csv into
one comparison. Raw MAE/RMSE aren't comparable across markets with very
different price levels (SP500 ~7000 vs. Nikkei225 ~40000), so this also
computes MAE as a % of each market's average close price ("relative MAE").

Usage:
    python plot_cross_market_summary.py --datasets SP500 SSE SZSE Nikkei225
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from results_io import load_results, results_dir  # noqa: E402

MODELS = ["kronos", "lag-llama", "timesfm", "itransformer"]
COLORS = {"kronos": "#d62728", "lag-llama": "#9467bd", "timesfm": "#2ca02c", "itransformer": "#1f77b4"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    for ds in args.datasets:
        summary = pd.read_csv(os.path.join(results_dir(ds), "phase1_summary.csv"), index_col=0)
        for m in MODELS:
            if m not in summary.index:
                continue
            actual_mean = load_results(m, ds)["actual_close"].mean()
            row = summary.loc[m].to_dict()
            row["dataset"] = ds
            row["model"] = m
            row["relative_mae_pct"] = 100 * row["mae"] / actual_mean
            rows.append(row)

    combined = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "cross_market_summary.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(combined[["dataset", "model", "mae", "relative_mae_pct", "dir_acc", "coverage_80"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    pivot_mae = combined.pivot(index="dataset", columns="model", values="relative_mae_pct")[MODELS]
    pivot_mae = pivot_mae.reindex(args.datasets)
    pivot_mae.plot(kind="bar", ax=axes[0], color=[COLORS[m] for m in MODELS])
    axes[0].set_ylabel("MAE as % of avg. close price")
    axes[0].set_title("Relative MAE by market (lower = better)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8)

    pivot_cov = combined.pivot(index="dataset", columns="model", values="coverage_80")[MODELS]
    pivot_cov = pivot_cov.reindex(args.datasets)
    pivot_cov.plot(kind="bar", ax=axes[1], color=[COLORS[m] for m in MODELS])
    axes[1].axhline(0.8, color="gray", linestyle="--", linewidth=1, label="target (80%)")
    axes[1].set_ylabel("80% interval coverage")
    axes[1].set_title("Uncertainty calibration by market (target: 0.80)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)

    plt.suptitle("Cross-market Phase 1 comparison (leakage-safe rolling windows, context=400/pred=20)", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "cross_market_summary.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
