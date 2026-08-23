"""
Phase 2 step 2: breaks Phase 1's already-computed results out by regime
(stable/medium/unstable, from compute_regime_features.py), pooled across
all markets, and runs pairwise Diebold-Mariano tests *within* each regime
bucket to see whether model rankings change by regime.

IMPORTANT CAVEAT (see compute_regime_features.py's crosstab output): regime
labels are heavily confounded with market identity in this dataset --
SP500's entire leakage-safe test period is uniformly "stable" (all 27
windows), Nikkei225's is 77% "unstable". This is a real property of the
fixed 2025-07-01-onward test window (its start date is fixed by Kronos's
pretraining cutoff, not chosen for regime diversity), not an analysis bug.
It means any regime-vs-market interaction found here cannot be cleanly
separated from a market-identity effect using this data alone -- flagged
explicitly in every printed table, and must be flagged in any write-up.

Usage:
    python analyze_phase2_regimes.py --datasets SP500 SSE SZSE Nikkei225
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
from results_io import load_results, compute_metrics  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from aggregate_results import diebold_mariano  # noqa: E402

MODELS = ["kronos", "lag-llama", "timesfm", "itransformer"]
COLORS = {"kronos": "#d62728", "lag-llama": "#9467bd", "timesfm": "#2ca02c", "itransformer": "#1f77b4"}
REGIME_ORDER = ["stable", "medium", "unstable"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
REGIME_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regime_features_all.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    regimes = pd.read_csv(REGIME_FEATURES_PATH)[["dataset", "window_id", "regime"]]

    print("=" * 70)
    print("CAVEAT: regime is confounded with market in this data (see crosstab)")
    print(pd.crosstab(regimes["dataset"], regimes["regime"]))
    print("=" * 70)

    all_results = []
    for ds in args.datasets:
        for m in MODELS:
            try:
                df = load_results(m, ds)
            except FileNotFoundError:
                continue
            df["dataset"] = ds
            all_results.append(df)
    results = pd.concat(all_results, ignore_index=True)
    results = results.merge(regimes, on=["dataset", "window_id"], how="left")

    print("\n" + "=" * 70)
    print("Per-model metrics by regime (pooled across all markets)")
    print("=" * 70)
    summary_rows = []
    for regime in REGIME_ORDER:
        for m in MODELS:
            sub = results[(results["regime"] == regime) & (results["model"] == m)]
            if sub.empty:
                continue
            metrics = compute_metrics(sub)
            summary_rows.append({"regime": regime, "model": m, **metrics})
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("Pairwise Diebold-Mariano test WITHIN each regime bucket")
    print("=" * 70)
    dm_rows = []
    for regime in REGIME_ORDER:
        regime_results = {m: results[(results["regime"] == regime) & (results["model"] == m)] for m in MODELS}
        for i in range(len(MODELS)):
            for j in range(i + 1, len(MODELS)):
                a, b = MODELS[i], MODELS[j]
                if regime_results[a].empty or regime_results[b].empty:
                    continue
                merged = regime_results[a].merge(
                    regime_results[b], on=["dataset", "window_id", "step_ahead", "target_date"],
                    suffixes=("_a", "_b"),
                )
                if merged.empty:
                    continue
                pred_len = int(merged["step_ahead"].max())
                errors_a = (merged["pred_close_a"] - merged["actual_close_a"]).values
                errors_b = (merged["pred_close_b"] - merged["actual_close_b"]).values
                dm_stat, p_value = diebold_mariano(errors_a, errors_b, h=pred_len)
                dm_rows.append({
                    "regime": regime, "model_a": a, "model_b": b, "n_points": len(merged),
                    "dm_statistic": dm_stat, "p_value": p_value, "significant_5pct": p_value < 0.05,
                })
    dm_df = pd.DataFrame(dm_rows)
    print(dm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary.to_csv(os.path.join(OUT_DIR, "phase2_regime_summary.csv"), index=False)
    dm_df.to_csv(os.path.join(OUT_DIR, "phase2_regime_dm_test.csv"), index=False)

    # Plot: grouped bars, MAE by regime, one group of bars per model
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot_mae = summary.pivot(index="regime", columns="model", values="mae").reindex(REGIME_ORDER)[MODELS]
    pivot_mae.plot(kind="bar", ax=axes[0], color=[COLORS[m] for m in MODELS])
    axes[0].set_ylabel("MAE (pooled across markets, raw price units)")
    axes[0].set_title("MAE by regime\n(NOTE: regime is confounded with market -- see caveat)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8)

    pivot_dir = summary.pivot(index="regime", columns="model", values="dir_acc").reindex(REGIME_ORDER)[MODELS]
    pivot_dir.plot(kind="bar", ax=axes[1], color=[COLORS[m] for m in MODELS])
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (50%)")
    axes[1].set_ylabel("Directional accuracy")
    axes[1].set_title("Directional accuracy by regime")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)

    plt.suptitle("Phase 2: performance by regime, pooled across SP500/SSE/SZSE/Nikkei225", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "phase2_regime_comparison.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")

    # Second plot: regime composition by market, to keep the confound visible
    fig2, ax = plt.subplots(figsize=(7, 4))
    comp = pd.crosstab(regimes["dataset"], regimes["regime"], normalize="index").reindex(columns=REGIME_ORDER)
    comp = comp.reindex(args.datasets)
    comp.plot(kind="barh", stacked=True, ax=ax, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_xlabel("Fraction of windows")
    ax.set_title("Regime composition by market\n(shows the confound: SP500=100% stable, Nikkei225=77% unstable)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_png2 = os.path.join(OUT_DIR, "phase2_regime_market_confound.png")
    plt.savefig(out_png2, dpi=150)
    print(f"Saved {out_png2}")


if __name__ == "__main__":
    main()
