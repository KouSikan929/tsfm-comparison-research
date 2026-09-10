"""
Phase 2 step 2: breaks Phase 1's already-computed results out by regime
(stable/medium/unstable, from compute_regime_features.py), pooled across
all markets, and runs pairwise Diebold-Mariano tests *within* each regime
bucket to see whether model rankings change by regime.

Uses the CAUSAL per-market regime thresholds by default
(regime_features_all_causal.csv -- see compute_regime_features.py), which
fixes a look-ahead issue in the original global-pooled-hindsight version
(--legacy-thresholds restores the old file/behavior for comparison; see
docs/methodology_experimental_setup.md section 10.5b for the full writeup
of what changed and why).

Even with causal thresholds, expect some residual regime/market
correlation to be a real property of the data (a market's own volatility
history can differ structurally from another market's), not necessarily an
analysis bug -- check the printed crosstab each run rather than assuming
either the old confound or a clean separation.

Usage:
    python analyze_phase2_regimes.py --datasets SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD
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

ALL_MODELS = ["kronos", "lag-llama", "timesfm", "itransformer"]
COLORS = {"kronos": "#d62728", "lag-llama": "#9467bd", "timesfm": "#2ca02c", "itransformer": "#1f77b4"}
REGIME_ORDER = ["stable", "medium", "unstable"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
CAUSAL_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regime_features_all_causal.csv")
LEGACY_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regime_features_all.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                         help="Subset of models to include (default: all 4).")
    parser.add_argument("--legacy-thresholds", action="store_true",
                         help="Use the old global-pooled-hindsight regime_features_all.csv instead "
                              "of the causal per-market version.")
    args = parser.parse_args()
    models = args.models
    features_path = LEGACY_FEATURES_PATH if args.legacy_thresholds else CAUSAL_FEATURES_PATH

    regimes = pd.read_csv(features_path)[["dataset", "window_id", "regime"]]

    print("=" * 70)
    print(f"Using: {'LEGACY global-pooled-hindsight' if args.legacy_thresholds else 'CAUSAL per-market'} thresholds ({features_path})")
    print("Regime composition by market (check for residual market confound):")
    print(pd.crosstab(regimes["dataset"], regimes["regime"]))
    print("=" * 70)

    all_results = []
    for ds in args.datasets:
        for m in models:
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
        for m in models:
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
        regime_results = {m: results[(results["regime"] == regime) & (results["model"] == m)] for m in models}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
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
    pivot_mae = summary.pivot(index="regime", columns="model", values="mae").reindex(REGIME_ORDER)[models]
    pivot_mae.plot(kind="bar", ax=axes[0], color=[COLORS[m] for m in models])
    axes[0].set_ylabel("MAE (pooled across markets, raw price units)")
    axes[0].set_title("MAE by regime")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8)

    pivot_dir = summary.pivot(index="regime", columns="model", values="dir_acc").reindex(REGIME_ORDER)[models]
    pivot_dir.plot(kind="bar", ax=axes[1], color=[COLORS[m] for m in models])
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (50%)")
    axes[1].set_ylabel("Directional accuracy")
    axes[1].set_title("Directional accuracy by regime")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8)

    threshold_kind = "legacy global-pooled" if args.legacy_thresholds else "causal per-market"
    plt.suptitle(f"Phase 2: performance by regime ({threshold_kind} thresholds), pooled across {len(args.datasets)} markets", fontsize=12)
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
    ax.set_title(f"Regime composition by market ({threshold_kind} thresholds)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_png2 = os.path.join(OUT_DIR, "phase2_regime_market_confound.png")
    plt.savefig(out_png2, dpi=150)
    print(f"Saved {out_png2}")


if __name__ == "__main__":
    main()
