"""
Phase 2 step 1: computes regime features (realized volatility, Hurst
exponent, efficiency ratio) for every window's *context* period, across all
given datasets, then buckets windows into stable/medium/unstable regimes
using GLOBAL tertile thresholds (pooled across all datasets/markets) on
realized volatility -- the primary, most standard "regime" axis in
finance. Pooling across markets rather than computing thresholds per-market
matters here: each market only has ~26-27 windows, too few to split into
three reliable buckets on its own, so pooling to ~107 total windows gives
each regime bucket a workable sample size.

Hurst and efficiency_ratio are also computed and saved, but used as
continuous covariates for correlation-style analysis (see
analyze_phase2_regimes.py) rather than additional bucketing dimensions --
with only ~107 total windows, splitting on more than one axis would leave
too few windows per cell.

Usage:
    python compute_regime_features.py --datasets SP500 SSE SZSE Nikkei225
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from regime_features import compute_window_features  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def compute_for_dataset(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(data_path(dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(dataset)

    rows = []
    for _, w in windows.iterrows():
        context = df.loc[w.context_start_idx:w.context_end_idx, "close"].values
        feats = compute_window_features(context)
        feats["dataset"] = dataset
        feats["window_id"] = int(w.window_id)
        feats["target_start_date"] = w.target_start_date
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    all_feats = pd.concat([compute_for_dataset(ds) for ds in args.datasets], ignore_index=True)

    # Global tertile thresholds on realized_vol, pooled across all markets/windows.
    q1, q2 = all_feats["realized_vol"].quantile([1 / 3, 2 / 3])
    def label_regime(v):
        if v <= q1:
            return "stable"
        elif v <= q2:
            return "medium"
        else:
            return "unstable"
    all_feats["regime"] = all_feats["realized_vol"].apply(label_regime)

    out_path = os.path.join(OUT_DIR, "regime_features_all.csv")
    all_feats.to_csv(out_path, index=False)

    print(f"Computed regime features for {len(all_feats)} windows across {len(args.datasets)} markets")
    print(f"Global realized_vol tertile thresholds: stable <= {q1:.4f} < medium <= {q2:.4f} < unstable")
    print("\nRegime counts:")
    print(all_feats["regime"].value_counts())
    print("\nRegime counts by market:")
    print(pd.crosstab(all_feats["dataset"], all_feats["regime"]))
    print("\nFeature summary by regime:")
    print(all_feats.groupby("regime")[["realized_vol", "hurst", "efficiency_ratio"]].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
