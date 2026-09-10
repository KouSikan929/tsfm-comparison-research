"""
Phase 2 step 1: computes the expanded regime feature set (see
scripts/common/regime_features.py) for every window's *context* period,
across all given datasets, then buckets windows into stable/medium/unstable
regimes on realized volatility (the primary regime axis).

CAUSAL THRESHOLDS (fixes a look-ahead issue in the original version -- see
docs/methodology_experimental_setup.md section 10.5b): thresholds are no
longer a single global tertile pooled across all TEST-period windows from
every market (which meant an early window's regime label depended in part on
a chronologically LATER window's realized volatility). Instead, each
market's own stable/medium/unstable cutoffs are calibrated from that
market's own REALIZED VOLATILITY computed over many overlapping 400-day
windows strictly WITHIN its train+val period (i.e. entirely before
TEST_START) -- history that is, by construction, available before every
single test window's forecast origin. This is both more causal (no test
window's classification depends on any other window, ever) and better
powered (a market's pre-test history gives ~400+ overlapping reference
windows at stride=5, vs. ~24-42 test windows total per market), which is
what motivated pooling across markets in the first place.

The previous global-pooled-hindsight output (`regime_features_all.csv`) is
left on disk unchanged as a labeled prior artifact; this script now writes
`regime_features_all_causal.csv` instead.

Usage:
    python compute_regime_features.py --datasets SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from regime_features import compute_window_features, realized_volatility  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_END = "2024-06-30"
TEST_START = "2025-07-01"

HIST_WINDOW_LEN = 400  # same context length used everywhere else in this project
HIST_STRIDE = 5        # dense enough for a robust per-market reference distribution


def historical_vol_thresholds(dataset: str) -> tuple[float, float]:
    """Rolling 400-day realized volatility computed at every HIST_STRIDE-th
    row of the pre-TEST_START history (train+val), tertile thresholds taken
    from that distribution. Entirely causal: every value used here is dated
    before TEST_START, hence before every single test window's target."""
    df = pd.read_csv(data_path(dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    pre_test = df[df["timestamps"] < TEST_START].reset_index(drop=True)

    vols = []
    for end in range(HIST_WINDOW_LEN, len(pre_test), HIST_STRIDE):
        window_prices = pre_test["close"].iloc[end - HIST_WINDOW_LEN:end].values
        vols.append(realized_volatility(window_prices))
    vols = np.array(vols)
    q1, q2 = np.quantile(vols, [1 / 3, 2 / 3])
    return float(q1), float(q2), len(vols)


def compute_for_dataset(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(data_path(dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(dataset)

    rows = []
    for _, w in windows.iterrows():
        ctx = df.loc[w.context_start_idx:w.context_end_idx]
        feats = compute_window_features(ctx["close"].values, ctx["volume"].values)
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

    print("Calibrating causal, per-market volatility thresholds from pre-TEST_START history...")
    thresholds = {}
    for ds in args.datasets:
        q1, q2, n_hist = historical_vol_thresholds(ds)
        thresholds[ds] = (q1, q2)
        print(f"  {ds:12s} stable <= {q1:.4f} < medium <= {q2:.4f} < unstable   (from {n_hist} historical reference windows)")

    def label_regime(row):
        q1, q2 = thresholds[row["dataset"]]
        v = row["realized_vol"]
        if v <= q1:
            return "stable"
        elif v <= q2:
            return "medium"
        else:
            return "unstable"

    all_feats["regime"] = all_feats.apply(label_regime, axis=1)

    out_path = os.path.join(OUT_DIR, "regime_features_all_causal.csv")
    all_feats.to_csv(out_path, index=False)

    print(f"\nComputed regime features for {len(all_feats)} windows across {len(args.datasets)} markets")
    print("\nRegime counts:")
    print(all_feats["regime"].value_counts())
    print("\nRegime counts by market:")
    print(pd.crosstab(all_feats["dataset"], all_feats["regime"]))
    feature_cols = ["realized_vol", "rolling_vol_20d", "hurst", "efficiency_ratio", "adf_stat",
                     "adf_pvalue", "kpss_stat", "kpss_pvalue", "skewness", "kurtosis",
                     "max_abs_return", "max_drawdown", "downside_vol", "volume_ratio"]
    print("\nFeature summary by regime:")
    print(all_feats.groupby("regime")[feature_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out_path}")
    print("(Previous global-pooled-threshold version, if present, is left unchanged at regime_features_all.csv)")


if __name__ == "__main__":
    main()
