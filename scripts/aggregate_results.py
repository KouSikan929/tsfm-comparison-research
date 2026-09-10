"""
Phase 1 aggregation: combines all four models' long-format results
(data/results/<model>_results.csv, produced by eval_kronos.py,
eval_lagllama.py, eval_timesfm.py, eval_itransformer.py) into one comparison
table with a pairwise Diebold-Mariano significance test on forecast errors.

Note on window overlap: windows.csv (scripts/common/windowing.py) uses
stride=10 with pred_len=20, so consecutive windows overlap by 10 rows -- the
per-point errors are NOT independent draws. The Diebold-Mariano test's
Newey-West-style variance estimate (with a lag matching pred_len) is exactly
the standard way to handle this serial correlation, which is why we use it
here instead of a naive t-test on raw error differences.

Only depends on pandas/numpy/scipy, so it works from any of the four conda
envs (or a plain env with those installed).

Usage:
    python aggregate_results.py --dataset SP500
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from results_io import load_results, compute_metrics, results_dir  # noqa: E402

MODELS = ["kronos", "lag-llama", "timesfm", "itransformer"]


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss differentials, with a
    Newey-West-style variance correction for serial correlation up to lag h-1
    (h = forecast horizon = pred_len, since windows overlap by more than one
    step -- see module docstring).

    Returns (dm_statistic, two_sided_p_value). A large |statistic| / small
    p-value means the two models' forecast errors are significantly
    different, not just numerically different.
    """
    d = errors_a ** 2 - errors_b ** 2  # loss differential (squared error)
    n = len(d)
    d_mean = d.mean()

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        if lag >= n:
            break
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / h) * cov

    var_d = max(var_d, 1e-12)  # guard against numerical negatives
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--models", nargs="+", default=MODELS,
                         help="Subset of models to compare (default: all 4). Use this to exclude a "
                              "model whose results are stale/out of sync with the others, e.g. "
                              "--models kronos lag-llama timesfm")
    args = parser.parse_args()
    dataset = args.dataset
    models = args.models

    print(f"Loading per-model results for {dataset}...")
    per_model = {}
    for m in models:
        path = os.path.join(results_dir(dataset), f"{m}_results.csv")
        if not os.path.exists(path):
            print(f"  [skip] {m}: no results file at {path}")
            continue
        per_model[m] = load_results(m, dataset)
        print(f"  {m}: {len(per_model[m])} rows, {per_model[m]['window_id'].nunique()} windows")

    if len(per_model) < 2:
        print("\nNeed at least 2 models' results to compare. Run the eval_*.py scripts first.")
        return

    print("\n" + "=" * 70)
    print("Per-model aggregate metrics")
    print("=" * 70)
    summary_rows = []
    for m, df in per_model.items():
        metrics = compute_metrics(df)
        summary_rows.append({"model": m, **metrics})
    summary = pd.DataFrame(summary_rows).set_index("model")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # Pairwise DM test needs identical (window_id, step_ahead) coverage across
    # models -- inner-join on the models being compared.
    print("\n" + "=" * 70)
    print("Pairwise Diebold-Mariano test (H0: equal forecast accuracy)")
    print("=" * 70)
    model_names = list(per_model.keys())
    pred_len = int(per_model[model_names[0]]["step_ahead"].max())

    dm_rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            merged = per_model[a].merge(
                per_model[b], on=["window_id", "step_ahead", "target_date"],
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue
            errors_a = (merged["pred_close_a"] - merged["actual_close_a"]).values
            errors_b = (merged["pred_close_b"] - merged["actual_close_b"]).values
            dm_stat, p_value = diebold_mariano(errors_a, errors_b, h=pred_len)
            dm_rows.append({
                "model_a": a, "model_b": b, "n_shared_points": len(merged),
                "dm_statistic": dm_stat, "p_value": p_value,
                "significant_5pct": p_value < 0.05,
            })

    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        print(dm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n(Positive dm_statistic => model_a has larger squared errors than model_b, i.e. model_b is better.)")
    else:
        print("No overlapping windows between any pair of models -- check window alignment.")

    out_path = os.path.join(results_dir(dataset), "phase1_summary.csv")
    summary.to_csv(out_path)
    dm_out_path = os.path.join(results_dir(dataset), "phase1_dm_test.csv")
    dm_df.to_csv(dm_out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(f"Saved DM test results to {dm_out_path}")


if __name__ == "__main__":
    main()
