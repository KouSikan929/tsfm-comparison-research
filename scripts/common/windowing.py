"""
Shared, model-agnostic rolling-window generator for Phase 1 evaluation.

Every model script (Kronos, Lag-Llama, TimesFM, iTransformer) reads the same
per-dataset windows CSV produced here, so all four are scored on *exactly*
the same context/target row ranges -- apples-to-apples comparison depends on
this.

Design:
  - Context (input) rows are allowed to reach back into the train/val period
    -- that's just historical input, not label leakage.
  - Target (forecast) rows must fall entirely within the leakage-safe test
    period (>= TEST_START, see scripts/download_data.py), since that's the
    window none of the three pretrained foundation models could have seen
    during pretraining.
  - Windows step forward by `stride` rows across the test period, each
    covering `pred_len` target rows preceded by `context_len` context rows.

Only depends on pandas, so it works unmodified inside every model's conda
env (kronos / lag-llama / timesfm / itransformer all have pandas installed).

Usage (as a library):
    from windowing import generate_windows, load_windows
    windows = generate_windows(df, context_len=400, pred_len=20, stride=10,
                                test_start="2025-07-01")

Usage (as a script, to (re)generate a dataset's windows CSV):
    python windowing.py --dataset SSE
    python windowing.py --dataset SP500 --context_len 400 --pred_len 20 --stride 10
"""
import argparse
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
WINDOWS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

CONTEXT_LEN = 400
PRED_LEN = 20
STRIDE = 10
TEST_START = "2025-07-01"


def windows_path(dataset: str) -> str:
    return os.path.join(WINDOWS_DIR, f"eval_windows_{dataset}.csv")


def data_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}_daily.csv")


def generate_windows(df: pd.DataFrame, context_len: int, pred_len: int, stride: int, test_start: str) -> pd.DataFrame:
    """Returns one row per window: window_id, context_start_idx/date,
    context_end_idx/date, target_start_idx/date, target_end_idx/date.

    Indices are positional (iloc-style) into `df`, which must be sorted
    ascending by date with a reset (0..n-1) integer index.
    """
    test_start_ts = pd.Timestamp(test_start)
    first_test_idx = df.index[df["timestamps"] >= test_start_ts][0]

    rows = []
    window_id = 0
    target_start_idx = first_test_idx
    while target_start_idx + pred_len <= len(df):
        context_start_idx = target_start_idx - context_len
        if context_start_idx < 0:
            target_start_idx += stride
            continue
        context_end_idx = target_start_idx - 1
        target_end_idx = target_start_idx + pred_len - 1

        rows.append({
            "window_id": window_id,
            "context_start_idx": context_start_idx,
            "context_end_idx": context_end_idx,
            "target_start_idx": target_start_idx,
            "target_end_idx": target_end_idx,
            "context_start_date": df["timestamps"].iloc[context_start_idx].date().isoformat(),
            "context_end_date": df["timestamps"].iloc[context_end_idx].date().isoformat(),
            "target_start_date": df["timestamps"].iloc[target_start_idx].date().isoformat(),
            "target_end_date": df["timestamps"].iloc[target_end_idx].date().isoformat(),
        })
        window_id += 1
        target_start_idx += stride

    return pd.DataFrame(rows)


def load_windows(dataset: str) -> pd.DataFrame:
    return pd.read_csv(windows_path(dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225 (must match data/raw/<dataset>_daily.csv)")
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--test_start", default=TEST_START)
    args = parser.parse_args()

    df = pd.read_csv(data_path(args.dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    windows = generate_windows(df, args.context_len, args.pred_len, args.stride, args.test_start)
    out_path = windows_path(args.dataset)
    windows.to_csv(out_path, index=False)

    print(f"[{args.dataset}] Generated {len(windows)} windows (context_len={args.context_len}, pred_len={args.pred_len}, stride={args.stride})")
    print(f"Test period: {args.test_start} -> {df['timestamps'].max().date()}")
    print(f"First window target: {windows['target_start_date'].iloc[0]} -> {windows['target_end_date'].iloc[0]}")
    print(f"Last window target:  {windows['target_start_date'].iloc[-1]} -> {windows['target_end_date'].iloc[-1]}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
