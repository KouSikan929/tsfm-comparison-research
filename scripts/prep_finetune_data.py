"""
Prepares a leakage-safe fine-tuning CSV for one dataset: every row with
`timestamps <= VAL_END` (i.e. the TRAIN+VAL period only -- see
scripts/download_data.py for the split boundaries), written to
data/finetune_data/<dataset>_finetune.csv in the same schema as
data/raw/<dataset>_daily.csv.

This is intentionally a SEPARATE step from the zero-shot pipeline's data:
fine-tuning must never see anything from TEST_START onward, or the
zero-shot-vs-fine-tuned comparison would be contaminated (the fine-tuned
model would have trained on exactly the period it's being evaluated on).
Every one of the three finetune_<model>.py scripts reads this file, so the
leakage boundary is enforced in exactly one place.

Usage:
    python prep_finetune_data.py --dataset SP500
"""
import argparse
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_data")

VAL_END = "2025-06-30"  # must match scripts/download_data.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(RAW_DIR, f"{args.dataset}_daily.csv"))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    finetune_df = df[df["timestamps"] <= VAL_END].copy()
    finetune_df["timestamps"] = finetune_df["timestamps"].dt.strftime("%Y-%m-%d")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.dataset}_finetune.csv")
    finetune_df.to_csv(out_path, index=False)
    print(f"[{args.dataset}] {len(finetune_df)} rows (<= {VAL_END}) -> {out_path}")
    print(f"  Range: {finetune_df['timestamps'].min()} -> {finetune_df['timestamps'].max()}")


if __name__ == "__main__":
    main()
