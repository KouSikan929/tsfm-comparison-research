"""
Reformats a data/raw/<dataset>_daily.csv into the layout iTransformer's
Dataset_Custom(DateSplit) expects: a 'date' column (renamed from
'timestamps') with the target ('close') column last.

Usage:
    python prep_itransformer_data.py --dataset SSE
"""
import argparse
import os

import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ITRANSFORMER_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "iTransformer", "dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(RAW_PATH, f"{args.dataset}_daily.csv"))
    df = df.rename(columns={"timestamps": "date"})
    cols = ["date", "open", "high", "low", "volume", "amount", "close"]
    df = df[cols]

    out_dir = os.path.join(ITRANSFORMER_DATASET_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {df.shape} to {out_path}")


if __name__ == "__main__":
    main()
