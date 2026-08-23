"""
Download daily OHLCV data and save it in the shared CSV format used across
all four models in this project (Kronos, Lag-Llama, TimesFM, iTransformer).

Output columns: timestamps, open, high, low, close, volume, amount
  - `amount` (turnover value) is approximated as close * volume, since most
    non-CN data sources (yfinance included) don't report it directly.

Usage:
    python download_data.py --ticker "^GSPC" --name SP500 \
        --start 2015-01-01 --end 2026-08-22
"""
import argparse
import os

import pandas as pd
import yfinance as yf

# Leakage-safe split boundaries agreed on for this project.
# See conversation notes: Kronos's own held-out window ends 2025-06-05,
# so the test window here starts safely after that.
TRAIN_END = "2024-06-30"
VAL_END = "2025-06-30"
TEST_START = "2025-07-01"


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # yfinance can return MultiIndex columns when tickers/interval combos vary; flatten defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "timestamps",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["amount"] = df["close"] * df["volume"]

    cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols].sort_values("timestamps").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="yfinance ticker symbol, e.g. '^GSPC', 'AAPL', 'BTC-USD'")
    parser.add_argument("--name", required=True, help="Short name used in the output filename, e.g. 'SP500'")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None, help="Defaults to today")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Downloading {args.ticker} from {args.start} to {end} ...")
    df = download(args.ticker, args.start, end)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}_daily.csv")
    df.to_csv(out_path, index=False)

    n_train = (df["timestamps"] <= TRAIN_END).sum()
    n_val = ((df["timestamps"] > TRAIN_END) & (df["timestamps"] <= VAL_END)).sum()
    n_test = (df["timestamps"] >= TEST_START).sum()

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Date range: {df['timestamps'].min()} -> {df['timestamps'].max()}")
    print(f"Split (informational, based on TRAIN_END={TRAIN_END}, VAL_END={VAL_END}, TEST_START={TEST_START}):")
    print(f"  train: {n_train} rows (<= {TRAIN_END})")
    print(f"  val:   {n_val} rows ({TRAIN_END} < t <= {VAL_END})")
    print(f"  test:  {n_test} rows (>= {TEST_START}, leakage-safe window)")


if __name__ == "__main__":
    main()
