"""Calendar splits, context-only features, and train-only preprocessing."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.regime_features import compute_window_features

FEATURES = (
    "realized_vol", "rolling_vol_20d", "hurst", "efficiency_ratio",
    "max_drawdown", "downside_vol", "adf_stat", "adf_pvalue",
    "kpss_stat", "kpss_pvalue", "skewness", "kurtosis",
    "max_abs_return", "volume_ratio",
)
COLUMNS = ("open", "high", "low", "close", "volume", "amount")
CLOSE = 3


def calendar_features(dates):
    """Kronos native fields and Lag-Llama's six second-frequency fields."""
    dates = pd.DatetimeIndex(dates)
    kronos = np.stack([dates.minute, dates.hour, dates.dayofweek,
                       dates.day, dates.month], axis=-1).astype(np.int64)
    lag = np.stack([dates.second / 59 - .5, dates.minute / 59 - .5,
                    dates.hour / 23 - .5, dates.dayofweek / 6 - .5,
                    (dates.day - 1) / 30 - .5,
                    (dates.dayofyear - 1) / 365 - .5], axis=-1).astype(np.float32)
    return kronos, lag


class WindowDataset(Dataset):
    def __init__(self, raw_dir, markets, split, context=400, horizon=20, stride=10,
                 train_end="2024-06-30", val_end="2025-06-30", limit_per_market=0):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if context < 21 or horizon < 1 or stride < 1:
            raise ValueError("context >= 21, horizon >= 1, stride >= 1 required")
        train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
        if not train_end < val_end < pd.Timestamp("2025-07-01"):
            raise ValueError("Require train_end < val_end < final test start 2025-07-01")
        if len(set(markets)) != len(markets) or not markets:
            raise ValueError("Provide distinct, nonempty market names")
        self.context, self.horizon = context, horizon
        self.markets, self.series, self.windows = list(markets), [], []
        self.window_ids = []
        for market_idx, market in enumerate(markets):
            if Path(market).name != market or "/" in market or "\\" in market:
                raise ValueError("Market must be a file stem")
            df = pd.read_csv(Path(raw_dir) / f"{market}_daily.csv")
            dates = pd.DatetimeIndex(pd.to_datetime(df.timestamps))
            if dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError(f"{market}: invalid / unsorted / duplicate dates")
            if "amount" not in df:
                df["amount"] = df.close * df.volume
            values = df[list(COLUMNS)].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all() or (values[:, :4] <= 0).any():
                raise ValueError(f"{market}: nonfinite inputs or nonpositive prices")
            if (values[:, 4:] < 0).any():
                raise ValueError(f"{market}: negative volume / amount")
            stamps, time_feats = calendar_features(dates)
            self.series.append((values, stamps, time_feats, dates))
            if split == "train":
                lo, hi = context, int(dates.searchsorted(train_end, side="right"))
            elif split == "val":
                lo = max(context, int(dates.searchsorted(train_end, side="right")))
                hi = int(dates.searchsorted(val_end, side="right"))
            else:
                lo = max(context, int(dates.searchsorted(pd.Timestamp("2025-07-01"))))
                hi = len(dates)
            starts = list(range(lo, hi - horizon + 1, stride))
            if limit_per_market:
                # Even temporal coverage for a smoke subset; no random split.
                indices = np.linspace(0, len(starts) - 1,
                                      min(limit_per_market, len(starts)), dtype=int)
                starts = [starts[i] for i in indices]
            if not starts:
                raise ValueError(f"{market}: no complete {split} windows")
            for start in starts:
                x = values[start-context:start]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    feats = compute_window_features(x[:, CLOSE].astype(float), x[:, 4].astype(float))
                self.windows.append((market_idx, start,
                                     np.array([feats[k] for k in FEATURES], dtype=np.float32)))
                self.window_ids.append((start-lo)//stride)
        self.features = np.stack([w[2] for w in self.windows])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        market, start, feats = self.windows[index]
        x, stamps, time_feats, _ = self.series[market]
        sl = slice(start-self.context, start)
        return {"context": torch.from_numpy(x[sl].copy()),
                "target": torch.from_numpy(x[start:start+self.horizon, CLOSE].copy()),
                "features": torch.from_numpy(feats.copy()),
                "stamps": torch.from_numpy(stamps[sl].copy()),
                "time_features": torch.from_numpy(time_feats[sl].copy()),
                "market": torch.tensor(market), "index": torch.tensor(index)}

    def sampling_weights(self):
        counts = np.bincount([w[0] for w in self.windows], minlength=len(self.markets))
        return torch.tensor([1 / counts[w[0]] for w in self.windows], dtype=torch.double)

    def manifest(self):
        out = {}
        for m, name in enumerate(self.markets):
            starts = [s for i, s, _ in self.windows if i == m]
            dates = self.series[m][3]
            out[name] = {"windows": len(starts),
                         "first_target": str(dates[min(starts)].date()),
                         "last_target": str(dates[max(starts)+self.horizon-1].date())}
        return out
