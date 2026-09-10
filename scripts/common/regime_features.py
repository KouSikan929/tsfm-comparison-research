"""
Phase 2: regime characterization metrics, computed on the *context* portion
of each rolling window (the 400 days of history the model actually sees,
not the forecast target -- these describe the input the model is
conditioning on, not the outcome it's predicting).

Organized into the six conceptual dimensions from the research plan's
revised Phase 3, each covered by one or more concrete features:

  1. Volatility            -- realized_vol, rolling_vol_20d
  2. Persistence / memory  -- hurst
  3. Stationarity           -- adf_stat, adf_pvalue, kpss_stat, kpss_pvalue
  4. Trend strength         -- efficiency_ratio
  5. Distributional         -- skewness, kurtosis, max_abs_return
  6. Market stress          -- max_drawdown, downside_vol

Plus one liquidity/attention indicator (volume_ratio) using raw volume
(never the synthetic `amount` column, which is close*volume, not a real
reported field -- see scripts/download_data.py).

Design notes:

- ADF/KPSS are run on the context's *log returns*, not the price level.
  ADF on a raw equity price level fails to reject the unit-root null almost
  everywhere (price levels are generically non-stationary/I(1) regardless of
  "regime"), so it has near-zero discriminating power for this purpose --
  documented as a known limitation in docs/methodology_experimental_setup.md
  before this fix. Returns are generically much closer to stationary, so the
  test statistic actually varies meaningfully window-to-window.

- Hurst remains a *persistence* measure, not a stationarity test -- the two
  are computed and reported separately, never conflated (also flagged in the
  methodology doc as a thing to keep distinct).

Only depends on numpy/scipy/statsmodels, so it works unmodified in any conda
env that has those three installed (the project's `base` env does).
"""
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, kpss


def realized_volatility(close_prices: np.ndarray, annualize: bool = True) -> float:
    log_returns = np.diff(np.log(close_prices))
    vol = np.std(log_returns, ddof=1)
    if annualize:
        vol *= np.sqrt(252)
    return vol


def rolling_vol_20d(close_prices: np.ndarray, annualize: bool = True) -> float:
    """Realized volatility over just the most recent 20 days of the context
    -- a "recent conditions" volatility signal distinct from the full
    400-day realized_volatility, which can smear a recent spike/calm patch
    across a much longer window."""
    recent = close_prices[-21:]  # 21 prices -> 20 returns
    if len(recent) < 3:
        return np.nan
    return realized_volatility(recent, annualize=annualize)


def hurst_exponent(close_prices: np.ndarray, min_lag: int = 2, max_lag: int = 100) -> float:
    log_prices = np.log(close_prices)
    max_lag = min(max_lag, len(log_prices) // 2)
    lags = np.arange(min_lag, max_lag)
    tau = np.array([np.std(log_prices[lag:] - log_prices[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return slope


def efficiency_ratio(close_prices: np.ndarray) -> float:
    net_change = abs(close_prices[-1] - close_prices[0])
    path_length = np.sum(np.abs(np.diff(close_prices)))
    if path_length == 0:
        return 0.0
    return net_change / path_length


def stationarity_tests(close_prices: np.ndarray) -> dict:
    """ADF and KPSS on log returns (not price level -- see module docstring).
    ADF null: unit root (non-stationary) -- low p-value => reject => stationary.
    KPSS null: stationary -- low p-value => reject => non-stationary.
    The two are complementary (opposite nulls), which is why both are run."""
    log_returns = np.diff(np.log(close_prices))
    out = {"adf_stat": np.nan, "adf_pvalue": np.nan, "kpss_stat": np.nan, "kpss_pvalue": np.nan}
    try:
        adf_res = adfuller(log_returns, autolag="AIC")
        out["adf_stat"], out["adf_pvalue"] = adf_res[0], adf_res[1]
    except Exception:
        pass
    try:
        kpss_res = kpss(log_returns, regression="c", nlags="auto")
        out["kpss_stat"], out["kpss_pvalue"] = kpss_res[0], kpss_res[1]
    except Exception:
        pass
    return out


def distributional_features(close_prices: np.ndarray) -> dict:
    log_returns = np.diff(np.log(close_prices))
    return {
        "skewness": float(scipy_stats.skew(log_returns)),
        "kurtosis": float(scipy_stats.kurtosis(log_returns)),  # excess kurtosis (0 = normal)
        "max_abs_return": float(np.max(np.abs(log_returns))),
    }


def max_drawdown(close_prices: np.ndarray) -> float:
    """Maximum peak-to-trough decline within the context, as a positive fraction."""
    running_max = np.maximum.accumulate(close_prices)
    drawdown = (running_max - close_prices) / running_max
    return float(np.max(drawdown))


def downside_volatility(close_prices: np.ndarray, annualize: bool = True) -> float:
    """Std of negative log returns only (semi-deviation) -- a stress-flavored
    volatility measure distinct from realized_volatility, which weights
    up-moves and down-moves equally."""
    log_returns = np.diff(np.log(close_prices))
    downside = log_returns[log_returns < 0]
    if len(downside) < 2:
        return 0.0
    vol = np.std(downside, ddof=1)
    if annualize:
        vol *= np.sqrt(252)
    return vol


def volume_ratio(volumes: np.ndarray) -> float:
    """Mean volume over the most recent 20 days of context, relative to the
    mean volume over the whole context -- a simple liquidity/attention
    indicator. Uses raw `volume` (real, reported), never the synthetic
    `amount` = close*volume column."""
    if len(volumes) < 21 or np.mean(volumes) == 0:
        return np.nan
    recent_mean = np.mean(volumes[-20:])
    full_mean = np.mean(volumes)
    return float(recent_mean / full_mean) if full_mean > 0 else np.nan


def compute_window_features(close_prices: np.ndarray, volumes: np.ndarray | None = None) -> dict:
    feats = {
        "realized_vol": realized_volatility(close_prices),
        "rolling_vol_20d": rolling_vol_20d(close_prices),
        "hurst": hurst_exponent(close_prices),
        "efficiency_ratio": efficiency_ratio(close_prices),
        "max_drawdown": max_drawdown(close_prices),
        "downside_vol": downside_volatility(close_prices),
    }
    feats.update(stationarity_tests(close_prices))
    feats.update(distributional_features(close_prices))
    if volumes is not None:
        feats["volume_ratio"] = volume_ratio(volumes)
    else:
        feats["volume_ratio"] = np.nan
    return feats
