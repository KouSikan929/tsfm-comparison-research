"""
Phase 2: regime characterization metrics, computed on the *context* portion
of each rolling window (the 400 days of history the model actually sees,
not the forecast target -- these describe the input the model is
conditioning on, not the outcome it's predicting).

Three characteristics, one per axis named in the research plan
(stationarity, volatility, trend strength):

1. Realized volatility -- annualized std of daily log returns. The
   standard, most direct "how turbulent is this market right now" measure.

2. Hurst exponent -- estimated via the scaling of the standard deviation of
   lagged log-price differences (a fast, numpy-only variance-scaling
   estimator; less statistically rigorous than full rescaled-range or DFA,
   but standard practice for this kind of screening use and fine given we
   only need a discriminating regime signal, not a publication-grade
   estimate). H < 0.5 => mean-reverting, H ~= 0.5 => random walk,
   H > 0.5 => trending/persistent.

   Note on why Hurst instead of a classic ADF/KPSS stationarity test: ADF
   applied to the raw price *level* of an equity index will fail to reject
   the unit-root null almost everywhere (price levels are generically
   non-stationary/I(1) regardless of "regime"), so it has close to zero
   discriminating power across windows here. Hurst captures the
   mean-reversion-vs-trending distinction that's actually informative for
   this analysis.

3. Efficiency Ratio (Kaufman) -- |net price change| / sum(|daily changes|)
   over the window. 0 = choppy/no net progress, 1 = a straight-line trend.
   This is "trend strength": how much of the window's total movement
   resulted in net directional progress.

Only depends on numpy, so it works unmodified in any conda env.
"""
import numpy as np


def realized_volatility(close_prices: np.ndarray, annualize: bool = True) -> float:
    log_returns = np.diff(np.log(close_prices))
    vol = np.std(log_returns, ddof=1)
    if annualize:
        vol *= np.sqrt(252)
    return vol


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


def compute_window_features(close_prices: np.ndarray) -> dict:
    return {
        "realized_vol": realized_volatility(close_prices),
        "hurst": hurst_exponent(close_prices),
        "efficiency_ratio": efficiency_ratio(close_prices),
    }
