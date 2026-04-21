"""
Technical indicators matching the Excel reference model.

Conventions per spec:
- SMA / EMA / RSI / MACD / Bollinger / Z-score: compute on adjusted close
- ADX / Stochastic / ATR: compute on raw H/L/C

All smoothing is explicit. Do NOT replace with pandas-ta without verifying that
pandas-ta's defaults match these (for ATR/ADX they do not — those default to
EMA, not Wilder).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----- Moving averages -------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema_spec(series: pd.Series, period: int) -> pd.Series:
    """
    EMA with alpha = 2/(N+1), seeded with SMA(N) at index N-1. Pandas' native
    `ewm(adjust=False)` seeds with the first observation, which diverges from
    Excel's seeding. We do the seeded version explicitly.
    """
    alpha = 2.0 / (period + 1)
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < period:
        return out
    vals = valid.to_numpy()
    smoothed = np.full(len(vals), np.nan)
    smoothed[period - 1] = vals[:period].mean()
    for i in range(period, len(vals)):
        smoothed[i] = alpha * vals[i] + (1.0 - alpha) * smoothed[i - 1]
    out.loc[valid.index] = smoothed
    return out


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder smoothing: seed with SMA(N), then avg_t = avg_{t-1}*(N-1)/N + x_t/N.
    Equivalent to an EMA with alpha = 1/N (seeded with SMA).
    """
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < period:
        return out
    vals = valid.to_numpy()
    smoothed = np.full(len(vals), np.nan)
    smoothed[period - 1] = vals[:period].mean()
    for i in range(period, len(vals)):
        smoothed[i] = smoothed[i - 1] * (period - 1) / period + vals[i] / period
    out.loc[valid.index] = smoothed
    return out


# ----- Bollinger / RSI / MACD ------------------------------------------------

def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=1)  # STDEV.S
    return mid + num_std * std, mid, mid - num_std * std


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema_spec(close, fast)
    ema_slow = ema_spec(close, slow)
    macd_line = ema_fast - ema_slow
    # Signal line: EMA(9) of macd_line. Seed with SMA(9) on first 9 valid
    # MACD values, which starts at the bar where slow EMA first valid + 8.
    signal_line = ema_spec(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ----- Stochastic / ADX / ATR (raw H/L/C) -----------------------------------

def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    rng = (highest - lowest).replace(0, np.nan)
    return 100 * (close - lowest) / rng


def stoch_d(k: pd.Series, period: int = 3) -> pd.Series:
    return k.rolling(period, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Returns (ADX, +DI, -DI). Wilder-smoothed throughout.
    """
    tr = true_range(high, low, close)
    up = high.diff()
    down = -low.diff()

    plus_dm_raw = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=high.index,
    )
    minus_dm_raw = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=high.index,
    )

    atr_val = wilder_smooth(tr, period)
    plus_dm_s = wilder_smooth(plus_dm_raw, period)
    minus_dm_s = wilder_smooth(minus_dm_raw, period)

    plus_di = 100 * plus_dm_s / atr_val
    minus_di = 100 * minus_dm_s / atr_val

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = wilder_smooth(dx, period)
    return adx_val, plus_di, minus_di


# ----- Z-score against 50-day mean (Range strategy) --------------------------

def zscore_dev(close: pd.Series, sma50: pd.Series, dev_window: int = 60) -> pd.Series:
    """
    Dev_t = (Close - SMA50) / SMA50
    Z_t   = (Dev_t - mean60(Dev)) / stdev.s60(Dev)
    """
    dev = (close - sma50) / sma50
    dev_mean = dev.rolling(dev_window, min_periods=dev_window).mean()
    dev_std = dev.rolling(dev_window, min_periods=dev_window).std(ddof=1)
    return (dev - dev_mean) / dev_std.replace(0, np.nan)
