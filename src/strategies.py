"""
Four strategies. Logic is reconciled against the reference Excel workbook
('End of day summary' sheet, columns CC/CJ/CU/DA). Score-to-signal mapping
uses the spec's uniform ±2 threshold — deliberately stricter than the Excel,
which used ±1 for Trend/Range. We keep the stricter threshold so the
contradiction-filter consensus gets cleaner, higher-conviction inputs.

Mapping (shared across all four):
  +4..+5 STRONG BUY   +2..+3 BUY   -1..+1 HOLD   -2..-3 SELL   -4..-5 STRONG SELL
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_from_score(score: int | float) -> str:
    """Shared vocabulary mapping per spec."""
    if pd.isna(score):
        return "HOLD"
    s = int(round(score))
    if s >= 4:
        return "STRONG BUY"
    if s >= 2:
        return "BUY"
    if s >= -1:
        return "HOLD"
    if s >= -3:
        return "SELL"
    return "STRONG SELL"


# ----- Strategy 1: TREND -----------------------------------------------------

def trend_score(row: pd.Series) -> int | float:
    """
    Four components, each ±1. Score range -4..+4.

    Per Excel:
      MA trend : UP (+1) or DOWN (-1) — no neutral
      BB pos   : OVERSOLD (+1), OVERBOUGHT (-1), else 0
      RSI zone : OVERSOLD (+1), OVERBOUGHT (-1), else 0
      MACD     : BULL (+1) or BEAR (-1) — no neutral
    """
    required = ["sma20", "sma50", "bb_upper", "bb_lower",
                "rsi14", "macd", "macd_signal", "close"]
    if any(pd.isna(row[c]) for c in required):
        return np.nan

    score = 0
    # MA trend: ±1, no neutral (Excel uses >= / < split with DOWN as default).
    score += 1 if row["sma20"] > row["sma50"] else -1

    # BB position.
    if row["close"] < row["bb_lower"]:
        score += 1
    elif row["close"] > row["bb_upper"]:
        score -= 1

    # RSI zone.
    if row["rsi14"] < 30:
        score += 1
    elif row["rsi14"] > 70:
        score -= 1

    # MACD: ±1, no neutral.
    score += 1 if row["macd"] > row["macd_signal"] else -1

    return score


# ----- Strategy 2: RANGE / MEAN REVERSION ------------------------------------

def range_score(row: pd.Series) -> int | float:
    """
    Z-score extremity (±2 or ±1), BB (±1), RSI (±1), Stoch %K (±1).
    Asymmetric trend filter sets signal to HOLD (via score=0) when we'd fight
    a strong trend.
    """
    required = ["zscore", "bb_upper", "bb_lower", "rsi14", "stoch_k",
                "adx14", "sma20", "sma50", "close"]
    if any(pd.isna(row[c]) for c in required):
        return np.nan

    score = 0
    z = row["zscore"]
    if z <= -2:
        score += 2
    elif z <= -1:
        score += 1
    elif z >= 2:
        score -= 2
    elif z >= 1:
        score -= 1

    if row["close"] < row["bb_lower"]:
        score += 1
    elif row["close"] > row["bb_upper"]:
        score -= 1

    if row["rsi14"] < 30:
        score += 1
    elif row["rsi14"] > 70:
        score -= 1

    if row["stoch_k"] < 20:
        score += 1
    elif row["stoch_k"] > 80:
        score -= 1

    # Asymmetric trend filter: zero the score if it opposes a strong trend.
    if row["adx14"] > 25:
        if row["sma20"] > row["sma50"] and score < 0:
            return 0
        if row["sma20"] < row["sma50"] and score > 0:
            return 0

    return score


# ----- Strategy 3: REVERSAL --------------------------------------------------

def reversal_score_row(i: int, df: pd.DataFrame) -> int | float:
    """
    Excel formula: (NearLow - NearHigh) + RSIDivergence + VolExhaust
                   + MACDTurn + StochHook

    Fixed vs my original:
      - NearLow / NearHigh weighted ±1 each (not ±2), so no single factor dominates.
      - VolExhaust uses LOW volume (< 0.8 x avg20) at an extreme — dried-up
        participation is the exhaustion signal, not a climactic spike.
      - MACDTurn is a 2-bar rule: histogram rose (or fell) while still on the
        wrong side of zero.
      - StochHook requires K AND D both in the extreme zone (30/70 thresholds),
        plus K crossing D.
    """
    if i < 60:
        return np.nan
    row = df.iloc[i]
    if any(pd.isna(row[c]) for c in ["rsi14", "macd_hist", "stoch_k", "stoch_d",
                                      "close", "volume"]):
        return np.nan

    score = 0
    close = row["close"]

    # Near 60-day extremes (±1 each per Excel, measured on raw L/H).
    low60 = df["low"].iloc[i - 59:i + 1].min()
    high60 = df["high"].iloc[i - 59:i + 1].max()
    near_low = close <= low60 * 1.02 if pd.notna(low60) else False
    near_high = close >= high60 * 0.98 if pd.notna(high60) else False
    if near_low:
        score += 1
    if near_high:
        score -= 1

    # RSI divergence within the 20-bar window. Compare RSI at the prior extreme
    # (20 bars ago) to RSI now.
    if i >= 20:
        past_rsi = df["rsi14"].iloc[i - 20]
        window20_low = df["close"].iloc[i - 19:i + 1].min()
        window20_high = df["close"].iloc[i - 19:i + 1].max()
        if pd.notna(past_rsi):
            if close <= window20_low and row["rsi14"] > past_rsi:
                score += 1
            elif close >= window20_high and row["rsi14"] < past_rsi:
                score -= 1

    # Volume exhaustion (LOW vol at extreme → exhaustion).
    avg_vol_20 = df["volume"].iloc[max(0, i - 19):i + 1].mean()
    if pd.notna(avg_vol_20) and row["volume"] < 0.8 * avg_vol_20:
        if near_low:
            score += 1
        if near_high:
            score -= 1

    # MACD histogram turn — 2-bar rule.
    if i >= 1:
        h0 = df["macd_hist"].iloc[i]
        h1 = df["macd_hist"].iloc[i - 1]
        if pd.notna(h0) and pd.notna(h1):
            if h0 > h1 and h0 < 0:
                score += 1
            elif h0 < h1 and h0 > 0:
                score -= 1

    # Stochastic hook — K crosses D in the extreme zone (both K and D in zone).
    if i >= 1:
        k0 = df["stoch_k"].iloc[i]
        d0 = df["stoch_d"].iloc[i]
        k1 = df["stoch_k"].iloc[i - 1]
        d1 = df["stoch_d"].iloc[i - 1]
        if all(pd.notna(x) for x in (k0, d0, k1, d1)):
            if k0 > d0 and k1 <= d1 and k0 < 30 and d0 < 30:
                score += 1
            elif k0 < d0 and k1 >= d1 and k0 > 70 and d0 > 70:
                score -= 1

    return score


# ----- Strategy 4: MOMENTUM --------------------------------------------------

def momentum_score_row(i: int, df: pd.DataFrame) -> int | float:
    """
    Excel: trendPt + bouncePt + volPt + stochPt + adxPt.

    Fixed vs my original:
      - 'Bounce up' = close > SMA100 AND in last 5 bars a LOW came within 2%
        of SMA100. Catches near-misses, not just actual penetrations.
      - 'Reject down' = close < SMA100 AND a HIGH within 2% of SMA100 in last 5.
      - stochPt gates on trend alignment (bull turn only counts in an uptrend).
    """
    if i < 100:
        return np.nan
    row = df.iloc[i]
    required = ["sma100", "close", "atr14", "stoch_k", "stoch_d",
                "adx14", "high", "low"]
    if any(pd.isna(row[c]) for c in required):
        return np.nan

    score = 0
    up = row["close"] > row["sma100"]
    dn = row["close"] < row["sma100"]

    # Trend vs SMA100.
    if up:
        score += 1
    elif dn:
        score -= 1

    # Bounce: price tagged within 2% of SMA100 in last 5 bars.
    look = df.iloc[max(0, i - 4):i + 1]
    sma100_now = row["sma100"]
    if up:
        tagged = (look["low"] <= sma100_now * 1.02).any()
        if tagged:
            score += 2
    elif dn:
        tagged = (look["high"] >= sma100_now * 0.98).any()
        if tagged:
            score -= 2

    # Volatility contraction aligned with trend.
    if i >= 20:
        atr_now = row["atr14"]
        atr_then = df["atr14"].iloc[i - 20]
        if pd.notna(atr_now) and pd.notna(atr_then) and atr_now < atr_then:
            if up:
                score += 1
            elif dn:
                score -= 1

    # Stochastic turn aligned with trend.
    if i >= 1:
        k0 = df["stoch_k"].iloc[i]
        d0 = df["stoch_d"].iloc[i]
        k1 = df["stoch_k"].iloc[i - 1]
        d1 = df["stoch_d"].iloc[i - 1]
        if all(pd.notna(x) for x in (k0, d0, k1, d1)):
            bull_turn = k0 > d0 and k1 <= d1 and k1 < 50
            bear_turn = k0 < d0 and k1 >= d1 and k1 > 50
            if bull_turn and up:
                score += 1
            elif bear_turn and dn:
                score -= 1

    # ADX gate.
    if row["adx14"] > 20:
        if up:
            score += 1
        elif dn:
            score -= 1

    return score
