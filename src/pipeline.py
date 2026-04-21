"""
End-to-end pipeline:
  1. Load cached OHLCV per ticker (or backfill if missing).
  2. Apply the bulk-day append.
  3. Compute indicators per ticker on adjusted close + raw H/L/C.
  4. Score each of the four strategies for every bar.
  5. Combine into a final consensus signal.
  6. Write signals.json (latest row per ticker) and tickers/{T}.json (history).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import consensus, fetch, indicators as ind, strategies as strat


SCHEMA_VERSION = "1.0.0"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach every indicator column the strategies need.

    Price basis convention (matches the Excel reference):
      - Smoothing / momentum indicators (SMA, EMA, RSI, MACD, BB bands themselves,
        Z-score baseline) use ADJUSTED close for dividend stability.
      - "Chart-visible" comparisons (BB position: is today's close above the upper
        band? Z-score: how far is today's close from SMA50?) use RAW close, because
        those are the prices a human sees on their chart.
      - H/L/C indicators (ADX, Stoch, ATR) use RAW as per spec.

    This deliberately mixes two price series for BB-position and Z-score. It is
    the convention the reference workbook uses; matching it keeps the pipeline's
    output in parity with that workbook. For tickers that have paid dividends,
    raw close > adjusted close, so BB/Z-score signals fire slightly more often
    than a purely-adjusted implementation would show.
    """
    df = df.copy()

    adj = df["adj_close"]
    raw = df["close"]

    # Adjusted-close smoothers.
    df["sma20"] = ind.sma(adj, 20)
    df["sma50"] = ind.sma(adj, 50)
    df["sma100"] = ind.sma(adj, 100)
    df["sma200"] = ind.sma(adj, 200)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = ind.bollinger(adj, 20, 2)
    df["rsi14"] = ind.rsi(adj, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = ind.macd(adj, 12, 26, 9)

    # Z-score: Dev = (raw_close - SMA50_of_adj) / SMA50_of_adj. Mixes series.
    df["zscore"] = ind.zscore_dev(raw, df["sma50"], 60)

    # Raw H/L/C indicators.
    df["stoch_k"] = ind.stoch_k(df["high"], df["low"], df["close"], 14)
    df["stoch_d"] = ind.stoch_d(df["stoch_k"], 3)
    df["adx14"], df["plus_di"], df["minus_di"] = ind.adx(
        df["high"], df["low"], df["close"], 14
    )
    df["atr14"] = ind.atr(df["high"], df["low"], df["close"], 14)

    # Strategies see `close` = RAW close for BB-position checks, consistent with
    # the Excel. Preserve adjusted close separately for anything that needs it.
    df["close_adj"] = adj
    # close stays as raw (already in the frame)

    return df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-bar strategy scores, per-bar signals, and the final signal."""
    df = df.copy()

    # Trend + Range are per-row; Reversal + Momentum need lookbacks.
    df["score_trend"] = df.apply(strat.trend_score, axis=1)
    df["score_range"] = df.apply(strat.range_score, axis=1)

    rev = []
    mom = []
    for i in range(len(df)):
        rev.append(strat.reversal_score_row(i, df))
        mom.append(strat.momentum_score_row(i, df))
    df["score_reversal"] = rev
    df["score_momentum"] = mom

    for name in ("trend", "range", "reversal", "momentum"):
        df[f"sig_{name}"] = df[f"score_{name}"].apply(strat.signal_from_score)

    def _consensus(row):
        return consensus.final_signal({
            "trend": row["sig_trend"],
            "range": row["sig_range"],
            "reversal": row["sig_reversal"],
            "momentum": row["sig_momentum"],
        })

    df["final_signal"] = df.apply(_consensus, axis=1)
    return df


def _safe_num(x):
    """JSON-friendly number: None for NaN/inf, rounded float otherwise."""
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return round(float(x), 4)


def _row_to_history(row: pd.Series) -> dict:
    """One row of per-ticker history, as it goes into tickers/{T}.json."""
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "close": _safe_num(row["close"]),
        "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
        "strategies": {
            "trend": row["sig_trend"],
            "range": row["sig_range"],
            "reversal": row["sig_reversal"],
            "momentum": row["sig_momentum"],
        },
        "scores": {
            "trend": _safe_num(row["score_trend"]),
            "range": _safe_num(row["score_range"]),
            "reversal": _safe_num(row["score_reversal"]),
            "momentum": _safe_num(row["score_momentum"]),
        },
        "final_signal": row["final_signal"],
        "indicators": {
            "sma20": _safe_num(row["sma20"]),
            "sma50": _safe_num(row["sma50"]),
            "sma100": _safe_num(row["sma100"]),
            "sma200": _safe_num(row["sma200"]),
            "bb_upper": _safe_num(row["bb_upper"]),
            "bb_lower": _safe_num(row["bb_lower"]),
            "rsi14": _safe_num(row["rsi14"]),
            "macd": _safe_num(row["macd"]),
            "macd_signal": _safe_num(row["macd_signal"]),
            "macd_hist": _safe_num(row["macd_hist"]),
            "adx14": _safe_num(row["adx14"]),
            "atr14": _safe_num(row["atr14"]),
            "stoch_k": _safe_num(row["stoch_k"]),
            "stoch_d": _safe_num(row["stoch_d"]),
            "zscore": _safe_num(row["zscore"]),
        },
    }


def process_ticker(ticker: str, df: pd.DataFrame) -> tuple[dict, dict] | None:
    """
    Returns (summary, history). `summary` is one row for signals.json,
    `history` is the full JSON blob for tickers/{T}.json. Returns None if the
    ticker has insufficient data (e.g. newly listed, <200 bars).
    """
    if df.empty or len(df) < 100:
        return None
    df = compute_indicators(df)
    df = compute_signals(df)

    # Only emit rows where final signal is defined (needs SMA100 at minimum).
    df = df.dropna(subset=["score_momentum"])
    if df.empty:
        return None

    latest = df.iloc[-1]
    if len(df) >= 2:
        prev_close = df["close"].iloc[-2]
        change_pct = (latest["close"] / prev_close - 1) * 100 if prev_close else None
    else:
        change_pct = None

    summary = {
        "ticker": ticker,
        "close": _safe_num(latest["close"]),
        "change_pct": _safe_num(change_pct),
        "final_signal": latest["final_signal"],
        "strategies": {
            "trend": latest["sig_trend"],
            "range": latest["sig_range"],
            "reversal": latest["sig_reversal"],
            "momentum": latest["sig_momentum"],
        },
        "scores": {
            "trend": _safe_num(latest["score_trend"]),
            "range": _safe_num(latest["score_range"]),
            "reversal": _safe_num(latest["score_reversal"]),
            "momentum": _safe_num(latest["score_momentum"]),
        },
    }

    history = {
        "ticker": ticker,
        "schema_version": SCHEMA_VERSION,
        "history": [_row_to_history(r) for _, r in df.iterrows()],
    }
    return summary, history


def run(tickers: list[str], cache_dir: Path, out_dir: Path,
        do_backfill: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tickers").mkdir(parents=True, exist_ok=True)

    # Step 1: ensure cache exists.
    if do_backfill:
        print(f"Backfilling {len(tickers)} tickers...")
        failures = fetch.backfill_universe(tickers, cache_dir)
        if failures:
            print(f"  {len(failures)} failures: {failures[:10]}...")

    # Step 2: bulk append today's bar to every ticker.
    print("Fetching bulk last-day data...")
    try:
        bulk = fetch.fetch_bulk_last_day()
        fetch.append_bulk_day(bulk, set(tickers), cache_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[bulk] failed: {e} — proceeding with cached data")

    # Step 3: process each ticker.
    summaries = []
    for t in tickers:
        df = fetch.load_cache(t, cache_dir)
        result = process_ticker(t, df)
        if result is None:
            continue
        summary, history = result
        summaries.append(summary)
        with open(out_dir / "tickers" / f"{t}.json", "w") as f:
            json.dump(history, f, separators=(",", ":"))

    # Step 4: write home-screen payload.
    trade_date = None
    if summaries:
        # Find the latest trade date from the first ticker's cache.
        any_ticker = next(iter(summaries))["ticker"]
        df = fetch.load_cache(any_ticker, cache_dir)
        if not df.empty:
            trade_date = df["date"].max().strftime("%Y-%m-%d")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trade_date": trade_date,
        "universe_size": len(summaries),
        "signals": summaries,
    }
    with open(out_dir / "signals.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Done. {len(summaries)} tickers processed.")
