"""
End-to-end pipeline:

  1. Ensure OHLCV cache per ticker (backfill if needed).
  2. Append today's bulk day.
  3. Compute indicators per ticker.
  4. Run primary + secondary strategies.
  5. Apply PSAR alignment filter.
  6. Expand entries into persistent BUY/SELL/None state.
  7. Compute per-ticker success rate + current position strength.
  8. Write signals.json (home-screen payload) and tickers/{T}.json.

Schema (v2.0.0) output fields:

signals.json:
  {
    "schema_version": "2.0.0",
    "generated_at": "...Z",
    "trade_date": "YYYY-MM-DD",
    "universe_size": N,
    "strategy_success_rate_52w": 0.68 or null,
    "strategy_trade_count_52w": 2847,
    "signals": [
      {
        "ticker": "AAPL",
        "close": 273.17,
        "change_pct": 0.09,
        "signal": "BUY" | "SELL" | null,
        "signal_source": "primary" | "secondary" | null,
        "signal_started_date": "YYYY-MM-DD" | null,
        "signal_trading_days": 5 | null,
        "position_strength": "Strong" | "Moderate" | "Close to flipping" | null,
        "ticker_success_rate_52w": 0.73 | null,
        "ticker_trade_count_52w": 8
      },
      ...
    ]
  }

tickers/AAPL.json:
  {
    "ticker": "AAPL",
    "schema_version": "2.0.0",
    "current": {...same as above minus ticker...},
    "success_rate": {"success_rate": 0.73, "trade_count": 8},
    "history": [
      {
        "date": "YYYY-MM-DD",
        "close": 273.17,
        "volume": 50000000,
        "signal": "BUY"|"SELL"|null,
        "psar": 245.3
      },
      ...
    ]
  }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import consensus, fetch, indicators as ind, strategies as strat


SCHEMA_VERSION = "2.0.0"

# Match the PHP: 260 trading days = 52 weeks of trading.
SUCCESS_WINDOW_BARS = 260
# Need enough bars to let pivots (±15+4) and PSAR warm up. The bars BEFORE the
# 260-window are used purely as warm-up so the 260 bars all have valid
# indicators; we only evaluate signals on the 260-window itself.
WARMUP_BARS = 60


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute only what the new strategy needs:
      - MACD line (12/26/9) on adjusted close
      - 14,3,3 slow stochastic on raw H/L/C
      - PSAR (Wilder defaults) on raw H/L

    The strategy layer also uses `high`, `low`, `close` — these come straight
    from the input df and are kept untouched.
    """
    df = df.copy()
    adj = df["adj_close"]

    # MACD on adjusted close.
    macd_line, macd_signal, _ = ind.macd(adj, 12, 26, 9)
    df["macd"] = macd_line
    # (macd_signal + hist are computed but unused by the pivot strategy; drop.)

    # 14,3,3 slow stochastic on raw H/L/C.
    df["stoch_k"] = ind.stoch_slow_k(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_d"] = ind.stoch_d(df["stoch_k"], 3)

    # Parabolic SAR on raw H/L.
    df["psar"] = ind.parabolic_sar(df["high"], df["low"])

    return df


def _safe_num(x):
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return round(float(x), 4)


def _latest_change_pct(df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    latest_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    if pd.isna(latest_close) or pd.isna(prev_close) or prev_close == 0:
        return None
    return round(float((latest_close / prev_close - 1.0) * 100.0), 4)


def process_ticker(ticker: str, df: pd.DataFrame) -> tuple[dict, dict, list[dict]] | None:
    """
    Run the strategy for one ticker.

    Returns (summary, history_payload, trades). summary is a row for signals.json;
    history_payload is the per-ticker file; trades is the list of completed
    trades in the 52w window (used to aggregate strategy-wide success rate).

    Returns None if the ticker has too little history.
    """
    if df.empty or len(df) < WARMUP_BARS + 20:
        return None

    # Compute indicators on the full cached history (necessary for MACD/PSAR
    # to warm up), then trim to the success-rate window for signal evaluation.
    df = df.sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    # Evaluate signals on the whole series (cheap), but we'll slice to the
    # 260-bar window for state/trade computations. This ensures any "recent"
    # signal still sees the full indicator warm-up behind it.
    df_full = df.copy()

    # Strategy layer expects columns: date, high, low, close, macd, stoch_k, stoch_d, psar.
    primary = strat.compute_primary_signals(df_full)
    primary = strat.apply_psar_alignment_filter(df_full, primary)
    secondary = strat.compute_trending_stocks_signals(df_full)

    # Slice to the 52-week window for state expansion + success rate. If we
    # have less than SUCCESS_WINDOW_BARS available, use everything we have.
    window = df_full.tail(SUCCESS_WINDOW_BARS).reset_index(drop=True)

    # Filter the entry dicts down to only dates in the window.
    window_dates = set(window["date"].astype(str).str[:10])
    primary_win = {d: v for d, v in primary.items() if d in window_dates}
    secondary_win = {d: v for d, v in secondary.items() if d in window_dates}

    result = consensus.process_ticker_signals(window, primary_win, secondary_win)
    if not result["active_status"]:
        return None

    current = result["current"]
    sr = result["success_rate"]

    latest_date_str = result["active_status"][-1]["date"]
    summary = {
        "ticker": ticker,
        "close": _safe_num(result["active_status"][-1]["close"]),
        "change_pct": _latest_change_pct(window),
        "signal": current["signal"],
        "signal_source": current["signal_source"],
        "signal_started_date": current["signal_started_date"],
        "signal_trading_days": current["signal_trading_days"],
        "position_strength": current["position_strength"],
        "ticker_success_rate_52w": sr["success_rate"],
        "ticker_trade_count_52w": sr["trade_count"],
    }

    # Per-ticker history payload is compact — only fields the app renders.
    hist_rows = []
    # Build a lookup from date -> volume from the window frame.
    vol_by_date = {
        str(d)[:10]: v
        for d, v in zip(window["date"], window["volume"])
    }

    for row in result["active_status"]:
        hist_rows.append({
            "date": row["date"],
            "close": _safe_num(row["close"]),
            "volume": int(vol_by_date[row["date"]])
                if row["date"] in vol_by_date and pd.notna(vol_by_date[row["date"]])
                else None,
            "signal": row["signal"],
            "psar": _safe_num(row["psar"]),
        })

    history_payload = {
        "ticker": ticker,
        "schema_version": SCHEMA_VERSION,
        "current": {
            "signal": current["signal"],
            "signal_source": current["signal_source"],
            "signal_started_date": current["signal_started_date"],
            "signal_trading_days": current["signal_trading_days"],
            "position_strength": current["position_strength"],
        },
        "success_rate": sr,
        "history": hist_rows,
    }

    return summary, history_payload, result["trades"]


def run(tickers: list[str], cache_dir: Path, out_dir: Path,
        do_backfill: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tickers").mkdir(parents=True, exist_ok=True)

    if do_backfill:
        print(f"Backfilling {len(tickers)} tickers...")
        failures = fetch.backfill_universe(tickers, cache_dir)
        if failures:
            print(f"  {len(failures)} failures: {failures[:10]}...")

    print("Fetching bulk last-day data...")
    try:
        bulk = fetch.fetch_bulk_last_day()
        fetch.append_bulk_day(bulk, set(tickers), cache_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[bulk] failed: {e} — proceeding with cached data")

    summaries = []
    all_trades: list[dict] = []
    for t in tickers:
        df = fetch.load_cache(t, cache_dir)
        result = process_ticker(t, df)
        if result is None:
            continue
        summary, history, trades = result
        summaries.append(summary)
        all_trades.extend(trades)
        with open(out_dir / "tickers" / f"{t}.json", "w") as f:
            json.dump(history, f, separators=(",", ":"))

    # Strategy-wide success rate aggregated across all tickers' trades.
    agg = consensus.aggregate_strategy_success_rate(all_trades)

    trade_date = None
    if summaries:
        any_ticker = summaries[0]["ticker"]
        df = fetch.load_cache(any_ticker, cache_dir)
        if not df.empty:
            trade_date = df["date"].max().strftime("%Y-%m-%d")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trade_date": trade_date,
        "universe_size": len(summaries),
        "strategy_success_rate_52w": agg["success_rate"],
        "strategy_trade_count_52w": agg["trade_count"],
        "signals": summaries,
    }
    with open(out_dir / "signals.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    buy_count = sum(1 for s in summaries if s["signal"] == "BUY")
    sell_count = sum(1 for s in summaries if s["signal"] == "SELL")
    none_count = sum(1 for s in summaries if s["signal"] is None)
    print(f"Done. {len(summaries)} tickers processed. "
          f"BUY={buy_count} SELL={sell_count} CASH={none_count}. "
          f"Strategy 52w success: {agg['success_rate']} across {agg['trade_count']} trades.")