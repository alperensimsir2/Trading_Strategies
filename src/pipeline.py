"""
End-to-end pipeline.

Schema v2.1.0 changes from v2.0.0:
  - Per-ticker entries gain: ticker_avg_hold_days, ticker_avg_gain_pct,
    ticker_avg_peak_gain_pct
  - signals.json wrapper gains: strategy_avg_hold_days, strategy_avg_gain_pct,
    strategy_avg_peak_gain_pct
  - ticker_success_rate_52w and ticker_trade_count_52w retained for backward
    compat (success_rate now informational only)

Outputs:

signals.json:
  {
    "schema_version": "2.1.0",
    "generated_at": "...Z",
    "trade_date": "YYYY-MM-DD",
    "universe_size": N,
    "strategy_trade_count_52w": int,
    "strategy_avg_hold_days": float | null,
    "strategy_avg_gain_pct": float | null,
    "strategy_avg_peak_gain_pct": float | null,
    "strategy_success_rate_52w": float | null,
    "signals": [
      {
        "ticker": "AAPL",
        "close": 273.17,
        "change_pct": 0.09,
        "signal": "BUY" | "SELL" | null,
        "signal_source": "primary" | "secondary" | null,
        "signal_started_date": "YYYY-MM-DD" | null,
        "signal_trading_days": int | null,
        "position_strength": "Strong" | "Moderate" | "Close to flipping" | null,
        "ticker_trade_count_52w": int,
        "ticker_avg_hold_days": float | null,
        "ticker_avg_gain_pct": float | null,
        "ticker_avg_peak_gain_pct": float | null,
        "ticker_success_rate_52w": float | null
      }
    ]
  }

tickers/{T}.json:
  {
    "ticker": "AAPL",
    "schema_version": "2.1.0",
    "current": {...},
    "metrics": {
      "trade_count": int,
      "avg_hold_days": float | null,
      "avg_gain_pct": float | null,
      "avg_peak_gain_pct": float | null,
      "success_rate": float | null
    },
    "history": [
      {"date": "YYYY-MM-DD", "close": ..., "volume": ..., "signal": ..., "psar": ...},
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


SCHEMA_VERSION = "2.1.0"
SUCCESS_WINDOW_BARS = 260
WARMUP_BARS = 60


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    adj = df["adj_close"]
    macd_line, _, _ = ind.macd(adj, 12, 26, 9)
    df["macd"] = macd_line
    df["stoch_k"] = ind.stoch_slow_k(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_d"] = ind.stoch_d(df["stoch_k"], 3)
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
    if df.empty or len(df) < WARMUP_BARS + 20:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    df_full = df.copy()
    primary = strat.compute_primary_signals(df_full)
    primary = strat.apply_psar_alignment_filter(df_full, primary)
    secondary = strat.compute_trending_stocks_signals(df_full)

    window = df_full.tail(SUCCESS_WINDOW_BARS).reset_index(drop=True)
    window_dates = set(window["date"].astype(str).str[:10])
    primary_win = {d: v for d, v in primary.items() if d in window_dates}
    secondary_win = {d: v for d, v in secondary.items() if d in window_dates}

    result = consensus.process_ticker_signals(window, primary_win, secondary_win)
    if not result["active_status"]:
        return None

    current = result["current"]
    metrics = result["metrics"]

    # If there's an active trade, pull its peak gain. The active trade is the
    # last entry in result["trades"] when is_active is True.
    active_peak_gain_pct = None
    if result["trades"] and result["trades"][-1].get("is_active"):
        active_peak_gain_pct = result["trades"][-1]["peak_gain_pct"]

    summary = {
        "ticker": ticker,
        "close": _safe_num(result["active_status"][-1]["close"]),
        "change_pct": _latest_change_pct(window),
        "signal": current["signal"],
        "signal_source": current["signal_source"],
        "signal_started_date": current["signal_started_date"],
        "signal_trading_days": current["signal_trading_days"],
        "position_strength": current["position_strength"],
        "active_peak_gain_pct": active_peak_gain_pct,
        "ticker_trade_count_52w": metrics["trade_count"],
        "ticker_avg_hold_days": metrics["avg_hold_days"],
        "ticker_avg_gain_pct": metrics["avg_gain_pct"],
        "ticker_avg_peak_gain_pct": metrics["avg_peak_gain_pct"],
        "ticker_success_rate_52w": metrics["success_rate"],
    }

    # Per-ticker history payload — minimal fields the app actually displays.
    vol_by_date = {
        str(d)[:10]: v for d, v in zip(window["date"], window["volume"])
    }
    hist_rows = []
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
            "peak_gain_pct": active_peak_gain_pct,
        },
        "metrics": {
            "trade_count": metrics["trade_count"],
            "avg_hold_days": metrics["avg_hold_days"],
            "avg_gain_pct": metrics["avg_gain_pct"],
            "avg_peak_gain_pct": metrics["avg_peak_gain_pct"],
            "success_rate": metrics["success_rate"],
        },
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

    agg = consensus.aggregate_strategy_metrics(all_trades)

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
        "strategy_trade_count_52w": agg["trade_count"],
        "strategy_avg_hold_days": agg["avg_hold_days"],
        "strategy_avg_gain_pct": agg["avg_gain_pct"],
        "strategy_avg_peak_gain_pct": agg["avg_peak_gain_pct"],
        "strategy_success_rate_52w": agg["success_rate"],
        "signals": summaries,
    }
    with open(out_dir / "signals.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    buy_count = sum(1 for s in summaries if s["signal"] == "BUY")
    sell_count = sum(1 for s in summaries if s["signal"] == "SELL")
    none_count = sum(1 for s in summaries if s["signal"] is None)
    print(f"Done. {len(summaries)} tickers processed. "
          f"BUY={buy_count} SELL={sell_count} CASH={none_count}.")
    print(f"Strategy 52w: {agg['trade_count']} trades, "
          f"avg hold {agg['avg_hold_days']}d, "
          f"avg gain {agg['avg_gain_pct']:+.2f}%, "
          f"avg peak gain {agg['avg_peak_gain_pct']:+.2f}%, "
          f"success {agg['success_rate']}.")