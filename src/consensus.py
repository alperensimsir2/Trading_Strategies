"""
Merge + state expansion + performance metrics.

This module used to compute cross-strategy "consensus" from four independent
strategies. The new model is different — there's a primary pivot-based
strategy and a secondary snap-back strategy. Both produce entry signals
(BUY/SELL on specific days). This module:

  1. Merges primary and secondary entries (secondary wins on same-day conflict).
  2. Expands entry events into a persistent active state per day: BUY, SELL,
     or None. Positions persist until PSAR flips against the trade (same exit
     rule as the reference PHP).
  3. Computes per-ticker success rate over the last 52 trading weeks (260 bars).
  4. Classifies current position strength based on distance from PSAR.

All of this is pure data transformation — no side effects, no I/O.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ===== Merge primary + secondary entries ====================================

def merge_entries(primary: dict[str, str],
                  secondary: dict[str, str]) -> dict[str, tuple[str, str]]:
    """
    Merge two signal dicts into one, keyed by date.

    Values are (side, source): side is 'BUY' or 'SELL', source is
    'primary' or 'secondary'.

    On same-day conflict, secondary wins. (This matches the PHP: trend
    continuation shouldn't be obscured by a coincident trend reversal.)
    """
    merged: dict[str, tuple[str, str]] = {}
    for date, side in primary.items():
        merged[date] = (side, "primary")
    for date, side in secondary.items():
        merged[date] = (side, "secondary")  # overwrites any primary on same day
    return merged


# ===== Expand entries to persistent BUY/SELL/None state ====================

def expand_signals_to_active_status(
    df_asc: pd.DataFrame,
    entry_signals: dict[str, tuple[str, str]],
) -> list[dict]:
    """
    Walk the ASC-ordered bars. At each bar:
      - If today is an entry day: set current state to the entry side
        (BUY or SELL), and remember the entry date + source.
      - Else if in BUY and close < PSAR: exit to cash (state = None).
      - Else if in SELL and close > PSAR: exit to cash (state = None).
      - Else: hold current state.

    Returns a list of dicts, one per bar, aligned with df_asc:
      {
        'date': 'YYYY-MM-DD',
        'close': float or None,
        'psar': float or None,
        'signal': 'BUY'|'SELL'|None,
        'signal_source': 'primary'|'secondary'|None,
        'signal_started_date': 'YYYY-MM-DD' or None,
      }

    The None values in `signal` represent "cash" days — not an active position.
    """
    from .strategies import _date_to_str  # local import to avoid cycles
    out: list[dict] = []

    state: Optional[str] = None
    source: Optional[str] = None
    started: Optional[str] = None

    for _, row in df_asc.iterrows():
        date = _date_to_str(row.get("date"))
        if date is None:
            continue

        close = row.get("close")
        psar = row.get("psar")
        close_v = float(close) if pd.notna(close) else None
        psar_v = float(psar) if pd.notna(psar) else None

        # Entry (including flip).
        if date in entry_signals:
            side, src = entry_signals[date]
            # If we're already in the same side, don't reset started/source —
            # this preserves "signal has been active since X" through re-entries.
            if state != side:
                state = side
                source = src
                started = date
        # Exit check (PSAR flip against position).
        elif state == "BUY" and close_v is not None and psar_v is not None and close_v < psar_v:
            state = None
            source = None
            started = None
        elif state == "SELL" and close_v is not None and psar_v is not None and close_v > psar_v:
            state = None
            source = None
            started = None

        out.append({
            "date": date,
            "close": close_v,
            "psar": psar_v,
            "signal": state,
            "signal_source": source,
            "signal_started_date": started,
        })

    return out


# ===== Trade reconstruction + success rate ==================================

def _extract_trades(active_status: list[dict]) -> list[dict]:
    """
    Walk the per-bar active status and extract completed trades.

    A trade runs from the day it entered (signal changed from None/flip) to
    the day it exited (signal went back to None due to PSAR flip). A trade
    where the side flips (BUY -> SELL or vice versa) produces TWO trades:
    the old position closed at the flip date, and a new one opened there.

    Returns list of dicts:
      {
        'entry_date': str,
        'entry_close': float,
        'exit_date': str,
        'exit_close': float,
        'side': 'BUY'|'SELL',
        'source': 'primary'|'secondary',
        'profitable': bool,  # BUY: exit>entry; SELL: exit<entry
        'pnl_pct': float,    # signed % return (positive = profitable for both sides)
      }

    Open (unfilled) trades at the end of the series are NOT included — we only
    count completed trades where we know the outcome.
    """
    trades: list[dict] = []
    prev_side: Optional[str] = None
    prev_source: Optional[str] = None
    entry_date: Optional[str] = None
    entry_close: Optional[float] = None

    for bar in active_status:
        side = bar["signal"]
        close = bar["close"]

        if side != prev_side:
            # Position change. Close out previous (if any).
            if prev_side in ("BUY", "SELL") and entry_date is not None and entry_close is not None and close is not None:
                if prev_side == "BUY":
                    pnl_pct = (close - entry_close) / entry_close * 100.0
                    profitable = close > entry_close
                else:  # SELL
                    pnl_pct = (entry_close - close) / entry_close * 100.0
                    profitable = close < entry_close
                trades.append({
                    "entry_date": entry_date,
                    "entry_close": entry_close,
                    "exit_date": bar["date"],
                    "exit_close": close,
                    "side": prev_side,
                    "source": prev_source or "primary",
                    "profitable": bool(profitable),
                    "pnl_pct": round(pnl_pct, 4),
                })
            # Open new (if we entered a position, not just exited to cash).
            if side in ("BUY", "SELL") and close is not None:
                entry_date = bar["date"]
                entry_close = close
                prev_source = bar.get("signal_source")
            else:
                entry_date = None
                entry_close = None
                prev_source = None
            prev_side = side

    return trades


def compute_ticker_success_rate(active_status: list[dict],
                                min_trades: int = 5) -> dict:
    """
    Compute success rate over the full active_status window (which should be
    the last 260 bars = 52 trading weeks).

    Returns:
      {
        'success_rate': float or None,  # 0.0-1.0, None if not enough trades
        'trade_count': int,
      }

    When trade_count < min_trades, success_rate is None (insufficient data).
    """
    trades = _extract_trades(active_status)
    count = len(trades)
    if count == 0:
        return {"success_rate": None, "trade_count": 0}
    if count < min_trades:
        return {"success_rate": None, "trade_count": count}

    wins = sum(1 for t in trades if t["profitable"])
    rate = wins / count
    return {"success_rate": round(rate, 4), "trade_count": count}


def aggregate_strategy_success_rate(all_trades: list[dict]) -> dict:
    """
    Compute strategy-wide success rate across trades from all tickers.

    Returns the same shape as compute_ticker_success_rate (success_rate is
    None if no trades at all).
    """
    count = len(all_trades)
    if count == 0:
        return {"success_rate": None, "trade_count": 0}
    wins = sum(1 for t in all_trades if t["profitable"])
    return {"success_rate": round(wins / count, 4), "trade_count": count}


# ===== Position strength classification =====================================

def classify_position_strength(close: Optional[float],
                               psar: Optional[float],
                               signal: Optional[str]) -> Optional[str]:
    """
    Classify how "safe" the current position is based on distance from PSAR.

    Buckets:
      - 'Strong'            : price is > 5% away from PSAR (lots of room)
      - 'Moderate'          : price is 2% to 5% away
      - 'Close to flipping' : price is < 2% away

    When signal is None (in cash), returns None.
    """
    if signal not in ("BUY", "SELL"):
        return None
    if close is None or psar is None or psar == 0:
        return None

    dist_pct = abs(close - psar) / abs(psar) * 100.0

    if dist_pct >= 5.0:
        return "Strong"
    if dist_pct >= 2.0:
        return "Moderate"
    return "Close to flipping"


def days_since(start_date_str: Optional[str],
               current_date_str: Optional[str]) -> Optional[int]:
    """
    Count trading days from start to current (inclusive of both ends → +1).

    Uses simple calendar-day arithmetic then divides by (7/5) as an approximation.
    This is close enough for UI display; for exactness, the caller should pass
    a pre-computed trading-days count.
    """
    if start_date_str is None or current_date_str is None:
        return None
    try:
        a = pd.Timestamp(start_date_str)
        b = pd.Timestamp(current_date_str)
    except Exception:
        return None
    delta_days = (b - a).days
    if delta_days < 0:
        return None
    # Approximate trading days = calendar days * 5/7.
    return int(round(delta_days * 5 / 7))


# ===== Convenience: run everything for one ticker ==========================

def process_ticker_signals(df_asc: pd.DataFrame,
                            primary_entries: dict[str, str],
                            secondary_entries: dict[str, str]) -> dict:
    """
    One-stop call combining merge + expansion + success rate + current
    position classification.

    Returns:
      {
        'active_status': [ {date, close, psar, signal, ...}, ... ],  # full history
        'trades': [ {entry_date, exit_date, side, ...}, ... ],
        'current': {
          'signal': 'BUY'|'SELL'|None,
          'signal_source': 'primary'|'secondary'|None,
          'signal_started_date': str or None,
          'signal_trading_days': int or None,
          'position_strength': 'Strong'|'Moderate'|'Close to flipping'|None,
        },
        'success_rate': {'success_rate': float|None, 'trade_count': int},
      }
    """
    merged = merge_entries(primary_entries, secondary_entries)
    active = expand_signals_to_active_status(df_asc, merged)

    if not active:
        return {
            "active_status": [],
            "trades": [],
            "current": {
                "signal": None,
                "signal_source": None,
                "signal_started_date": None,
                "signal_trading_days": None,
                "position_strength": None,
            },
            "success_rate": {"success_rate": None, "trade_count": 0},
        }

    latest = active[-1]
    trades = _extract_trades(active)
    sr = compute_ticker_success_rate(active)

    current = {
        "signal": latest["signal"],
        "signal_source": latest["signal_source"],
        "signal_started_date": latest["signal_started_date"],
        "signal_trading_days": days_since(latest["signal_started_date"], latest["date"]),
        "position_strength": classify_position_strength(
            latest["close"], latest["psar"], latest["signal"]
        ),
    }

    return {
        "active_status": active,
        "trades": trades,
        "current": current,
        "success_rate": sr,
    }