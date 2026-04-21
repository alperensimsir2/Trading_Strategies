"""
EODHD client. Endpoints used:

- /eod/{TICKER}.US                 individual ticker EOD history (one call each)
- /eod-bulk-last-day/US            all US tickers, one day (one call total)
- /fundamentals/GSPC.INDX          S&P 500 index constituents

API pricing note: the bulk endpoint counts as ~100 API calls (varies by plan —
check your dashboard). Still cheaper than 500 individual calls.

Caching strategy:
  - First run: call individual EOD per ticker to build ~260-bar history. Cache
    it as Parquet or JSON on disk (or R2).
  - Daily runs: call bulk endpoint once, append today's row per ticker.
  - Warmup safety: if the cache is missing or stale (>7 days old), refetch the
    whole history for that ticker.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

EODHD_KEY = os.environ.get("EODHD_API_KEY")
BASE = "https://eodhd.com/api"
DEFAULT_TIMEOUT = 30


def _require_key():
    if not EODHD_KEY:
        raise RuntimeError("EODHD_API_KEY environment variable is not set")


def fetch_ticker_eod(ticker: str, days: int = 300) -> pd.DataFrame:
    """
    Fetch EOD history for a single ticker. Returns a DataFrame indexed by date
    with columns: open, high, low, close, adjusted_close, volume.
    """
    _require_key()
    url = f"{BASE}/eod/{ticker}.US"
    params = {"api_token": EODHD_KEY, "fmt": "json", "period": "d"}
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"adjusted_close": "adj_close"})
    return df.tail(days).reset_index(drop=True)


def fetch_bulk_last_day() -> list[dict]:
    """
    One day of EOD for every US ticker. Returns a list of dicts; each has
    keys like: code, exchange_short_name, date, open, high, low, close,
    adjusted_close, volume.
    """
    _require_key()
    url = f"{BASE}/eod-bulk-last-day/US"
    params = {"api_token": EODHD_KEY, "fmt": "json"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_sp500_constituents() -> list[str]:
    """
    Returns the list of S&P 500 ticker codes via EODHD's index fundamentals.
    Response shape uses integer-keyed Components dict with Code / Name fields.
    If this endpoint's shape drifts, keep a static fallback in tickers.txt.
    """
    _require_key()
    url = f"{BASE}/fundamentals/GSPC.INDX"
    params = {"api_token": EODHD_KEY, "fmt": "json"}
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    components = data.get("Components", {}) or data.get("General", {}).get("Components", {})
    return sorted({v["Code"] for v in components.values() if "Code" in v})


# ----- Local cache -----------------------------------------------------------

def load_cache(ticker: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_cache(ticker: str, df: pd.DataFrame, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{ticker}.parquet", index=False)


def backfill_universe(tickers: Iterable[str], cache_dir: Path,
                      days: int = 300, sleep_s: float = 0.1) -> list[str]:
    """
    One-time backfill: fetch 300 bars per ticker and write to cache. Returns
    the list of tickers that failed.
    """
    failures = []
    for t in tickers:
        try:
            df = fetch_ticker_eod(t, days=days)
            if df.empty:
                failures.append(t)
                continue
            save_cache(t, df, cache_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[backfill] {t}: {e}")
            failures.append(t)
        time.sleep(sleep_s)  # rate-limit cushion
    return failures


def append_bulk_day(bulk_rows: list[dict], tickers: set[str], cache_dir: Path) -> None:
    """
    Append today's row from the bulk endpoint into each ticker's cache.
    Skips tickers not in the S&P 500 universe. De-dupes by date.
    """
    by_code: dict[str, dict] = {r["code"]: r for r in bulk_rows if r.get("code") in tickers}
    for ticker, row in by_code.items():
        df = load_cache(ticker, cache_dir)
        new_row = {
            "date": pd.to_datetime(row["date"]),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "adj_close": row.get("adjusted_close"),
            "volume": row.get("volume"),
        }
        if df.empty:
            df = pd.DataFrame([new_row])
        else:
            # De-dupe: replace existing bar for same date, else append.
            df = df[df["date"] != new_row["date"]]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        # Keep a rolling window to bound disk usage.
        df = df.tail(400).reset_index(drop=True)
        save_cache(ticker, df, cache_dir)
