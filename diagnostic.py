"""Diagnose strategy performance per source and per ticker."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src import consensus, fetch, indicators as ind, strategies as strat, pipeline

TICKERS_FILE = Path("tickers.txt")
CACHE_DIR = Path("data/cache")

tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]

all_trades = []
per_ticker_stats = []

for ticker in tickers:
    df = fetch.load_cache(ticker, CACHE_DIR)
    if df.empty or len(df) < 80:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    df = pipeline.compute_indicators(df)

    primary = strat.compute_primary_signals(df)
    primary = strat.apply_psar_alignment_filter(df, primary)
    secondary = strat.compute_trending_stocks_signals(df)

    window = df.tail(260).reset_index(drop=True)
    window_dates = set(window["date"].astype(str).str[:10])
    primary_win = {d: v for d, v in primary.items() if d in window_dates}
    secondary_win = {d: v for d, v in secondary.items() if d in window_dates}

    result = consensus.process_ticker_signals(window, primary_win, secondary_win)
    trades = result["trades"]
    all_trades.extend(trades)

    if len(trades) >= 3:
        wins = sum(1 for t in trades if t["profitable"])
        avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades)
        per_ticker_stats.append({
            "ticker": ticker,
            "trades": len(trades),
            "wins": wins,
            "rate": wins / len(trades),
            "avg_pnl_pct": avg_pnl,
        })

# Overall
print(f"\n=== OVERALL ===")
print(f"Total trades: {len(all_trades)}")
wins = sum(1 for t in all_trades if t["profitable"])
avg_pnl = sum(t["pnl_pct"] for t in all_trades) / len(all_trades)
print(f"Wins: {wins} ({100 * wins / len(all_trades):.1f}%)")
print(f"Avg PnL: {avg_pnl:+.2f}%")
print(f"Total PnL (sum): {sum(t['pnl_pct'] for t in all_trades):+.2f}%")

# By source
print(f"\n=== BY SOURCE ===")
by_source = defaultdict(list)
for t in all_trades:
    by_source[t["source"]].append(t)
for source, trades in by_source.items():
    wins = sum(1 for t in trades if t["profitable"])
    avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0
    print(f"{source:10s}: {len(trades)} trades, {wins} wins ({100 * wins / len(trades):.1f}%), avg PnL {avg_pnl:+.2f}%")

# By side
print(f"\n=== BY SIDE ===")
by_side = defaultdict(list)
for t in all_trades:
    by_side[t["side"]].append(t)
for side, trades in by_side.items():
    wins = sum(1 for t in trades if t["profitable"])
    avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0
    print(f"{side:5s}: {len(trades)} trades, {wins} wins ({100 * wins / len(trades):.1f}%), avg PnL {avg_pnl:+.2f}%")

# Win/loss distribution
wins_only = [t for t in all_trades if t["profitable"]]
losses_only = [t for t in all_trades if not t["profitable"]]
avg_win = sum(t["pnl_pct"] for t in wins_only) / len(wins_only) if wins_only else 0
avg_loss = sum(t["pnl_pct"] for t in losses_only) / len(losses_only) if losses_only else 0
print(f"\n=== ASYMMETRY ===")
print(f"Avg WIN:  +{avg_win:.2f}%")
print(f"Avg LOSS: {avg_loss:.2f}%")
if avg_loss != 0:
    print(f"Win/Loss ratio: {abs(avg_win / avg_loss):.2f}")
print(f"\nA strategy can be profitable with low win rate IF avg win >> avg loss.")
print(f"Total return scenario: {wins} × {avg_win:.2f}% + {len(losses_only)} × {avg_loss:.2f}% = {sum(t['pnl_pct'] for t in all_trades):+.2f}%")

# Top/bottom tickers
print(f"\n=== TOP 5 TICKERS BY SUCCESS RATE (min 3 trades) ===")
per_ticker_stats.sort(key=lambda x: -x["rate"])
for s in per_ticker_stats[:5]:
    print(f"  {s['ticker']:6s}: {s['wins']}/{s['trades']} ({100 * s['rate']:.0f}%) avg PnL {s['avg_pnl_pct']:+.2f}%")
print(f"\n=== BOTTOM 5 TICKERS ===")
for s in per_ticker_stats[-5:]:
    print(f"  {s['ticker']:6s}: {s['wins']}/{s['trades']} ({100 * s['rate']:.0f}%) avg PnL {s['avg_pnl_pct']:+.2f}%")