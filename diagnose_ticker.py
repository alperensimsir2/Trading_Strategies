"""Print every trade for a single ticker so you can verify by eye."""

import sys
from pathlib import Path

from src import consensus, fetch, pipeline, strategies as strat

ticker = sys.argv[1] if len(sys.argv) > 1 else "AIZ"

# Auto-detect cache directory - try common locations.
candidates = [Path("cache"), Path("data/cache"), Path("data")]
cache_dir = None
for c in candidates:
    if c.exists() and any(c.iterdir()):
        cache_dir = c
        break

if cache_dir is None:
    print(f"ERROR: No cache directory found. Looked in: {[str(c) for c in candidates]}")
    print(f"Files in current directory:")
    for p in Path(".").iterdir():
        print(f"  {p}")
    sys.exit(1)

print(f"Using cache directory: {cache_dir}")
print(f"Cache contains {len(list(cache_dir.iterdir()))} files")

df = fetch.load_cache(ticker, cache_dir)
print(f"Loaded {len(df)} bars for {ticker}")

if df.empty:
    print(f"ERROR: Cache for {ticker} is empty!")
    sample_files = sorted([p.name for p in cache_dir.iterdir()])[:5]
    print(f"Sample files in cache: {sample_files}")
    sys.exit(1)

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

print(f"\n{ticker}: {len(trades)} trades over 52 weeks\n")
print(f"{'#':>3} {'Side':>4} {'Entry':>10} {'EntryPx':>8} {'Exit':>10} {'ExitPx':>8} "
      f"{'Realized':>9} {'Peak':>8} {'Hold':>5} {'Result':>5}")
print("-" * 90)

for i, t in enumerate(trades, 1):
    print(f"{i:3d} {t['side']:>4s} {t['entry_date']:>10s} {t['entry_close']:>8.2f} "
          f"{t['exit_date']:>10s} {t['exit_close']:>8.2f} "
          f"{t['pnl_pct']:>+8.2f}% {t['peak_gain_pct']:>+7.2f}% {t['hold_days']:>4d}d "
          f"{'WIN' if t['profitable'] else 'LOSS':>5s}")

n = len(trades)
if n:
    wins = sum(1 for t in trades if t["profitable"])
    avg_pnl = sum(t["pnl_pct"] for t in trades) / n
    avg_peak = sum(t["peak_gain_pct"] for t in trades) / n
    avg_hold = sum(t["hold_days"] for t in trades) / n
    print("-" * 90)
    print(f"\n  Total: {n} trades, {wins} wins ({100*wins/n:.0f}%)")
    print(f"  Avg realized: {avg_pnl:+.2f}%   Avg peak: {avg_peak:+.2f}%   Avg hold: {avg_hold:.1f}d")
else:
    print("(no trades found)")