# S&P 500 Daily Signals

Backend pipeline that computes daily BUY/SELL signals for the S&P 500 from a
four-strategy composite (Trend, Range/Mean-Reversion, Reversal, Momentum),
following the Excel reference model.

Runs on GitHub Actions cron. No server. Writes static JSON that mobile clients
fetch from Cloudflare R2 (or a gh-pages branch).

## Layout

```
.
├── .github/workflows/daily-signals.yml   # cron trigger
├── schema/                                # JSON Schema contracts for mobile app
│   ├── signals.schema.json                # home-screen payload
│   └── ticker.schema.json                 # per-ticker history payload
├── src/
│   ├── indicators.py                      # SMA/EMA/RSI/MACD/BB/ADX/ATR/Stoch/Z-score
│   ├── strategies.py                      # 4 strategies from Part 1 spec
│   ├── consensus.py                       # contradiction-filter final signal
│   ├── fetch.py                           # EODHD API (bulk + individual EOD)
│   ├── pipeline.py                        # orchestrator
│   └── __main__.py                        # `python -m src` entry point
├── tickers.txt                            # S&P 500 universe, one ticker per line
└── requirements.txt
```

## Design decisions worth knowing

**Indicators are handwritten, not pandas-ta.** `pandas-ta`'s ATR/ADX default to
EMA smoothing, not Wilder's. Your Excel model uses Wilder. `src/indicators.py`
implements Wilder smoothing explicitly (seed with SMA(N), then recursive
`avg_t = avg_{t-1}·(N-1)/N + x_t/N`). That matches the spec exactly. Verify on
one ticker against your workbook before trusting the output.

**Adjusted vs raw prices.** Per spec:
- SMA, EMA, RSI, MACD, Bollinger Bands, Z-score → adjusted close
- ADX, Stochastic %K, ATR → raw H/L/C

EODHD returns both `close` and `adjusted_close` per bar.

**History is cached, not refetched daily.** Initial run backfills ~260 bars per
ticker via individual `/eod/{TICKER}.US` calls (one-time cost, ~500 API calls).
Subsequent runs use `/eod-bulk-last-day/US` (one call, all tickers) and append
one row to each ticker's cached history. Indicators recompute against the
rolling window.

**Final signal preserves contradictions.** If strategies disagree, the output is
`NO SIGNAL`, not an averaged score. This is a load-bearing part of the model —
do not replace it with a weighted average in a future refactor.

## Setup

```bash
pip install -r requirements.txt
export EODHD_API_KEY=your_key_here
python -m src --backfill       # first run: build history cache
python -m src                  # subsequent runs: append + recompute
```

Outputs land in `out/`:
- `out/signals.json` — home-screen list of all tickers + latest signal
- `out/tickers/{TICKER}.json` — per-ticker history for the detail screen

## Cron schedule

`.github/workflows/daily-signals.yml` runs at 00:30 UTC Tue–Sat. That's ~3.5
hours after US market close in both EDT and EST, which is enough time for
EODHD's EOD data to settle.

## Deploying JSON

Two options, both wired in the workflow as commented blocks:
1. **Cloudflare R2** via `aws s3 sync` (S3-compatible). Fastest for mobile
   clients. Requires R2 credentials as repo secrets.
2. **gh-pages branch** via `peaceiris/actions-gh-pages`. Simpler, free. Higher
   latency, lower cache hit rate.

## Next things to build

- Constituent refresh: S&P 500 membership changes quarterly. Add a weekly job
  that diffs the index and adds/removes tickers from `tickers.txt`.
- Data-quality checks: reject days where >5% of tickers have missing bulk data.
- Signal-stability feature for the app: flag tickers whose signal changed today
  vs yesterday.
