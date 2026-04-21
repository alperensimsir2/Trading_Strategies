"""
CLI: `python -m src` (daily) or `python -m src --backfill` (one-time history build).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import fetch, pipeline


def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        print(f"{path} not found; falling back to EODHD constituent lookup")
        try:
            return fetch.fetch_sp500_constituents()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Could not load tickers: {e}")
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="Fetch full history per ticker before running.")
    ap.add_argument("--tickers", default="tickers.txt", help="Path to ticker list.")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "out"))
    args = ap.parse_args()

    tickers = load_tickers(Path(args.tickers))
    pipeline.run(
        tickers=tickers,
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        do_backfill=args.backfill,
    )


if __name__ == "__main__":
    main()
