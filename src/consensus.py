"""
Final consensus signal across the four strategies.

CRITICAL: This is intentionally NOT an average. When strategies contradict
(one says BUY, another says SELL), the output is NO SIGNAL. That disagreement
carries information — a naive average would blur it. Do not refactor this to a
weighted average in a future cleanup pass.
"""

from __future__ import annotations

BULL = {"BUY", "STRONG BUY"}
BEAR = {"SELL", "STRONG SELL"}


def final_signal(strategy_signals: dict[str, str]) -> str:
    """
    strategy_signals: e.g. {"trend": "BUY", "range": "HOLD",
                            "reversal": "STRONG BUY", "momentum": "BUY"}
    """
    bull_count = sum(1 for s in strategy_signals.values() if s in BULL)
    bear_count = sum(1 for s in strategy_signals.values() if s in BEAR)
    strong_bull = sum(1 for s in strategy_signals.values() if s == "STRONG BUY")
    strong_bear = sum(1 for s in strategy_signals.values() if s == "STRONG SELL")

    if bull_count > 0 and bear_count > 0:
        return "NO SIGNAL"  # contradicted
    if bull_count == 0 and bear_count == 0:
        return "HOLD"  # all neutral

    if bull_count > 0:
        if strong_bull >= 2:
            return "STRONG BUY"
        if bull_count >= 2:
            return "BUY"
        return "WEAK BUY"  # only one bullish, rest HOLD
    else:
        if strong_bear >= 2:
            return "STRONG SELL"
        if bear_count >= 2:
            return "SELL"
        return "WEAK SELL"
