"""Issue #8's mechanics: drawdown episodes on an index, and the dip-adding ladder.

An episode starts when the index first closes at or below (1 - dd) of its trailing rolling
peak; after a trigger, no new episode arms until the index closes back within `rearm` of its
(then-current) rolling peak. The ladder buys equal tranches at the trigger and at each deeper
frozen step below the SAME peak, never sells, and is judged on whether the account (credited
stock value plus idle cash) returns to its initial capital within the horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS = (0.8, 0.7, 0.6)        # frozen: tranches at -20% (the trigger), -30%, -40% of peak
DIV_CREDIT = 0.02              # the deliberately generous dividend credit, per ~250d year


def episodes(close: pd.Series, dd: float = 0.2, peak_window: int = 500,
             rearm: float = 0.95) -> pd.DataFrame:
    """All episode triggers for one index. Returns date, iloc, and the peak the drawdown is
    measured against (frozen for the whole episode)."""
    c = close.dropna()
    peak = c.rolling(peak_window, min_periods=1).max()
    armed = True
    rows = []
    for i in range(len(c)):
        if armed and c.iloc[i] <= (1.0 - dd) * peak.iloc[i]:
            rows.append({"date": c.index[i], "iloc": i, "peak": float(peak.iloc[i])})
            armed = False
        elif not armed and c.iloc[i] >= rearm * peak.iloc[i]:
            armed = True
    return pd.DataFrame(rows)


def ladder_outcome(close: pd.Series, trigger_iloc: int, peak: float, horizon: int = 500,
                   steps: tuple[float, ...] = STEPS, div_credit: float = 0.0) -> dict:
    """One episode's ladder, marked daily over `horizon` trading days from the trigger.

    Equal tranches of 1/len(steps): the first at the trigger close, each further one at the
    first close at or below step*peak; unfilled tranches idle at 0%. `div_credit` compounds
    each filled tranche's value by (1+div_credit) per 250 traded days SINCE ITS FILL (the
    generous variant). Returns fills, whether/when the account recovered its initial capital,
    terminal account value, the lump-sum terminal value, and the fill dates for the
    market-alternative leg.
    """
    c = close.dropna()
    path = c.iloc[trigger_iloc:trigger_iloc + 1 + horizon].to_numpy(dtype=float)
    n = len(steps)
    fill_day = np.full(n, -1)
    fill_day[0] = 0
    for j, step in enumerate(steps[1:], start=1):
        hits = np.nonzero(path <= step * peak)[0]
        if len(hits):
            fill_day[j] = int(hits[0])
    t = np.arange(len(path))
    value = np.zeros(len(path))
    for j in range(n):
        if fill_day[j] < 0:
            value += 1.0 / n                                   # idle cash
        else:
            live = t >= fill_day[j]
            grow = (path / path[fill_day[j]]) * (1.0 + div_credit) ** ((t - fill_day[j]) / 250.0)
            value += np.where(live, grow, 1.0) / n             # cash until the fill, then stock
    rec = np.nonzero(value[1:] >= 1.0)[0]                      # day 0 trivially == 1.0
    recovered_day = int(rec[0] + 1) if len(rec) else -1
    return {"n_fills": int((fill_day >= 0).sum()),
            "recovered_day": recovered_day,
            "recovered": recovered_day > 0,
            "terminal": float(value[-1]),
            "lump_terminal": float(path[-1] / path[0]),
            "n_days": len(path) - 1,
            "fill_dates": [c.index[trigger_iloc + d] if d >= 0 else None for d in fill_day]}


def schedule_into(bench: pd.Series, fill_dates: list, horizon_end) -> float:
    """The market-alternative leg: the same tranche schedule invested into `bench` on the
    same dates (unfilled tranches stay cash), marked at `horizon_end`."""
    b = bench.dropna()
    n = len(fill_dates)
    end_pos = b.index.searchsorted(horizon_end, side="right") - 1
    if end_pos < 0:
        return float("nan")
    end_px = float(b.iloc[end_pos])
    total = 0.0
    for d in fill_dates:
        if d is None:
            total += 1.0 / n
            continue
        pos = b.index.searchsorted(d, side="left")
        if pos >= len(b):
            total += 1.0 / n
            continue
        total += (end_px / float(b.iloc[pos])) / n
    return total
