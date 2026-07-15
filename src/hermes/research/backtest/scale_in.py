"""Issue #3's execution comparison: inverted-pyramid scale-in vs lump-sum at the same entry.

One event = one (name, entry close). Both arms start with one unit of cash. The lump-sum arm
invests all of it at the entry close; the pyramid arm invests a third at the entry close and a
third at the first close at or below each subsequent step down. Tranches never filled stay in
cash at 0%. Buys pay the proportional buy rate; both arms are marked to market at the horizon
without a liquidation leg (they hold identical exit treatment, and the second-order difference
-- the pyramid holds less stock and would pay less to exit -- favors lump-sum, i.e. runs
AGAINST the friend's claim, so ignoring it cannot flip a CONFIRMED verdict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS = (0.925, 0.85)      # frozen: tranches at entry, entry-7.5%, entry-15%


def pyramid_vs_lump(path: np.ndarray, buy_rate: float, steps: tuple[float, ...] = STEPS) -> dict:
    """One event. `path` = the name's own traded closes from the entry close onward (entry
    at path[0]), already truncated to the horizon. Returns both arms' net account values
    (initial cash = 1) and how many pyramid tranches filled."""
    p0 = float(path[0])
    n_tranches = 1 + len(steps)
    lump = (1.0 - buy_rate) * float(path[-1]) / p0

    value = (1.0 - buy_rate) / n_tranches * float(path[-1]) / p0   # tranche 1 fills at entry
    filled = 1
    for step in steps:
        hits = np.nonzero(path <= p0 * step)[0]
        if len(hits):
            fill = float(path[hits[0]])
            value += (1.0 - buy_rate) / n_tranches * float(path[-1]) / fill
            filled += 1
        else:
            value += 1.0 / n_tranches                              # unfilled tranche stays cash
    return {"lump": lump, "pyramid": value, "filled": filled}


def run_events(events: pd.DataFrame, close: pd.DataFrame, horizon: int, buy_rate: float,
               max_entry_wait: int = 10) -> pd.DataFrame:
    """Run every (code, date) signal event. Entry is the name's first traded close AFTER the
    signal day (waiting past `max_entry_wait` calendar rows drops the event -- a suspension
    straddling the signal is not a fillable entry). The horizon is `horizon` of the name's own
    traded days, shorter if the series ends (delisting: both arms mark at the last real close).
    """
    rows = []
    for code, grp in events.groupby("code"):
        s = close[code].dropna()
        pos = s.index.get_indexer(grp["date"])
        for d, i in zip(grp["date"], pos):
            if i < 0:
                continue
            e = i + 1
            if e >= len(s) or (s.index[e] - d).days > max_entry_wait:
                continue
            path = s.iloc[e:e + 1 + horizon].to_numpy(dtype=float)
            if len(path) < 2:
                continue
            r = pyramid_vs_lump(path, buy_rate)
            rows.append({"code": code, "date": d, "entry_date": s.index[e],
                         "n_days": len(path) - 1, **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["diff"] = out["pyramid"] - out["lump"]
    return out
