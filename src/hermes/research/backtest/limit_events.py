"""Limit-up event mechanics for the pre-registered study (issue #1): after the seal, what's left?

A close locked at the up-limit has no sellers, so a buy cannot fill THAT day. The study's entry
is therefore the first later close that is NOT limit-locked -- the first genuinely fillable EOD
print -- and everything the stock did while sealed is recorded as the part you cannot get (the
A-share analog of the US overnight gap in the sibling plutus gap studies).

Three pieces, all pure and unit-tested:

  - `fresh_events`  -- the STARTS of limit-locked streaks (flag == direction today, not
    yesterday). Follow-on locked days belong to the same event, not new ones.
  - `resolve_entries` -- for each event, the first later date whose close is not locked in the
    event's direction (searched within `max_wait` days), plus the abnormal return accrued from
    the event close to that entry close (`missed`).
  - `event_cars` -- cumulative ABNORMAL return over each horizon, accruing from the day AFTER
    the entry close (buying at a close earns nothing that day), minus a flat round-trip cost.
    A horizon that runs off the panel is NaN, never a truncated average; a name that delists
    mid-horizon contributes the days it actually traded (the delisting is in the return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fresh_events(flags: pd.DataFrame, direction: int = 1) -> pd.DataFrame:
    """[code, date] rows where a limit-locked streak STARTS: flag equals `direction` today and
    did not the day before. NaN flags (no bar) never match."""
    hit = flags.eq(direction)
    fresh = hit & ~hit.shift(1, fill_value=False)
    ev = fresh.stack()
    ev = ev[ev].reset_index()
    ev.columns = ["date", "code", "_"]
    return ev[["code", "date"]].sort_values(["date", "code"]).reset_index(drop=True)


def resolve_entries(events: pd.DataFrame, flags: pd.DataFrame, abn: pd.DataFrame,
                    direction: int = 1, max_wait: int = 60) -> pd.DataFrame:
    """Attach [entry_date, wait, missed] to each event.

    entry_date = the first date after the event whose flag is NOT `direction` (the seal broke or
    never re-formed; a NaN flag -- suspension -- does not qualify: nothing traded). `wait` is in
    trading days. `missed` = the abnormal return summed over (event, entry] -- what happened
    between the locked close you saw and the first close you could buy. Events with no fillable
    close within `max_wait` days keep NaT/NaN and are dropped by event_cars (their count is the
    caller's to report)."""
    idx = flags.index
    fl = flags.to_numpy()
    ab = abn.to_numpy()
    col = {c: j for j, c in enumerate(flags.columns)}
    entry, wait, missed = [], [], []
    for code, d in zip(events["code"], events["date"]):
        i, j = idx.get_loc(d), col[code]
        e, w = pd.NaT, np.nan
        for k in range(i + 1, min(i + 1 + max_wait, len(idx))):
            f = fl[k, j]
            if np.isfinite(f) and f != direction:
                e, w = idx[k], k - i
                break
        entry.append(e)
        wait.append(w)
        if e is pd.NaT or (isinstance(e, float) and np.isnan(e)):
            missed.append(np.nan)
        else:
            seg = ab[i + 1:idx.get_loc(e) + 1, j]
            missed.append(float(np.nansum(seg)))
    out = events.copy()
    out["entry_date"], out["wait"], out["missed"] = entry, wait, missed
    return out


def event_cars(events: pd.DataFrame, abn: pd.DataFrame, horizons: tuple[int, ...],
               rt_cost: float) -> pd.DataFrame:
    """Per-event cumulative abnormal returns from the entry close, net of `rt_cost` (one flat
    round trip per event). Events without an entry are dropped. For each horizon h: `car_h`
    (net), NaN where the full window does not fit inside the panel; `n_days_h` = days the
    position actually survived (fewer than h = the name delisted; those days still count)."""
    ev = events.dropna(subset=["entry_date"]).reset_index(drop=True)
    idx = abn.index
    ab = abn.to_numpy()
    col = {c: j for j, c in enumerate(abn.columns)}
    pos = idx.searchsorted(pd.DatetimeIndex(ev["entry_date"]), side="left")
    for h in horizons:
        cars, nd = np.full(len(ev), np.nan), np.zeros(len(ev), dtype=int)
        for r, (code, p) in enumerate(zip(ev["code"], pos)):
            j = col.get(code)
            if j is None or p + h >= len(idx):
                continue
            seg = ab[p + 1:p + 1 + h, j]
            ok = ~np.isnan(seg)
            cars[r] = float(seg[ok].sum()) - rt_cost
            nd[r] = int(ok.sum())
        ev[f"car_{h}"] = cars
        ev[f"n_days_{h}"] = nd
    return ev
