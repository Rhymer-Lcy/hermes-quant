"""Realistic A-share dividend accounting on top of an adjusted-price book. Issue #24.

The lake is forward-adjusted (BaoStock ``adjustflag="2"``), and that series is a TOTAL-RETURN
series: on an ex-date the adjusted return already contains ``dps / raw_close[t-1]``. So the engine
has always reinvested every dividend -- at the ex-date close, instantly, PRE-TAX, in fractional
shares, into the same name. A real individual account gets none of those four terms for free.

This module measures the difference without touching the engine. Two facts make that possible:

1. the gross dividend enters the adjusted return as exactly ``dps / raw_close[t-1]``, so
   withholding a rate ``tau`` is the closed-form perturbation ``adj_ret - tau * dps/raw_prev``;
2. that perturbation can be expressed as a multiplicative correction on the price panel, which is
   EXACTLY 1.0 when ``tau == 0`` -- so the neutral setting is bit-identical to canonical by
   construction, not by tolerance.

Stock dividends are not in the cash-dividend lake and would otherwise read as price collapses, so
the share multiplier is derived from the two price series rather than assumed (`share_multiplier`).

Nothing here is imported by the deployed path; it is research-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Caishui [2015] No. 101 -- differentiated individual income tax on listed-share dividends.
# The holding period runs from purchase to TRANSFER (sale): CSDCC withholds nothing at the
# ex-date for holdings of a year or less and settles the liability when the shares are sold.
RATE_SHORT = 0.20      # h <= 1 month   -- 100% of the dividend is taxable at 20%
RATE_MID = 0.10        # 1 month < h <= 1 year -- 50% inclusion, so 10% effective
RATE_LONG = 0.00       # h > 1 year     -- exempt


@dataclass(frozen=True)
class TaxRule:
    """A frozen dividend-tax convention. `kind` is "statutory" or "flat"."""

    name: str
    kind: str = "statutory"
    flat_rate: float = 0.0
    boundary: str = "calendar_month"   # "calendar_month" | "days30"
    measure: str = "sale"              # "sale" | "ex_date"

    def rate(self, buy: pd.Timestamp, sale: pd.Timestamp, ex: pd.Timestamp) -> float:
        """Effective tax rate on a dividend received by a lot bought at `buy` and sold at `sale`
        (the window end stands in for a lot still open), with ex-date `ex`."""
        if self.kind == "flat":
            return self.flat_rate
        end = ex if self.measure == "ex_date" else sale
        if self.boundary == "days30":
            one_month = buy + pd.Timedelta(days=30)
        else:
            one_month = buy + pd.DateOffset(months=1)
        if end <= one_month:
            return RATE_SHORT
        return RATE_MID if end <= buy + pd.DateOffset(years=1) else RATE_LONG


# The pre-registered sensitivity grid (issue #24), frozen before any outcome was computed.
TAX_RULES: dict[str, TaxRule] = {
    "tau-A": TaxRule("statutory, FIFO, buy->sale, calendar-month boundary (PRIMARY)"),
    "tau-B": TaxRule("flat 20% on every dividend (worst case)", kind="flat", flat_rate=0.20),
    "tau-C": TaxRule("flat 10% on every dividend", kind="flat", flat_rate=0.10),
    "tau-D": TaxRule("0% -- today's model (reference cell)", kind="flat", flat_rate=0.0),
    "tau-E": TaxRule("statutory, one-month boundary = 30 calendar days", boundary="days30"),
    "tau-F": TaxRule("statutory, holding period measured buy->ex-date", measure="ex_date"),
    "tau-G": TaxRule("institutional holder -- exempt", kind="flat", flat_rate=0.0),
}
PRIMARY_RULE = "tau-A"


# --------------------------------------------------------------------------- events


def dividend_events(adj_close: pd.DataFrame, raw_close: pd.DataFrame,
                    dividends: pd.DataFrame) -> pd.DataFrame:
    """One row per (code, ex-date) that lands on a tradable bar of `adj_close`.

    Columns: code, date, dps, raw_prev, raw_now, adj_ret, dyield, m.

    ``dyield = dps / raw_close[t-1]`` is the fraction of the adjusted return that IS the dividend.
    ``m`` is the derived share multiplier (see `share_multiplier`): 1.0 for a pure cash dividend,
    ~2.0 for a 10-for-10 stock dividend."""
    div = dividends.loc[dividends["dps"] > 0, ["code", "ex_date", "dps"]].copy()
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    div = div.groupby(["code", "ex_date"], as_index=False)["dps"].sum()

    codes = [c for c in adj_close.columns if c in raw_close.columns]
    rows = []
    for code, grp in div.groupby("code", sort=True):
        if code not in codes:
            continue
        a = adj_close[code]
        r = raw_close[code]
        pos_a = {d: i for i, d in enumerate(a.index)}
        pos_r = {d: i for i, d in enumerate(r.index)}
        for ex, dps in zip(grp["ex_date"], grp["dps"]):
            ia, ir = pos_a.get(ex), pos_r.get(ex)
            if ia is None or ir is None or ia == 0 or ir == 0:
                continue
            a1, a0 = a.iloc[ia], a.iloc[ia - 1]
            r1, r0 = r.iloc[ir], r.iloc[ir - 1]
            if not np.isfinite([a1, a0, r1, r0]).all() or min(a0, r0, r1) <= 0:
                continue
            adj_ret = a1 / a0 - 1.0
            rows.append({"code": code, "date": ex, "dps": float(dps),
                         "raw_prev": float(r0), "raw_now": float(r1),
                         "adj_ret": float(adj_ret), "dyield": float(dps) / float(r0),
                         "m": share_multiplier(adj_ret, float(r0), float(r1), float(dps))})
    out = pd.DataFrame(rows, columns=["code", "date", "dps", "raw_prev", "raw_now",
                                      "adj_ret", "dyield", "m"])
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def share_multiplier(adj_ret: float, raw_prev: float, raw_now: float, dps: float) -> float:
    """Shares after the corporate action per share before it, derived from the two price series.

    Per OLD share the holder ends the ex-date with ``m`` new shares worth ``raw_now`` each plus
    ``dps`` in cash, and the adjusted (total-return) series prices that whole package:

        1 + adj_ret = (m * raw_now + dps) / raw_prev

    A pure cash dividend gives m == 1; a 10-for-10 stock dividend gives m == 2."""
    return ((1.0 + adj_ret) * raw_prev - dps) / raw_now


# --------------------------------------------------------------------------- FIFO lots


def fifo_lots(trades: list[dict], window_end: pd.Timestamp) -> pd.DataFrame:
    """Match the engine's fills into FIFO lots: columns code, buy_date, sell_date, shares.

    A lot still open at the end of the window carries ``sell_date = window_end`` and
    ``open = True`` -- the holder has not transferred, so no rate is yet determined; continuing
    the lot to the window edge is the neutral convention and its materiality is reported."""
    fills = sorted(enumerate(trades), key=lambda kv: (kv[1]["date"], kv[0]))
    books: dict[str, list[list]] = {}
    lots: list[dict] = []
    for _i, t in fills:
        code, sh, d = t["code"], int(t["shares"]), pd.Timestamp(t["date"])
        book = books.setdefault(code, [])
        if sh > 0:
            book.append([d, sh])
        elif sh < 0:
            need = -sh
            while need > 0:
                if not book:
                    raise ValueError(f"FIFO underflow: sell of {code} on {d.date()} with no lot")
                buy_d, avail = book[0]
                take = min(avail, need)
                lots.append({"code": code, "buy_date": buy_d, "sell_date": d,
                             "shares": take, "open": False})
                need -= take
                if take == avail:
                    book.pop(0)
                else:
                    book[0][1] -= take
    for code, book in books.items():
        for buy_d, rem in book:
            if rem > 0:
                lots.append({"code": code, "buy_date": buy_d, "sell_date": window_end,
                             "shares": rem, "open": True})
    out = pd.DataFrame(lots, columns=["code", "buy_date", "sell_date", "shares", "open"])
    return out.sort_values(["code", "buy_date"]).reset_index(drop=True)


def lots_entitled(lots: pd.DataFrame, code: str, ex: pd.Timestamp) -> pd.DataFrame:
    """The lots on the register for an ex-date dividend.

    The register closes at the end of the prior trading day, and the engine trades at the close.
    So a lot bought AT the ex-date close is too late, and a lot sold at the prior close is already
    gone: entitlement is ``buy_date < ex <= sell_date``."""
    sub = lots[lots["code"] == code]
    return sub[(sub["buy_date"] < ex) & (sub["sell_date"] >= ex)]


def _lots_by_code(lots: pd.DataFrame) -> dict[str, tuple]:
    """Pre-group the lot book into per-code arrays -- `event_tax_table` is called 7 rules x 7
    tiers deep, and re-filtering the whole frame per event is what makes that quadratic."""
    out = {}
    for code, g in lots.groupby("code", sort=False):
        out[code] = (g["buy_date"].to_numpy(), g["sell_date"].to_numpy(),
                     g["shares"].to_numpy(dtype=float), g["open"].to_numpy(dtype=bool))
    return out


# --------------------------------------------------------------------------- tax per event


def event_tax_table(lots: pd.DataFrame, events: pd.DataFrame, rule: TaxRule,
                    open_rate: float | None = None) -> pd.DataFrame:
    """Share-weighted effective tax rate per (code, ex-date) the book was actually on the register
    for, plus the share counts feeding the weighting.

    Returns events restricted to entitled rows, with `tau`, `engine_shares` and the share of
    entitled shares in each statutory band.

    `open_rate` overrides the rate on lots still open at the window end WITHOUT disturbing their
    entitlement -- the sensitivity handle for a convention the pre-registration left open. Note
    that forcing a high `open_rate` is deliberately NOT statutorily reachable: under the statute a
    lot cannot both collect a dividend months after purchase and be taxed at the one-month rate,
    so such a column is a strict upper bound, not a scenario."""
    if lots.empty or events.empty:
        return events.assign(tau=np.nan, engine_shares=0.0).iloc[:0]
    by_code = _lots_by_code(lots)
    ev = events[events["code"].isin(by_code)]
    rows = []
    for rec in ev.itertuples(index=False):
        buys, sells, shares, is_open = by_code[rec.code]
        ex = np.datetime64(rec.date)
        keep = (buys < ex) & (sells >= ex)          # entitlement: see `lots_entitled`
        if not keep.any():
            continue
        sh = shares[keep]
        rates = np.array([
            open_rate if (open_rate is not None and o) else rule.rate(pd.Timestamp(b),
                                                                     pd.Timestamp(s), rec.date)
            for b, s, o in zip(buys[keep], sells[keep], is_open[keep])])
        total = sh.sum()
        rows.append({"code": rec.code, "date": rec.date, "dps": rec.dps,
                     "dyield": rec.dyield, "engine_shares": total,
                     "tau": float((sh * rates).sum() / total),
                     "w_short": float(sh[rates == RATE_SHORT].sum() / total),
                     "w_mid": float(sh[rates == RATE_MID].sum() / total),
                     "w_long": float(sh[rates == RATE_LONG].sum() / total)})
    return pd.DataFrame(rows, columns=["code", "date", "dps", "dyield", "engine_shares",
                                       "tau", "w_short", "w_mid", "w_long"])


# --------------------------------------------------------------------------- net panel


def net_close_panel(adj_close: pd.DataFrame, taxed: pd.DataFrame,
                    events: pd.DataFrame) -> pd.DataFrame:
    """The adjusted panel with the dividend tax withheld, as a multiplicative correction.

        net_ret = adj_ret - tau * dyield
        factor  = (1 + net_ret) / (1 + adj_ret) = 1 - tau * dyield / (1 + adj_ret)
        net_close[t] = adj_close[t] * cumprod(factor)[t]

    With ``tau == 0`` every factor is exactly 1.0, so the cumulative product is exactly 1.0 and
    the returned panel is BIT-IDENTICAL to `adj_close` (Gate P). Names and dates with no taxed
    event are untouched for the same reason."""
    if taxed.empty:
        return adj_close.copy()
    adj_ret = events.set_index(["code", "date"])["adj_ret"]
    corr = pd.DataFrame(1.0, index=adj_close.index, columns=adj_close.columns)
    touched = taxed[taxed["tau"] != 0.0]
    for rec in touched.itertuples(index=False):
        ar = float(adj_ret.loc[(rec.code, rec.date)])
        corr.at[rec.date, rec.code] = 1.0 - rec.tau * rec.dyield / (1.0 + ar)
    return adj_close * corr.cumprod()


# --------------------------------------------------------------------------- holdings & leakage


def daily_shares(trades: list[dict], index: pd.DatetimeIndex,
                 columns) -> pd.DataFrame:
    """Engine share holdings per (date, code), folded from the fill log and held forward."""
    flow = pd.DataFrame(0.0, index=index, columns=list(columns))
    for t in trades:
        d, c = pd.Timestamp(t["date"]), t["code"]
        if d in flow.index and c in flow.columns:
            flow.at[d, c] += float(t["shares"])
    return flow.cumsum()


def leakage_stream(shares: pd.DataFrame, adj_close: pd.DataFrame,
                   taxed: pd.DataFrame) -> pd.Series:
    """Cash tax paid per date: ``tau * dps * real_shares``, computed in adjusted space as
    ``position_value[t-1] * tau * dyield`` -- the two are identical because
    ``position_value[t-1] = real_shares * raw_close[t-1]``."""
    leak = pd.Series(0.0, index=adj_close.index)
    if taxed.empty:
        return leak
    pos = {d: i for i, d in enumerate(adj_close.index)}
    for rec in taxed.itertuples(index=False):
        i = pos.get(rec.date)
        if i is None or i == 0 or rec.tau == 0.0:
            continue
        prev = adj_close.index[i - 1]
        val = float(shares.at[prev, rec.code]) * float(adj_close.at[prev, rec.code])
        if np.isfinite(val) and val > 0:
            leak.iloc[i] += val * rec.tau * rec.dyield
    return leak


def gross_dividend_stream(shares: pd.DataFrame, adj_close: pd.DataFrame,
                          taxed: pd.DataFrame) -> pd.Series:
    """Gross (pre-tax) dividend income per date, on the same basis as `leakage_stream`.

    Its annual total over average equity is the book's realised gross dividend yield -- the
    external magnitude check against published index yields."""
    gross = pd.Series(0.0, index=adj_close.index)
    if taxed.empty:
        return gross
    pos = {d: i for i, d in enumerate(adj_close.index)}
    for rec in taxed.itertuples(index=False):
        i = pos.get(rec.date)
        if i is None or i == 0:
            continue
        prev = adj_close.index[i - 1]
        val = float(shares.at[prev, rec.code]) * float(adj_close.at[prev, rec.code])
        if np.isfinite(val) and val > 0:
            gross.iloc[i] += val * rec.dyield
    return gross


def band_dividend_cash(shares: pd.DataFrame, adj_close: pd.DataFrame,
                       taxed: pd.DataFrame) -> dict[str, float]:
    """Dividend CASH falling in each statutory holding-period band.

    Weighting by share count alone is wrong here and silently so: shares of different names carry
    different prices, so a share-weighted band split is not a currency amount. The weight is the
    same position value that `leakage_stream` taxes."""
    out = {"short": 0.0, "mid": 0.0, "long": 0.0, "total": 0.0}
    if taxed.empty:
        return out
    pos = {d: i for i, d in enumerate(adj_close.index)}
    for rec in taxed.itertuples(index=False):
        i = pos.get(rec.date)
        if i is None or i == 0:
            continue
        prev = adj_close.index[i - 1]
        val = float(shares.at[prev, rec.code]) * float(adj_close.at[prev, rec.code])
        if not np.isfinite(val) or val <= 0:
            continue
        cash = val * rec.dyield
        out["short"] += cash * rec.w_short
        out["mid"] += cash * rec.w_mid
        out["long"] += cash * rec.w_long
        out["total"] += cash
    return out


def closed_form_terminal(equity: pd.Series, leak: pd.Series, *, lag: bool = True) -> float:
    """Terminal wealth after removing the tax leakage from a canonical curve:

        E_real[T] = E_canon[T] * prod_t (1 - leak_t / E_canon[t-1])

    `lag=False` divides by ``E_canon[t]`` instead -- the form written in the pre-registration.
    Both are reported; the leak is a withdrawal against the PRIOR close, so `lag=True` is the
    correct one and the difference between them is a diagnostic, not a choice."""
    base = equity.shift(1) if lag else equity
    ratio = 1.0 - (leak / base).fillna(0.0)
    return float(equity.iloc[-1] * ratio.clip(lower=0.0).prod())
