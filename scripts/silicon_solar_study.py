"""Industrial silicon as the single top-timing indicator for the solar sector. Issue #18.

A friend's macro call, stated as a one-indicator rule (paraphrased from the chat, translated):
"Watch one thing -- the industrial-silicon price. It peaks about four years out, and only then
does the sector start to decline. Last time industrial silicon took off, it hung solar on the
mountain [it marked the solar top]."

This is NOT a cross-sectional stock rule like issues #2-#17; it is a MACRO timing claim with a
single modern cycle, so it cannot yield a historical statistical verdict. It is registered
PROSPECTIVELY: the operationalization is frozen here, a baseline is snapshotted, and the verdict is
evaluated only when the NEXT silicon peak confirms (earliest ~2030-2032). Frozen before any
silicon/solar co-movement was examined.

  claim          the NEXT industrial-silicon peak comes ~4 years out and leads a solar decline
  silicon        GFEX industrial-silicon futures main-continuous (SI0), monthly last close.
                 The series starts 2022-12-22; the 2021-22 boom peak PREDATES it, so the one
                 episode the friend cites is NOT measurable on free data -- history here is N=1,
                 descriptive only, and contaminated (the friend named that episode after the fact).
  solar equity   CSI Solar Industry index 931151, monthly close (robustness: PV-ETF 515790)
  benchmark      CSI 300 (000300), monthly close -- price index vs price index, so no dividend
                 asymmetry (the #12 trap is avoided by construction); pulled every run via a
                 csindex-led fallback chain and cached, entering the verdict only at the P2 step
  peak(t)        a month whose silicon close is the max of the centered 25-month window [t-12,t+12]
                 AND exceeds the registration-month close; confirmable only 12 months after t
  P1 timing      the next confirmed peak (first such t after the registration month) falls in
                 [2029-01, 2031-12] -- four years out, plus or minus one year
  P2 mechanism   the 12-month-forward return of 931151 minus 000300, measured from that peak, is
                 negative (the silicon top leads the solar top)
  verdict        CONFIRMED iff P1 and P2; REJECTED if the peak lands outside the window, or the
                 12m-forward solar excess is >= 0, or no peak above the registration-month level
                 forms by 2033-12
  status         forward-tracking; re-run monthly to refresh the baseline and the peak search

    conda activate hermes
    python scripts/silicon_solar_study.py
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd

from hermes.io import atomic_to_parquet, atomic_write_text
from hermes.paths import PAPER_DIR, PARQUET_DIR, ensure_dirs

REG_DATE = "2026-07-19"                 # registration (claim) date -- frozen
REG_MONTH = pd.Timestamp("2026-07-31")  # month-end the registration snaps to
WIN = 12                                # half-window (months) for the centered-peak definition
PEAK_WINDOW = ("2029-01", "2031-12")    # P1: four years out, +/- one year
DEGENERATE_BY = pd.Timestamp("2033-12-31")  # no new peak above reg level by here -> REJECTED
LAKE = PARQUET_DIR / "silicon_solar_monthly.parquet"
REG_JSON = PAPER_DIR / "silicon_solar_registration.json"


class ForwardDataUnavailable(RuntimeError):
    """No usable prices AND no cache -- a transient data/network condition, not a bug (mirrors
    plutus-quant's guard and hermes' BaoStockUnavailable)."""


def _retry(fn, tries: int = 4, pause: float = 2.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:      # noqa: BLE001 -- akshare raises many transient network types
            last = e
            time.sleep(pause)
    raise RuntimeError(f"pull failed after {tries} tries: {last}")


def _monthly_close(df: pd.DataFrame, date_col: str, close_col: str) -> pd.Series:
    s = df[[date_col, close_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    return s.set_index(date_col)[close_col].astype(float).sort_index().resample("ME").last()


def _pull_hs300(today: str):
    """CSI 300 monthly close via a fallback chain led by the authoritative csindex.com.cn feed
    (the eastmoney push2 endpoint is flaky). Returns (series, tag) or (None, None); the benchmark
    enters the verdict only at the P2 step, so a miss on any run is non-fatal -- it is pulled every
    run so its history accrues from registration rather than being fetched once, years out."""
    import akshare as ak
    attempts = [
        ("csindex", lambda: ak.stock_zh_index_hist_csindex(symbol="000300", start_date="20180101",
                                                           end_date=today), "日期", "收盘"),
        ("csi000300", lambda: ak.stock_zh_index_daily_em(symbol="csi000300", start_date="20180101",
                                                        end_date=today), "date", "close"),
        ("510300_etf", lambda: ak.fund_etf_hist_em(symbol="510300", period="daily",
                                                  start_date="20180101", end_date=today,
                                                  adjust="qfq"), "日期", "收盘"),
    ]
    for tag, fn, dc, cc in attempts:
        try:
            return _monthly_close(_retry(fn, tries=2), dc, cc), tag
        except Exception:
            continue
    return None, None


def pull_panel() -> pd.DataFrame:
    """Fresh monthly panel (silicon, solar, solar_etf, hs300). Silicon and solar are required;
    if either fails and no cache exists, raise ForwardDataUnavailable. Merges onto any cached
    LAKE so a transient miss on one series keeps the last good column."""
    import akshare as ak
    today = date.today().strftime("%Y%m%d")
    cached = pd.read_parquet(LAKE) if LAKE.exists() else None

    cols: dict[str, pd.Series] = {}
    try:
        si = _retry(lambda: ak.futures_main_sina(symbol="SI0", start_date="20221201",
                                                 end_date=today))
        cols["silicon"] = _monthly_close(si, "日期", "收盘价")
    except Exception:
        pass
    for pull, dc, cc in [                    # solar: csindex.com.cn primary (authoritative, resilient)
        (lambda: ak.stock_zh_index_hist_csindex(symbol="931151", start_date="20180101",
                                                end_date=today), "日期", "收盘"),
        (lambda: ak.stock_zh_index_daily_em(symbol="csi931151", start_date="20180101",
                                            end_date=today), "date", "close"),  # eastmoney fallback
    ]:
        try:
            cols["solar"] = _monthly_close(_retry(pull, tries=2), dc, cc)
            break
        except Exception:
            continue
    try:                                     # sina ETF history (raw close) -- reliable robustness leg
        etf = _retry(lambda: ak.fund_etf_hist_sina(symbol="sh515790"))
        cols["solar_etf"] = _monthly_close(etf, "date", "close")
    except Exception:
        pass
    hs, _tag = _pull_hs300(today)            # CSI 300 benchmark: cached now so its history accrues
    if hs is not None:                       # from registration, though it enters the verdict at P2
        cols["hs300"] = hs

    fresh = pd.DataFrame(cols) if cols else None
    if fresh is None and cached is None:
        raise ForwardDataUnavailable("no silicon/solar prices and no cache")
    if cached is None:
        panel = fresh
    elif fresh is None:
        panel = cached
    else:                                    # prefer fresh columns, keep cached where fresh missing
        panel = cached.copy()
        for c in fresh.columns:
            panel[c] = fresh[c].combine_first(panel[c]) if c in panel else fresh[c]
        panel = panel.reindex(panel.index.union(fresh.index)).sort_index()
    if "silicon" not in panel or "solar" not in panel:
        raise ForwardDataUnavailable("silicon or solar column missing after merge")
    return panel.sort_index()


def confirmed_peaks_after(silicon: pd.Series, ref_level: float, after: pd.Timestamp) -> list:
    """Months t (strictly after `after`) whose close is the max of the centered window
    [t-WIN, t+WIN] AND exceeds ref_level. Only months with a fully observed forward window
    (t+WIN present) are confirmable, so a peak is not counted until 12 months have passed."""
    s = silicon.dropna()
    out = []
    for i in range(len(s)):
        ts = s.index[i]
        if ts <= after:
            continue
        lo, hi = i - WIN, i + WIN
        if lo < 0 or hi >= len(s):           # window not fully observed -> not yet confirmable
            continue
        val = float(s.iloc[i])
        if val >= float(s.iloc[lo:hi + 1].max()) and val > ref_level:
            out.append((ts, val))
    return out


def _dd(series: pd.Series) -> dict:
    s = series.dropna()
    hi_idx = s.idxmax()
    return {"high": round(float(s.max()), 2), "high_month": hi_idx.strftime("%Y-%m"),
            "current": round(float(s.iloc[-1]), 2),
            "drawdown_from_high": round(float(s.iloc[-1] / s.max() - 1.0), 4)}


def build_report(panel: pd.DataFrame) -> dict:
    silicon, solar = panel["silicon"].dropna(), panel["solar"].dropna()
    ref_level = float(silicon.reindex([REG_MONTH]).iloc[0]) if REG_MONTH in silicon.index \
        else float(silicon.iloc[-1])

    # descriptive-only historical read (N=1, contaminated) -----------------------------------
    overlap = pd.concat([silicon, solar], axis=1, keys=["silicon", "solar"]).dropna()
    r = np.log(overlap).diff().dropna()
    corr = float(r["silicon"].corr(r["solar"])) if len(r) > 2 else float("nan")
    silicon_peak_predates = bool(silicon.index.min() > solar.idxmax())

    # forward peak search (empty until a post-registration peak confirms, ~2030+) --------------
    peaks = confirmed_peaks_after(silicon, ref_level, REG_MONTH)
    exceeded = bool((silicon.loc[silicon.index > REG_MONTH] > ref_level).any())

    if peaks:
        pk_month, pk_val = peaks[0]
        in_window = (pd.Timestamp(PEAK_WINDOW[0]) <= pk_month
                     <= pd.Timestamp(PEAK_WINDOW[1]) + pd.offsets.MonthEnd(0))
        fwd = solar.index[solar.index >= pk_month + pd.DateOffset(months=WIN)]
        status = "PEAK_CONFIRMED_AWAITING_P2" if len(fwd) == 0 else "EVALUABLE"
    else:
        pk_month, pk_val, in_window, status = None, None, None, (
            "AWAITING_PEAK" if date.today() <= DEGENERATE_BY.date()
            else "REJECTED_NO_PEAK_BY_DEADLINE")

    return {
        "mode": "forward_registration",
        "issue": 18,
        "registration_date": REG_DATE,
        "as_of": panel.index.max().strftime("%Y-%m"),
        "run_date": date.today().strftime("%Y-%m-%d"),
        "frozen_operationalization": {
            "silicon": "GFEX industrial-silicon futures main-continuous (SI0), monthly last close",
            "solar_equity": "CSI Solar Industry index 931151, monthly close",
            "benchmark": "CSI 300 (000300), monthly close (cached from registration, used at P2)",
            "peak_def": "centered 25-month max [t-12,t+12] and > registration-month close",
            "P1_timing_window": list(PEAK_WINDOW),
            "P2_mechanism": "931151 minus 000300 return over the 12 months after the peak < 0",
            "degenerate_reject_by": DEGENERATE_BY.strftime("%Y-%m"),
        },
        "baseline_snapshot": {
            "reference_month": REG_MONTH.strftime("%Y-%m"),
            "silicon_ref_close": round(ref_level, 2),
            "silicon": _dd(silicon),
            "solar": _dd(solar),
            "solar_etf": _dd(panel["solar_etf"]) if "solar_etf" in panel else None,
        },
        "descriptive_historical_N1": {
            "note": ("DESCRIPTIVE ONLY, NOT A VERDICT. One modern cycle; contaminated (the friend "
                     "cited the 2021-22 episode after the fact). The cited silicon takeoff is NOT "
                     "measurable on free data: silicon futures began 2022-12, after the solar top."),
            "overlap_months": int(len(overlap)),
            "silicon_futures_start": silicon.index.min().strftime("%Y-%m"),
            "solar_series_high_month": solar.idxmax().strftime("%Y-%m"),
            "silicon_start_after_solar_top": silicon_peak_predates,
            "monthly_logret_corr_over_overlap": round(corr, 3),
        },
        "forward_status": {
            "status": status,
            "silicon_exceeded_ref_since_registration": exceeded,
            "next_confirmed_peak_month": pk_month.strftime("%Y-%m") if pk_month else None,
            "next_confirmed_peak_close": round(pk_val, 2) if pk_val is not None else None,
            "next_confirmed_peak_in_P1_window": in_window,
        },
    }


def _fmt(report: dict) -> str:
    b, d, f = (report["baseline_snapshot"], report["descriptive_historical_N1"],
               report["forward_status"])
    L = []
    L.append(f"SILICON -> SOLAR top-timing (issue #18) -- forward registration, as of "
             f"{report['as_of']} (run {report['run_date']})")
    L.append("\n  BASELINE (frozen at registration):")
    L.append(f"    silicon SI0  ref {b['silicon_ref_close']:>9.1f}  |  series high "
             f"{b['silicon']['high']:.0f} ({b['silicon']['high_month']}), now "
             f"{b['silicon']['current']:.0f}, {b['silicon']['drawdown_from_high']:+.1%} from high")
    L.append(f"    solar 931151 {'':>9}  |  series high {b['solar']['high']:.1f} "
             f"({b['solar']['high_month']}), now {b['solar']['current']:.1f}, "
             f"{b['solar']['drawdown_from_high']:+.1%} from high")
    L.append("\n  DESCRIPTIVE (N=1, NOT a verdict):")
    L.append(f"    silicon futures start {d['silicon_futures_start']}; solar top "
             f"{d['solar_series_high_month']}; silicon data begins AFTER the solar top: "
             f"{d['silicon_start_after_solar_top']}")
    L.append(f"    -> the friend's cited episode is UNVERIFIABLE on free data; overlap "
             f"{d['overlap_months']}m, monthly log-return corr {d['monthly_logret_corr_over_overlap']}")
    L.append("\n  FORWARD STATUS:")
    L.append(f"    {f['status']} -- next confirmed peak: {f['next_confirmed_peak_month']} "
             f"(in [2029-01,2031-12]: {f['next_confirmed_peak_in_P1_window']}); silicon back above "
             f"ref since registration: {f['silicon_exceeded_ref_since_registration']}")
    return "\n".join(L)


def main() -> None:
    ensure_dirs()
    panel = pull_panel()
    atomic_to_parquet(panel, LAKE)
    report = build_report(panel)
    atomic_write_text(json.dumps(report, ensure_ascii=False, indent=2), REG_JSON)
    print(_fmt(report))
    print(f"\nsaved -> {LAKE}\nsaved -> {REG_JSON}")


if __name__ == "__main__":
    main()
