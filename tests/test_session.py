"""Expected-session logic: a validated calendar in Beijing time decides which session the live
record must hold; a stale or incomplete lake is classified, never passed as current."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from hermes.data.sources.baostock_source import BaoStockUnavailable
from hermes.live import session as ss

START, END = date(2026, 9, 1), date(2026, 11, 2)
HOLIDAY = {date(2026, 10, d) for d in range(1, 8)}          # National Day closure, 2026-10-01..07


def rows(start=START, end=END, holiday=HOLIDAY):
    out, d = [], start
    while d <= end:
        out.append((d.isoformat(), "1" if d.weekday() < 5 and d not in holiday else "0"))
        d += timedelta(days=1)
    return out


def cal():
    return ss.validate_calendar(rows(), START, END)


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ss.BEIJING)


# ---- calendar validation -------------------------------------------------------------------
def test_a_real_shaped_calendar_validates():
    c = cal()
    assert c.is_session(date(2026, 9, 30)) and not c.is_session(date(2026, 10, 5))
    assert c.previous_session(date(2026, 10, 8)) == date(2026, 9, 30)


@pytest.mark.parametrize("mutate, match", [
    (lambda r: [], "empty"),
    (lambda r: r[:10] + r[11:], "not one row per day"),
    (lambda r: r[:10] + [r[10]] + r[10:], "not one row per day"),
    (lambda r: [(d, "2" if i == 3 else f) for i, (d, f) in enumerate(r)], "outside 0/1"),
    (lambda r: [(d, "1") if d == "2026-09-05" else (d, f) for d, f in r], "weekend"),
    (lambda r: [(d, "0") if "2026-09-07" <= d <= "2026-09-25" else (d, f) for d, f in r], "consecutive closed"),
    (lambda r: [("2026/09/01", r[0][1])] + r[1:], "unparseable"),
])
def test_a_malformed_calendar_is_rejected(mutate, match):
    with pytest.raises(ss.CalendarInvalid, match=match):
        ss.validate_calendar(mutate(rows()), START, END)


# ---- expected session ----------------------------------------------------------------------
@pytest.mark.parametrize("now, want", [
    (at(2026, 10, 26, 18, 0), date(2026, 10, 26)),     # Monday, publication due
    (at(2026, 10, 26, 17, 59), date(2026, 10, 23)),    # Monday before due -> Friday
    (at(2026, 10, 26, 9, 0), date(2026, 10, 23)),      # Monday morning -> Friday
    (at(2026, 10, 24, 12, 0), date(2026, 10, 23)),     # Saturday -> Friday
    (at(2026, 10, 25, 23, 0), date(2026, 10, 23)),     # Sunday -> Friday
    (at(2026, 10, 5, 20, 0), date(2026, 9, 30)),       # inside the holiday -> last session before it
    (at(2026, 10, 8, 17, 0), date(2026, 9, 30)),       # first session back, before due
    (at(2026, 10, 8, 19, 0), date(2026, 10, 8)),       # first session back, after due
])
def test_expected_session(now, want):
    assert ss.expected_session(cal(), now) == want


def test_the_machine_zone_does_not_matter():
    """06:00 US Eastern on 2026-10-26 is 18:00 in Beijing: the session of 10-26 is due."""
    eastern = datetime(2026, 10, 26, 6, 0, tzinfo=ss.timezone(timedelta(hours=-4)))
    assert ss.expected_session(cal(), eastern) == date(2026, 10, 26)


def test_a_date_outside_the_window_is_refused():
    with pytest.raises(ss.CalendarInvalid, match="outside the calendar window"):
        ss.expected_session(cal(), at(2026, 12, 1, 20))


# ---- corroboration with the lake -----------------------------------------------------------
def lake_dates(upto=date(2026, 10, 23), drop=(), add=()):
    ds = [d for d in cal().sessions_between(START, upto) if d not in drop] + list(add)
    return pd.to_datetime(sorted(ds))


def test_lake_and_calendar_agree():
    ss.corroborate_with_lake(cal(), lake_dates(), date(2026, 10, 23))


def test_a_lake_bar_on_a_holiday_contradicts_the_calendar():
    with pytest.raises(ss.CalendarInvalid, match="bars on non-session days"):
        ss.corroborate_with_lake(cal(), lake_dates(add=[date(2026, 10, 5)]), date(2026, 10, 23))


def test_a_session_missing_from_the_lake_contradicts_the_calendar():
    with pytest.raises(ss.CalendarInvalid, match="sessions without bars"):
        ss.corroborate_with_lake(cal(), lake_dates(drop=[date(2026, 10, 14)]), date(2026, 10, 23))


# ---- assessment ----------------------------------------------------------------------------
MEMBERS = [f"sh.6{i:05d}" for i in range(200)]


def panel(last: date, missing_on_last: int = 0):
    idx = pd.to_datetime(cal().sessions_between(date(2026, 10, 12), last))
    p = pd.DataFrame(1.0, index=idx, columns=MEMBERS)
    if missing_on_last:
        p.iloc[-1, :missing_on_last] = np.nan
    return p


def test_current_session_with_a_few_names_absent():
    chk = ss.assess_session(panel(date(2026, 10, 26), missing_on_last=1), MEMBERS, cal(),
                            at(2026, 10, 26, 18, 0))
    assert chk.evidence == ss.CURRENT_SESSION and chk.session_date == date(2026, 10, 26)
    assert chk.coverage == pytest.approx(199 / 200)


def test_an_old_lake_with_full_coverage_is_stale_not_current():
    """The audit's defect: the old gate measured coverage on the lake's own last date."""
    chk = ss.assess_session(panel(date(2026, 10, 23)), MEMBERS, cal(), at(2026, 10, 26, 18, 0))
    assert chk.evidence == ss.STALE_SOURCE and chk.session_date is None
    assert chk.expected_session == date(2026, 10, 26) and chk.lake_latest == date(2026, 10, 23)


def test_a_half_published_session_is_incomplete():
    chk = ss.assess_session(panel(date(2026, 10, 26), missing_on_last=100), MEMBERS, cal(),
                            at(2026, 10, 26, 18, 30))
    assert chk.evidence == ss.INCOMPLETE_PUBLICATION and chk.session_date is None


def test_a_weekend_run_on_fridays_data_is_current():
    chk = ss.assess_session(panel(date(2026, 10, 23)), MEMBERS, cal(), at(2026, 10, 24, 10))
    assert chk.evidence == ss.CURRENT_SESSION and chk.session_date == date(2026, 10, 23)


def test_bars_published_before_the_due_time_count():
    chk = ss.assess_session(panel(date(2026, 10, 26)), MEMBERS, cal(), at(2026, 10, 26, 17, 0))
    assert chk.evidence == ss.CURRENT_SESSION and chk.session_date == date(2026, 10, 26)


def test_a_bar_dated_on_a_holiday_or_in_the_future_is_rejected():
    p = panel(date(2026, 10, 23))
    holiday = p.copy()
    holiday.loc[pd.Timestamp("2026-10-24")] = 1.0
    with pytest.raises(ss.CalendarInvalid, match="not a session day"):
        ss.assess_session(holiday, MEMBERS, cal(), at(2026, 10, 24, 20))
    with pytest.raises(ss.CalendarInvalid, match="after today"):
        ss.assess_session(panel(date(2026, 10, 26)), MEMBERS, cal(), at(2026, 10, 25, 20))


def test_report_fields_are_iso_strings():
    chk = ss.assess_session(panel(date(2026, 10, 26)), MEMBERS, cal(), at(2026, 10, 26, 18, 5))
    f = chk.report_fields()
    assert f == {"evidence": "current_session", "expected_session": "2026-10-26",
                 "session_date": "2026-10-26", "session_coverage": 1.0,
                 "session_checked_at": "2026-10-26T18:05:00+08:00"}


# ---- fetching -------------------------------------------------------------------------------
class _RS:
    def __init__(self, rows, code="0"):
        self._rows, self.error_code, self.error_msg = list(rows), code, "boom"

    def next(self):
        return bool(self._rows)

    def get_row_data(self):
        return list(self._rows.pop(0))


def _fake_bs(monkeypatch, rs):
    import contextlib

    import baostock

    from hermes.data.sources import baostock_source as bss
    monkeypatch.setattr(bss, "session", contextlib.nullcontext)
    monkeypatch.setattr(baostock, "query_trade_dates", lambda start_date, end_date: rs(start_date, end_date))


def test_fetch_builds_the_window_around_beijing_today(monkeypatch):
    seen = {}

    def rs(s, e):
        seen["range"] = (s, e)
        return _RS(rows(date.fromisoformat(s), date.fromisoformat(e), HOLIDAY))
    _fake_bs(monkeypatch, rs)
    c = ss.fetch_calendar(at(2026, 10, 26, 18))
    assert seen["range"] == ("2026-09-16", "2026-11-02") and c.is_session(date(2026, 10, 26))


def test_a_failed_calendar_query_is_transient(monkeypatch):
    _fake_bs(monkeypatch, lambda s, e: _RS([], code="10002007"))
    with pytest.raises(BaoStockUnavailable, match="trade calendar query failed"):
        ss.fetch_calendar(at(2026, 10, 26, 18))


def test_an_empty_calendar_response_is_invalid(monkeypatch):
    _fake_bs(monkeypatch, lambda s, e: _RS([]))
    with pytest.raises(ss.CalendarInvalid, match="empty"):
        ss.fetch_calendar(at(2026, 10, 26, 18))
