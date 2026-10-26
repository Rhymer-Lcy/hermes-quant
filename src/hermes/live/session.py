"""Which trading session should the live record contain right now? Answered in Asia/Shanghai time
from a validated exchange calendar -- never from the machine's local date, and never from one
reference stock's latest bar.

  * Clock: Beijing time as a fixed UTC+08:00 offset (China observes no DST, so the offset is exact
    and needs no tz database). The machine runs on US Eastern time, so `date.today()` there can be a
    day behind the session being recorded.
  * Calendar: BaoStock `query_trade_dates` over a window around today, validated before use
    (`validate_calendar`): one row per calendar day with no gaps or duplicates, a 0/1 flag, no
    Saturday or Sunday session, and no run of closed days longer than any real A-share holiday. After
    the refresh it is cross-checked against the lake itself (`corroborate_with_lake`): the dates the
    lake holds bars for must be exactly the calendar's sessions.
  * Expected session: today if today is a session and its EOD publication is due (PUBLICATION_DUE),
    else the previous session. A weekend or holiday therefore expects the last session before it.
  * Assessment (`assess_session`): classifies the lake against that expectation as
    current_session / stale_source / incomplete_publication. The live refresh refuses (retryable,
    exit 75) anything but current_session before a canonical record is written.

A calendar that cannot be fetched is transient (BaoStockUnavailable, retried). A calendar that
fails validation or contradicts the lake raises CalendarInvalid, which is fatal: the record stops
until a person looks, rather than trusting either source."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

BEIJING = timezone(timedelta(hours=8), "Asia/Shanghai")
PUBLICATION_DUE = time(18, 0)     # 15:00 close + the ~2-3 h BaoStock EOD posting window (paper_trading.md)
CALENDAR_LOOKBACK_DAYS = 40
CALENDAR_LOOKAHEAD_DAYS = 7
MAX_CLOSED_RUN_DAYS = 12          # longer than any Spring Festival / National Day closure incl. weekends
MIN_SESSION_COVERAGE = 0.90       # share of current members that must carry the session's bar

CURRENT_SESSION = "current_session"
STALE_SOURCE = "stale_source"
INCOMPLETE_PUBLICATION = "incomplete_publication"
HISTORICAL_RECOMPUTE = "historical_recompute"


class CalendarInvalid(RuntimeError):
    """The trade calendar failed validation or contradicts the lake (fatal: needs a person)."""


def beijing_now() -> datetime:
    return datetime.now(BEIJING)


@dataclass(frozen=True)
class TradeCalendar:
    start: date
    end: date
    sessions: frozenset

    def is_session(self, d: date) -> bool:
        self._check_range(d)
        return d in self.sessions

    def previous_session(self, d: date) -> date:
        """The last session strictly before `d` inside the calendar window."""
        self._check_range(d)
        earlier = [s for s in self.sessions if s < d]
        if not earlier:
            raise CalendarInvalid(f"no session in the calendar window before {d}")
        return max(earlier)

    def sessions_between(self, first: date, last: date) -> list[date]:
        return sorted(s for s in self.sessions if first <= s <= last)

    def _check_range(self, d: date) -> None:
        if not self.start <= d <= self.end:
            raise CalendarInvalid(f"{d} lies outside the calendar window {self.start} .. {self.end}")


def validate_calendar(rows: list[tuple[str, str]], start: date, end: date) -> TradeCalendar:
    """Build a TradeCalendar from (calendar_date, is_trading_day) rows, or raise CalendarInvalid."""
    if not rows:
        raise CalendarInvalid(f"empty trade calendar for {start} .. {end}")
    try:
        parsed = [(datetime.strptime(d, "%Y-%m-%d").date(), flag) for d, flag in rows]
    except (TypeError, ValueError) as exc:
        raise CalendarInvalid(f"unparseable calendar date: {exc}") from exc
    bad_flags = sorted({flag for _, flag in parsed} - {"0", "1"})
    if bad_flags:
        raise CalendarInvalid(f"is_trading_day values outside 0/1: {bad_flags}")
    days = [d for d, _ in parsed]
    expected = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if days != expected:
        missing = sorted(set(expected) - set(days))
        extra = sorted(set(days) - set(expected))
        raise CalendarInvalid(f"calendar is not one row per day for {start} .. {end}: "
                              f"missing {missing[:5]}, unexpected {extra[:5]}, "
                              f"{len(days) - len(set(days))} duplicate(s)")
    sessions = frozenset(d for d, flag in parsed if flag == "1")
    weekend = sorted(d for d in sessions if d.weekday() >= 5)
    if weekend:
        raise CalendarInvalid(f"calendar marks weekend days as sessions: {weekend[:5]}")
    if not sessions:
        raise CalendarInvalid(f"calendar has no session in {start} .. {end}")
    run = longest = 0
    for d in expected:
        run = 0 if d in sessions else run + 1
        longest = max(longest, run)
    if longest > MAX_CLOSED_RUN_DAYS:
        raise CalendarInvalid(f"calendar shows {longest} consecutive closed days "
                              f"(more than any real closure, {MAX_CLOSED_RUN_DAYS})")
    return TradeCalendar(start=start, end=end, sessions=sessions)


def fetch_calendar(now: datetime | None = None) -> TradeCalendar:
    """The validated calendar around `now` (Beijing). Fetch failures are transient
    (BaoStockUnavailable); a response that fails validation raises CalendarInvalid."""
    import baostock as bs

    from ..data.sources import baostock_source as bss

    now = now or beijing_now()
    start = now.date() - timedelta(days=CALENDAR_LOOKBACK_DAYS)
    end = now.date() + timedelta(days=CALENDAR_LOOKAHEAD_DAYS)
    with bss.session():
        rs = bs.query_trade_dates(start_date=start.isoformat(), end_date=end.isoformat())
        if rs.error_code != "0":
            raise bss.BaoStockUnavailable(f"trade calendar query failed: {rs.error_code} {rs.error_msg}")
        rows = []
        while rs.next():
            row = rs.get_row_data()
            rows.append((row[0], row[1]))
    return validate_calendar(rows, start, end)


def expected_session(cal: TradeCalendar, now: datetime, due: time = PUBLICATION_DUE) -> date:
    """Today if it is a session whose EOD publication is due, else the previous session."""
    today = now.astimezone(BEIJING).date()
    if cal.is_session(today) and now.astimezone(BEIJING).time() >= due:
        return today
    return cal.previous_session(today)


def corroborate_with_lake(cal: TradeCalendar, lake_dates, upto: date) -> None:
    """The lake's bar dates and the calendar's sessions must coincide on [cal.start, upto]."""
    held = {pd.Timestamp(d).date() for d in lake_dates}
    lo = cal.start
    window = {d for d in held if lo <= d <= upto}
    sessions = set(cal.sessions_between(lo, upto))
    extra, missing = sorted(window - sessions), sorted(sessions - window)
    if extra or missing:
        raise CalendarInvalid(f"calendar and lake disagree on {lo} .. {upto}: bars on non-session "
                              f"days {extra[:5]}, sessions without bars {missing[:5]}")


@dataclass(frozen=True)
class SessionCheck:
    evidence: str                 # current_session / stale_source / incomplete_publication
    expected_session: date
    session_date: date | None     # the session the lake was verified to hold (None unless current)
    lake_latest: date
    coverage: float               # share of members with a bar on the assessed date
    checked_at: datetime

    def report_fields(self) -> dict:
        return {"evidence": self.evidence,
                "expected_session": self.expected_session.isoformat(),
                "session_date": self.session_date.isoformat() if self.session_date else None,
                "session_coverage": round(self.coverage, 4),
                "session_checked_at": self.checked_at.isoformat(timespec="seconds")}


def assess_session(panel: pd.DataFrame, members: list[str], cal: TradeCalendar, now: datetime,
                   min_coverage: float = MIN_SESSION_COVERAGE) -> SessionCheck:
    """Classify the lake (a date x code close panel) against the expected session.

    current_session: the lake's latest date is a session no earlier than the expected one (a run
    before PUBLICATION_DUE may already hold today's bars) and at least `min_coverage` of `members`
    carry a bar on it. stale_source: the lake ends before the expected session. Raises
    CalendarInvalid for a bar dated on a non-session day or in the future."""
    now = now.astimezone(BEIJING)
    expected = expected_session(cal, now)
    latest = pd.Timestamp(panel.index.max()).date()
    if latest > now.date():
        raise CalendarInvalid(f"the lake holds a bar dated {latest}, after today {now.date()}")
    if not cal.is_session(latest):
        raise CalendarInvalid(f"the lake's latest bar {latest} is not a session day")
    cols = [c for c in members if c in panel.columns]
    if latest < expected:                       # no member can carry a bar the lake does not reach
        return SessionCheck(STALE_SOURCE, expected, None, latest, 0.0, now)
    cov = float(panel.loc[pd.Timestamp(latest), cols].notna().mean()) if cols else 0.0
    if cov < min_coverage:
        return SessionCheck(INCOMPLETE_PUBLICATION, expected, None, latest, cov, now)
    return SessionCheck(CURRENT_SESSION, expected, latest, latest, cov, now)
