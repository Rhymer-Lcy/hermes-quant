"""Coverage of the expected session counts EVERY current index member. A member with no lake file
used to fall out of both sides of the ratio (load_close_panel skips a missing file), so a lake missing
whole members read as fully covered. The four ways a member can lack the session's bar are reported
separately: no lake file (its own blocking decision), a failed pull this run, a verified delisting
(the only case removed from the denominator), and an unexplained absence. A suspended day is a row
carrying the previous close and counts as covered."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from hermes.live import feed
from hermes.live import session as ss

START, END = date(2026, 9, 1), date(2026, 11, 2)
HOLIDAY = {date(2026, 10, d) for d in range(1, 8)}


def cal():
    rows, d = [], START
    while d <= END:
        rows.append((d.isoformat(), "1" if d.weekday() < 5 and d not in HOLIDAY else "0"))
        d += timedelta(days=1)
    return ss.validate_calendar(rows, START, END)


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ss.BEIJING)


MEMBERS = [f"sh.6{i:05d}" for i in range(200)]
SESSION = date(2026, 10, 26)
NOW = at(2026, 10, 26, 18, 30)


def lake(absent_files=0, no_bar=0, suspended=0):
    """Close and volume panels as load_close_panel would build them: a member without a file has no
    column at all; `no_bar` members have a file but no row on the session; `suspended` members carry
    the previous close with no volume."""
    idx = pd.to_datetime(cal().sessions_between(date(2026, 10, 12), SESSION))
    cols = MEMBERS[absent_files:]
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    volume = pd.DataFrame(1e6, index=idx, columns=cols)
    gone = cols[:no_bar]
    close.loc[idx[-1], gone] = np.nan
    volume.loc[idx[-1], gone] = np.nan
    halted = cols[no_bar:no_bar + suspended]
    volume.loc[idx[-1], halted] = 0.0
    return close, volume


def assess(close, volume=None, **kw):
    return ss.assess_session(close, MEMBERS, cal(), NOW, volume=volume, **kw)


# ---- the false positive this gate closes ---------------------------------------------------
def test_members_without_a_lake_file_stay_in_the_denominator():
    """21 of 200 members have no file. The old ratio was 179/179 = 100% and the session passed."""
    close, volume = lake(absent_files=21)
    chk = assess(close, volume)
    assert chk.coverage == pytest.approx(179 / 200)
    assert chk.evidence != ss.CURRENT_SESSION
    assert chk.completeness.missing_file == tuple(MEMBERS[:21])


def test_one_missing_lake_file_blocks_although_the_ratio_would_pass():
    """199/200 clears 90%, but a current member with no data at all cannot be classified (suspended?
    delisted? never pulled?): its own decision, retryable."""
    close, volume = lake(absent_files=1)
    chk = assess(close, volume)
    assert chk.coverage == pytest.approx(199 / 200)
    assert chk.evidence == ss.INCOMPLETE_LAKE and chk.session_date is None
    assert chk.completeness.missing_file == (MEMBERS[0],)


def test_the_legacy_ratio_counts_a_member_without_a_column_as_uncovered():
    close, _ = lake(absent_files=21)
    _latest, cov = feed.latest_coverage(close, MEMBERS)
    assert cov == pytest.approx(179 / 200)


# ---- legitimate states that must not fail --------------------------------------------------
def test_suspended_members_count_as_covered_and_are_reported():
    close, volume = lake(suspended=40)                 # 20% suspended: still a complete session
    chk = assess(close, volume)
    assert chk.evidence == ss.CURRENT_SESSION and chk.coverage == 1.0
    assert len(chk.completeness.suspended) == 40


def test_a_complete_lake_is_current_without_any_lookup():
    close, volume = lake()

    def never(codes):
        raise AssertionError("no member lacks the bar, so nothing needs a lookup")

    chk = assess(close, volume, terminal=never)
    assert chk.evidence == ss.CURRENT_SESSION and chk.coverage == 1.0


def test_verified_delistings_leave_the_denominator():
    close, volume = lake(no_bar=25)                    # 175/200 = 87.5% < 90% before the lookup
    gone = MEMBERS[:25]
    chk = assess(close, volume, terminal=lambda codes: {c: pd.Timestamp("2026-10-19") for c in codes})
    assert chk.completeness.terminal == tuple(gone)
    assert chk.coverage == 1.0 and chk.evidence == ss.CURRENT_SESSION


def test_a_delisting_dated_after_the_session_does_not_leave_the_denominator():
    close, volume = lake(no_bar=25)
    chk = assess(close, volume, terminal=lambda codes: {c: pd.Timestamp("2026-10-27") for c in codes})
    assert chk.completeness.terminal == () and len(chk.completeness.no_bar) == 25
    assert chk.evidence == ss.INCOMPLETE_PUBLICATION


def test_a_failed_pull_is_reported_and_counted_uncovered():
    close, volume = lake(no_bar=5)
    chk = assess(close, volume, pull_errors=set(MEMBERS[:5]), terminal=lambda codes: {})
    assert chk.completeness.pull_error == tuple(MEMBERS[:5])
    assert chk.coverage == pytest.approx(195 / 200) and chk.evidence == ss.CURRENT_SESSION


def test_a_failed_pull_is_never_taken_for_a_delisting():
    close, volume = lake(no_bar=25)
    seen = []

    def terminal(codes):
        seen.extend(codes)
        return {c: pd.Timestamp("2026-10-19") for c in codes}

    chk = assess(close, volume, pull_errors=set(MEMBERS[:25]), terminal=terminal)
    assert seen == [] and chk.completeness.terminal == ()
    assert chk.evidence == ss.INCOMPLETE_PUBLICATION


def test_a_half_published_session_is_decided_without_hundreds_of_lookups():
    close, volume = lake(no_bar=100)

    def never(codes):
        raise AssertionError(f"{len(codes)} lookups on a half-published day")

    chk = assess(close, volume, terminal=never)
    assert chk.evidence == ss.INCOMPLETE_PUBLICATION and chk.coverage == pytest.approx(0.5)


def test_the_report_names_every_category():
    close, volume = lake(absent_files=1, no_bar=2, suspended=3)
    f = assess(close, volume, pull_errors={MEMBERS[1]}, terminal=lambda codes: {}).report_fields()
    assert f["session_members"] == 200 and f["session_members_with_bar"] == 197
    assert f["session_suspended"] == 3
    assert f["session_missing_files"] == [MEMBERS[0]]
    assert f["session_pull_errors"] == [MEMBERS[1]]
    assert f["session_terminal"] == [] and f["session_without_bar"] == [MEMBERS[2]]
