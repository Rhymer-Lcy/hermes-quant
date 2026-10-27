"""End-to-end gate for scripts/paper_live.py on a synthetic lake (tmp_path; nothing real is read or
written): the canonical record is written only with current-session evidence; a stale, incomplete
or unverifiable lake leaves every canonical file untouched and exits 75 (or 1 for a contradictory
calendar); a --as-of replay never rewrites the canonical directory."""
import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes.data import lake as lake_mod
from hermes.data.sources.baostock_source import BaoStockUnavailable
from hermes.live import feed, paper
from hermes.live import session as ss

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper_live.py"
HOLIDAY = {date(2026, 10, d) for d in range(1, 8)}
CODES = [f"sh.6000{i:02d}" for i in range(12)]
TIERS = ["--tiers", "100000", "1000000"]


def sessions(first=date(2026, 2, 2), last=date(2026, 11, 6)):
    out, d = [], first
    while d <= last:
        if d.weekday() < 5 and d not in HOLIDAY:
            out.append(d)
        d += timedelta(days=1)
    return out


def calendar_rows(start, end):
    out, d = [], start
    while d <= end:
        out.append((d.isoformat(), "1" if d.weekday() < 5 and d not in HOLIDAY else "0"))
        d += timedelta(days=1)
    return out


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ss.BEIJING)


class Env:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.daily = tmp_path / "daily"
        self.daily.mkdir()
        self.paper_dir = tmp_path / "paper"
        self.paper_dir.mkdir()
        spec = importlib.util.spec_from_file_location("paper_live_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        rng = np.random.default_rng(7)
        days = pd.to_datetime([d for d in sessions() if d <= date(2026, 10, 26)])
        self.frames = {}
        for i, code in enumerate(CODES):
            close = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
            self.frames[code] = pd.DataFrame({"date": days, "code": code, "close": close,
                                              "peTTM": 5.0 + 2 * i, "volume": 1e6})
        month_ends = pd.Series(days, index=days).groupby(days.to_period("M")).max()
        self.mdf = pd.DataFrame([(d, c) for d in month_ends for c in CODES], columns=["date", "code"])
        self.mdf = self.mdf[self.mdf["date"] <= "2026-09-30"].reset_index(drop=True)
        self.membership = tmp_path / "membership.parquet"
        self.mdf.to_parquet(self.membership, index=False)
        self.calendar_calls = 0
        mp = monkeypatch
        mp.setattr(lake_mod, "PARQUET_DIR", tmp_path)
        for mod in (paper, self.mod):
            mp.setattr(mod, "MEMBERSHIP_PARQUET", self.membership)
            mp.setattr(mod, "PAPER_DIR", self.paper_dir)
        mp.setattr(paper, "ensure_dirs", lambda: None)
        mp.setattr(feed, "extend_membership",
                   lambda end: (self.mdf, sorted(self.mdf["code"].unique()), []))
        self.pull_errors: set[str] = set()
        self.delisted: dict[str, date] = {}
        self.terminal_calls: list[list[str]] = []
        self.terminal_override = None
        mp.setattr(feed, "update_daily_bars", self.pull_summary)
        mp.setattr(feed, "terminal_dates", self.terminal)
        mp.setattr(ss, "fetch_calendar", self.fetch)
        self.now = None
        self.calendar_override = None

    def pull_summary(self, union, end):
        return pd.DataFrame({"code": union, "rows": 1,
                             "status": ["error: 10002007" if c in self.pull_errors else "ok" for c in union],
                             "mode": "append:+1"})

    def terminal(self, codes):
        self.terminal_calls.append(list(codes))
        if isinstance(self.terminal_override, Exception):
            raise self.terminal_override
        return {c: pd.Timestamp(self.delisted[c]) if c in self.delisted else None for c in codes}

    def fetch(self, now):
        self.calendar_calls += 1
        if isinstance(self.calendar_override, Exception):
            raise self.calendar_override
        start, end = now.date() - timedelta(days=40), now.date() + timedelta(days=7)
        return ss.validate_calendar(calendar_rows(start, end), start, end)

    def lake_through(self, last: date, missing_on_last=0, extra_bar=None, absent=(), suspended=()):
        """`absent`: codes with NO row on `last`; `suspended`: a row on `last` carrying the previous
        close and no volume (how BaoStock serves a suspended day)."""
        for code, df in self.frames.items():
            df = df[df["date"] <= pd.Timestamp(last)].copy()
            if extra_bar is not None:
                df = pd.concat([df, df.tail(1).assign(date=pd.Timestamp(extra_bar))], ignore_index=True)
            if code in CODES[:missing_on_last] or code in absent:
                df = df[df["date"] < pd.Timestamp(last)]
            if code in suspended:
                df.loc[df.index[-1], ["close", "volume"]] = [df["close"].iloc[-2], np.nan]
            df.to_parquet(self.daily / f"{code.replace('.', '_')}.parquet", index=False)

    def frames_truncated(self, code, last: date):
        """Rewrite one code's lake file to end on `last` (absent on every later session)."""
        df = self.frames[code]
        df[df["date"] <= pd.Timestamp(last)].to_parquet(
            self.daily / f"{code.replace('.', '_')}.parquet", index=False)

    def clock(self, now):
        self.mp.setattr(ss, "beijing_now", lambda: now)
        self.mp.setattr(paper, "beijing_now", lambda: now)

    def canonical(self):
        return {p.name: p.read_bytes() for p in sorted(self.paper_dir.iterdir())}

    def report(self, tier=100000):
        import json
        return json.loads((self.paper_dir / f"report_{tier}.json").read_text("utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def seed_canonical(env):
    """A previously written canonical record (the 10-23 one) that a refused run must not touch."""
    env.lake_through(date(2026, 10, 23))
    env.clock(at(2026, 10, 23, 18, 30))
    assert env.mod.run(TIERS) == 0
    return env.canonical()


def test_current_session_is_written_with_its_evidence(env):
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 0
    r = env.report()
    assert (r["as_of"], r["evidence"], r["session_date"], r["expected_session"]) == \
        ("2026-10-26", "current_session", "2026-10-26", "2026-10-26")
    assert r["run_date"] == "2026-10-26" and r["session_coverage"] == 1.0


def not_held(env, tier=1000000):
    held = set(env.report(tier)["positions"])
    return next(c for c in CODES if c not in held)


def test_a_member_without_a_lake_file_blocks_the_record(env, capsys):
    """11 of 12 members carry the session; the twelfth has no lake file at all. The old ratio dropped
    it from the denominator (11/11 = 100%) and wrote the record."""
    before = seed_canonical(env)
    code = not_held(env)
    env.lake_through(date(2026, 10, 26))
    (env.daily / f"{code.replace('.', '_')}.parquet").unlink()
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 75
    err = capsys.readouterr().err
    assert "incomplete_lake" in err and f"no lake file: ['{code}']" in err
    assert env.canonical() == before


def test_a_failed_pull_within_tolerance_is_written_and_reported(env):
    seed_canonical(env)
    code = not_held(env)
    env.lake_through(date(2026, 10, 26), absent=[code])
    env.pull_errors = {code}
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 0
    r = env.report(1000000)
    assert r["session_pull_errors"] == [code] and r["session_missing_files"] == []
    assert r["session_coverage"] == pytest.approx(11 / 12, abs=1e-4)


def test_stale_lake_exits_75_and_writes_nothing(env, capsys):
    before = seed_canonical(env)
    env.clock(at(2026, 10, 26, 18, 30))                 # Monday's session is due; lake ends Friday
    assert env.mod.run(TIERS) == 75
    assert "stale_source: the 2026-10-26 session is due but the lake ends 2026-10-23" in capsys.readouterr().err
    assert env.canonical() == before


def test_incomplete_publication_exits_75_and_writes_nothing(env, capsys):
    before = seed_canonical(env)
    env.lake_through(date(2026, 10, 26), missing_on_last=6)    # half the members lack Monday's bar
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 75
    assert "incomplete_publication" in capsys.readouterr().err
    assert env.canonical() == before


@pytest.mark.parametrize("now, lake_last", [(at(2026, 10, 24, 12), date(2026, 10, 23)),   # Saturday
                                            (at(2026, 10, 5, 20), date(2026, 9, 30)),     # holiday
                                            (at(2026, 10, 26, 17), date(2026, 10, 23))])  # before due
def test_no_newer_session_due_is_current(env, now, lake_last):
    env.lake_through(lake_last)
    env.clock(now)
    assert env.mod.run(TIERS) == 0
    r = env.report()
    assert r["evidence"] == "current_session" and r["session_date"] == lake_last.isoformat()


def test_unavailable_calendar_exits_75_and_writes_nothing(env, capsys):
    before = seed_canonical(env)
    env.lake_through(date(2026, 10, 26))
    env.calendar_override = BaoStockUnavailable("trade calendar query failed: 10002007")
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 75
    assert "trade calendar query failed" in capsys.readouterr().err
    assert env.canonical() == before


def test_calendar_contradicting_the_lake_is_fatal(env, capsys):
    before = seed_canonical(env)
    env.lake_through(date(2026, 10, 23), extra_bar=date(2026, 10, 24))   # a bar on a Saturday
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 1
    assert "CalendarInvalid: the lake's latest bar 2026-10-24 is not a session day" in capsys.readouterr().err
    assert env.canonical() == before


def test_a_session_missing_from_the_lake_is_fatal(env, capsys):
    """Every member lacks 2026-10-21 although the calendar says it was a session."""
    before = seed_canonical(env)
    env.lake_through(date(2026, 10, 26))
    for f in env.daily.glob("*.parquet"):
        df = pd.read_parquet(f)
        df[df["date"] != pd.Timestamp("2026-10-21")].to_parquet(f, index=False)
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 1
    assert "sessions without bars [datetime.date(2026, 10, 21)]" in capsys.readouterr().err
    assert env.canonical() == before


def test_no_refresh_on_a_stale_lake_writes_nothing(env, capsys):
    before = seed_canonical(env)
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(["--no-refresh", *TIERS]) == 75
    assert "stale_source" in capsys.readouterr().err
    assert env.canonical() == before


def test_no_refresh_on_a_current_lake_writes(env):
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 19))
    assert env.mod.run(["--no-refresh", *TIERS]) == 0
    assert env.report()["evidence"] == "current_session"


def test_replay_never_rewrites_the_canonical_record(env):
    before = seed_canonical(env)
    assert env.mod.run(["--as-of", "2026-09-30", "--no-refresh", *TIERS]) == 0
    assert env.canonical() == before and env.calendar_calls == 1   # only the seeding run asked


def test_replay_into_an_explicit_directory(env):
    before = seed_canonical(env)
    out = env.tmp / "replay"
    assert env.mod.run(["--as-of", "2026-09-30", "--no-refresh", "--out-dir", str(out), *TIERS]) == 0
    import json
    r = json.loads((out / "report_100000.json").read_text("utf-8"))
    assert r["as_of"] == "2026-09-30" and r["evidence"] == "historical_recompute"
    assert env.canonical() == before


def test_replay_into_the_canonical_directory_is_refused(env):
    before = seed_canonical(env)
    with pytest.raises(SystemExit) as exc:
        env.mod.run(["--as-of", "2026-09-30", "--no-refresh", "--out-dir", str(env.paper_dir), *TIERS])
    assert exc.value.code == 2 and env.canonical() == before


def _lake_bytes(env):
    return {p.name: p.read_bytes() for p in sorted(env.daily.iterdir())} | {
        "membership": env.membership.read_bytes()}


@pytest.mark.parametrize("extra", [[], ["--out-dir", "REPLAY"], ["--backtest"]])
def test_as_of_without_no_refresh_is_refused_before_any_pull(env, extra):
    """A replay must not pull, extend the membership or touch the lake: --as-of needs --no-refresh."""
    before = seed_canonical(env)
    lake_before = _lake_bytes(env)
    calls = []
    env.mp.setattr(env.mod, "refresh", lambda *a, **k: calls.append("refresh"))
    argv = ["--as-of", "2026-09-30", *[str(env.tmp / x) if x == "REPLAY" else x for x in extra], *TIERS]
    with pytest.raises(SystemExit) as exc:
        env.mod.run(argv)
    assert exc.value.code == 2 and calls == []
    assert env.canonical() == before and _lake_bytes(env) == lake_before
    assert not (env.tmp / "REPLAY").exists()


def test_the_canonical_directory_is_recognised_under_another_spelling(env):
    before = seed_canonical(env)
    variants = [str(env.paper_dir).upper(), str(env.paper_dir / ".." / "paper"),
                str(env.paper_dir).replace("\\", "/")]
    for v in variants:
        with pytest.raises(SystemExit) as exc:
            env.mod.run(["--as-of", "2026-09-30", "--no-refresh", "--out-dir", v, *TIERS])
        assert exc.value.code == 2, v
    assert env.canonical() == before


def test_rerun_of_the_same_session_is_byte_identical(env):
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 0
    first = env.canonical()
    assert env.mod.run(TIERS) == 0
    assert env.canonical() == first


# ---- crash inside the canonical write: torn states are detectable and self-heal --------------
def test_a_clean_write_commits_a_consistent_manifest(env):
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 0
    assert paper.verify_canonical_manifest(env.paper_dir) == ("consistent", [])
    import json
    m = json.loads((env.paper_dir / paper.CANONICAL_MANIFEST).read_text("utf-8"))
    assert (m["as_of"], m["evidence"], m["tiers"]) == ("2026-10-26", "current_session", [100000, 1000000])
    assert len(m["files"]) == 6


@pytest.mark.parametrize("fail_at", range(1, 8))          # 2 tiers x 3 files, then the manifest
def test_a_crash_at_every_write_boundary_is_detected_and_heals(env, fail_at):
    seed_canonical(env)                                    # a committed 10-23 record
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 18, 30))
    real_parquet, real_text = paper.atomic_to_parquet, paper.atomic_write_text
    count = {"n": 0}

    def faulty(real):
        def write(*a, **k):
            count["n"] += 1
            if count["n"] == fail_at:
                raise OSError(f"injected failure at write {fail_at}")
            return real(*a, **k)
        return write
    env.mp.setattr(paper, "atomic_to_parquet", faulty(real_parquet))
    env.mp.setattr(paper, "atomic_write_text", faulty(real_text))
    assert env.mod.run(TIERS) == 1                          # a write error is fatal, not retried
    state, problems = paper.verify_canonical_manifest(env.paper_dir)
    if fail_at == 1:                                        # nothing was replaced yet
        assert state == "consistent" and env.report()["as_of"] == "2026-10-23"
    else:
        assert state == "incomplete" and problems
    env.mp.setattr(paper, "atomic_to_parquet", real_parquet)
    env.mp.setattr(paper, "atomic_write_text", real_text)
    assert env.mod.run(TIERS) == 0                          # the next run rewrites everything
    assert paper.verify_canonical_manifest(env.paper_dir) == ("consistent", [])
    assert env.report()["as_of"] == "2026-10-26"


def test_a_tier_outside_the_last_commit_is_reported(env):
    env.lake_through(date(2026, 10, 26))
    env.clock(at(2026, 10, 26, 18, 30))
    assert env.mod.run(TIERS) == 0
    assert env.mod.run(["--tiers", "100000"]) == 0          # a manual subset run commits one tier
    state, problems = paper.verify_canonical_manifest(env.paper_dir)
    assert state == "incomplete" and problems == ["report_1000000.json: not covered by the last commit"]


# ---- right edge: a held name without a bar on the session ----------------------------------
def held_name(env, tier=1000000):
    held = sorted(env.report(tier)["positions"])
    assert held, "the synthetic book must hold names for these tests to mean anything"
    return held[0]


def monday_with(env, **lake):
    before = seed_canonical(env)
    code = held_name(env)
    env.lake_through(date(2026, 10, 26), **{k: [code] for k in lake})
    env.clock(at(2026, 10, 26, 18, 30))
    return before, code


def test_held_name_pull_error_blocks_although_the_pull_is_above_98pct(env, capsys):
    before, code = monday_with(env, absent=True)
    env.pull_errors = {code}                       # 1 of 12 failed: the session gate alone passes
    assert env.mod.run(TIERS) == 75
    assert f"'{code}': 'pull_error'" in capsys.readouterr().err
    assert env.canonical() == before and env.terminal_calls == []


def test_held_name_without_a_bar_or_a_delisting_record_blocks(env, capsys):
    before, code = monday_with(env, absent=True)
    assert env.mod.run(TIERS) == 75
    assert f"'{code}': 'missing_published_bar'" in capsys.readouterr().err
    assert env.canonical() == before and env.terminal_calls == [[code]]


def test_a_delisting_after_the_session_is_not_verified(env):
    before, code = monday_with(env, absent=True)
    env.delisted = {code: date(2026, 10, 27)}
    assert env.mod.run(TIERS) == 75 and env.canonical() == before


def test_a_failed_delisting_lookup_is_transient(env):
    before, code = monday_with(env, absent=True)
    env.terminal_override = BaoStockUnavailable("stock_basic query failed: 10002007")
    assert env.mod.run(TIERS) == 75 and env.canonical() == before


def test_a_verified_delisting_is_written_with_the_engines_exit(env):
    before, code = monday_with(env, absent=True)
    env.delisted = {code: date(2026, 10, 26)}
    assert env.mod.run(TIERS) == 0
    r = env.report(1000000)
    assert r["right_edge_exits"] == [code] and code not in r["positions"]
    sells = [t for t in r["today_trades"] if t["code"] == code]
    last_close = env.frames[code].set_index("date")["close"].loc["2026-10-23"]
    assert len(sells) == 1 and sells[0]["shares"] < 0 and sells[0]["price"] == pytest.approx(last_close)


def two_sessions_absent(env):
    """Canonical written on 10-22; then a held name has no bar on 10-23 or 10-26 (Monday)."""
    env.lake_through(date(2026, 10, 22))
    env.clock(at(2026, 10, 22, 18, 30))
    assert env.mod.run(TIERS) == 0
    before = env.canonical()
    code = held_name(env)
    env.lake_through(date(2026, 10, 26))
    env.frames_truncated(code, date(2026, 10, 22))
    env.clock(at(2026, 10, 26, 18, 30))
    return before, code


def test_a_name_absent_for_two_sessions_still_blocks_on_the_second(env, capsys):
    """The engine dates the forced exit on the FIRST bar the name lacks (10-23), so on 10-26 it is not
    among today's fills; the check must still find it."""
    before, code = two_sessions_absent(env)
    assert env.mod.run(TIERS) == 75
    assert f"'{code}': 'missing_published_bar'" in capsys.readouterr().err
    assert env.canonical() == before


def test_an_earlier_verified_delisting_is_rechecked_and_kept(env):
    before, code = two_sessions_absent(env)
    env.delisted = {code: date(2026, 10, 23)}
    assert env.mod.run(TIERS) == 0
    r = env.report(1000000)
    assert r["right_edge_exits"] == [code] and code not in r["positions"]
    assert [t for t in r["today_trades"] if t["code"] == code] == []    # sold on 10-23, not today
    assert env.terminal_calls == [[code]]


def test_a_suspended_held_name_is_kept_without_any_check(env):
    before, code = monday_with(env, suspended=True)
    assert env.mod.run(TIERS) == 0
    r = env.report(1000000)
    assert r["right_edge_exits"] == [] and code in r["positions"] and env.terminal_calls == []


def test_classification_order():
    summary = pd.DataFrame({"code": ["a", "b", "c", "d"], "status": ["error: x", "ok", "ok", "ok"]})
    seen = []

    def terminal(codes):
        seen.append(codes)
        return {"b": pd.Timestamp("2026-10-26"), "c": pd.Timestamp("2026-10-27"), "d": None}
    disp = feed.classify_right_edge(["a", "b", "c", "d"], "2026-10-26", summary, terminal)
    assert disp == {"a": "pull_error", "b": "verified_terminal", "c": "missing_published_bar",
                    "d": "missing_published_bar"}
    assert seen == [["b", "c", "d"]]                # a pull error is never looked up
