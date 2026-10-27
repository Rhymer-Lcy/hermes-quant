"""The D=5 shadow's right edge (scripts/paper_shadow_d5.py) on synthetic panels; nothing real is read
or written. The shadow can hold a name the canonical D=1 book does not, so the canonical run's
right-edge check never sees it: a shadow-only holding without a bar on the latest session must be a
verified delisting before ANY tier (or the manifest) is written, and a --as-of replay writes nothing."""
import importlib.util
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes.data.sources.baostock_source import BaoStockUnavailable
from hermes.live import feed
from hermes.live import shadow as sh

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper_shadow_d5.py"
CODES = [f"sh.6000{i:02d}" for i in range(12)]
DROPPED = CODES[10]                         # enters the D=5 book only, then loses its last bar
TIERS = ["--tiers", "100000", "1000000"]


def panels(last_bar_missing=True, end="2026-09-30", missing_bars=1):
    """D=1's September signal bar is 08-31 and D=5's is 09-04; DROPPED jumps into the top ten from
    09-02, so only the D=5 book buys it. Its 09-30 bar is then absent (not a delisting)."""
    dates = pd.bdate_range("2026-06-18", end)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(10 + rng.random((len(dates), len(CODES))).cumsum(0) * 0.01,
                         index=dates, columns=CODES)
    signal = pd.DataFrame(np.tile(np.arange(12, 0, -1, dtype=float), (len(dates), 1)),
                          index=dates, columns=CODES)
    signal.loc[signal.index >= "2026-09-02", DROPPED] = 100.0
    if last_bar_missing:
        close.loc[close.index[-missing_bars:], DROPPED] = np.nan
    return close, signal, (lambda _d: set(CODES))


def test_the_shadow_reports_a_right_edge_exit_the_canonical_book_does_not_hold():
    r = sh.shadow_step(1_000_000, persist=False, panels=panels())
    assert DROPPED not in r["d1_positions"]       # the canonical check would never see it
    assert [t["code"] for t in r["today_trades"] if t["shares"] < 0] == [DROPPED]
    assert r["right_edge_exits"] == [DROPPED]


def test_a_name_absent_for_two_sessions_is_still_reported():
    """The forced exit is dated on the first missing bar (09-29), not on the latest one."""
    r = sh.shadow_step(1_000_000, persist=False, panels=panels(missing_bars=2))
    assert [t["code"] for t in r["today_trades"] if t["shares"] < 0] == []
    assert r["right_edge_exits"] == [DROPPED]


def test_a_bar_present_on_the_session_is_not_a_right_edge_exit():
    r = sh.shadow_step(1_000_000, persist=False, panels=panels(last_bar_missing=False))
    assert r["right_edge_exits"] == []
    assert DROPPED in r["positions"]


def test_a_rebalance_sell_with_its_bar_is_not_a_right_edge_exit():
    """The panel ends on D=5's September execution bar (09-07): the shadow sells a name that has its
    bar there, which is an ordinary rebalance fill."""
    r = sh.shadow_step(1_000_000, persist=False, panels=panels(last_bar_missing=False, end="2026-09-07"))
    assert [t["code"] for t in r["today_trades"] if t["shares"] < 0] == [CODES[9]]
    assert r["right_edge_exits"] == []


class Driver:
    def __init__(self, tmp_path, monkeypatch, pnl):
        spec = importlib.util.spec_from_file_location("paper_shadow_d5_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.dir = tmp_path / "d5"
        self.delisted: dict[str, date] = {}
        self.terminal_calls: list[list[str]] = []
        mp = monkeypatch
        for name in ("gate_production_parity", "gate_no_lookahead"):
            mp.setattr(self.mod, name, lambda *a, **k: None)
        mp.setattr(self.mod, "canonical_at_fork", lambda tiers: {})
        mp.setattr(self.mod, "gate_common_state", lambda *a, **k: None)
        mp.setattr(self.mod, "build_panels", lambda as_of=None: pnl)
        mp.setattr(self.mod, "fingerprint_canonical", lambda: {})
        mp.setattr(self.mod, "assert_canonical_untouched", lambda before: None)
        mp.setattr(self.mod, "head_commit", lambda: "0" * 40)
        prov = {"runtime_code_sha": "0" * 40, "runtime_worktree_clean": True}
        mp.setattr(self.mod, "runtime_provenance", lambda: prov)
        mp.setattr(sh, "runtime_provenance", lambda: prov)
        mp.setattr(self.mod, "SHADOW_DIR", self.dir)
        mp.setattr(sh, "SHADOW_DIR", self.dir)
        mp.setattr(feed, "terminal_dates", self.terminal)

    def terminal(self, codes):
        self.terminal_calls.append(list(codes))
        return {c: pd.Timestamp(self.delisted[c]) for c in codes if c in self.delisted}

    def files(self):
        return sorted(p.name for p in self.dir.glob("*")) if self.dir.exists() else []


def test_a_shadow_only_name_without_its_bar_blocks_every_write(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())
    with pytest.raises(BaoStockUnavailable, match=DROPPED):
        d.mod.main(TIERS)
    assert d.terminal_calls == [[DROPPED]]
    assert d.files() == []                        # no tier, and no manifest either


def test_a_verified_delisting_is_written_for_every_tier(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())
    d.delisted[DROPPED] = date(2026, 9, 30)
    d.mod.main(TIERS)
    assert d.files() == ["curve_100000.parquet", "curve_1000000.parquet", "manifest.json",
                         "report_100000.json", "report_1000000.json",
                         "trades_100000.parquet", "trades_1000000.parquet"]


def test_a_delisting_dated_after_the_session_does_not_verify_the_exit(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())
    d.delisted[DROPPED] = date(2026, 10, 9)
    with pytest.raises(BaoStockUnavailable):
        d.mod.main(TIERS)
    assert d.files() == []


def test_no_right_edge_exit_writes_without_asking_baostock(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels(last_bar_missing=False))
    d.mod.main(TIERS)
    assert d.terminal_calls == []
    assert "report_1000000.json" in d.files()


def test_a_hung_delisting_lookup_times_out_and_writes_nothing(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())
    release = threading.Event()
    monkeypatch.setattr(feed, "terminal_dates", lambda codes: release.wait(30) and {})
    monkeypatch.setattr(d.mod, "TERMINAL_LOOKUP_TIMEOUT_S", 0.2)
    t0 = time.perf_counter()
    try:
        with pytest.raises(BaoStockUnavailable, match="did not answer"):
            d.mod.main(TIERS)
    finally:
        release.set()
    assert time.perf_counter() - t0 < 20
    assert d.files() == []


def test_a_failed_delisting_lookup_is_raised_on_the_callers_thread(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())

    def fail(codes):
        raise BaoStockUnavailable("stock_basic query failed: 10002007")

    monkeypatch.setattr(feed, "terminal_dates", fail)
    with pytest.raises(BaoStockUnavailable, match="10002007"):
        d.mod.main(TIERS)
    assert d.files() == []


def test_a_dry_run_reports_the_refusal_and_writes_nothing(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels())
    with pytest.raises(BaoStockUnavailable):
        d.mod.main(TIERS + ["--dry-run"])
    assert d.files() == []


def test_an_as_of_replay_is_never_persisted(tmp_path, monkeypatch):
    d = Driver(tmp_path, monkeypatch, panels(last_bar_missing=False))
    d.mod.main(TIERS + ["--as-of", "2026-09-30"])
    assert d.files() == []
