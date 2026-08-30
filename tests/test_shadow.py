"""Forward-shadow invariants (issue #22). The shadow races a historical champion that FAILED
issue #21's battery, so its only value is being unfalsifiable after the fact -- which requires the
common state at the fork bar to be exact, the fork to be strictly forward, and the canonical
record to be untouchable. All three are asserted here rather than trusted.
"""
import json

import numpy as np
import pandas as pd
import pytest

from hermes.live import shadow as sh
from hermes.live.strategy import ALL_TIERS, DEPLOYED
from hermes.paths import PAPER_DIR
from hermes.research.backtest.schedule import (assert_no_lookahead, calendar_rebalance_schedule,
                                               splice_schedule)

FORK = pd.Timestamp(sh.SHADOW_INCEPTION_ASOF)


# --- splice_schedule -----------------------------------------------------------------

def test_splice_takes_baseline_through_the_switch_and_candidate_after():
    dates = pd.bdate_range("2026-06-18", "2026-10-30")
    at = list(dates).index(FORK)
    base = calendar_rebalance_schedule(dates, 1)
    cand = calendar_rebalance_schedule(dates, 5)
    sp = splice_schedule(base, cand, at, dates)
    assert {e: s for e, s in sp.items() if e <= at} == {e: s for e, s in base.items() if e <= at}
    assert {e: s for e, s in sp.items() if e > at} == {e: s for e, s in cand.items() if e > at}


def test_splice_never_duplicates_or_drops_a_month():
    dates = pd.bdate_range("2026-06-18", "2026-12-31")
    at = list(dates).index(FORK)
    sp = splice_schedule(calendar_rebalance_schedule(dates, 1),
                         calendar_rebalance_schedule(dates, 5), at, dates)
    months = [(dates[e].year, dates[e].month) for e in sp]
    assert len(months) == len(set(months)), "at most one rebalance per month"
    # every month after the panel's first must still be represented
    all_months = sorted({(d.year, d.month) for d in dates})[1:]
    assert set(months) == set(all_months)


def test_splice_raises_when_a_month_straddles_the_switch():
    """The one real hazard: baseline exec before the switch, candidate exec after -> two in a month.

    The panel must start in July so August's D=1 execution (the 3rd) has a preceding signal bar --
    on a panel starting in August it would be bar 0 and be skipped, and no straddle would form."""
    dates = pd.bdate_range("2026-07-01", "2026-09-30")
    at = list(dates).index(pd.Timestamp("2026-08-04"))     # between D=1's (8/03) and D=5's (8/05)
    with pytest.raises(ValueError, match="twice in month"):
        splice_schedule(calendar_rebalance_schedule(dates, 1),
                        calendar_rebalance_schedule(dates, 5), at, dates)


def test_splice_signal_strictly_precedes_execution():
    dates = pd.bdate_range("2026-06-18", "2026-12-31")
    at = list(dates).index(FORK)
    sp = splice_schedule(calendar_rebalance_schedule(dates, 1),
                         calendar_rebalance_schedule(dates, 5), at, dates)
    assert_no_lookahead(sp, dates)
    assert all(s < e for e, s in sp.items())


# --- the fork itself -----------------------------------------------------------------

def test_shadow_schedule_refuses_to_fork_off_a_missing_inception_bar():
    """If the fork bar is absent the experiment would silently start somewhere else."""
    dates = pd.bdate_range("2026-06-18", "2026-08-27")      # stops one bar short of the fork
    with pytest.raises(sh.ShadowGateError, match="not a bar in the panel"):
        sh.shadow_schedule(dates)


def test_first_post_fork_execution_is_the_september_d5_date():
    """September 2026: D=1 trades on the 1st; the 5th is a Saturday, so D=5 trades on Monday the
    7th. The first real divergence is therefore 2026-09-01, when D=1 rebalances and the shadow
    does not."""
    dates = pd.bdate_range("2026-06-18", "2026-09-30")
    sp = sh.shadow_schedule(dates)
    sept = sorted(dates[e] for e in sp if dates[e].month == 9)
    assert sept == [pd.Timestamp("2026-09-07")]
    d1_sept = sorted(dates[e] for e in calendar_rebalance_schedule(dates, 1) if dates[e].month == 9)
    assert d1_sept == [pd.Timestamp("2026-09-01")]
    assert pd.Timestamp("2026-09-01") not in sept, "the shadow must NOT trade on D=1's date"


def test_weekend_and_holiday_gaps_resolve_forward_within_the_month():
    """A closure spanning the 5th pushes execution to the first bar after it, still in the month."""
    dates = pd.DatetimeIndex(["2026-08-27", "2026-08-28",                       # fork bar
                              "2026-10-01", "2026-10-09", "2026-10-12"])        # National Day gap
    sp = sh.shadow_schedule(dates)
    october = sorted(dates[e] for e in sp if dates[e].month == 10)
    assert october == [pd.Timestamp("2026-10-09")]          # first bar on/after the 5th


def test_everything_through_the_fork_bar_is_the_canonical_schedule():
    dates = pd.bdate_range("2026-06-18", "2026-12-31")
    at = list(dates).index(FORK)
    sp = sh.shadow_schedule(dates)
    base = calendar_rebalance_schedule(dates, 1)
    assert {e: s for e, s in sp.items() if e <= at} == {e: s for e, s in base.items() if e <= at}


# --- common state, on the real ledger -------------------------------------------------

@pytest.mark.skipif(not (PAPER_DIR / "report_1000000.json").exists(),
                    reason="canonical paper ledger not present in this checkout")
def test_common_state_matches_the_canonical_ledger_in_every_tier():
    """GATE 2 as a test: through the fork bar the shadow book IS the canonical book."""
    close, signal, asof_fn = sh.build_panels()
    pre = close.index <= FORK
    sp = sh.shadow_schedule(close.index)
    pre_sched = {e: s for e, s in sp.items() if close.index[e] <= FORK}
    for cap in ALL_TIERS:
        ref = json.loads((PAPER_DIR / f"report_{cap}.json").read_text(encoding="utf-8"))
        res = sh.run_book(close.loc[pre], signal.loc[pre], cap, asof_fn, pre_sched)
        assert float(res.equity.iloc[-1]) == pytest.approx(ref["equity"], abs=1e-9), cap
        assert sh._positions(res.trades) == {k: int(v) for k, v in ref["positions"].items()}, cap
        assert len(res.trades) == ref["n_trades_total"], cap


@pytest.mark.skipif(not (PAPER_DIR / "report_1000000.json").exists(),
                    reason="canonical paper ledger not present in this checkout")
def test_shadow_return_is_zero_at_inception_for_every_tier():
    """The shadow record must start at 0% on the common bar -- no inherited performance."""
    panels = sh.build_panels(as_of=sh.SHADOW_INCEPTION_ASOF)
    for cap in ALL_TIERS:
        r = sh.shadow_step(cap, as_of=sh.SHADOW_INCEPTION_ASOF, persist=False, panels=panels)
        assert r["total_return_since_shadow_inception"] == pytest.approx(0.0, abs=1e-12), cap
        assert r["d1_return_since_shadow_inception"] == pytest.approx(0.0, abs=1e-12), cap
        assert r["excess_return_vs_d1"] == pytest.approx(0.0, abs=1e-12), cap
        assert r["equity"] == pytest.approx(r["d1_equity_same_asof"], abs=1e-9), cap
        assert r["positions"] == r["d1_positions"], cap
        assert r["n_trades_since_shadow_inception"] == 0, cap


@pytest.mark.skipif(not (PAPER_DIR / "report_1000000.json").exists(),
                    reason="canonical paper ledger not present in this checkout")
def test_shadow_step_is_idempotent_apart_from_the_wall_clock():
    """Recompute-from-seed: two runs on the same lake agree on everything but the run timestamp."""
    panels = sh.build_panels(as_of=sh.SHADOW_INCEPTION_ASOF)
    volatile = {"run_date", "lake_lag_days", "fresh"}
    a = sh.shadow_step(1_000_000, as_of=sh.SHADOW_INCEPTION_ASOF, persist=False, panels=panels)
    b = sh.shadow_step(1_000_000, as_of=sh.SHADOW_INCEPTION_ASOF, persist=False, panels=panels)
    assert {k: v for k, v in a.items() if k not in volatile} == \
           {k: v for k, v in b.items() if k not in volatile}


# --- isolation ------------------------------------------------------------------------

def test_shadow_output_directory_is_outside_the_canonical_tree():
    assert sh.SHADOW_DIR.resolve() != PAPER_DIR.resolve()
    assert PAPER_DIR.resolve() not in sh.SHADOW_DIR.resolve().parents
    assert sh.SHADOW_DIR.name == sh.SHADOW_ID and sh.SHADOW_DIR.parent.name == "paper_shadow"


@pytest.mark.skipif(not (PAPER_DIR / "report_1000000.json").exists(),
                    reason="canonical paper ledger not present in this checkout")
def test_persisting_the_shadow_does_not_touch_the_canonical_ledger(tmp_path):
    """HARD TEST: a real persist run, fingerprinted before and after."""
    before = sh.fingerprint_canonical()
    panels = sh.build_panels(as_of=sh.SHADOW_INCEPTION_ASOF)
    sh.shadow_step(1_000_000, as_of=sh.SHADOW_INCEPTION_ASOF, persist=True,
                   out_dir=tmp_path, panels=panels)
    sh.assert_canonical_untouched(before)               # raises if anything moved
    assert (tmp_path / "report_1000000.json").exists()
    assert not list(PAPER_DIR.glob("*shadow*")), "no shadow artefact may land in results/paper"


def test_canonical_write_protection_detects_a_change():
    """The guard must fire on any mismatch, including in a checkout with NO ledger.

    `results/` is gitignored, so a CI runner has an empty canonical directory. An earlier version
    took its baseline from `fingerprint_canonical()` and mutated the first key, which raised
    StopIteration on an empty dict rather than exercising the guard -- the failure that turned CI
    red once Lint stopped masking the Test step. Injecting a sentinel key makes the comparison
    mismatch regardless of what is on disk."""
    fabricated = {**sh.fingerprint_canonical(), "__sentinel_not_on_disk__": (-1, -1)}
    with pytest.raises(sh.ShadowGateError, match="WAS MODIFIED"):
        sh.assert_canonical_untouched(fabricated)


# --- manifest -------------------------------------------------------------------------

def test_manifest_freezes_the_experiment_and_declares_its_status():
    m = sh.build_manifest("deadbeef")
    assert m["candidate_calendar_day"] == 5 and m["baseline_calendar_day"] == 1
    assert m["shadow_inception_asof"] == "2026-08-28"
    assert m["paper_inception"] == "2026-06-18"
    assert m["capital_tiers"] == list(ALL_TIERS)
    assert m["strategy_spec"]["n_hold"] == DEPLOYED.n_hold
    assert m["strategy_spec"]["value_weight"] == DEPLOYED.value_weight
    for flag in ("retrospective_counterfactual_before_inception_is_not_part_of_forward_record",
                 "candidate_parameters_are_frozen",
                 "results_paper_is_canonical_and_must_not_be_rewritten"):
        assert m[flag] is True


def test_manifest_validation_rejects_frozen_field_drift(tmp_path):
    p = tmp_path / "manifest.json"
    frozen = sh.build_manifest("aaa")
    p.write_text(json.dumps(frozen), encoding="utf-8")
    assert sh.validate_manifest(p, sh.build_manifest("bbb"))    # commit may move; frozen fields agree
    drifted = dict(frozen, candidate_calendar_day=8)            # someone quietly switched the day
    p.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(sh.ShadowGateError, match="frozen-field drift"):
        sh.validate_manifest(p, sh.build_manifest("bbb"))


def test_manifest_absent_returns_the_expected_without_writing(tmp_path):
    p = tmp_path / "manifest.json"
    assert sh.validate_manifest(p, sh.build_manifest("aaa"))["candidate_calendar_day"] == 5
    assert not p.exists(), "validation must not create the manifest as a side effect"


# --- small helpers --------------------------------------------------------------------

def test_avg_names_held_counts_only_the_forward_window():
    dates = pd.bdate_range("2026-08-24", "2026-09-02")
    trades = [{"date": dates[0], "code": "a", "shares": 100},
              {"date": dates[0], "code": "b", "shares": 100},
              {"date": dates[-1], "code": "b", "shares": -100}]
    # before the fork two names are held; the sell lands on the last bar
    got = sh._avg_names_held_since(trades, dates, dates[-2])
    assert got == pytest.approx((2 + 1) / 2)


def test_drawdown_since_is_measured_from_the_records_own_start():
    eq = pd.Series([100.0, 90.0, 95.0], index=pd.bdate_range("2026-09-01", periods=3))
    assert sh._drawdown_since(eq) == pytest.approx(-0.10)
    assert sh._drawdown_since(pd.Series(dtype=float)) == 0.0


def test_positions_fold_drops_closed_names():
    trades = [{"code": "a", "shares": 100}, {"code": "b", "shares": 200},
              {"code": "a", "shares": -100}]
    assert sh._positions(trades) == {"b": 200}


def test_shadow_step_refuses_a_lake_that_ends_before_inception():
    close = pd.DataFrame(np.ones((3, 1)), index=pd.bdate_range("2026-06-18", periods=3),
                         columns=["x"])
    with pytest.raises(sh.ShadowGateError, match="before the shadow inception"):
        sh.shadow_step(10_000, persist=False, panels=(close, close, lambda _d: {"x"}))
