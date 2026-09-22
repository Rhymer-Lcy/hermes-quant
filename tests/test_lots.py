"""Integer-lot allocator invariants (issue #25). Research-only code, but the engine imports it
behind an opt-in flag, so the default path's byte-identity is asserted here too.

Synthetic inputs throughout, so a failure points at the allocator rather than at the lake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.lots import (_key, _trade_cash, allocate, baseline_lots,
                                           build_inputs, ideal_lots, solve, solve_lambda_scan)

COSTS = AShareCosts()
LOT, SLIP = COSTS.lot_size, COSTS.slip


def mk(prices: dict[str, float], equity: float, cash: float | None = None,
       positions: dict[str, int] | None = None, weights: dict[str, float] | None = None):
    codes = list(prices)
    w = weights or {c: 1.0 / len(codes) for c in codes}
    return build_inputs(codes, pd.Series(prices), w, positions or {}, equity,
                        equity if cash is None else cash, COSTS, SLIP, LOT)


# ----------------------------------------------------------------- baseline reproduction


def test_baseline_lots_reproduce_the_production_floor_rule():
    """`baseline_lots` must be the production rule, or every comparison in the study is void."""
    from hermes.research.backtest.portfolio import target_shares
    prices = {"a": 3.07, "b": 12.4, "c": 57.55, "d": 4.61}
    equity, gross = 100_000.0, 100_000.0
    w = {c: 0.25 for c in prices}
    raw = pd.Series(prices)
    prod = target_shares(gross, w, 1.0, raw, SLIP, LOT)
    inp = mk(prices, equity)
    mine = {c: int(s) for c, s in zip(inp.codes, baseline_lots(inp))}
    assert mine == prod


def test_baseline_leaves_a_whole_slice_in_cash_when_one_lot_exceeds_it():
    """The ZERO-SLICE channel, in miniature: a 1,000 slice cannot buy a lot of a 60 CNY stock."""
    inp = mk({"cheap": 3.0, "dear": 60.0}, 2_000.0)
    b = baseline_lots(inp)
    assert b[list(inp.codes).index("dear")] == 0
    assert _trade_cash(inp, b) > 900          # the dear name's entire slice stays in cash


# ----------------------------------------------------------------- feasibility


def test_solution_is_always_affordable_including_fees():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(3, 11))
        prices = {f"s{i}": float(rng.uniform(2.0, 120.0)) for i in range(n)}
        equity = float(rng.uniform(8_000, 400_000))
        inp = mk(prices, equity)
        s = solve(inp)
        assert _trade_cash(inp, s) >= -1e-9, (prices, equity)


def test_no_negative_cash_when_starting_from_existing_holdings():
    prices = {"a": 5.0, "b": 25.0, "c": 80.0}
    pos = {"a": 2000, "b": 300}
    equity = 2000 * 5.0 + 300 * 25.0 + 1_000.0
    inp = mk(prices, equity, cash=1_000.0, positions=pos)
    s = solve(inp)
    assert _trade_cash(inp, s) >= -1e-9


def test_insufficient_capital_yields_an_empty_but_feasible_book():
    inp = mk({"a": 300.0, "b": 500.0}, 1_000.0)
    s = solve(inp)
    assert (s == 0).all()
    assert _trade_cash(inp, s) >= -1e-9


# ----------------------------------------------------------------- lot compliance & shape


def test_every_holding_is_a_whole_number_of_lots():
    rng = np.random.default_rng(11)
    for _ in range(25):
        prices = {f"s{i}": float(rng.uniform(2, 90)) for i in range(int(rng.integers(3, 11)))}
        inp = mk(prices, float(rng.uniform(10_000, 500_000)))
        s = solve(inp)
        assert np.all(s % LOT == 0)
        assert np.all(s >= 0)


def test_allocator_never_invents_a_name_outside_the_basket_or_the_book():
    prices = {"a": 5.0, "b": 9.0, "c": 40.0}
    w = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    got = allocate(90_000.0, w, 1.0, pd.Series(prices), SLIP, LOT, COSTS, 90_000.0, {}, 90_000.0)
    assert set(got) <= set(prices)


def test_a_held_name_outside_the_basket_is_represented_so_its_proceeds_count():
    prices = {"a": 5.0, "b": 9.0, "old": 20.0}
    w = {"a": 0.5, "b": 0.5}
    pos = {"old": 500}
    eq = 500 * 20.0 + 1_000.0
    got = allocate(eq, w, 1.0, pd.Series(prices), SLIP, LOT, COSTS, 1_000.0, pos, eq)
    assert "old" in got and got["old"] == 0, "the exited name must be targeted to zero"


# ----------------------------------------------------------------- optimality vs baseline


def test_candidate_is_never_worse_than_baseline_under_the_frozen_key():
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(3, 11))
        prices = {f"s{i}": float(rng.uniform(2.0, 150.0)) for i in range(n)}
        inp = mk(prices, float(rng.uniform(8_000, 600_000)))
        b = baseline_lots(inp)
        if _trade_cash(inp, b) < 0:
            continue
        assert _key(inp, solve(inp)) <= _key(inp, b)


def test_candidate_buys_an_expensive_name_the_baseline_drops():
    """The behaviour the study exists to measure: redistribute rather than sit in cash."""
    inp = mk({"a": 3.0, "b": 12.0, "c": 60.0}, 10_000.0)
    b, s = baseline_lots(inp), solve(inp)
    i = list(inp.codes).index("c")
    assert b[i] == 0 and s[i] > 0
    assert _trade_cash(inp, s) < _trade_cash(inp, b)      # less idle cash


def test_floor_ceil_is_not_a_sufficient_search_space():
    """Counter-example to the convenient assumption, asserted rather than asserted-away."""
    found = False
    rng = np.random.default_rng(5)
    for _ in range(80):
        n = int(rng.integers(4, 11))
        prices = {f"s{i}": float(rng.uniform(2.0, 120.0)) for i in range(n)}
        inp = mk(prices, float(rng.uniform(10_000, 300_000)))
        k = solve(inp) / LOT
        x = ideal_lots(inp)
        if np.any(k > np.ceil(x) + 1e-9) or np.any(k < np.floor(x) - 1e-9):
            found = True
            break
    assert found, "expected at least one optimum outside the floor/ceil box"


def test_solve_is_at_least_as_good_as_the_independent_frontier_solver():
    rng = np.random.default_rng(13)
    for _ in range(30):
        n = int(rng.integers(3, 11))
        prices = {f"s{i}": float(rng.uniform(2.0, 100.0)) for i in range(n)}
        inp = mk(prices, float(rng.uniform(10_000, 250_000)))
        alt = solve_lambda_scan(inp)
        if alt is None:
            continue
        assert _key(inp, solve(inp)) <= _key(inp, alt)


# ----------------------------------------------------------------- determinism


def test_result_is_deterministic_and_idempotent():
    prices = {"a": 4.2, "b": 17.9, "c": 63.1, "d": 8.8}
    inp = mk(prices, 120_000.0)
    first = solve(inp)
    for _ in range(4):
        assert np.array_equal(solve(inp), first)
    # feeding the solution back as the starting book must not move it
    inp2 = mk(prices, 120_000.0, cash=float(_trade_cash(inp, first)),
              positions={c: int(s) for c, s in zip(inp.codes, first)})
    assert np.array_equal(solve(inp2), first)


def test_tie_break_order_is_the_frozen_one():
    """Lower objective wins first; among equals, lower residual cash, then cost, then turnover."""
    inp = mk({"a": 10.0, "b": 10.0}, 20_000.0)
    x = np.array([1000.0, 1000.0])
    y = np.array([900.0, 1100.0])
    kx, ky = _key(inp, x), _key(inp, y)
    assert kx[0] <= ky[0]
    assert len(kx) == 5 and isinstance(kx[4], tuple)


# ----------------------------------------------------------------- engine integration


def test_engine_default_path_is_byte_identical_without_the_allocator():
    """`allocator=None` must leave the production engine untouched."""
    from hermes.research.backtest.portfolio import signal_portfolio_backtest
    idx = pd.bdate_range("2024-01-01", periods=140)
    rng = np.random.default_rng(0)
    cols = [f"s{i}" for i in range(12)]
    px = pd.DataFrame(20 + np.cumsum(rng.normal(0, 0.3, (len(idx), len(cols))), axis=0),
                      index=idx, columns=cols).clip(lower=1.0)
    sg = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    a = signal_portfolio_backtest(px, sg, 100_000, 10, collect_trades=True)
    b = signal_portfolio_backtest(px, sg, 100_000, 10, collect_trades=True, allocator=None)
    assert a.equity.equals(b.equity)
    assert a.trades == b.trades
    assert a.total_costs == b.total_costs


def test_unknown_allocator_name_raises():
    from hermes.research.backtest.portfolio import signal_portfolio_backtest
    idx = pd.bdate_range("2024-01-01", periods=60)
    px = pd.DataFrame(10.0, index=idx, columns=["a", "b"])
    sg = pd.DataFrame(1.0, index=idx, columns=["a", "b"])
    with pytest.raises(ValueError, match="unknown allocator"):
        signal_portfolio_backtest(px, sg, 50_000, 2, allocator="nope")


def test_cwil_path_runs_and_holds_no_negative_cash_in_the_engine():
    from hermes.research.backtest.portfolio import signal_portfolio_backtest
    idx = pd.bdate_range("2024-01-01", periods=140)
    rng = np.random.default_rng(1)
    cols = [f"s{i}" for i in range(12)]
    px = pd.DataFrame(np.abs(30 + np.cumsum(rng.normal(0, 0.5, (len(idx), len(cols))), axis=0)),
                      index=idx, columns=cols).clip(lower=1.0)
    sg = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    r = signal_portfolio_backtest(px, sg, 30_000, 10, collect_trades=True, allocator="cwil")
    assert (r.equity > 0).all()
    assert r.avg_names_held > 0


def test_alloc_log_is_opt_in_and_records_the_pre_trade_state():
    from hermes.research.backtest.portfolio import signal_portfolio_backtest
    idx = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(2)
    cols = [f"s{i}" for i in range(11)]
    px = pd.DataFrame(np.abs(25 + np.cumsum(rng.normal(0, 0.4, (len(idx), len(cols))), axis=0)),
                      index=idx, columns=cols).clip(lower=1.0)
    sg = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    log: list = []
    signal_portfolio_backtest(px, sg, 100_000, 10, alloc_log=log)
    assert log and {"date", "equity", "cash", "weights", "positions", "price", "desired"} <= set(
        log[0])
    assert log[0]["equity"] > 0


# ----------------------------------------------------------------- boundary cases


def test_expensive_stock_small_account_does_not_crash_and_stays_feasible():
    inp = mk({"a": 1800.0, "b": 4.0}, 20_000.0)
    s = solve(inp)
    assert _trade_cash(inp, s) >= -1e-9
    assert np.all(s % LOT == 0)


def test_minimum_commission_is_charged_on_a_tiny_buy():
    inp = mk({"a": 2.0}, 1_000.0)
    s = solve(inp)
    spent = 1_000.0 - _trade_cash(inp, s)
    notional = float(s[0] * inp.price[0] * (1 + SLIP))
    assert spent >= notional + 4.99, "the CNY 5 minimum commission must be paid"


def test_empty_basket_is_handled():
    inp = build_inputs([], pd.Series(dtype=float), {}, {}, 10_000.0, 10_000.0, COSTS, SLIP, LOT)
    assert solve(inp).size == 0
    assert allocate(0.0, {}, 1.0, pd.Series(dtype=float), SLIP, LOT, COSTS, 0.0, {}, 0.0) == {}


def test_zero_equity_is_handled():
    got = allocate(0.0, {"a": 1.0}, 1.0, pd.Series({"a": 5.0}), SLIP, LOT, COSTS, 0.0, {}, 0.0)
    assert got == {}
