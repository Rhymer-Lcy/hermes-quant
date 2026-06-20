"""Paper-trading ledger: idempotent folding + the anti-skew parity guarantee (the paper
ledger must reproduce the research engine's equity exactly), and the deployed-spec lock."""
import numpy as np
import pandas as pd

from hermes.live.ledger import LedgerState, fold_day
from hermes.live.paper import ledger_equity, replay
from hermes.live.strategy import DEPLOYED, deployed_signal
from hermes.research.factors import library as fl


def test_fold_day_buy_then_sell_and_marks():
    s0 = LedgerState(seed_cash=10_000.0)
    # Buy 100 @ 50 with 5 fee -> cash 10000 - 5000 - 5 = 4995; mark @ 60 -> 4995 + 6000.
    s1 = fold_day(s0, "2020-01-02", [{"code": "x", "shares": 100, "price": 50.0, "fee": 5.0}],
                  {"x": 60.0})
    assert s1.positions == {"x": 100}
    assert abs(s1.cash - 4995.0) < 1e-9
    assert abs(s1.equity_curve[-1][1] - (4995.0 + 6000.0)) < 1e-9
    # Sell 100 @ 70 with 5 fee -> cash 4995 + 7000 - 5 = 11990; no positions left.
    s2 = fold_day(s1, "2020-01-03", [{"code": "x", "shares": -100, "price": 70.0, "fee": 5.0}],
                  {"x": 70.0})
    assert s2.positions == {}
    assert abs(s2.cash - 11_990.0) < 1e-9
    # Immutability: earlier states are untouched by later folds.
    assert s0.positions == {} and abs(s0.cash - 10_000.0) < 1e-9
    assert s1.positions == {"x": 100}


def test_fold_day_nan_mark_contributes_zero():
    s = fold_day(LedgerState(seed_cash=1_000.0), "2020-01-02",
                 [{"code": "d", "shares": 100, "price": 5.0, "fee": 0.0}], {"d": float("nan")})
    # cash 1000 - 500 = 500; NaN mark -> position valued at 0 (matches engine _hold_value).
    assert abs(s.equity_curve[-1][1] - 500.0) < 1e-9


def test_replay_reproduces_engine_equity_exactly():
    # Small synthetic panel: 3 names over ~3 months of business days, monotone prices and a
    # fixed ranking a>b>c so top-2 is stable. Replay's ledger equity must match the engine.
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    n = len(dates)
    price = pd.DataFrame({
        "a": np.linspace(10.0, 13.0, n),
        "b": np.linspace(20.0, 24.0, n),
        "c": np.linspace(30.0, 33.0, n),
    }, index=dates)
    signal = pd.DataFrame({"a": 3.0, "b": 2.0, "c": 1.0}, index=dates)

    ledger, res = replay(price, signal, 1_000_000.0, n_hold=2)
    led = ledger_equity(ledger).reindex(res.equity.index)
    assert float(np.abs(led.values - res.equity.values).max()) < 1e-6
    assert len(ledger.folded_days) == n            # every trading day folded
    assert res.trades                              # at least one rebalance happened


def test_replay_idempotent_rebuild():
    # Re-running replay on the same inputs yields an identical equity curve (idempotence).
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    price = pd.DataFrame({"a": np.linspace(10, 12, len(dates)),
                          "b": np.linspace(20, 19, len(dates))}, index=dates)
    signal = pd.DataFrame({"a": 2.0, "b": 1.0}, index=dates)
    e1 = ledger_equity(replay(price, signal, 100_000.0, n_hold=1)[0])
    e2 = ledger_equity(replay(price, signal, 100_000.0, n_hold=1)[0])
    assert e1.equals(e2)


def test_deployed_signal_is_the_documented_blend():
    # Anti-skew lock: the single-source deployed_signal must equal the documented formula --
    # value (1/PE) + 1m-reversal, restricted to PIT members BEFORE a 5:1 blend. If someone
    # edits the spec, this test forces the docs/research to move with it.
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    n = len(dates)
    close = pd.DataFrame({"a": np.linspace(10, 13, n), "b": np.linspace(20, 18, n),
                          "c": np.linspace(30, 33, n)}, index=dates)
    pe = pd.DataFrame({"a": 8.0, "b": 12.0, "c": 25.0}, index=dates)
    members = {"a", "b", "c"}

    def asof(_):
        return members

    got = deployed_signal(close, pe, asof)
    ep = fl.restrict_to_universe(fl.earnings_yield(pe), asof)
    rev = fl.restrict_to_universe(-fl.trailing_return(close, DEPLOYED.reversal_lookback), asof)
    want = fl.blend([ep, rev], [DEPLOYED.value_weight, DEPLOYED.reversal_weight])
    pd.testing.assert_frame_equal(got, want)
    assert (DEPLOYED.n_hold, DEPLOYED.rebalance_band, DEPLOYED.weight_asof) == (10, 0, None)
