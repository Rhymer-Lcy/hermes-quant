"""IF short-hedge overlay: passthrough at ratio 0, drawdown removal on a beta-1 book, and the
small-account contract-granularity infeasibility."""
import numpy as np
import pandas as pd

from hermes.research.backtest.hedge import IF_MULT, hedge_overlay, max_drawdown


def _crash_index():
    dates = pd.bdate_range("2020-01-01", periods=60)
    idx = np.concatenate([np.linspace(4000, 5000, 30), np.linspace(5000, 3000, 30)])  # +25% then -40%
    return pd.Series(idx, index=dates)


def test_hedge_ratio_zero_is_passthrough():
    idx = _crash_index()
    book = pd.Series(np.linspace(1e6, 1.1e6, len(idx)), index=idx.index)
    hedged, nc, eff = hedge_overlay(book, idx, 0.0)
    assert nc == 0.0 and eff == 0.0
    pd.testing.assert_series_equal(hedged, book)


def test_full_hedge_removes_market_drawdown_on_beta1_book():
    idx = _crash_index()
    cap = 50_000_000.0                                  # large enough for fine contract granularity
    book = cap * idx / idx.iloc[0]                       # beta-1: book tracks the index exactly
    assert max_drawdown(book) < -0.35                    # unhedged tracks the index's ~-40% crash
    hedged, nc, eff = hedge_overlay(book, idx, 1.0, annual_cost=0.0)
    assert nc > 10                                       # ~cap/(idx*mult) ≈ 50M/1.2M ≈ many contracts
    assert abs(eff - 1.0) < 0.05                          # effective hedge ≈ target after rounding
    assert max_drawdown(hedged) > -0.10                  # market drawdown hedged away (-40% -> ~-5%)


def test_small_account_cannot_hedge_with_if():
    idx = _crash_index()
    book = pd.Series(np.full(len(idx), 100_000.0), index=idx.index)   # 100k « 1.2M contract notional
    hedged, nc, eff = hedge_overlay(book, idx, 1.0)
    assert nc == 0.0                                     # round(100k / 1.2M) = 0 contracts
    assert np.allclose(hedged.to_numpy(), book.to_numpy())   # no hedge possible -> equity unchanged
