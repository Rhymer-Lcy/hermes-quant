"""Unit tests for realistic A-share dividend accounting (issue #24).

Every test uses synthetic data whose answer is computable by hand, so a failure points at the
code rather than at the lake. Two of these are regression tests for defects the study's own
review actually found, and they are marked as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research.backtest.dividends import (PRIMARY_RULE, RATE_LONG, RATE_MID, RATE_SHORT,
                                                TAX_RULES, TaxRule, band_dividend_cash,
                                                closed_form_terminal, daily_shares,
                                                dividend_events, event_tax_table, fifo_lots,
                                                gross_dividend_stream, leakage_stream,
                                                lots_entitled, net_close_panel, share_multiplier)

TS = pd.Timestamp


# ----------------------------------------------------------------- share multiplier


def test_share_multiplier_is_one_for_a_pure_cash_dividend():
    # 10.00 -> 9.50 ex-date with DPS 0.50: the total return is exactly 0, so no shares were issued.
    assert share_multiplier(0.0, 10.0, 9.50, 0.50) == pytest.approx(1.0)


def test_share_multiplier_recovers_a_ten_for_ten_stock_dividend():
    # 10-for-10: price halves to 5.00, no cash. One old share becomes two new ones.
    assert share_multiplier(0.0, 10.0, 5.0, 0.0) == pytest.approx(2.0)


def test_share_multiplier_recovers_a_mixed_stock_and_cash_action():
    # 10-for-5 (m = 1.5) plus DPS 0.20 out of a 12.00 close: post price 7.8667 on zero total return.
    raw_now = (12.0 - 0.20) / 1.5
    assert share_multiplier(0.0, 12.0, raw_now, 0.20) == pytest.approx(1.5)


def test_share_multiplier_matches_its_defining_identity():
    m, raw_prev, raw_now, dps = 1.3, 8.0, 6.0, 0.15
    adj_ret = (m * raw_now + dps) / raw_prev - 1.0
    assert share_multiplier(adj_ret, raw_prev, raw_now, dps) == pytest.approx(m)


# ----------------------------------------------------------------- tax bands


@pytest.mark.parametrize(("sell", "expected"), [
    ("2026-06-20", RATE_SHORT),   # well inside one month
    ("2026-07-02", RATE_SHORT),   # EXACTLY one month -- the statute says "within one month (incl.)"
    ("2026-07-03", RATE_MID),     # one day past it
    ("2027-06-02", RATE_MID),     # exactly one year -- still the 10% band
    ("2027-06-03", RATE_LONG),    # past one year -- exempt
])
def test_statutory_bands_on_the_calendar_month_boundary(sell, expected):
    rule = TAX_RULES[PRIMARY_RULE]
    assert rule.rate(TS("2026-06-02"), TS(sell), TS("2026-06-10")) == expected


def test_thirty_day_boundary_actually_diverges_from_calendar_month_arithmetic():
    """A short February is where the two conventions genuinely disagree: one calendar month
    after 01-31 is 02-28 (28 days), while 30 days is 03-02. A sale on 03-01 falls on opposite
    sides of the two boundaries -- which is the whole reason tau-E is in the frozen grid."""
    buy, sell = TS("2026-01-31"), TS("2026-03-01")
    assert TAX_RULES["tau-A"].rate(buy, sell, buy) == RATE_MID      # past 02-28
    assert TAX_RULES["tau-E"].rate(buy, sell, buy) == RATE_SHORT    # inside 03-02
    # And where a month is 31 days long the disagreement runs the other way: 07-01 + 30d is
    # 07-31, but one calendar month is 08-01, so a sale on 08-01 splits them.
    buy2, sell2 = TS("2026-07-01"), TS("2026-08-01")
    assert TAX_RULES["tau-A"].rate(buy2, sell2, buy2) == RATE_SHORT  # inside 08-01
    assert TAX_RULES["tau-E"].rate(buy2, sell2, buy2) == RATE_MID    # past 07-31


def test_thirty_day_and_calendar_month_agree_away_from_the_boundary():
    buy = TS("2026-01-05")
    for sell, expect in (("2026-01-20", RATE_SHORT), ("2026-06-01", RATE_MID)):
        assert TAX_RULES["tau-A"].rate(buy, TS(sell), buy) == expect
        assert TAX_RULES["tau-E"].rate(buy, TS(sell), buy) == expect


def test_ex_date_measure_can_differ_from_sale_measure():
    buy, ex, sell = TS("2026-01-05"), TS("2026-01-20"), TS("2026-09-01")
    assert TAX_RULES["tau-A"].rate(buy, sell, ex) == RATE_MID     # measured to the sale
    assert TAX_RULES["tau-F"].rate(buy, sell, ex) == RATE_SHORT   # measured to the ex-date


def test_flat_rules_ignore_the_holding_period():
    for key, expect in (("tau-B", 0.20), ("tau-C", 0.10), ("tau-D", 0.0), ("tau-G", 0.0)):
        r = TAX_RULES[key]
        assert r.rate(TS("2015-01-01"), TS("2025-01-01"), TS("2020-01-01")) == expect
        assert r.rate(TS("2026-01-01"), TS("2026-01-02"), TS("2026-01-02")) == expect


# ----------------------------------------------------------------- FIFO lots


def _t(date, code, shares):
    return {"date": TS(date), "code": code, "shares": shares, "price": 10.0, "fee": 0.0}


def test_fifo_splits_a_sell_across_two_buy_lots_in_order():
    trades = [_t("2026-01-05", "a", 100), _t("2026-02-05", "a", 200), _t("2026-03-05", "a", -150)]
    lots = fifo_lots(trades, TS("2026-06-30"))
    closed = lots[~lots["open"]].sort_values("buy_date")
    assert list(closed["shares"]) == [100, 50]
    assert list(closed["buy_date"]) == [TS("2026-01-05"), TS("2026-02-05")]
    assert (closed["sell_date"] == TS("2026-03-05")).all()
    still_open = lots[lots["open"]]
    assert list(still_open["shares"]) == [150]
    assert list(still_open["buy_date"]) == [TS("2026-02-05")]


def test_open_lots_carry_the_window_end_and_are_flagged():
    lots = fifo_lots([_t("2026-01-05", "a", 100)], TS("2026-06-30"))
    assert lots["open"].all()
    assert lots["sell_date"].iloc[0] == TS("2026-06-30")


def test_fully_closed_position_leaves_no_open_lot():
    trades = [_t("2026-01-05", "a", 100), _t("2026-02-05", "a", -100)]
    lots = fifo_lots(trades, TS("2026-06-30"))
    assert len(lots) == 1 and not lots["open"].any()


def test_selling_more_than_held_raises_rather_than_silently_netting():
    with pytest.raises(ValueError, match="FIFO underflow"):
        fifo_lots([_t("2026-01-05", "a", 100), _t("2026-02-05", "a", -200)], TS("2026-06-30"))


# ----------------------------------------------------------------- register entitlement


def test_entitlement_excludes_a_lot_bought_on_the_ex_date():
    lots = fifo_lots([_t("2026-03-10", "a", 100)], TS("2026-06-30"))
    assert lots_entitled(lots, "a", TS("2026-03-10")).empty       # register closed the day before
    assert len(lots_entitled(lots, "a", TS("2026-03-11"))) == 1


def test_entitlement_excludes_a_lot_sold_on_the_register_day_but_keeps_the_ex_date_seller():
    trades = [_t("2026-01-05", "a", 100), _t("2026-03-09", "a", -100)]
    lots = fifo_lots(trades, TS("2026-06-30"))
    assert lots_entitled(lots, "a", TS("2026-03-10")).empty       # sold at the register close
    trades2 = [_t("2026-01-05", "a", 100), _t("2026-03-10", "a", -100)]
    lots2 = fifo_lots(trades2, TS("2026-06-30"))
    assert len(lots_entitled(lots2, "a", TS("2026-03-10"))) == 1  # still on the register


# ----------------------------------------------------------------- synthetic panel fixtures


@pytest.fixture
def panel():
    """Two names over six bars. 'a' goes ex 0.50 on a 10.00 close on bar 3; 'b' never pays."""
    idx = pd.date_range("2026-03-02", periods=6, freq="B")
    raw = pd.DataFrame({"a": [10.0, 10.0, 10.0, 9.5, 9.5, 9.5],
                        "b": [20.0] * 6}, index=idx)
    adj = pd.DataFrame({"a": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],   # total return flat
                        "b": [20.0] * 6}, index=idx)
    div = pd.DataFrame({"code": ["a"], "ex_date": [idx[3]], "dps": [0.50]})
    return adj, raw, div, idx


def test_dividend_events_reads_the_yield_off_the_unadjusted_prior_close(panel):
    adj, raw, div, idx = panel
    ev = dividend_events(adj, raw, div)
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row["code"] == "a" and row["date"] == idx[3]
    assert row["dyield"] == pytest.approx(0.05)      # 0.50 / 10.00
    assert row["adj_ret"] == pytest.approx(0.0)
    assert row["m"] == pytest.approx(1.0)


def test_net_panel_is_bit_identical_when_tau_is_zero(panel):
    adj, raw, div, _ = panel
    ev = dividend_events(adj, raw, div)
    lots = fifo_lots([_t("2026-03-02", "a", 100)], TS("2026-03-09"))
    taxed = event_tax_table(lots, ev, TAX_RULES["tau-D"])
    assert net_close_panel(adj, taxed, ev).equals(adj)     # GATE P, in miniature


def test_net_panel_withholds_exactly_tau_times_the_dividend(panel):
    adj, raw, div, idx = panel
    ev = dividend_events(adj, raw, div)
    lots = fifo_lots([_t("2026-03-02", "a", 100)], TS("2026-03-09"))
    taxed = event_tax_table(lots, ev, TAX_RULES["tau-B"])     # flat 20%
    net = net_close_panel(adj, taxed, ev)
    # adj_ret was 0, so the net return is -0.20 * 0.05 = -1.0%, applied once and carried forward.
    assert net["a"].iloc[2] == pytest.approx(10.0)
    for i in (3, 4, 5):
        assert net["a"].iloc[i] == pytest.approx(10.0 * 0.99)
    assert net["b"].equals(adj["b"])                          # a non-payer is untouched


def test_a_name_the_book_does_not_hold_is_never_taxed(panel):
    adj, raw, div, _ = panel
    ev = dividend_events(adj, raw, div)
    lots = fifo_lots([_t("2026-03-02", "b", 100)], TS("2026-03-09"))   # holds 'b', not 'a'
    taxed = event_tax_table(lots, ev, TAX_RULES["tau-B"])
    assert taxed.empty
    assert net_close_panel(adj, taxed, ev).equals(adj)


def test_event_tax_table_share_weights_two_lots_in_different_bands(panel):
    adj, raw, div, idx = panel
    ev = dividend_events(adj, raw, div)
    # 100 shares held since 2025 (>1 year -> exempt) and 300 bought days ago (<=1m -> 20%).
    trades = [_t("2025-01-06", "a", 100), _t("2026-03-03", "a", 300)]
    lots = fifo_lots(trades, TS("2026-03-20"))
    taxed = event_tax_table(lots, ev, TAX_RULES[PRIMARY_RULE])
    row = taxed.iloc[0]
    assert row["engine_shares"] == pytest.approx(400.0)
    assert row["tau"] == pytest.approx((100 * RATE_LONG + 300 * RATE_SHORT) / 400)
    assert row["w_short"] == pytest.approx(0.75)
    assert row["w_long"] == pytest.approx(0.25)


def test_open_rate_override_keeps_entitlement(panel):
    """Regression: an earlier sensitivity forced open lots into the 20% band by collapsing
    sell_date onto buy_date, which silently REVOKED their entitlement instead of re-rating them."""
    adj, raw, div, _ = panel
    ev = dividend_events(adj, raw, div)
    lots = fifo_lots([_t("2025-01-06", "a", 100)], TS("2026-03-20"))   # open, >1y -> exempt
    assert event_tax_table(lots, ev, TAX_RULES[PRIMARY_RULE]).iloc[0]["tau"] == RATE_LONG
    forced = event_tax_table(lots, ev, TAX_RULES[PRIMARY_RULE], open_rate=RATE_SHORT)
    assert len(forced) == 1                       # still entitled -- the bug dropped the row
    assert forced.iloc[0]["tau"] == RATE_SHORT
    assert forced.iloc[0]["engine_shares"] == pytest.approx(100.0)


# ----------------------------------------------------------------- cash streams


def test_leakage_and_gross_streams_are_the_actual_cash(panel):
    adj, raw, div, idx = panel
    ev = dividend_events(adj, raw, div)
    trades = [_t("2026-03-02", "a", 100)]
    lots = fifo_lots(trades, TS("2026-03-20"))
    taxed = event_tax_table(lots, ev, TAX_RULES["tau-B"])
    shares = daily_shares(trades, idx, adj.columns)
    gross = gross_dividend_stream(shares, adj, taxed)
    leak = leakage_stream(shares, adj, taxed)
    assert gross.sum() == pytest.approx(100 * 0.50)          # 100 shares x DPS 0.50
    assert leak.sum() == pytest.approx(100 * 0.50 * 0.20)
    assert gross.loc[idx[3]] == pytest.approx(50.0)          # credited on the ex-date bar


def test_band_cash_weights_by_value_not_by_share_count():
    """Regression: the study first split the bands by share count, which is not a currency
    amount -- a cheap name and an expensive one contribute equally per share. Caught by an
    independent brute-force recomputation, and the split moved by 5 pp when fixed."""
    idx = pd.date_range("2026-03-02", periods=3, freq="B")
    adj = pd.DataFrame({"cheap": [2.0] * 3, "dear": [200.0] * 3}, index=idx)
    shares = pd.DataFrame({"cheap": [100.0] * 3, "dear": [100.0] * 3}, index=idx)
    taxed = pd.DataFrame([
        {"code": "cheap", "date": idx[2], "dps": 0.1, "dyield": 0.05, "engine_shares": 100.0,
         "tau": RATE_SHORT, "w_short": 1.0, "w_mid": 0.0, "w_long": 0.0},
        {"code": "dear", "date": idx[2], "dps": 10.0, "dyield": 0.05, "engine_shares": 100.0,
         "tau": RATE_LONG, "w_short": 0.0, "w_mid": 0.0, "w_long": 1.0},
    ])
    bands = band_dividend_cash(shares, adj, taxed)
    # Equal share counts, but 'dear' pays 100x the cash: the long band must dominate.
    assert bands["short"] == pytest.approx(100 * 2.0 * 0.05)
    assert bands["long"] == pytest.approx(100 * 200.0 * 0.05)
    assert bands["short"] / bands["total"] == pytest.approx(1 / 101)
    assert bands["short"] / bands["total"] != pytest.approx(0.5)     # the share-weighted answer


def test_daily_shares_folds_fills_and_holds_them_forward():
    idx = pd.date_range("2026-03-02", periods=4, freq="B")
    trades = [_t("2026-03-03", "a", 100), _t("2026-03-04", "a", 50), _t("2026-03-05", "a", -150)]
    sh = daily_shares(trades, idx, ["a"])
    assert list(sh["a"]) == [0.0, 100.0, 150.0, 0.0]


def test_closed_form_terminal_removes_a_known_leak():
    idx = pd.date_range("2026-03-02", periods=4, freq="B")
    eq = pd.Series([100.0, 110.0, 121.0, 133.1], index=idx)
    leak = pd.Series([0.0, 0.0, 12.1, 0.0], index=idx)     # 10% of the prior close, 110.0
    assert closed_form_terminal(eq, leak) == pytest.approx(133.1 * (1 - 12.1 / 110.0))
    assert closed_form_terminal(eq, leak, lag=False) == pytest.approx(133.1 * (1 - 12.1 / 121.0))
    assert closed_form_terminal(eq, pd.Series(0.0, index=idx)) == pytest.approx(133.1)


# ----------------------------------------------------------------- frozen grid integrity


def test_the_pre_registered_grid_is_intact():
    """The seven conventions were frozen in issue #24 before any result existed."""
    assert set(TAX_RULES) == {"tau-A", "tau-B", "tau-C", "tau-D", "tau-E", "tau-F", "tau-G"}
    assert PRIMARY_RULE == "tau-A"
    assert (RATE_SHORT, RATE_MID, RATE_LONG) == (0.20, 0.10, 0.00)
    a = TAX_RULES["tau-A"]
    assert (a.kind, a.boundary, a.measure) == ("statutory", "calendar_month", "sale")


def test_unknown_rule_shape_is_not_silently_treated_as_statutory():
    rule = TaxRule("nonsense", kind="flat", flat_rate=0.33)
    assert rule.rate(TS("2020-01-01"), TS("2026-01-01"), TS("2021-01-01")) == 0.33


def test_empty_inputs_do_not_raise():
    idx = pd.date_range("2026-03-02", periods=3, freq="B")
    adj = pd.DataFrame({"a": [1.0, 1.0, 1.0]}, index=idx)
    empty_lots = fifo_lots([], TS("2026-03-04"))
    ev = dividend_events(adj, adj, pd.DataFrame(columns=["code", "ex_date", "dps"]))
    assert ev.empty and empty_lots.empty
    taxed = event_tax_table(empty_lots, ev, TAX_RULES["tau-A"])
    assert taxed.empty
    assert net_close_panel(adj, taxed, ev).equals(adj)
    assert band_dividend_cash(pd.DataFrame(0.0, index=idx, columns=["a"]), adj, taxed)["total"] == 0
    assert np.isclose(leakage_stream(pd.DataFrame(0.0, index=idx, columns=["a"]),
                                     adj, taxed).sum(), 0.0)
