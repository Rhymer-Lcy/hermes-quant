"""A-share cost model: commission floor, sell-side stamp tax, zero-cost sanity."""
import math

from hermes.research.backtest.frictions import ZERO_COSTS, AShareCosts


def test_commission_min_floor():
    c = AShareCosts()
    turnover = 1_000.0  # 0.025% * 1000 = 0.25 -> floored to the 5-RMB minimum
    assert math.isclose(c.buy_fees(turnover), 5.0 + turnover * c.transfer_fee_rate)


def test_commission_above_floor():
    c = AShareCosts()
    turnover = 100_000.0  # 0.025% * 100000 = 25 > 5, so no floor
    assert math.isclose(c.buy_fees(turnover), 25.0 + turnover * c.transfer_fee_rate)


def test_stamp_tax_is_sell_side_only():
    c = AShareCosts()
    turnover = 100_000.0
    # sell minus buy fee, commission cancels, leaves exactly the stamp tax
    assert math.isclose(c.sell_fees(turnover) - c.buy_fees(turnover), turnover * c.stamp_tax_sell)


def test_zero_turnover_no_fee():
    c = AShareCosts()
    assert c.buy_fees(0.0) == 0.0
    assert c.sell_fees(0.0) == 0.0


def test_zero_costs_are_zero():
    assert ZERO_COSTS.buy_fees(1e6) == 0.0
    assert ZERO_COSTS.sell_fees(1e6) == 0.0
    assert ZERO_COSTS.slip == 0.0


def test_slip_bps_conversion():
    assert math.isclose(AShareCosts(slippage_bps=5.0).slip, 5e-4)
