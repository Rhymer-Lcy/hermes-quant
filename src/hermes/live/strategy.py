"""The DEPLOYED strategy spec -- ONE definition, shared by research demos AND live paper
trading, so the served signal can never drift from the researched one (train/serve skew is
the dominant silent alpha-killer; see execution/__init__.py).

Deployed = value (earnings yield) + a LIGHT 1-month-reversal tilt (blend 5:1), equal weight,
top-10, monthly. Established in docs/multi_factor.md: the 4/1-9/1 region is a robust plateau
(net Calmar ~0.32 > A2's 0.30), inverse-vol does NOT stack on top of the reversal tilt, and
the turnover buffer (rebalance_band) hurts value rotation -- so both stay off here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..research.factors import library as fl


@dataclass(frozen=True)
class DeployedStrategy:
    n_hold: int = 10
    value_weight: float = 5.0          # value : reversal = 5 : 1 (the plateau read)
    reversal_weight: float = 1.0
    reversal_lookback: int = 20        # 1-month (≈20 trading days) reversal
    rebalance_band: int = 0            # turnover buffer OFF (hurts value rotation)
    weight_asof = None                 # equal weight (inverse-vol does not stack)


DEPLOYED = DeployedStrategy()


def deployed_signal(close: pd.DataFrame, pe_ttm: pd.DataFrame, members_asof,
                    spec: DeployedStrategy = DEPLOYED) -> pd.DataFrame:
    """The deployed score panel: value (1/PE) + a light 1-month-reversal tilt, each
    restricted to the PIT members BEFORE the cross-sectional blend (IRON RULE 1 -- else the
    survivorship union leaks into the z-scores). Higher score = more attractive."""
    ep = fl.restrict_to_universe(fl.earnings_yield(pe_ttm), members_asof)
    rev = fl.restrict_to_universe(-fl.trailing_return(close, spec.reversal_lookback), members_asof)
    return fl.blend([ep, rev], [spec.value_weight, spec.reversal_weight])
