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

# Paper-trading capital tiers (元), grouped small / medium / large. Chosen to be informative,
# not round-number filler:
#   small  [1万, 3万, 5万]  -- brackets the feasibility knee: 1万 is INFEASIBLE (100-share lots +
#                            5元 min commission prevent a 10-name book), 3万 is the floor, 5万 is
#                            comfortably viable. This is the research-critical band.
#   medium [10万, 50万]     -- the working regime (book fully diversified).
#   large  [100万, 500万]   -- capacity reference; the strategy SATURATES by ~10万 (these add no
#                            new behaviour) and assumes negligible market impact -- true for HS300
#                            large caps at <=50万/name, but NOT modeled (flat 5bps slippage, no
#                            size-dependent impact term). Read them as "does it scale", not research.
CAPITAL_TIERS: dict[str, list[int]] = {
    "small": [10_000, 30_000, 50_000],
    "medium": [100_000, 500_000],
    "large": [1_000_000, 5_000_000],
}
ALL_TIERS: list[int] = [v for tier in CAPITAL_TIERS.values() for v in tier]
TIER_LABEL: dict[int, str] = {v: label for label, tier in CAPITAL_TIERS.items() for v in tier}


def deployed_signal(close: pd.DataFrame, pe_ttm: pd.DataFrame, members_asof,
                    spec: DeployedStrategy = DEPLOYED) -> pd.DataFrame:
    """The deployed score panel: value (1/PE) + a light 1-month-reversal tilt, each
    restricted to the PIT members BEFORE the cross-sectional blend (IRON RULE 1 -- else the
    survivorship union leaks into the z-scores). Higher score = more attractive."""
    ep = fl.restrict_to_universe(fl.earnings_yield(pe_ttm), members_asof)
    rev = fl.restrict_to_universe(-fl.trailing_return(close, spec.reversal_lookback), members_asof)
    return fl.blend([ep, rev], [spec.value_weight, spec.reversal_weight])
