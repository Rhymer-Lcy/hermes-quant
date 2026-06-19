"""Offline research: factor construction, model training, backtest, evaluation.

This package is the part that runs on the V100x8 cluster (or locally for small
models). It emits signals/scores; it does NOT place orders — execution lives in
`hermes.execution` (vnpy).
"""

# Capital tiers (RMB) for cross-account analysis. Chosen to span the regimes where
# A-share frictions change character, aligned with real thresholds:
#   10_000     small  -- ~1 lot ≈ whole account; diversification impossible
#   50_000     small  -- 100-share lot + 5元 min commission dominate
#   100_000    medium -- common miniQMT live threshold; ~5-10 names feasible
#   500_000    medium -- 合格投资者 / legacy QMT threshold
#   1_000_000  large  -- full diversification; costs ~proportional
#   3_000_000  large  -- 专业投资者 threshold
CAPITAL_TIERS = [10_000, 50_000, 100_000, 500_000, 1_000_000, 3_000_000]

