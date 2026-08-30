"""The eight pre-registered verdict gates of the rebalance-timing study (issue #21).

REGRESSION: an adversarial audit of the first implementation showed the gate battery could print
CONFIRMED for a day whose OWN Holm-corrected p was 1.0 -- check A tested "some day is significant"
while checks C-H were read at the CAGR champion, so the eight-gate conjunction certified a
disjunction. `test_significance_from_another_day_cannot_confirm_the_champion` reproduces exactly
the fabricated case the audit used. The gates now bind one day at a time.

The study lives in scripts/ (no package surface), so it is loaded by path rather than imported.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rebalance_timing_study",
    Path(__file__).resolve().parents[1] / "scripts" / "rebalance_timing_study.py")
rts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rts)

DAYS = rts.DAYS
REGIME_NAMES = [nm for _lo, _hi, nm in rts.REGIMES]


def _frames(*, champ=17, sig_day=9, champ_gross=+0.004, champ_regimes_ok=True):
    """A synthetic sweep: `champ` is the CAGR argmax, `sig_day` is the only Holm rejection."""
    net = pd.DataFrame(index=pd.Index(DAYS, name="D"))
    net["cagr"] = 0.1049
    net.loc[champ, "cagr"] = 0.1120
    net["dCAGR"] = net["cagr"] - net.loc[1, "cagr"]
    net["dSharpe"] = 0.0
    net["dCalmar"] = 0.0
    net.loc[champ, ["dSharpe", "dCalmar"]] = 0.05
    net["dMaxDD"] = 0.0

    gross = pd.DataFrame(index=net.index)
    gross["dCAGR"] = 0.0
    gross.loc[champ, "dCAGR"] = champ_gross

    paired = pd.DataFrame(index=pd.Index([d for d in DAYS if d != 1], name="D"))
    paired["reject"] = False
    paired.loc[sig_day, "reject"] = True
    paired["mean_diff"] = 0.0
    paired.loc[sig_day, "mean_diff"] = 0.0009
    paired.loc[champ, "mean_diff"] = 0.0006
    paired["gross_mean"] = 0.0005

    reg = pd.DataFrame(index=net.index)
    for nm in REGIME_NAMES:
        reg[f"{nm} dCAGR"] = 0.0
        reg.loc[champ, f"{nm} dCAGR"] = 0.01 if champ_regimes_ok else -0.01

    sec = pd.DataFrame(index=net.index)
    sec["dCAGR"] = 0.0
    sec.loc[champ, "dCAGR"] = 0.006

    tiers = {}
    for cap in rts.ROBUST_TIERS:
        t = pd.DataFrame(index=net.index)
        t["dCAGR"] = 0.0
        t.loc[champ, "dCAGR"] = 0.004
        tiers[cap] = t
    return net, gross, paired, reg, sec, tiers


def test_significance_from_another_day_cannot_confirm_the_champion():
    """The audit's exact counter-example: D=17 is the CAGR champion and passes the day-level
    checks, but only D=9 survives Holm. No day may then pass all eight."""
    tab = rts.gate_table(*_frames(champ=17, sig_day=9))
    assert not tab.loc[17, "A_holm"], "champion must not inherit another day's significance"
    assert tab.loc[9, "A_holm"], "the genuinely significant day should still show A"
    assert not tab["ALL_EIGHT"].any(), "no single day passes all eight, so nothing is confirmed"


def test_a_gate_requires_the_days_own_rejection_and_a_positive_mean():
    net, gross, paired, reg, sec, tiers = _frames(champ=17, sig_day=17)
    paired.loc[17, "mean_diff"] = -0.0004                 # rejected but in the WRONG direction
    tab = rts.gate_table(net, gross, paired, reg, sec, tiers)
    assert not tab.loc[17, "A_holm"]


def test_b_gate_reads_the_gross_sweep_so_a_cost_only_gain_fails():
    """A net-only improvement that is merely lower turnover must not pass B."""
    tab_ok = rts.gate_table(*_frames(champ=17, sig_day=17, champ_gross=+0.004))
    assert tab_ok.loc[17, "B_gross"]
    tab_bad = rts.gate_table(*_frames(champ=17, sig_day=17, champ_gross=-0.001))
    assert not tab_bad.loc[17, "B_gross"], "gross must agree in direction, not just net"


def test_all_eight_can_pass_when_one_day_genuinely_satisfies_every_gate():
    """The battery must not be vacuously unpassable -- a day meeting every criterion confirms."""
    net, gross, paired, reg, sec, tiers = _frames(champ=17, sig_day=17)
    for d in (16, 17, 18):                                 # build a real 3-day plateau around it
        net.loc[d, "cagr"] = 0.1120
    net["dCAGR"] = net["cagr"] - net.loc[1, "cagr"]
    tab = rts.gate_table(net, gross, paired, reg, sec, tiers)
    assert tab.loc[17, "ALL_EIGHT"], tab.loc[17].to_dict()


def test_plateau_gate_rejects_an_isolated_spike():
    net, gross, paired, reg, sec, tiers = _frames(champ=17, sig_day=17)
    tab = rts.gate_table(net, gross, paired, reg, sec, tiers)
    assert not tab.loc[17, "C_plateau"], "a lone winner with ordinary neighbours is not a plateau"


def test_regime_gate_requires_every_regime_to_agree():
    tab = rts.gate_table(*_frames(champ=17, sig_day=17, champ_regimes_ok=False))
    assert not tab.loc[17, "D_regimes"]


def test_baseline_day_is_not_a_candidate_against_itself():
    tab = rts.gate_table(*_frames())
    assert rts.BASELINE_D not in tab.index


def test_plateau_runs_finds_maximal_contiguous_runs():
    better = pd.Series({d: d in {3, 4, 5, 9, 20, 21} for d in DAYS})
    assert rts.plateau_runs(better) == [[3, 4, 5], [9], [20, 21]]


def test_holm_is_monotone_and_never_less_strict_than_bonferroni_at_the_minimum():
    p = {d: v for d, v in zip(range(2, 8), [0.001, 0.01, 0.02, 0.30, 0.40, 0.90])}
    out = rts.holm(p)
    adj = out["p_holm"].tolist()
    assert adj == sorted(adj), "Holm adjusted p must be non-decreasing in raw p"
    assert out.loc[2, "p_holm"] == pytest.approx(0.001 * len(p))   # smallest gets full Bonferroni
    assert out["p_holm"].max() <= 1.0


def test_carry_in_drawdown_uses_the_global_high_water_mark():
    """docs/oos_decay.md's correction: a window-local mark forgives losses carried into the window."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    eq = pd.Series([100.0, 120.0, 90.0, 95.0, 80.0, 85.0], index=idx)
    carried = rts.carry_in_drawdown(eq, "2020-01-03", "2020-01-06")
    assert carried == pytest.approx(80.0 / 120.0 - 1.0)            # measured against the 120 peak
    local = (eq.loc["2020-01-03":] / eq.loc["2020-01-03":].cummax() - 1).min()
    assert carried < local, "the carried-in mark must be the stricter of the two"
