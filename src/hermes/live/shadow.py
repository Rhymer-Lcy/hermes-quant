"""Forward SHADOW paper records: a frozen candidate schedule racing the canonical D=1. Issue #22.

WHAT THIS IS. Issue #21 enumerated all 31 calendar rebalance days and closed REJECTED / NOT
IDENTIFIED: D=5 was the historical CAGR champion (+12.67% vs +10.49%) but failed the pre-registered
battery -- HAC t = 1.508, Holm p = 1.0, +0.06 pp/yr on 2016-2025, unstable across regimes, and the
bootstrap champion wandered. A historical maximum is not validated alpha. This module exists to put
that maximum in front of data it cannot be fitted to.

WHAT THIS IS NOT. It is not a recommendation, not a strategy change, and not a second paper
account. `results/paper/` remains the ONLY canonical forward record; the shadow writes to a
physically separate tree and asserts, at runtime, that it never touched the canonical one.

THE FORK. Both books replay the SAME history through the SAME engine from PAPER_INCEPTION
(2026-06-18) on the canonical schedule; only executions strictly after SHADOW_INCEPTION_ASOF
(2026-08-28) use the candidate. So at the shadow's inception bar the two books hold identical
positions, shares, cash and equity by construction, and every later difference is genuine forward
evidence. The earlier 2026-06-18 -> 2026-08-28 D=5 counterfactual lives in
results/backtests/counterfactual/ and is NOT part of this record: splicing it on would hand the
candidate a differently-chosen book at inception and destroy the comparison.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from ..data.lake import load_close_panel
from ..data.membership import MEMBERSHIP_PARQUET, membership_lookup
from ..io import atomic_to_parquet
from ..paths import PAPER_DIR, REPO_ROOT, RESULTS_DIR
from ..research.backtest.frictions import AShareCosts
from ..research.backtest.portfolio import signal_portfolio_backtest
from ..research.backtest.schedule import (assert_no_lookahead, calendar_rebalance_schedule,
                                          splice_schedule)
from .strategy import ALL_TIERS, DEPLOYED, PAPER_INCEPTION, DeployedStrategy, deployed_signal

#: The bar at which the shadow and the canonical record share one state and then fork. Frozen in
#: issue #22 and never recomputed from the lake -- moving it would silently redefine the experiment.
SHADOW_INCEPTION_ASOF = "2026-08-28"
CANDIDATE_DAY = 5
BASELINE_DAY = 1
SHADOW_ID = "d5"
PREREG_ISSUE = "https://github.com/Rhymer-Lcy/hermes-quant/issues/22"
SCHEMA_VERSION = 2

#: Provenance SHAs, kept strictly separate from the experiment's frozen PARAMETERS. Code history is
#: not strategy definition: a runtime SHA may move with maintenance, the candidate day may not.
#:   PREREGISTRATION_BASE_SHA   repository HEAD when issue #22 was registered and the manifest first
#:                              frozen -- 12:18:01, while the shadow sources were still uncommitted.
#:                              This is what v1's `created_at_commit` actually recorded.
#:   INITIAL_IMPLEMENTATION_SHA the first committed revision at which the shadow's three blocking
#:                              gates can be executed end to end. 0eb165a added live/shadow.py but
#:                              not the runner that hosts the gates; 03acb02 added
#:                              scripts/paper_shadow_d5.py (gate_production_parity /
#:                              gate_common_state / gate_no_lookahead) and its wrapper, so 03acb02
#:                              is the first revision that can actually run and pass them.
PREREGISTRATION_BASE_SHA = "ca0d64bb038bb26734bf2aa3274f19d452b0f09f"
INITIAL_IMPLEMENTATION_SHA = "03acb02"

SHADOW_ROOT = RESULTS_DIR / "paper_shadow"
SHADOW_DIR = SHADOW_ROOT / SHADOW_ID


def _positions(trades) -> dict[str, int]:
    """Final book folded from the engine's fill log (what live.paper.replay does day by day)."""
    pos: dict[str, int] = {}
    for t in trades:
        pos[t["code"]] = pos.get(t["code"], 0) + int(t["shares"])
    return {c: s for c, s in sorted(pos.items()) if s > 0}


def _drawdown_since(equity: pd.Series) -> float:
    """Drawdown measured from the series' own start -- correct here because the shadow record
    genuinely BEGINS at inception (unlike a sub-window of an ongoing account, where a carried-in
    high-water mark is required; see docs/oos_decay.md)."""
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0


def _avg_names_held_since(trades, dates: pd.DatetimeIndex, since: pd.Timestamp) -> float:
    """Mean number of names held per bar, counted only from `since`.

    The engine's `avg_names_held` covers the whole replay, which for a shadow spans the shared
    pre-inception history too; the forward record needs the post-fork figure, so the book is folded
    from the fill log bar by bar. This is the same fold live.paper.replay performs, restricted to
    the forward window.
    """
    by_day: dict[pd.Timestamp, list[dict]] = {}
    for tr in trades:
        by_day.setdefault(tr["date"], []).append(tr)
    pos: dict[str, int] = {}
    counts = []
    for d in dates:
        for tr in by_day.get(d, ()):
            pos[tr["code"]] = pos.get(tr["code"], 0) + int(tr["shares"])
        if d >= since:
            counts.append(sum(1 for v in pos.values() if v > 0))
    return float(sum(counts) / len(counts)) if counts else 0.0


def runtime_provenance() -> dict:
    """The HEAD and worktree state of the code producing THIS run.

    Recorded per report so a result years from now can be traced to the revision that computed it.
    A persisted forward run on a dirty tree would be unauditable evidence, so the runner refuses
    it; a dry run may proceed, since it writes nothing.

    The repository is addressed with `git -C REPO_ROOT`, never via the process cwd. The scheduled
    task registers no WorkingDirectory, so a cwd-relative `git` ran in System32, failed, and
    reported the tree DIRTY when it was clean -- which silently blocked every persisted run.
    `runtime_git_ok` now records whether git answered at all, so "unreadable" and "dirty" can
    never again be the same signal."""
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:                       # noqa: BLE001 -- provenance is best-effort
            return None
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"runtime_code_sha": sha or "unknown",
            "runtime_git_ok": sha is not None and status is not None,
            "runtime_worktree_clean": bool(sha) and status == ""}


class ShadowGateError(RuntimeError):
    """A blocking gate failed: no shadow output may be written."""


def build_panels(as_of: str | None = None, spec: DeployedStrategy = DEPLOYED):
    """The same panels live.paper.live_step builds: full-history signal, then sliced at inception."""
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof_fn = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        close, pe = close.loc[close.index <= cutoff], pe.loc[pe.index <= cutoff]
    signal = deployed_signal(close, pe, asof_fn, spec)          # lookbacks satisfied on full history
    incept = pd.Timestamp(PAPER_INCEPTION)
    return close.loc[close.index >= incept], signal.loc[signal.index >= incept], asof_fn


def shadow_schedule(dates: pd.DatetimeIndex, candidate_day: int = CANDIDATE_DAY) -> dict[int, int]:
    """Canonical through the shadow-inception bar, candidate strictly after it.

    Raises if the inception bar is not in the panel: silently forking somewhere else would be the
    one failure mode that invalidates the whole comparison."""
    fork = pd.Timestamp(SHADOW_INCEPTION_ASOF)
    at = dates.searchsorted(fork, side="right") - 1
    if at < 0 or dates[at] != fork:
        raise ShadowGateError(
            f"shadow inception bar {SHADOW_INCEPTION_ASOF} is not a bar in the panel "
            f"(nearest {dates[at].date() if at >= 0 else 'none'}); refusing to fork elsewhere")
    sch = splice_schedule(calendar_rebalance_schedule(dates, BASELINE_DAY),
                          calendar_rebalance_schedule(dates, candidate_day), at, dates)
    assert_no_lookahead(sch, dates)
    return sch


def run_book(close, signal, capital, asof_fn, sched, spec: DeployedStrategy = DEPLOYED,
             costs: AShareCosts | None = None):
    """One book through the production engine. `initial_rebalance=True` invests the seed on the
    inception bar, exactly as the canonical paper account does."""
    return signal_portfolio_backtest(
        close, signal, capital, n_hold=spec.n_hold, costs=costs, members_asof=asof_fn,
        weight_asof=spec.weight_asof, rebalance_band=spec.rebalance_band,
        collect_trades=True, initial_rebalance=True, schedule=sched)


def fingerprint_canonical() -> dict[str, tuple[int, int]]:
    """Size + mtime of every canonical artefact, so an accidental write is provable rather than
    merely improbable. Re-checked after the run."""
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(PAPER_DIR.glob("*")) if p.is_file()}


def assert_canonical_untouched(before: dict[str, tuple[int, int]]) -> None:
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(PAPER_DIR.glob("*")) if p.is_file()}
    if after != before:
        changed = sorted(set(before) ^ set(after)) or [k for k in before if before[k] != after.get(k)]
        raise ShadowGateError(f"results/paper WAS MODIFIED by the shadow run: {changed}")


# --- manifest ------------------------------------------------------------------------

#: Fields a normal run may only VALIDATE. The two provenance anchors join them: the baseline and
#: the implementation revision are historical facts, not settings, and a run must never rewrite
#: them. `experiment_freeze_sha` is also frozen once written. `runtime_code_sha` is deliberately
#: absent -- it belongs to the per-run report, not the manifest.
FROZEN_FIELDS = ("shadow_id", "candidate_calendar_day", "baseline_calendar_day",
                 "shadow_inception_asof", "pre_registration_issue", "paper_inception",
                 "capital_tiers", "strategy_spec", "cost_model",
                 "preregistration_base_sha", "initial_implementation_sha")


def build_manifest(commit: str, spec: DeployedStrategy = DEPLOYED,
                   costs: AShareCosts | None = None, *,
                   experiment_freeze_sha: str | None = None,
                   legacy_created_at_commit: str | None = None) -> dict:
    """The frozen experiment definition plus its provenance. `commit` is the CURRENT HEAD, recorded
    only as the freeze SHA on a first write; it is never allowed to overwrite a frozen field."""
    c = costs or AShareCosts()
    return {
        "schema_version": SCHEMA_VERSION,
        "shadow_id": SHADOW_ID,
        "candidate_calendar_day": CANDIDATE_DAY,
        "baseline_calendar_day": BASELINE_DAY,
        "shadow_inception_asof": SHADOW_INCEPTION_ASOF,
        "pre_registration_issue": PREREG_ISSUE,
        "paper_inception": PAPER_INCEPTION,
        # --- provenance (code history; NOT experiment parameters) ---
        "preregistration_base_sha": PREREGISTRATION_BASE_SHA,
        "initial_implementation_sha": INITIAL_IMPLEMENTATION_SHA,
        "experiment_freeze_sha": experiment_freeze_sha or commit,
        "legacy_created_at_commit": legacy_created_at_commit,
        "provenance_note": (
            "v1 carried a single `created_at_commit` = the git HEAD at freeze time, recorded on a "
            "DIRTY tree while the shadow sources were still uncommitted. It denoted the "
            "pre-implementation baseline, never the implementation. v2 separates the base, the "
            "implementation, the freeze and the per-run runtime SHA; the v1 value is preserved "
            "verbatim in legacy_created_at_commit."),
        "capital_tiers": list(ALL_TIERS),
        "strategy_spec": {k: v for k, v in asdict(spec).items()},
        "cost_model": {"commission_rate": c.commission_rate, "min_commission": c.min_commission,
                       "stamp_tax_sell": c.stamp_tax_sell, "transfer_fee_rate": c.transfer_fee_rate,
                       "slippage_bps": c.slippage_bps, "lot_size": c.lot_size},
        "retrospective_counterfactual_before_inception_is_not_part_of_forward_record": True,
        "candidate_parameters_are_frozen": True,
        "results_paper_is_canonical_and_must_not_be_rewritten": True,
    }


def validate_manifest(path: Path, expected: dict) -> dict:
    """Read the on-disk manifest and refuse to run if a FROZEN field drifted.

    Non-frozen fields (the commit that last wrote it) may move; the frozen ones define the
    experiment and a silent edit is exactly what this guards. The existing file is never rewritten
    by a normal run."""
    if not path.exists():
        return expected
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    # A v1 file predates the two provenance anchors; their absence is a schema gap to migrate,
    # not drift to refuse. Every other frozen field must still match exactly.
    v1 = on_disk.get("schema_version", 1) < SCHEMA_VERSION
    drift = {k: (on_disk.get(k), expected[k]) for k in FROZEN_FIELDS
             if not (v1 and k not in on_disk) and on_disk.get(k) != expected[k]}
    if drift:
        raise ShadowGateError(f"manifest frozen-field drift {drift}; refusing to run. "
                              f"Retire the shadow deliberately instead of editing it.")
    return on_disk


# --- the daily step ------------------------------------------------------------------

def shadow_step(capital: float, *, as_of: str | None = None, spec: DeployedStrategy = DEPLOYED,
                costs: AShareCosts | None = None, persist: bool = True,
                candidate_day: int = CANDIDATE_DAY, out_dir: Path | None = None,
                panels=None) -> dict:
    """One tier's shadow report, with the canonical D=1 comparator computed on the same panel.

    Both books are recomputed from the seed every run (idempotent), and everything is normalised to
    the shadow-inception bar so the record starts at 0% there. Costs and trade counts are reported
    SINCE INCEPTION by differencing against a run truncated at that bar, which uses the engine's own
    accounting rather than re-deriving slippage from fills.
    """
    close, signal, asof_fn = panels if panels is not None else build_panels(as_of, spec)
    fork = pd.Timestamp(SHADOW_INCEPTION_ASOF)
    if close.index[-1] < fork:
        raise ShadowGateError(f"lake ends {close.index[-1].date()}, before the shadow inception "
                              f"{SHADOW_INCEPTION_ASOF}")
    sched_cand = shadow_schedule(close.index, candidate_day)
    sched_base = calendar_rebalance_schedule(close.index, BASELINE_DAY)

    books = {}
    for tag, sch in (("shadow", sched_cand), ("d1", sched_base)):
        full = run_book(close, signal, capital, asof_fn, sch, spec, costs)
        pre = close.index <= fork
        pre_res = run_book(close.loc[pre], signal.loc[pre], capital, asof_fn,
                           {e: s for e, s in sch.items() if close.index[e] <= fork}, spec, costs)
        eq_since = full.equity.loc[full.equity.index >= fork]
        books[tag] = {
            "res": full, "eq_since": eq_since,
            "start_equity": float(full.equity.loc[fork]),
            "equity": float(full.equity.iloc[-1]),
            "ret": float(full.equity.iloc[-1] / full.equity.loc[fork] - 1.0),
            "dd": _drawdown_since(eq_since),
            "costs": float(full.total_costs - pre_res.total_costs),
            "trades_since": [t for t in full.trades if t["date"] > fork],
            "positions": _positions(full.trades),
        }
    s, b = books["shadow"], books["d1"]
    today = close.index[-1]
    run_dt = date.today()
    lag = (run_dt - today.date()).days
    report = {
        "shadow_id": SHADOW_ID, "status": "forward_shadow",
        "candidate_calendar_day": candidate_day, "baseline_calendar_day": BASELINE_DAY,
        "pre_registration_issue": PREREG_ISSUE,
        "shadow_inception_asof": SHADOW_INCEPTION_ASOF,
        **runtime_provenance(),
        "as_of": today.strftime("%Y-%m-%d"), "run_date": run_dt.strftime("%Y-%m-%d"),
        "lake_lag_days": lag, "fresh": lag <= 4,
        "capital_tier": int(capital),
        "starting_equity_at_shadow_inception": s["start_equity"],
        "equity": s["equity"],
        "total_return_since_shadow_inception": s["ret"],
        "max_drawdown_since_shadow_inception": s["dd"],
        "n_positions": len(s["positions"]),
        "avg_names_held": _avg_names_held_since(s["res"].trades, close.index, fork),
        "positions": s["positions"],
        "n_trades_since_shadow_inception": len(s["trades_since"]),
        "total_costs_since_shadow_inception": s["costs"],
        "today_trades": [{**t, "date": t["date"].strftime("%Y-%m-%d")}
                         for t in s["res"].trades if t["date"] == today],
        # baseline comparator on the identical panel, so one JSON answers the whole question
        "d1_equity_same_asof": b["equity"],
        "d1_return_since_shadow_inception": b["ret"],
        "excess_return_vs_d1": s["ret"] - b["ret"],
        "d1_max_drawdown_since_shadow_inception": b["dd"],
        "drawdown_diff_vs_d1": s["dd"] - b["dd"],
        "d1_costs_since_shadow_inception": b["costs"],
        "cost_diff_vs_d1": s["costs"] - b["costs"],
        "d1_positions": b["positions"],
    }
    if persist:
        d = out_dir or SHADOW_DIR
        d.mkdir(parents=True, exist_ok=True)
        (d / f"report_{int(capital)}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        atomic_to_parquet(s["eq_since"].rename("equity").to_frame(),
                          d / f"curve_{int(capital)}.parquet", index=True)
        tdf = pd.DataFrame(s["trades_since"]) if s["trades_since"] else \
            pd.DataFrame(columns=["date", "code", "shares", "price", "fee"])
        atomic_to_parquet(tdf, d / f"trades_{int(capital)}.parquet", index=False)
    return report
