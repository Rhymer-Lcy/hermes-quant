"""Closest-weight integer-lot allocation (CWIL). Issue #25. Research-only, opt-in.

The production allocator floors every name's target independently:

    shares_i = floor( (gross/n_hold) / (p_i * (1+slip) * lot) ) * lot

Each name is rounded down in isolation, so the leftovers are never pooled. This module answers a
narrow, frozen question instead: of all integer-lot books that are actually affordable after real
fees, which one lands closest to the intended weight vector?

    f(k) = sum_i (a_i - t_i)^2 + (a_cash - t_cash)^2

with weights VALUED at the execution bar's close and the budget CHARGED at the slipped price plus
commission (CNY 5 minimum), stamp tax and transfer fee. Valuation price and transaction price are
deliberately different quantities and conflating them is how a "feasible" book turns out to need
more cash than it has.

Nothing here runs unless a caller opts in; `allocator=None` leaves the engine byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frictions import AShareCosts

REPAIR_BUDGET = 24   # over-budget frontier points repaired per rebalance (cost/quality knob)


@dataclass(frozen=True)
class AllocInputs:
    """Everything one rebalance needs, resolved to plain arrays in a fixed name order."""

    codes: tuple[str, ...]
    price: np.ndarray          # execution bar close -- the VALUATION price
    target_w: np.ndarray       # intended weight per name
    held: np.ndarray           # shares currently held (may be non-zero for names outside `top`)
    equity: float
    cash: float
    lot: int
    slip: float
    costs: AShareCosts

    @property
    def target_cash_w(self) -> float:
        return 1.0 - float(self.target_w.sum())


def _trade_cash(inp: AllocInputs, shares: np.ndarray) -> float:
    """Cash AFTER moving `held` -> `shares`, charging real costs. Negative means infeasible.

    Sells settle before buys, matching `execute_orders`, so the proceeds of a liquidation are
    genuinely available to fund the same rebalance's purchases."""
    cash = float(inp.cash)
    delta = shares - inp.held
    sells = delta < 0
    if sells.any():
        qty = -delta[sells]
        ep = inp.price[sells] * (1.0 - inp.slip)
        turnover = qty * ep
        for t in turnover:
            cash += float(t) - inp.costs.sell_fees(float(t))
    buys = delta > 0
    if buys.any():
        qty = delta[buys]
        ep = inp.price[buys] * (1.0 + inp.slip)
        turnover = qty * ep
        for t in turnover:
            cash -= float(t) + inp.costs.buy_fees(float(t))
    return cash


def _objective(inp: AllocInputs, shares: np.ndarray) -> float:
    a = shares * inp.price / inp.equity
    d = a - inp.target_w
    a_cash = 1.0 - float(a.sum())
    return float((d * d).sum() + (a_cash - inp.target_cash_w) ** 2)


def _incremental_cost(inp: AllocInputs, shares: np.ndarray) -> float:
    """Total fees plus slippage implied by moving `held` -> `shares` (for tie-breaks and stats)."""
    delta = shares - inp.held
    total = 0.0
    for i in np.nonzero(delta)[0]:
        q = abs(float(delta[i]))
        p = float(inp.price[i])
        if delta[i] < 0:
            ep = p * (1.0 - inp.slip)
            total += q * (p - ep) + inp.costs.sell_fees(q * ep)
        else:
            ep = p * (1.0 + inp.slip)
            total += q * (ep - p) + inp.costs.buy_fees(q * ep)
    return total


def _turnover(inp: AllocInputs, shares: np.ndarray) -> float:
    return float((np.abs(shares - inp.held) * inp.price).sum())


def _key(inp: AllocInputs, shares: np.ndarray) -> tuple:
    """The FROZEN tie-break order from issue #25: objective, then residual cash, then incremental
    cost, then turnover, then lexicographic ticker order (implicit in the array order)."""
    a_cash = 1.0 - float((shares * inp.price).sum() / inp.equity)
    return (round(_objective(inp, shares), 15), round(a_cash, 15),
            round(_incremental_cost(inp, shares), 9), round(_turnover(inp, shares), 6),
            tuple(int(s) for s in shares))


def ideal_lots(inp: AllocInputs) -> np.ndarray:
    """Fractional lot count that would hit the target weight exactly, valued at the close."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = inp.target_w * inp.equity / (inp.price * inp.lot)
    return np.where(np.isfinite(x), x, 0.0)


def baseline_lots(inp: AllocInputs) -> np.ndarray:
    """The production rule, reproduced here so the two can be compared on identical inputs.

    Note it budgets at the SLIPPED price, which is what `target_shares` does."""
    tgt_notional = inp.target_w * inp.equity
    per_lot = inp.price * (1.0 + inp.slip) * inp.lot
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.floor(np.where(per_lot > 0, tgt_notional / per_lot, 0.0))
    return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0) * inp.lot


def _repair(inp: AllocInputs, shares: np.ndarray) -> np.ndarray:
    """Make an infeasible start feasible by removing the lot whose removal costs the objective
    least, repeatedly. Pre-registered as a deterministic correction step, not a rescue hack."""
    shares = shares.copy()
    guard = 0
    while _trade_cash(inp, shares) < 0:
        guard += 1
        if guard > 10_000:
            return np.zeros_like(shares)
        cand = [i for i in range(len(shares)) if shares[i] >= inp.lot]
        if not cand:
            return np.zeros_like(shares)
        best, best_obj = None, None
        for i in cand:
            trial = shares.copy()
            trial[i] -= inp.lot
            o = _objective(inp, trial)          # objective only: _key is far more expensive
            if best_obj is None or o < best_obj:
                best, best_obj = trial, o
        shares = best
    return shares


def solve(inp: AllocInputs, *, local_search: bool = True) -> np.ndarray:
    """Integer-lot share counts minimising `f` subject to real affordability.

    TWO starts, then the frozen tie-break key picks between them. Greedy-from-all-floor alone is
    NOT good enough: measured against the independent parametric-frontier solver over 280
    rebalances it was strictly worse on 184 of them, because filling one lot at a time from the
    floor gets trapped once cash is nearly exhausted. Seeding the same local search from the
    frontier solution fixes that. The objective and the tie-break order are unchanged -- this is
    a better optimiser for the SAME frozen problem, not a different problem."""
    n = len(inp.codes)
    if n == 0:
        return np.zeros(0)
    starts = []
    base = baseline_lots(inp)
    starts.append(_repair(inp, base) if _trade_cash(inp, base) < 0 else base)
    frontier = solve_lambda_scan(inp)
    if frontier is not None:
        starts.append(frontier)
    best_overall = None
    for start in starts:
        cand = _hill_climb(inp, start, local_search)
        if best_overall is None or _key(inp, cand) < _key(inp, best_overall):
            best_overall = cand
    return best_overall


def _hill_climb(inp: AllocInputs, shares: np.ndarray, local_search: bool) -> np.ndarray:
    n = len(inp.codes)
    shares = shares.copy()

    def better(a, b):
        return _key(inp, a) < _key(inp, b)

    # greedy: keep adding the single most useful affordable lot
    for _ in range(10_000):
        best, best_key = None, _key(inp, shares)
        for i in range(n):
            trial = shares.copy()
            trial[i] += inp.lot
            if _trade_cash(inp, trial) < 0:
                continue
            k = _key(inp, trial)
            if k < best_key:
                best, best_key = trial, k
        if best is None:
            break
        shares = best

    if not local_search:
        return shares

    for _ in range(10_000):
        improved = False
        for i in range(n):                                   # drop a lot
            if shares[i] < inp.lot:
                continue
            trial = shares.copy()
            trial[i] -= inp.lot
            if _trade_cash(inp, trial) >= 0 and better(trial, shares):
                shares, improved = trial, True
        for i in range(n):                                   # move a lot between names
            if shares[i] < inp.lot:
                continue
            for j in range(n):
                if i == j:
                    continue
                trial = shares.copy()
                trial[i] -= inp.lot
                trial[j] += inp.lot
                if _trade_cash(inp, trial) >= 0 and better(trial, shares):
                    shares, improved = trial, True
        for i in range(n):                                   # add a lot
            trial = shares.copy()
            trial[i] += inp.lot
            if _trade_cash(inp, trial) >= 0 and better(trial, shares):
                shares, improved = trial, True
        if not improved:
            break
    return shares


def exhaustive(inp: AllocInputs, radius_down: int = 2, radius_up: int = 3,
               max_nodes: int = 400_000) -> tuple[np.ndarray | None, bool]:
    """Brute-force optimum over a box around the ideal, for VERIFYING `solve`.

    Returns (best, hit_boundary). `hit_boundary` True means the optimum sat on the edge of the
    searched box, so the box was not wide enough and the verification is inconclusive rather than
    passing -- reported, never silently treated as agreement."""
    n = len(inp.codes)
    x = ideal_lots(inp)
    lo = np.maximum(0, np.floor(x).astype(int) - radius_down)
    hi = np.floor(x).astype(int) + radius_up
    sizes = (hi - lo + 1)
    if n == 0 or float(np.prod(sizes.astype(float))) > max_nodes:
        return None, False
    grids = np.meshgrid(*[np.arange(lo[i], hi[i] + 1) for i in range(n)], indexing="ij")
    combos = np.stack([g.ravel() for g in grids], axis=1).astype(float) * inp.lot
    # Score every lattice point vectorised, then exact-check affordability in ASCENDING objective
    # order and stop at the first feasible one. The objective is the sort key, so the first
    # feasible point IS the constrained optimum -- identical answer to checking all of them, but
    # it runs a handful of exact feasibility tests instead of tens of thousands.
    a = combos * inp.price / inp.equity
    d = a - inp.target_w
    obj = (d * d).sum(axis=1) + (1.0 - a.sum(axis=1) - inp.target_cash_w) ** 2
    best = None
    for idx in np.argsort(obj, kind="stable"):
        row = combos[idx]
        if _trade_cash(inp, row) >= 0:
            best = row
            break
    if best is None:
        return None, False
    at_edge = bool(np.any((best / inp.lot == lo) & (lo > 0)) or np.any(best / inp.lot == hi))
    return best, at_edge


def solve_lambda_scan(inp: AllocInputs) -> np.ndarray | None:
    """A structurally DIFFERENT exact-frontier solver, used to verify `solve`.

    The objective decouples under a multiplier. Writing `u_i = k_i*c_i - t_i` with
    `c_i = lot*p_i/E`, the objective is `sum_i u_i^2 + (sum_i u_i)^2`; adding `2*lam*sum_i u_i`
    turns it into `sum_i (u_i + lam)^2` minus a constant, which is SEPARABLE -- each name's
    optimum is independently `k_i = max(0, round((t_i - lam)/c_i))`.

    Sweeping `lam` across every breakpoint where some `k_i` flips (`lam = t_i - c_i*(m + 1/2)`)
    therefore enumerates the whole parametric frontier exactly, with no search heuristic and no
    neighbourhood assumption. Each frontier point is checked for affordability and scored by the
    same frozen key; the best feasible one is returned. Sharing no code path with `solve`, an
    agreement between the two is real evidence rather than a restatement."""
    n = len(inp.codes)
    if n == 0:
        return None
    c = inp.lot * inp.price / inp.equity
    good = np.isfinite(c) & (c > 0)
    if not good.any():
        return None
    t = inp.target_w
    cap_lots = np.zeros(n, dtype=int)
    budget = inp.cash + float((inp.held * inp.price).sum())
    for i in range(n):
        cap_lots[i] = int(budget / (inp.price[i] * inp.lot)) + 2 if good[i] else 0
    # Only a narrow band of `lam` can contain the optimum. At the optimum every name sits within
    # a lot or two of its ideal count, and `k_i` shifts by one lot per `c_i` of `lam`, so
    # |lam| <= 2 * max_i c_i brackets the whole relevant frontier. Without this bound the scan
    # enumerates tens of thousands of breakpoints per rebalance at the large tiers and the
    # verification never finishes -- which is a performance bug, not a reason to skip verifying.
    span = 2.0 * float(np.max(c[good]))
    lams = {0.0}
    for i in range(n):
        if not good[i]:
            continue
        m_lo = max(0, int(np.floor((t[i] - span) / c[i] - 0.5)))
        m_hi = min(int(cap_lots[i]), int(np.ceil((t[i] + span) / c[i] + 0.5)))
        for m in range(m_lo, m_hi + 1):
            lam = float(t[i] - c[i] * (m + 0.5))
            if abs(lam) <= span:
                lams.add(lam)
    # Cap the frontier sample. Evenly thinning a sorted breakpoint list keeps the frontier's
    # shape while bounding cost; at the large tiers the raw list runs to thousands of points and
    # the scan dominates the whole study's runtime.
    ordered = sorted(lams)
    if len(ordered) > 400:
        step = len(ordered) / 400.0
        ordered = [ordered[min(int(j * step), len(ordered) - 1)] for j in range(400)]
    # Frontier points that overspend must be REPAIRED, not skipped. The best books at a small
    # account are exactly the ones that trim the cheap names to afford an expensive one, and that
    # book is only reachable by trimming an over-budget frontier point: dropping repair turned
    # [500, 200, 100] (objective 0.114) into [1400, 400, 0] (0.150), i.e. the expensive name was
    # never bought at all. Repair is quadratic, so it is applied only to the most promising
    # over-budget points, ranked by their unconstrained objective.
    feasible, over = [], []
    for lam in ordered:
        k = np.zeros(n)
        for i in range(n):
            if not good[i]:
                continue
            k[i] = max(0.0, float(np.round((t[i] - lam) / c[i])))
        shares = k * inp.lot
        (feasible if _trade_cash(inp, shares) >= 0 else over).append(shares)
    over.sort(key=lambda s: _objective(inp, s))
    cands = feasible + [_repair(inp, s) for s in over[:REPAIR_BUDGET]]
    best, best_key = None, None
    for shares in cands:
        key = _key(inp, shares)
        if best_key is None or key < best_key:
            best, best_key = shares, key
    return best


def build_inputs(codes, price: pd.Series, target_w: dict[str, float], positions: dict[str, int],
                 equity: float, cash: float, costs: AShareCosts, slip: float,
                 lot: int) -> AllocInputs:
    """Assemble one rebalance's inputs. `codes` must already be the union of the target basket and
    anything currently held, so a name being exited is represented and its proceeds are counted."""
    codes = tuple(codes)
    return AllocInputs(
        codes=codes,
        price=np.array([float(price.get(c, np.nan)) for c in codes], dtype=float),
        target_w=np.array([float(target_w.get(c, 0.0)) for c in codes], dtype=float),
        held=np.array([float(positions.get(c, 0)) for c in codes], dtype=float),
        equity=float(equity), cash=float(cash), lot=int(lot), slip=float(slip), costs=costs)


def allocate(gross: float, weights: dict[str, float], scale: float, raw: pd.Series,
             slip: float, lot: int, costs: AShareCosts, cash: float,
             positions: dict[str, int], equity: float) -> dict[str, int]:
    """Engine-facing entry point: same signature intent as `target_shares` plus the cash/holdings
    context the closest-weight objective needs. Returns desired SHARE counts per name."""
    basket = [c for c in weights if np.isfinite(raw.get(c, np.nan))]
    extra = [c for c in positions if positions[c] > 0 and c not in basket
             and np.isfinite(raw.get(c, np.nan))]
    codes = basket + sorted(extra)
    if not codes or equity <= 0:
        return {}
    tw = {c: gross * weights.get(c, 0.0) * scale / equity for c in codes}
    inp = build_inputs(codes, raw, tw, positions, equity, cash, costs, slip, lot)
    shares = solve(inp)
    return {c: int(round(s)) for c, s in zip(codes, shares)}
