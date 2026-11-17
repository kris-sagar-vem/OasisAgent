"""Allocation strategies. Baselines obey the same hard rules; only the ordering policy differs."""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

from .constraints import feasibility
from .model import Line, Lot, Plan, Problem, Rejection, Zone
from .registry import strategy
from .scoring import score_zones

INF = float("inf")
LotSort = Callable[[Zone, list[Lot]], list[Lot]]


def _expiry_key(lot: Lot) -> float:
    return lot.expiry_h if lot.expiry_h is not None else INF


def _capacity_rejections(problem: Problem) -> list[Rejection]:
    return [Rejection("*", z.zone_id, "CAPACITY", f"{z.name} can receive at most {z.room} more units")
            for z in problem.zones if z.open_need > 0 and z.room < z.open_need]


def _fill(problem: Problem, name: str, params: dict, zone_order: Sequence[Zone], lot_sort: LotSort,
          policy: str, ranks: dict[str, int] | None = None) -> Plan:
    allowed, rejections = feasibility(problem)
    rejections += _capacity_rejections(problem)
    left = {lt.lot_id: lt.quantity for lt in problem.lots}
    lines: list[Line] = []
    for z in zone_order:
        room = z.room
        if room <= 0:
            continue
        for lot in lot_sort(z, [lt for lt in problem.lots if allowed[(lt.lot_id, z.zone_id)]]):
            if room == 0:
                break
            if left[lot.lot_id] == 0:
                rejections.append(Rejection(lot.lot_id, z.zone_id, "SUPPLY_EXHAUSTED",
                                            f"{lot.lot_id} was used up by zones served earlier"))
                continue
            give = min(room, left[lot.lot_id])
            left[lot.lot_id] -= give
            room -= give
            rank = f" (rank #{ranks[z.zone_id]})" if ranks else ""
            lines.append(Line(lot.lot_id, z.zone_id, give,
                              f"{z.name}{rank} gets {give} units of {lot.lot_id}: every rule allows it; "
                              f"order used: {policy}."))
    return Plan(name, dict(params), lines, rejections)


def _by_id(z: Zone, lots: list[Lot]) -> list[Lot]:
    return sorted(lots, key=lambda lt: lt.lot_id)


@strategy("nearest_first")
def nearest_first(problem: Problem, params: dict) -> Plan:
    def dist(z: Zone) -> float:
        return min((problem.travel_h(lt, z) for lt in problem.lots), default=INF)
    order = sorted(problem.zones, key=lambda z: (dist(z), z.zone_id))
    return _fill(problem, "nearest_first", params, order, _by_id, "nearest zone first, lots in ID order")


@strategy("fcfs")
def fcfs(problem: Problem, params: dict) -> Plan:
    order = sorted(problem.zones, key=lambda z: (z.request_seq, z.zone_id))
    return _fill(problem, "fcfs", params, order, _by_id, "first request first, lots in ID order")


@strategy("priority_greedy")
def priority_greedy(problem: Problem, params: dict) -> Plan:
    scored = score_zones(problem)
    ranks = {s.zone_id: s.rank for s in scored}
    order = [problem.zone(s.zone_id) for s in scored]

    def soonest_expiry(z: Zone, lots: list[Lot]) -> list[Lot]:
        return sorted(lots, key=lambda lt: (_expiry_key(lt), problem.travel_h(lt, z), lt.lot_id))
    return _fill(problem, "priority_greedy", params, order, soonest_expiry,
                 "highest priority zone first, soonest-expiring lot first", ranks)


def _exhaustion(problem: Problem, plan_lines: Iterable[Line], allowed: dict) -> list[Rejection]:
    lines = list(plan_lines)
    used: dict[str, int] = {}
    got: dict[str, int] = {}
    for ln in lines:
        used[ln.lot_id] = used.get(ln.lot_id, 0) + ln.units
        got[ln.zone_id] = got.get(ln.zone_id, 0) + ln.units
    out = []
    for z in problem.zones:
        if got.get(z.zone_id, 0) >= z.room:
            continue
        for lt in problem.lots:
            if allowed[(lt.lot_id, z.zone_id)] and used.get(lt.lot_id, 0) >= lt.quantity:
                out.append(Rejection(lt.lot_id, z.zone_id, "SUPPLY_EXHAUSTED",
                                     f"{lt.lot_id} was fully allocated to other zones"))
    return out


@strategy("fair_lp")
def fair_lp(problem: Problem, params: dict) -> Plan:
    """Integer programme: maximise priority-weighted delivery with a fairness floor alpha.

    max sum P_z x_lz + lambda * perishable units used, subject to lot supply, zone room,
    and sum_l x_lz >= ceil(alpha * open_need_z) (capped by room). If the floor is
    infeasible, the largest feasible alpha (step 0.01) is used and the plan is flagged.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    alpha = float(params.get("alpha", 0.25))
    lam = float(params.get("lambda_perishable", 1.0))
    scores = {s.zone_id: s.score for s in score_zones(problem)}
    allowed, rejections = feasibility(problem)
    rejections += _capacity_rejections(problem)
    pairs = [(lt, z) for z in problem.zones for lt in problem.lots if allowed[(lt.lot_id, z.zone_id)]]
    if not pairs:
        return Plan("fair_lp", dict(params), [], rejections, ["NO_FEASIBLE_PAIRS"])

    n = len(pairs)
    c = np.zeros(n)
    for k, (lt, z) in enumerate(pairs):
        tie = 1e-4 / (1.0 + _expiry_key(lt)) if lt.perishable else 0.0  # soonest expiry first
        c[k] = -(scores[z.zone_id] + (lam if lt.perishable else 0.0) + tie)
    lot_idx = {lt.lot_id: i for i, lt in enumerate(problem.lots)}
    zone_idx = {z.zone_id: i for i, z in enumerate(problem.zones)}
    a_lot = np.zeros((len(problem.lots), n))
    a_zone = np.zeros((len(problem.zones), n))
    for k, (lt, z) in enumerate(pairs):
        a_lot[lot_idx[lt.lot_id], k] = 1
        a_zone[zone_idx[z.zone_id], k] = 1
    q = np.array([lt.quantity for lt in problem.lots], dtype=float)
    room = np.array([z.room for z in problem.zones], dtype=float)

    def solve(a: float):
        floors = np.array([min(math.ceil(a * z.open_need - 1e-9), z.room) for z in problem.zones], dtype=float)
        cons = [LinearConstraint(a_lot, -np.inf, q), LinearConstraint(a_zone, floors, room)]
        res = milp(c, integrality=np.ones(n), bounds=Bounds(0, np.inf), constraints=cons)
        return res if res.success else None

    flags: list[str] = []
    res, used_alpha = solve(alpha), alpha
    if res is None:
        used_alpha = math.floor(alpha * 100) / 100
        while res is None and used_alpha > 0:
            used_alpha = round(used_alpha - 0.01, 2)
            res = solve(max(used_alpha, 0.0))
        flags.append(f"FLOOR_INFEASIBLE: alpha {alpha:g} reduced to {max(used_alpha, 0.0):g}")
    if res is None:
        return Plan("fair_lp", dict(params), [], rejections, flags + ["SOLVER_FAILED"])

    ranks = {s.zone_id: s.rank for s in score_zones(problem)}
    lines = []
    for k, (lt, z) in enumerate(pairs):
        units = int(round(res.x[k]))
        if units > 0:
            lines.append(Line(lt.lot_id, z.zone_id, units,
                              f"{z.name} (rank #{ranks[z.zone_id]}) gets {units} units of {lt.lot_id}: every rule "
                              f"allows it; chosen by the fairness optimiser with a floor of "
                              f"{round(max(used_alpha, 0.0) * 100):g}% of need."))
    lines.sort(key=lambda ln: (ranks[ln.zone_id], ln.lot_id))
    rejections += _exhaustion(problem, lines, allowed)
    out_params = dict(params, alpha_used=max(used_alpha, 0.0))
    return Plan("fair_lp", out_params, lines, rejections, flags)


def run_strategy(problem: Problem, name: str, params: dict | None = None) -> Plan:
    from .registry import STRATEGIES
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy: {name}")
    return STRATEGIES[name](problem, dict(params or {}))
