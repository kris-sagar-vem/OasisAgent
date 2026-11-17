"""Hard rules. Every (lot, zone) pair gets a Verdict from every configured constraint."""
from __future__ import annotations

from .model import COLD_HANDLING, OK, Lot, Problem, Rejection, Verdict, Zone
from .registry import CONSTRAINTS, constraint


@constraint("handling")
def handling(zone: Zone, lot: Lot, problem: Problem, params: dict) -> Verdict:
    if lot.handling in zone.handling:
        return OK
    code = "COLD_CHAIN" if lot.handling in COLD_HANDLING else "HANDLING"
    return Verdict(False, code, f"{zone.name} cannot handle {lot.handling} supply")


@constraint("expiry")
def expiry(zone: Zone, lot: Lot, problem: Problem, params: dict) -> Verdict:
    if lot.expiry_h is None:
        return OK
    t = problem.travel_h(lot, zone)
    margin = float(params.get("safety_margin_h", 0.0))
    if t + margin >= lot.expiry_h:
        return Verdict(False, "EXPIRY", f"{lot.lot_id} needs {t:g} h to arrive but expires in {lot.expiry_h:g} h")
    return OK


@constraint("deadline")
def deadline(zone: Zone, lot: Lot, problem: Problem, params: dict) -> Verdict:
    if zone.deadline_h is None:
        return OK
    t = problem.travel_h(lot, zone)
    if t > zone.deadline_h:
        return Verdict(False, "DEADLINE", f"{lot.lot_id} needs {t:g} h but {zone.name}'s deadline is {zone.deadline_h:g} h")
    return OK


@constraint("eligibility")
def eligibility(zone: Zone, lot: Lot, problem: Problem, params: dict) -> Verdict:
    if (lot.category, zone.zone_type) in problem.denied:
        return Verdict(False, "ELIGIBILITY", f"{lot.category} is not allowed for {zone.zone_type} zones")
    return OK


@constraint("distance")
def distance(zone: Zone, lot: Lot, problem: Problem, params: dict) -> Verdict:
    max_h = float(params["max_h"])
    t = problem.travel_h(lot, zone)
    if t > max_h:
        return Verdict(False, "DISTANCE", f"{lot.lot_id} needs {t:g} h, over the {max_h:g} h limit")
    return OK


def feasibility(problem: Problem) -> tuple[dict[tuple[str, str], bool], list[Rejection]]:
    """Return allowed[(lot_id, zone_id)] and one Rejection per failed rule."""
    allowed: dict[tuple[str, str], bool] = {}
    rejections: list[Rejection] = []
    for z in problem.zones:
        for lot in problem.lots:
            ok = True
            for c in problem.pack.constraints:
                v = CONSTRAINTS[c.name](z, lot, problem, c.params)
                if not v.ok:
                    ok = False
                    rejections.append(Rejection(lot.lot_id, z.zone_id, v.code, v.detail))
            allowed[(lot.lot_id, z.zone_id)] = ok
    return allowed, rejections
