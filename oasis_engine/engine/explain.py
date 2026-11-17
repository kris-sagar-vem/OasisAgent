"""Explanations generated from data, never written by the model (R2, R3)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .model import Plan, Problem
from .scoring import score_zones
from .strategies import run_strategy


def _apply_change(problem: Problem, change: dict[str, Any]) -> tuple[Problem, str]:
    """Return a modified in-memory copy of the problem. Nothing is written anywhere."""
    kind = change.get("kind")
    if kind == "add_handling":
        z = problem.zone(change["zone_id"])
        nz = replace(z, handling=z.handling | {change["handling"]})
        zones = tuple(nz if x.zone_id == z.zone_id else x for x in problem.zones)
        return replace(problem, zones=zones), f"if {z.name} could handle {change['handling']} supply"
    if kind == "lot_delta":
        lt = problem.lot(change["lot_id"])
        nl = replace(lt, quantity=max(0, lt.quantity + int(change["delta"])))
        lots = tuple(nl if x.lot_id == lt.lot_id else x for x in problem.lots)
        return replace(problem, lots=lots), f"if {lt.lot_id} had {nl.quantity} units instead of {lt.quantity}"
    if kind == "weights":
        pack = problem.pack.with_weights({k: float(v) for k, v in change["weights"].items()})
        shown = ", ".join(f"{t.term} {t.weight:.2f}" for t in pack.scoring)
        return replace(problem, pack=pack), f"with weights {shown}"
    raise ValueError(f"unknown what-if kind: {kind!r}; use add_handling, lot_delta or weights")


def what_if(problem: Problem, strategy: str, params: dict, change: dict[str, Any],
            focus_zone: str | None = None) -> dict[str, Any]:
    from .metrics import compute_metrics
    before = run_strategy(problem, strategy, params)
    changed, desc = _apply_change(problem, change)
    after = run_strategy(changed, strategy, params)
    b, a = before.by_zone(), after.by_zone()
    diffs = []
    for z in problem.zones:
        if b.get(z.zone_id, 0) != a.get(z.zone_id, 0):
            diffs.append({"zone_id": z.zone_id, "name": z.name, "before": b.get(z.zone_id, 0),
                          "after": a.get(z.zone_id, 0), "delta": a.get(z.zone_id, 0) - b.get(z.zone_id, 0)})
    out: dict[str, Any] = {
        "change": change,
        "description": desc,
        "zone_changes": diffs,
        "metrics_before": compute_metrics(problem, before),
        "metrics_after": compute_metrics(changed, after),
    }
    if focus_zone:
        z = problem.zone(focus_zone)
        nb, na = b.get(focus_zone, 0), a.get(focus_zone, 0)
        out["sentence"] = (f"{desc[0].upper() + desc[1:]}, {z.name} would receive {na} units "
                           f"instead of {nb} under the same policy.")
    else:
        out["sentence"] = f"{desc[0].upper() + desc[1:]}, {len(diffs)} zones would change allocation."
    return out


def explain_zone(problem: Problem, plan: Plan, zone_id: str, with_what_if: bool = True) -> dict[str, Any]:
    z = problem.zone(zone_id)
    scored = {s.zone_id: s for s in score_zones(problem)}
    got = plan.by_zone().get(zone_id, 0)
    mine = [ln for ln in plan.lines if ln.zone_id == zone_id]
    rej = [r for r in plan.rejections if r.zone_id == zone_id]
    exhausted = sorted({r.lot_id for r in rej if r.code == "SUPPLY_EXHAUSTED"})
    used_by: dict[str, list[dict[str, Any]]] = {}
    for ln in plan.lines:
        if ln.lot_id in exhausted and ln.zone_id != zone_id:
            used_by.setdefault(ln.lot_id, []).append(
                {"zone_id": ln.zone_id, "name": problem.zone(ln.zone_id).name, "units": ln.units,
                 "rank": scored[ln.zone_id].rank})

    reasons = []
    for r in rej:
        if r.code == "SUPPLY_EXHAUSTED":
            who = ", ".join(f"{u['name']} ({u['units']})" for u in used_by.get(r.lot_id, []))
            reasons.append(f"{r.lot_id}: SUPPLY_EXHAUSTED, used by {who or 'other zones'}")
        else:
            reasons.append(f"{r.lot_id if r.lot_id != '*' else 'all lots'}: {r.code}, {r.detail}")

    short = z.open_need - got
    if short <= 0:
        sentence = f"{z.name} is fully served: {got} of {z.open_need} units."
    else:
        pct = round(100 * got / z.open_need) if z.open_need else 100
        sentence = (f"{z.name} gets {got} of {z.open_need} units ({pct}%), {short} short. "
                    f"Reasons: {'; '.join(reasons) if reasons else 'none recorded'}.")

    what_ifs = []
    if with_what_if and short > 0:
        tried = set()
        for r in rej:
            if r.code in ("COLD_CHAIN", "HANDLING"):
                h = problem.lot(r.lot_id).handling
                if h not in tried:
                    tried.add(h)
                    what_ifs.append(what_if(problem, plan.strategy, plan.params,
                                            {"kind": "add_handling", "zone_id": zone_id, "handling": h},
                                            focus_zone=zone_id))
    return {
        "zone_id": zone_id,
        "name": z.name,
        "rank": scored[zone_id].rank,
        "score": scored[zone_id].score,
        "rank_sentence": scored[zone_id].sentence,
        "need": z.open_need,
        "allocated": got,
        "shortfall": max(short, 0),
        "lines": [{"lot_id": ln.lot_id, "units": ln.units, "sentence": ln.sentence} for ln in mine],
        "rejections": [{"lot_id": r.lot_id, "code": r.code, "detail": r.detail} for r in rej],
        "used_by": used_by,
        "sentence": sentence,
        "what_if": what_ifs,
    }
