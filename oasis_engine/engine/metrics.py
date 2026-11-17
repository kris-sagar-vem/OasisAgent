"""Outcome metrics and the baseline comparison (requirement R4)."""
from __future__ import annotations

from typing import Any

from .model import Plan, Problem
from .registry import METRICS, metric


def _got(plan: Plan) -> dict[str, int]:
    return plan.by_zone()


@metric("delivered", "units delivered", better="higher")
def delivered(problem: Problem, plan: Plan) -> float:
    return float(sum(ln.units for ln in plan.lines))


@metric("spoiled", "perishable units left unassigned", better="lower")
def spoiled(problem: Problem, plan: Plan) -> float:
    used = plan.by_lot()
    return float(sum(lt.quantity - used.get(lt.lot_id, 0) for lt in problem.lots if lt.perishable))


@metric("unmet", "total unmet need", better="lower")
def unmet(problem: Problem, plan: Plan) -> float:
    got = _got(plan)
    return float(sum(z.open_need - got.get(z.zone_id, 0) for z in problem.zones))


@metric("vwu", "vulnerability-weighted unmet need", better="lower")
def vwu(problem: Problem, plan: Plan) -> float:
    got = _got(plan)
    return float(sum(z.vulnerability * (z.open_need - got.get(z.zone_id, 0)) for z in problem.zones))


@metric("worst_off_coverage", "worst-off zone coverage", better="higher")
def worst_off_coverage(problem: Problem, plan: Plan) -> float:
    got = _got(plan)
    covs = [got.get(z.zone_id, 0) / z.open_need for z in problem.zones if z.open_need > 0]
    return min(covs) if covs else 1.0


@metric("most_vulnerable_coverage", "coverage of the most vulnerable zone", better="higher")
def most_vulnerable_coverage(problem: Problem, plan: Plan) -> float:
    candidates = [z for z in problem.zones if z.open_need > 0]
    if not candidates:
        return 1.0
    z = sorted(candidates, key=lambda z: (-z.vulnerability, z.zone_id))[0]
    return _got(plan).get(z.zone_id, 0) / z.open_need


def compute_metrics(problem: Problem, plan: Plan, names: list[str] | None = None) -> dict[str, float]:
    names = names or problem.pack.metrics.show
    return {n: round(METRICS[n].fn(problem, plan), 4) for n in names}


def improvement_pct(name: str, method: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    if METRICS[name].better == "lower":
        return round((baseline - method) / abs(baseline) * 100, 1)
    return round((method - baseline) / abs(baseline) * 100, 1)


def headline_sentence(name: str, method_name: str, method: float, base_name: str, base: float) -> str:
    spec = METRICS[name]
    pct = improvement_pct(name, method, base)
    if method == base:
        return f"Under the same hard rules, {method_name} and {base_name} tie on {spec.label} at {method:.1f}."
    better = (method < base) if spec.better == "lower" else (method > base)
    direction = "lower" if method < base else "higher"
    verdict = "better" if better else "worse"
    return (f"Under the same hard rules, {method_name} changes {spec.label} from {base:.1f} ({base_name}) "
            f"to {method:.1f}, which is {abs(pct):.1f}% {direction} ({verdict}).")


def compare(problem: Problem, plans: dict[str, Plan], main: str) -> dict[str, Any]:
    """Metrics table for every plan plus headline sentences versus each baseline."""
    pack = problem.pack
    rows = {name: compute_metrics(problem, p) for name, p in plans.items()}
    head = pack.metrics.headline
    headlines = []
    for name, m in rows.items():
        if name == main:
            continue
        headlines.append({
            "baseline": name,
            "metric": head,
            "baseline_value": m[head],
            "method_value": rows[main][head],
            "improvement_pct": improvement_pct(head, rows[main][head], m[head]),
            "sentence": headline_sentence(head, main, rows[main][head], name, m[head]),
        })
    return {
        "main_strategy": main,
        "headline_metric": head,
        "metrics": rows,
        "metric_labels": {n: METRICS[n].label for n in pack.metrics.show},
        "headlines": headlines,
    }
