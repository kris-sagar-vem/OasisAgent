"""Triage: transparent weighted score P = sum_k w_k * f_k(zone), each f_k in [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass

from .model import Problem, Zone
from .registry import SCORING_TERMS, scoring_term


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@scoring_term("vulnerability", label="vulnerability")
def vulnerability(zone: Zone, problem: Problem, params: dict) -> float:
    return _clip(zone.vulnerability)


@scoring_term("urgency", label="deadline urgency")
def urgency(zone: Zone, problem: Problem, params: dict) -> float:
    """1 - min(deadline, horizon) / horizon: a shorter deadline gives a higher value."""
    if zone.deadline_h is None:
        return 0.0
    horizon = float(params.get("horizon_h", 72))
    return _clip(1.0 - min(max(zone.deadline_h, 0.0), horizon) / horizon)


@scoring_term("need_share", label="size of need")
def need_share(zone: Zone, problem: Problem, params: dict) -> float:
    biggest = max((z.open_need for z in problem.zones), default=0)
    return 0.0 if biggest == 0 else _clip(zone.open_need / biggest)


@scoring_term("attr_value", label="indicator")
def attr_value(zone: Zone, problem: Problem, params: dict) -> float:
    """A 0-1 column from the data (for example a BQML forecast or an index)."""
    v = zone.attrs.get(params["column"])
    return 0.0 if v in (None, "") else _clip(float(v))


@scoring_term("attr_share", label="indicator share")
def attr_share(zone: Zone, problem: Problem, params: dict) -> float:
    """A non-negative column divided by its maximum across zones."""
    col = params["column"]
    vals = [float(z.attrs.get(col) or 0) for z in problem.zones]
    biggest = max(vals, default=0.0)
    v = float(zone.attrs.get(col) or 0)
    return 0.0 if biggest <= 0 else _clip(v / biggest)


@scoring_term("condition", label="current conditions")
def condition(zone: Zone, problem: Problem, params: dict) -> float:
    """An external condition factor (for example hazard from weather), already in [0, 1]."""
    return _clip(problem.conditions.values.get(zone.zone_id, {}).get(params.get("factor", "hazard"), 0.0))


@dataclass(frozen=True)
class ScoredZone:
    zone_id: str
    name: str
    rank: int
    score: float
    values: dict[str, float]
    contributions: dict[str, float]
    sentence: str


def _label(term: str, params: dict) -> str:
    return str(params.get("label") or SCORING_TERMS[term].label)


def score_zones(problem: Problem) -> list[ScoredZone]:
    """Score and rank every zone. Ties: higher vulnerability first, then zone_id."""
    raw = []
    for z in problem.zones:
        values, contrib = {}, {}
        for t in problem.pack.scoring:
            v = SCORING_TERMS[t.term].fn(z, problem, t.params)
            values[t.term] = round(v, 6)
            contrib[t.term] = round(t.weight * v, 6)
        raw.append((z, round(sum(contrib.values()), 6), values, contrib))
    raw.sort(key=lambda x: (-x[1], -x[0].vulnerability, x[0].zone_id))
    params_by_term = {t.term: t.params for t in problem.pack.scoring}
    out = []
    for rank, (z, score, values, contrib) in enumerate(raw, start=1):
        parts = [f"{_label(k, params_by_term[k])} adds {c:.2f}"
                 for k, c in sorted(contrib.items(), key=lambda kv: (-kv[1], kv[0])) if c > 0]
        reason = ", ".join(parts) if parts else "no scoring term applies"
        sentence = f"{z.name} ranks #{rank} with a score of {score:.2f}: {reason}."
        out.append(ScoredZone(z.zone_id, z.name, rank, score, values, contrib, sentence))
    return out
