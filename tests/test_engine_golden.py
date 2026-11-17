"""Golden tests on the PREP_GUIDE section 3 example. Expected numbers are HAND-CALCULATED there."""
import csv
from pathlib import Path

import pytest

from allocator.engine.constraints import feasibility
from allocator.engine.explain import explain_zone, what_if
from allocator.engine.loader import build_problem
from allocator.engine.metrics import compare, compute_metrics
from allocator.engine.pack import load_pack, validate_pack
from allocator.engine.scoring import score_zones
from allocator.engine.strategies import run_strategy


def _read(pack, entity):
    path = Path(pack.source_dir) / "seed" / f"{pack.entities[entity].table}.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def problem():
    pack = load_pack("toy")
    assert validate_pack(pack) == []
    return build_problem(pack, {e: _read(pack, e) for e in pack.entities})


def test_scores_and_ranks_hand_calculated(problem):
    s = {z.zone_id: z for z in score_zones(problem)}
    assert [z.zone_id for z in sorted(s.values(), key=lambda z: z.rank)] == ["Z2", "Z3", "Z1", "Z4"]
    assert s["Z2"].score == pytest.approx(0.85, abs=1e-6)
    assert s["Z3"].score == pytest.approx(0.70, abs=1e-6)
    assert s["Z1"].score == pytest.approx(0.40, abs=1e-6)
    assert s["Z4"].score == pytest.approx(0.35, abs=1e-6)
    assert s["Z2"].sentence == ("Eastside ranks #1 with a score of 0.85: vulnerability adds 0.45, "
                                "deadline urgency adds 0.25, size of need adds 0.15.")


def test_nearest_first_hand_calculated(problem):
    plan = run_strategy(problem, "nearest_first")
    assert plan.by_zone() == {"Z1": 40, "Z3": 30}
    m = compute_metrics(problem, plan)
    assert m == {"delivered": 70.0, "spoiled": 10.0, "unmet": 50.0, "vwu": 37.0,
                 "worst_off_coverage": 0.0, "most_vulnerable_coverage": 0.0}


def test_priority_greedy_hand_calculated(problem):
    plan = run_strategy(problem, "priority_greedy")
    assert plan.by_zone() == {"Z1": 20, "Z2": 30, "Z3": 30}
    assert plan.by_lot() == {"L1": 50, "L2": 30}
    m = compute_metrics(problem, plan)
    assert m == {"delivered": 80.0, "spoiled": 0.0, "unmet": 40.0, "vwu": 14.0,
                 "worst_off_coverage": 0.0, "most_vulnerable_coverage": 1.0}


@pytest.mark.parametrize("alpha,expected,vwu,worst", [
    (0.25, {"Z1": 15, "Z2": 30, "Z3": 30, "Z4": 5}, 12.5, 0.25),
    (0.0, {"Z1": 20, "Z2": 30, "Z3": 30}, 14.0, 0.0),
])
def test_fair_lp_acceptance(problem, alpha, expected, vwu, worst):
    plan = run_strategy(problem, "fair_lp", {"alpha": alpha})
    assert plan.by_zone() == expected
    m = compute_metrics(problem, plan)
    assert m["vwu"] == vwu and m["worst_off_coverage"] == worst
    assert not plan.flags


@pytest.mark.parametrize("name", ["nearest_first", "fcfs", "priority_greedy", "fair_lp"])
def test_hard_rules_never_broken(problem, name):
    plan = run_strategy(problem, name, {"alpha": 0.25})
    allowed, _ = feasibility(problem)
    for ln in plan.lines:
        assert allowed[(ln.lot_id, ln.zone_id)], ln
    for lot_id, used in plan.by_lot().items():
        assert used <= problem.lot(lot_id).quantity
    for zone_id, got in plan.by_zone().items():
        assert got <= problem.zone(zone_id).room


def test_infeasible_floor_is_flagged(problem):
    plan = run_strategy(problem, "fair_lp", {"alpha": 0.9})
    assert plan.flags and plan.flags[0].startswith("FLOOR_INFEASIBLE")
    assert sum(plan.by_zone().values()) == 80


def test_compare_headline(problem):
    plans = {n: run_strategy(problem, n) for n in ["priority_greedy", "nearest_first", "fcfs"]}
    out = compare(problem, plans, "priority_greedy")
    nf = next(h for h in out["headlines"] if h["baseline"] == "nearest_first")
    assert (nf["baseline_value"], nf["method_value"], nf["improvement_pct"]) == (37.0, 14.0, 62.2)
    assert "from 37.0 (nearest_first) to 14.0" in nf["sentence"]


def test_explain_why_not(problem):
    plan = run_strategy(problem, "nearest_first")
    e = explain_zone(problem, plan, "Z2")
    codes = {(r["lot_id"], r["code"]) for r in e["rejections"]}
    assert codes == {("L2", "COLD_CHAIN"), ("L1", "SUPPLY_EXHAUSTED")}
    assert e["allocated"] == 0 and e["shortfall"] == 30
    assert e["used_by"]["L1"] == [{"zone_id": "Z1", "name": "Downtown", "units": 40, "rank": 3},
                                  {"zone_id": "Z3", "name": "Riverside", "units": 10, "rank": 2}]


def test_what_if_is_in_memory_only(problem):
    out = what_if(problem, "nearest_first", {}, {"kind": "lot_delta", "lot_id": "L1", "delta": 30}, "Z2")
    # Hand check: L1 = 80. Nearest-first: Z1 40, Z3 30 (all from L1), Z2 gets the last 10.
    assert out["zone_changes"] == [{"zone_id": "Z2", "name": "Eastside", "before": 0, "after": 10, "delta": 10}]
    assert problem.lot("L1").quantity == 50  # original untouched


def test_weights_what_if_renormalises(problem):
    out = what_if(problem, "priority_greedy", {}, {"kind": "weights", "weights": {"urgency": 0.6}})
    # Hand check: 0.5, 0.6, 0.2 sum to 1.3 -> 0.38, 0.46, 0.15.
    assert out["description"] == "with weights vulnerability 0.38, urgency 0.46, need_share 0.15"
