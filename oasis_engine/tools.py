"""Agent tools: thin wrappers over the Engine facade. No tool computes anything itself.

Rules enforced here (see PREP_GUIDE section 4 and 9):
- Only `commit_plan` changes operational state, and only through the gateway.
- No tool can record an approval in out_of_band mode; approval is CLI-only.
- External text never becomes an instruction; tools return structured data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from google.adk.tools import ToolContext

from .engine.service import EngineError, get_engine


def _safe(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except EngineError as e:
        return {"error": str(e)}
    except KeyError as e:
        return {"error": str(e).strip("'\"")}


def _actor(tool_context: Optional[ToolContext]) -> str:
    uid = getattr(tool_context, "user_id", None) if tool_context else None
    return f"agent:{uid or 'unknown'}"


# ---- read tools --------------------------------------------------------------
def current_time() -> dict[str, Any]:
    """Return the current date and time in UTC. Use this instead of guessing any date or time."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds"), "date": now.date().isoformat(),
            "weekday": now.strftime("%A")}


def rank_zones(top_n: int = 10) -> dict[str, Any]:
    """Rank zones by priority score with a plain-English reason for each.

    Args:
        top_n: How many top-ranked zones to return.
    """
    return _safe(get_engine().rank, top_n)


def compare_methods() -> dict[str, Any]:
    """Compare the main allocation method against the baselines (nearest-first, first-come
    first-served, ...) under the same hard rules. Returns metrics and headline sentences."""
    return _safe(get_engine().compare)


def explain_zone(zone_id: str, plan_id: str = "") -> dict[str, Any]:
    """Explain why a zone gets what it gets: its rank, every allocation line, every rejected
    lot with a reason code, who used the supply instead, and a what-if when a rule blocked it.

    Args:
        zone_id: The zone ID, for example Z08.
        plan_id: Optional plan ID to explain a saved proposal instead of a fresh run.
    """
    return _safe(get_engine().explain_zone, zone_id, plan_id or None)


def what_if(kind: str, zone_id: str = "", handling: str = "", lot_id: str = "",
            delta: int = 0) -> dict[str, Any]:
    """Run an in-memory scenario. Nothing is saved or dispatched.

    Args:
        kind: One of add_handling (zone_id + handling), lot_delta (lot_id + delta).
        zone_id: Zone to change or focus on.
        handling: Handling capability to add, for example chilled.
        lot_id: Lot whose quantity changes.
        delta: Units to add (negative to remove).
    """
    if kind == "add_handling":
        change = {"kind": kind, "zone_id": zone_id, "handling": handling}
    elif kind == "lot_delta":
        change = {"kind": kind, "lot_id": lot_id, "delta": int(delta)}
    else:
        return {"error": "kind must be add_handling or lot_delta"}
    return _safe(get_engine().what_if, change, zone_id or None)


def policy_dial(alphas: Optional[list[float]] = None, weights: str = "") -> dict[str, Any]:
    """Read-only policy trade-offs. Either sweep the fairness floor alpha to show what
    fairness costs, or re-rank with different scoring weights. Nothing is saved.

    Args:
        alphas: Fairness floors to try, each between 0 and 1, for example [0, 0.25, 0.5].
        weights: Optional new weights as "term=value,term=value", for example
            "vulnerability=0.6,urgency=0.2". They are renormalised to sum to 1.
    """
    if weights:
        try:
            w = {k.strip(): float(v) for k, v in (p.split("=") for p in weights.split(",") if p.strip())}
        except ValueError:
            return {"error": "weights must look like term=0.5,term=0.3"}
        return _safe(get_engine().policy_dial, weights=w)
    return _safe(get_engine().policy_dial, alphas=alphas)


def get_state() -> dict[str, Any]:
    """Current zones (need, served, open need), supply lots and recent plans."""
    return _safe(get_engine().state)


def get_plan(plan_id: str) -> dict[str, Any]:
    """Fetch a saved plan by ID, with its status, allocations and metrics.

    Args:
        plan_id: The plan ID, for example P-7F3K2A.
    """
    return _safe(get_engine().get_plan, plan_id)


def verify_audit() -> dict[str, Any]:
    """Verify the tamper-evident audit hash chain."""
    return _safe(get_engine().verify_audit)


# ---- write path ----------------------------------------------------------------
def propose_plan(tool_context: ToolContext, strategy: str = "") -> dict[str, Any]:
    """Create an immutable plan proposal. This moves NO supply; a human must approve it.

    Args:
        strategy: Optional strategy name; leave empty for the pack's main method.
    """
    return _safe(get_engine().propose, strategy or None, None, _actor(tool_context))


def reject_plan(plan_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
    """Mark a proposed plan as rejected. This never moves supply.

    Args:
        plan_id: The plan to reject.
        reason: Short reason in plain English.
    """
    return _safe(get_engine().reject, plan_id, reason, _actor(tool_context))


def commit_plan(plan_id: str, approval_code: str, tool_context: ToolContext) -> dict[str, Any]:
    """Dispatch an approved plan. The gateway refuses unless a human approved this exact plan,
    the approval code matches, and the data has not changed since the plan was made.

    Args:
        plan_id: The plan to commit.
        approval_code: The 6-character code shown with the plan.
    """
    engine = get_engine()
    approver = None
    if engine.approval_mode == "confirm":
        conf = getattr(tool_context, "tool_confirmation", None)
        if not (conf and conf.confirmed):
            return {"status": "REJECTED", "reason": "Human confirmation is required and was not given."}
        approver = f"human:{getattr(tool_context, 'user_id', None) or 'unknown'}"
    return _safe(engine.commit, plan_id, approval_code, _actor(tool_context), approver)


def _get_agent_tool(name: str):
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import agent.tools as at
    return getattr(at, name)


def predict_heat_offset(latitude: float, longitude: float) -> dict[str, Any]:
    """Call BigQuery ML ML.EXPLAIN_PREDICT against Dallas heat offset model.

    Returns the predicted microclimate temperature offset above city baseline and feature attributions.

    Args:
        latitude: Latitude coordinate in Dallas (e.g. 32.78).
        longitude: Longitude coordinate in Dallas (e.g. -96.80).
    """
    _eho = _get_agent_tool("explain_heat_offset")
    return _safe(_eho, latitude, longitude)


def get_heat_frequency() -> dict[str, Any]:
    """Retrieve 7 years of station temperature history for Dallas (days >= 95F and >= 100F)."""
    _ghf = _get_agent_tool("get_heat_frequency")
    return _safe(_ghf)


def run_equity_audit(selected_tract_ids: list[str] | None = None) -> dict[str, Any]:
    """Audit proposed cooling center sites against Dallas citywide demographic baselines."""
    _rea = _get_agent_tool("run_equity_audit")
    return _safe(_rea, selected_tract_ids)



READ_TOOLS = [current_time, rank_zones, compare_methods, explain_zone, what_if, policy_dial,
              get_state, get_plan, verify_audit, predict_heat_offset, get_heat_frequency, run_equity_audit]
WRITE_TOOLS = [propose_plan, reject_plan, commit_plan]
REVIEWER_TOOLS = [compare_methods, policy_dial, run_equity_audit]

