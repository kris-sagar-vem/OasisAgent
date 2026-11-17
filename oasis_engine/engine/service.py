"""Engine facade: the single API used by agent tools, the CLI, the guardian workflow and tests.

Every result carries a `source` block (provenance): which store, pack, data versions and
query job IDs produced the numbers. The agent must show numbers only from these results.
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict
from datetime import datetime
from typing import Any

from .explain import explain_zone as _explain_zone
from .explain import what_if as _what_if
from .gateway import build_proposal, utcnow, verify_chain
from .metrics import compare as _compare
from .metrics import compute_metrics
from .model import Line, Plan, Problem, Rejection
from .pack import Pack, load_pack, validate_pack
from .scoring import score_zones
from .strategies import run_strategy

APPROVAL_MODES = ("confirm", "out_of_band")


class EngineError(ValueError):
    """A user-facing error with a plain-English message."""


class Engine:
    def __init__(self, pack: Pack, store: Any, approval_mode: str = "out_of_band") -> None:
        if approval_mode not in APPROVAL_MODES:
            raise EngineError(f"APPROVAL_MODE must be one of {APPROVAL_MODES}, got {approval_mode!r}")
        self.pack = pack
        self.store = store
        self.approval_mode = approval_mode

    # ---- helpers -------------------------------------------------------------
    def _problem(self) -> Problem:
        return self.store.load_problem()

    def _conditions_age_min(self, problem: Problem, now: datetime) -> float | None:
        if not problem.conditions.observed_at:
            return None
        return round((now - datetime.fromisoformat(problem.conditions.observed_at)).total_seconds() / 60, 1)

    def source(self, problem: Problem) -> dict[str, Any]:
        now = utcnow()
        age = self._conditions_age_min(problem, now)
        limit = self.pack.conditions.max_age_minutes
        return {
            "computed_by": "allocation engine (deterministic code, not the language model)",
            "store": self.store.kind,
            "pack_id": self.pack.meta.id,
            "pack_hash": self.pack.pack_hash(),
            "as_of": now.isoformat(timespec="seconds"),
            "data_versions": {"lots": sum(lt.version for lt in problem.lots),
                              "zones": sum(z.version for z in problem.zones),
                              "conditions": problem.conditions.version},
            "conditions_source": problem.conditions.source,
            "conditions_age_minutes": age,
            "conditions_stale": bool(limit is not None and age is not None and age > limit),
            "job_ids": self.store.last_job_ids(),
            "disclaimer": self.pack.meta.disclaimer,
        }

    def _strategy(self, strategy: str | None, params: dict | None) -> tuple[str, dict]:
        name = strategy or self.pack.strategy.name
        merged = dict(self.pack.strategy.params) if name == self.pack.strategy.name else {}
        merged.update(params or {})
        return name, merged

    def _plan_from_proposal(self, plan_id: str) -> Plan:
        p = self.store.get_proposal(plan_id)
        if p is None:
            raise EngineError(f"No plan with ID {plan_id}.")
        return Plan(p.strategy, p.params, list(p.lines), list(p.rejections), list(p.flags))

    # ---- read tools ----------------------------------------------------------------
    def rank(self, top_n: int | None = None) -> dict[str, Any]:
        problem = self._problem()
        scored = score_zones(problem)
        rows = [{"rank": s.rank, "zone_id": s.zone_id, "name": s.name, "score": s.score,
                 "open_need": problem.zone(s.zone_id).open_need, "contributions": s.contributions,
                 "sentence": s.sentence} for s in scored[: top_n or len(scored)]]
        return {"weights": self.pack.weights(), "zones": rows, "source": self.source(problem)}

    def compare(self, strategy: str | None = None, params: dict | None = None) -> dict[str, Any]:
        problem = self._problem()
        name, merged = self._strategy(strategy, params)
        plans = {name: run_strategy(problem, name, merged)}
        for b in self.pack.strategy.baselines:
            if b != name:
                plans[b] = run_strategy(problem, b, {})
        out = _compare(problem, plans, name)
        out["flags"] = plans[name].flags
        out["source"] = self.source(problem)
        return out

    def explain_zone(self, zone_id: str, plan_id: str | None = None) -> dict[str, Any]:
        problem = self._problem()
        try:
            problem.zone(zone_id)
        except KeyError:
            raise EngineError(f"No zone with ID {zone_id}. Use rank to list zone IDs.") from None
        if plan_id:
            plan = self._plan_from_proposal(plan_id)
        else:
            name, merged = self._strategy(None, None)
            plan = run_strategy(problem, name, merged)
        out = _explain_zone(problem, plan, zone_id)
        out["plan_id"] = plan_id
        out["source"] = self.source(problem)
        return out

    def what_if(self, change: dict[str, Any], focus_zone: str | None = None) -> dict[str, Any]:
        """In-memory scenario. Nothing is written anywhere."""
        problem = self._problem()
        name, merged = self._strategy(None, None)
        try:
            out = _what_if(problem, name, merged, change, focus_zone)
        except KeyError as e:
            raise EngineError(str(e).strip("'\"")) from None
        out["written"] = False
        out["source"] = self.source(problem)
        return out

    def policy_dial(self, weights: dict[str, float] | None = None,
                    alphas: list[float] | None = None) -> dict[str, Any]:
        """Read-only. Either sweep the fairness floor alpha (what does fairness cost?) or
        re-rank under different policy weights. Nothing is written."""
        if alphas is not None or not weights:
            return self._alpha_sweep(alphas)
        return self._weight_dial(weights)

    def _alpha_sweep(self, alphas: list[float] | None) -> dict[str, Any]:
        problem = self._problem()
        alphas = sorted({round(float(a), 2) for a in (alphas or [0, 0.1, 0.2, 0.3, 0.4, 0.5])})
        if any(not 0 <= a <= 1 for a in alphas):
            raise EngineError("alpha values must be between 0 and 1")
        lam = float(self.pack.strategy.params.get("lambda_perishable", 1.0))
        rows = []
        for a in alphas:
            plan = run_strategy(problem, "fair_lp", {"alpha": a, "lambda_perishable": lam})
            m = compute_metrics(problem, plan)
            rows.append({"alpha": a, "alpha_used": plan.params.get("alpha_used", a), "vwu": m["vwu"],
                         "worst_off_coverage": m["worst_off_coverage"], "delivered": m["delivered"],
                         "spoiled": m["spoiled"], "flags": plan.flags})
        first, last = rows[0], rows[-1]
        sentence = (f"Raising the fairness floor from {first['alpha']:g} to {last['alpha']:g} moves "
                    f"vulnerability-weighted unmet need from {first['vwu']:g} to {last['vwu']:g} and "
                    f"worst-off coverage from {first['worst_off_coverage']:g} to {last['worst_off_coverage']:g}. "
                    f"Choosing alpha is a policy decision, not a fact.")
        return {"mode": "alpha_sweep", "rows": rows, "sentence": sentence, "written": False,
                "source": self.source(problem)}

    def _weight_dial(self, weights: dict[str, float]) -> dict[str, Any]:
        problem = self._problem()
        try:
            alt_pack = self.pack.with_weights(weights)
        except ValueError as e:
            raise EngineError(str(e)) from None
        from dataclasses import replace
        alt = replace(problem, pack=alt_pack)
        name, merged = self._strategy(None, None)
        base_rank = {s.zone_id: s.rank for s in score_zones(problem)}
        alt_scored = score_zones(alt)
        moves = [{"zone_id": s.zone_id, "name": s.name, "rank_before": base_rank[s.zone_id], "rank_after": s.rank}
                 for s in alt_scored if s.rank != base_rank[s.zone_id]]
        m_before = compute_metrics(problem, run_strategy(problem, name, merged))
        m_after = compute_metrics(alt, run_strategy(alt, name, merged))
        shown = ", ".join(f"{t.term} {t.weight:.2f}" for t in alt_pack.scoring)
        return {
            "mode": "weights",
            "weights_before": self.pack.weights(),
            "weights_after": alt_pack.weights(),
            "rank_changes": moves,
            "metrics_before": m_before,
            "metrics_after": m_after,
            "sentence": f"With weights {shown}, {len(moves)} zones change rank. Weights are policy choices, not facts.",
            "written": False,
            "source": self.source(problem),
        }

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        p = self.store.get_proposal(plan_id)
        if p is None:
            raise EngineError(f"No plan with ID {plan_id}.")
        problem = self._problem()
        d = self._plan_card(p, problem)
        d["status"] = p.status
        return d

    def state(self) -> dict[str, Any]:
        problem = self._problem()
        return {
            "zones": [{"zone_id": z.zone_id, "name": z.name, "need": z.need, "served": z.served,
                       "open_need": z.open_need} for z in problem.zones],
            "lots": [{"lot_id": lt.lot_id, "quantity": lt.quantity, "category": lt.category,
                      "expiry_h": lt.expiry_h, "handling": lt.handling} for lt in problem.lots],
            "recent_plans": [{"plan_id": p.plan_id, "status": p.status, "strategy": p.strategy,
                              "created_at": p.created_at} for p in self.store.list_proposals(5)],
            "source": self.source(problem),
        }

    def verify_audit(self) -> dict[str, Any]:
        return verify_chain(self.store.audit_rows())

    def validate(self) -> list[str]:
        return validate_pack(self.pack, self.store.available_columns())

    # ---- write path ----------------------------------------------------------------
    def _plan_card(self, p: Any, problem: Problem) -> dict[str, Any]:
        unit = self.pack.word("unit", "units")
        by_zone = p.units_by_zone()
        headline = self.pack.metrics.headline
        heads = []
        for base, bm in p.baseline_metrics.items():
            from .metrics import headline_sentence
            heads.append(headline_sentence(headline, p.strategy, p.metrics[headline], base, bm[headline]))
        return {
            "plan_id": p.plan_id,
            "plan_hash": p.plan_hash,
            "approval_code": p.approval_code,
            "strategy": p.strategy,
            "weights": p.weights,
            "total_units": sum(by_zone.values()),
            "allocations": [{"zone_id": z, "name": problem.zone(z).name if z in {x.zone_id for x in problem.zones} else z,
                             "units": u} for z, u in sorted(by_zone.items(), key=lambda kv: -kv[1])],
            "lines": [asdict(ln) for ln in p.lines],
            "rejection_count": len(p.rejections),
            "metrics": p.metrics,
            "baseline_metrics": p.baseline_metrics,
            "headlines": heads,
            "flags": p.flags,
            "unit_word": unit,
        }

    def propose(self, strategy: str | None = None, params: dict | None = None,
                actor: str = "agent") -> dict[str, Any]:
        """Append an immutable proposal. Moves no supply."""
        problem = self._problem()
        name, merged = self._strategy(strategy, params)
        plan = run_strategy(problem, name, merged)
        if not plan.lines:
            raise EngineError("The engine found no feasible allocation; nothing to propose.")
        metrics = compute_metrics(problem, plan)
        baselines = {b: compute_metrics(problem, run_strategy(problem, b, {}))
                     for b in self.pack.strategy.baselines if b != name}
        p = build_proposal(problem, plan, metrics, baselines, created_by=actor)
        self.store.save_proposal(p)
        card = self._plan_card(p, problem)
        card["status"] = "PROPOSED"
        card["next_step"] = self._next_step(p.plan_id, p.approval_code)
        card["source"] = self.source(problem)
        return card

    def _next_step(self, plan_id: str, code: str) -> str:
        if self.approval_mode == "confirm":
            return (f"Nothing has moved yet. To dispatch, a human must confirm plan {plan_id} "
                    f"with approval code {code}; the system will ask for confirmation.")
        return (f"Nothing has moved yet. To dispatch, a human runs: make approve PLAN={plan_id} BY=<name>, "
                f"then asks to commit {plan_id} with approval code {code}.")

    def approve(self, plan_id: str, approver: str, channel: str = "cli") -> dict[str, Any]:
        """Human approval. Exposed to the CLI only, never as an agent tool."""
        p = self.store.get_proposal(plan_id)
        if p is None:
            raise EngineError(f"No plan with ID {plan_id}.")
        return self.store.record_approval(plan_id, p.plan_hash, approver, channel)

    def commit(self, plan_id: str, approval_code: str, actor: str = "agent",
               approver: str | None = None) -> dict[str, Any]:
        if approver and self.approval_mode != "confirm":
            raise EngineError("Inline approval is only allowed in APPROVAL_MODE=confirm.")
        r = self.store.commit(plan_id, approval_code, actor, approver=approver)
        out = r.to_dict()
        out["sentence"] = {
            "APPLIED": f"Plan {plan_id} is committed. Supply and zone records were updated in one step, "
                       f"and notifications were queued.",
            "ALREADY_APPLIED": f"Plan {plan_id} was already committed. Nothing changed the second time.",
            "STALE": f"Plan {plan_id} was not applied because the data changed since it was made: "
                     f"{'; '.join(r.details)}. Propose a new plan.",
            "REJECTED": f"Plan {plan_id} was not applied: {r.reason}.",
            "NOT_FOUND": f"No plan with ID {plan_id}.",
        }[r.status]
        return out

    def reject(self, plan_id: str, reason: str, actor: str = "agent") -> dict[str, Any]:
        out = self.store.reject(plan_id, reason, actor)
        out["plan_id"] = plan_id
        out["moved_supply"] = False
        return out

    # ---- ingest (conditions provider or CLI; never an agent tool) -----------------------
    def ingest_conditions(self, rows: list[dict[str, Any]], source: str,
                          observed_at: str | None = None) -> dict[str, Any]:
        """Conditions enter as numbers only. Strings that are not numbers are rejected (S7, prompt injection)."""
        clean = []
        m = self.pack.entities.get("conditions")
        if m is None:
            raise EngineError("This pack has no conditions entity.")
        zone_ids = {z.zone_id for z in self._problem().zones}
        for r in rows:
            z, f, v = str(r.get("zone_id", "")), str(r.get("factor", "")), r.get("value")
            if z not in zone_ids:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if not 0.0 <= val <= 1.0 or not f.replace("_", "").isalnum():
                continue
            clean.append({m.col("zone_id"): z, m.col("factor"): f, m.col("value"): str(val)})
        return self.store.update_conditions(clean, source, observed_at or utcnow().isoformat(),
                                            actor=f"conditions:{source}")


_ENGINES: dict[tuple[str, str, str], Engine] = {}
_LOCK = threading.Lock()


def get_engine(pack: str | None = None, store: str | None = None,
               approval_mode: str | None = None) -> Engine:
    """One engine per (pack, store, mode), built from env vars PACK, STORE, APPROVAL_MODE."""
    from ..stores import make_store
    key = (pack or os.environ.get("PACK", "food_equity"),
           (store or os.environ.get("STORE", "memory")).lower(),
           approval_mode or os.environ.get("APPROVAL_MODE", "out_of_band"))
    with _LOCK:
        if key not in _ENGINES:
            p = load_pack(key[0])
            errors = validate_pack(p)
            if errors:
                raise EngineError("Pack is invalid: " + "; ".join(errors))
            _ENGINES[key] = Engine(p, make_store(p, key[1]), key[2])
        return _ENGINES[key]


def reset_engines() -> None:
    with _LOCK:
        _ENGINES.clear()


__all__ = ["Engine", "EngineError", "get_engine", "reset_engines", "Line", "Rejection"]
