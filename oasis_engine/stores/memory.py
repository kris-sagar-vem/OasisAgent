"""In-memory store with the same semantics as BigQuery: atomic, idempotent, stale-guarded commit.

Used by every offline test and the offline demo. State lives in this process only.
"""
from __future__ import annotations

import copy
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..engine.gateway import (CommitResult, Proposal, audit_row, code_matches, outbox_messages,
                              stale_reasons, utcnow)
from ..engine.loader import build_problem
from ..engine.model import Problem
from ..engine.pack import Pack
from .base import Store


class InjectedFailure(RuntimeError):
    """Raised by tests to prove a half-finished commit leaves no trace."""


class InMemoryStore(Store):
    kind = "memory"

    def __init__(self, pack: Pack, seed_dir: Path | None = None,
                 clock: Callable[[], datetime] = utcnow) -> None:
        super().__init__(pack)
        self.seed_dir = seed_dir or (Path(pack.source_dir) / "seed" if pack.source_dir else None)
        self.clock = clock
        self._lock = threading.RLock()
        self.fail_after_step: int | None = None  # test hook
        self.reset()

    # ---- setup -------------------------------------------------------------
    def _read_csv(self, entity: str) -> tuple[list[dict[str, Any]], set[str]]:
        if self.seed_dir is None:
            return [], set()
        path = self.seed_dir / f"{self.pack.entities[entity].table}.csv"
        if not path.exists():
            return [], set()
        with path.open() as f:
            reader = csv.DictReader(f)
            return list(reader), set(reader.fieldnames or [])

    def reset(self) -> None:
        with self._lock:
            self._tables: dict[str, list[dict[str, Any]]] = {}
            self._columns: dict[str, set[str]] = {}
            for entity in self.pack.entities:
                rows, cols = self._read_csv(entity)
                self._tables[entity], self._columns[entity] = rows, cols
            zm, lm = self.pack.entities["zones"], self.pack.entities["lots"]
            for r in self._tables["zones"]:
                r.setdefault(zm.col("served"), "0")
                r.setdefault(zm.col("version"), "1")
            for r in self._tables["lots"]:
                r.setdefault(lm.col("version"), "1")
            self._cond_meta = {"version": 1 if self._tables.get("conditions") else 0,
                               "observed_at": self.clock().isoformat() if self._tables.get("conditions") else None,
                               "source": "seed"}
            self._proposals: dict[str, Proposal] = {}
            self._approvals: list[dict[str, Any]] = []
            self._dispatch: list[dict[str, Any]] = []
            self._outbox: list[dict[str, Any]] = []
            self._audit: list[dict[str, Any]] = []

    # ---- reads -------------------------------------------------------------
    def load_problem(self) -> Problem:
        with self._lock:
            return build_problem(self.pack, copy.deepcopy(self._tables), dict(self._cond_meta))

    def available_columns(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._columns.items() if v}

    def get_proposal(self, plan_id: str) -> Proposal | None:
        with self._lock:
            p = self._proposals.get(plan_id)
            return copy.deepcopy(p) if p else None

    def list_proposals(self, limit: int = 10) -> list[Proposal]:
        with self._lock:
            ps = sorted(self._proposals.values(), key=lambda p: p.created_at, reverse=True)
            return copy.deepcopy(ps[:limit])

    def audit_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._audit)

    def outbox(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._outbox)

    # ---- helpers -------------------------------------------------------------
    def _audit_append(self, audit: list[dict[str, Any]], actor: str, event_type: str,
                      plan_id: str | None, payload: dict[str, Any]) -> None:
        prev = audit[-1]["row_hash"] if audit else "GENESIS"
        audit.append(audit_row(prev, len(audit) + 1, self.clock().isoformat(), actor, event_type, plan_id, payload))

    def _step(self, n: int) -> None:
        if self.fail_after_step is not None and n >= self.fail_after_step:
            raise InjectedFailure(f"injected failure after step {n}")

    # ---- append-only writes ----------------------------------------------------
    def save_proposal(self, p: Proposal) -> None:
        with self._lock:
            if p.plan_id in self._proposals:
                raise ValueError(f"duplicate plan_id {p.plan_id}")
            self._proposals[p.plan_id] = copy.deepcopy(p)
            self._audit_append(self._audit, p.created_by, "PROPOSED", p.plan_id,
                               {"plan_hash": p.plan_hash, "strategy": p.strategy, "metrics": p.metrics})

    def record_approval(self, plan_id: str, plan_hash: str, approver: str, channel: str) -> dict[str, Any]:
        with self._lock:
            p = self._proposals.get(plan_id)
            if p is None:
                return {"ok": False, "reason": "NOT_FOUND"}
            if p.plan_hash != plan_hash:
                return {"ok": False, "reason": "PLAN_HASH_MISMATCH"}
            if p.status not in ("PROPOSED", "APPROVED"):
                return {"ok": False, "reason": f"STATUS_{p.status}"}
            self._approvals.append({"plan_id": plan_id, "plan_hash": plan_hash, "approver": approver,
                                    "decision": "APPROVE", "channel": channel,
                                    "decided_at": self.clock().isoformat()})
            p.status = "APPROVED"
            self._audit_append(self._audit, approver, "APPROVED", plan_id, {"channel": channel})
            return {"ok": True, "status": "APPROVED"}

    def reject(self, plan_id: str, reason: str, actor: str) -> dict[str, Any]:
        with self._lock:
            p = self._proposals.get(plan_id)
            if p is None:
                return {"ok": False, "status": "NOT_FOUND"}
            if p.status not in ("PROPOSED", "APPROVED"):
                return {"ok": False, "status": p.status, "reason": f"cannot reject a {p.status} plan"}
            p.status = "REJECTED"
            self._audit_append(self._audit, actor, "REJECTED", plan_id, {"reason": reason})
            return {"ok": True, "status": "REJECTED"}

    # ---- the only operational mutation path -----------------------------------
    def commit(self, plan_id: str, approval_code: str, actor: str,
               approver: str | None = None) -> CommitResult:
        with self._lock:
            p = self._proposals.get(plan_id)
            if p is None:
                return CommitResult("NOT_FOUND", plan_id, "no such plan")
            if any(d["idempotency_key"] == p.idempotency_key for d in self._dispatch):
                return CommitResult("ALREADY_APPLIED", plan_id, "this plan was already committed; nothing changed")
            if p.status == "REJECTED":
                return CommitResult("REJECTED", plan_id, "plan was rejected")
            if p.status == "STALE":
                return CommitResult("STALE", plan_id, "plan is stale; propose a new plan")
            if not code_matches(approval_code, p.approval_code):
                self._audit_append(self._audit, actor, "COMMIT_REFUSED", plan_id, {"reason": "BAD_APPROVAL_CODE"})
                return CommitResult("REJECTED", plan_id, "BAD_APPROVAL_CODE")
            pending_approval = None
            if approver:
                pending_approval = {"plan_id": plan_id, "plan_hash": p.plan_hash, "approver": approver,
                                    "decision": "APPROVE", "channel": "adk_confirmation",
                                    "decided_at": self.clock().isoformat()}
            if pending_approval is None and not any(a["plan_id"] == plan_id and a["plan_hash"] == p.plan_hash and a["decision"] == "APPROVE"
                       for a in self._approvals):
                self._audit_append(self._audit, actor, "COMMIT_REFUSED", plan_id, {"reason": "NOT_APPROVED"})
                return CommitResult("REJECTED", plan_id, "NOT_APPROVED: no human approval recorded for this plan")

            zm, lm = self.pack.entities["zones"], self.pack.entities["lots"]
            lots = {r[lm.col("lot_id")]: r for r in self._tables["lots"]}
            zones = {r[zm.col("zone_id")]: r for r in self._tables["zones"]}
            lots_now = {k: (int(r[lm.col("version")]), int(float(r[lm.col("quantity")]))) for k, r in lots.items()}
            zones_now = {k: int(r[zm.col("version")]) for k, r in zones.items()}
            now = self.clock()
            reasons = stale_reasons(p, lots_now, zones_now, int(self._cond_meta["version"]),
                                    self._cond_meta["observed_at"], self.pack.conditions.max_age_minutes, now)
            if reasons:
                p.status = "STALE"
                self._audit_append(self._audit, actor, "STALE", plan_id, {"reasons": reasons})
                return CommitResult("STALE", plan_id, "data changed since the plan was made", reasons)

            before = self._snapshot_for(p)
            # Work on copies; swap only if every step succeeds (all-or-nothing).
            tables = copy.deepcopy(self._tables)
            audit = copy.deepcopy(self._audit)
            approvals = copy.deepcopy(self._approvals)
            if pending_approval:
                approvals.append(pending_approval)
                self._audit_append(audit, approver, "APPROVED", plan_id, {"channel": "adk_confirmation"})
            dispatch = copy.deepcopy(self._dispatch)
            outbox = copy.deepcopy(self._outbox)
            proposals = copy.deepcopy(self._proposals)
            lots_c = {r[lm.col("lot_id")]: r for r in tables["lots"]}
            zones_c = {r[zm.col("zone_id")]: r for r in tables["zones"]}
            for lot_id, units in p.units_by_lot().items():
                r = lots_c[lot_id]
                r[lm.col("quantity")] = str(int(float(r[lm.col("quantity")])) - units)
                r[lm.col("version")] = str(int(r[lm.col("version")]) + 1)
            self._step(1)
            for zone_id, units in p.units_by_zone().items():
                r = zones_c[zone_id]
                r[zm.col("served")] = str(int(float(r[zm.col("served")] or 0)) + units)
                r[zm.col("version")] = str(int(r[zm.col("version")]) + 1)
            self._step(2)
            proposals[plan_id].status = "COMMITTED"
            for ln in p.lines:
                dispatch.append({"plan_id": plan_id, "idempotency_key": p.idempotency_key, "lot_id": ln.lot_id,
                                 "zone_id": ln.zone_id, "units": ln.units, "committed_at": now.isoformat()})
            names = {k: r.get(zm.col("name"), k) for k, r in zones_c.items()}
            outbox.extend(outbox_messages(p, names, self.pack.word("unit", "units")))
            self._step(3)
            self._audit_append(audit, actor, "COMMITTED", plan_id,
                               {"idempotency_key": p.idempotency_key, "units": sum(p.units_by_lot().values())})
            self._tables, self._audit, self._dispatch = tables, audit, dispatch
            self._approvals = approvals
            self._outbox, self._proposals = outbox, proposals
            return CommitResult("APPLIED", plan_id, "committed atomically", [], before, self._snapshot_for(p))

    def _snapshot_for(self, p: Proposal) -> dict[str, Any]:
        zm, lm = self.pack.entities["zones"], self.pack.entities["lots"]
        lots = {r[lm.col("lot_id")]: r for r in self._tables["lots"]}
        zones = {r[zm.col("zone_id")]: r for r in self._tables["zones"]}
        return {
            "lots": {k: int(float(lots[k][lm.col("quantity")])) for k in sorted(p.units_by_lot())},
            "zones_served": {k: int(float(zones[k][zm.col("served")] or 0)) for k in sorted(p.units_by_zone())},
        }

    # ---- ingest paths -------------------------------------------------------------
    def update_conditions(self, rows: list[dict[str, Any]], source: str, observed_at: str,
                          actor: str) -> dict[str, Any]:
        with self._lock:
            self._tables["conditions"] = [dict(r) for r in rows]
            self._cond_meta = {"version": int(self._cond_meta["version"]) + 1,
                               "observed_at": observed_at, "source": source}
            self._audit_append(self._audit, actor, "CONDITIONS_UPDATED", None,
                               {"source": source, "rows": len(rows), "version": self._cond_meta["version"]})
            return dict(self._cond_meta)

    def record_field_update(self, lot_id: str, delta: int, actor: str, note: str) -> dict[str, Any]:
        with self._lock:
            lm = self.pack.entities["lots"]
            for r in self._tables["lots"]:
                if r[lm.col("lot_id")] == lot_id:
                    old = int(float(r[lm.col("quantity")]))
                    r[lm.col("quantity")] = str(max(0, old + int(delta)))
                    r[lm.col("version")] = str(int(r[lm.col("version")]) + 1)
                    self._audit_append(self._audit, actor, "FIELD_UPDATE", None,
                                       {"lot_id": lot_id, "delta": int(delta), "note": note})
                    return {"lot_id": lot_id, "before": old, "after": int(r[lm.col("quantity")])}
            raise KeyError(f"unknown lot_id: {lot_id}")

    def mark_outbox_sent(self, plan_id: str) -> int:
        with self._lock:
            n = 0
            for m in self._outbox:
                if m["plan_id"] == plan_id and m["status"] == "PENDING":
                    m["status"] = "SENT"
                    n += 1
            return n
