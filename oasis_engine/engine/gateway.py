"""Commit gateway: the ONLY path that changes operational state.

Builds immutable proposals (plan hash, approval code, idempotency key, expected versions)
and holds the pure checks every store must enforce inside its transaction.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .model import Line, Plan, Problem, Rejection

STATUSES = ("PROPOSED", "APPROVED", "REJECTED", "STALE", "COMMITTED")
RESULTS = ("APPLIED", "ALREADY_APPLIED", "STALE", "REJECTED", "NOT_FOUND")
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def approval_code_for(plan_hash: str) -> str:
    n = int(sha256("approve:" + plan_hash), 16)
    out = []
    for _ in range(6):
        n, r = divmod(n, len(_CODE_ALPHABET))
        out.append(_CODE_ALPHABET[r])
    return "".join(out)


@dataclass
class Proposal:
    plan_id: str
    created_at: str
    pack_id: str
    pack_hash: str
    strategy: str
    params: dict[str, Any]
    weights: dict[str, float]
    plan_hash: str
    approval_code: str
    idempotency_key: str
    status: str
    lines: list[Line]
    rejections: list[Rejection]
    metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    expected_versions: dict[str, Any]
    created_by: str
    flags: list[str] = field(default_factory=list)

    def units_by_lot(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ln in self.lines:
            out[ln.lot_id] = out.get(ln.lot_id, 0) + ln.units
        return out

    def units_by_zone(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ln in self.lines:
            out[ln.zone_id] = out.get(ln.zone_id, 0) + ln.units
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Proposal":
        d = dict(d)
        d["lines"] = [Line(**x) for x in d["lines"]]
        d["rejections"] = [Rejection(**x) for x in d["rejections"]]
        return cls(**d)


@dataclass
class CommitResult:
    status: str
    plan_id: str
    reason: str = ""
    details: list[str] = field(default_factory=list)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_proposal(problem: Problem, plan: Plan, metrics: dict[str, float],
                   baseline_metrics: dict[str, dict[str, float]], created_by: str,
                   now: datetime | None = None) -> Proposal:
    now = now or utcnow()
    pack = problem.pack
    versions = problem.versions()
    body = {
        "pack_hash": pack.pack_hash(),
        "strategy": plan.strategy,
        "params": plan.params,
        "weights": pack.weights(),
        "lines": plan.canonical_lines(),
        "expected_versions": versions,
    }
    plan_hash = sha256(canonical_json(body))
    plan_id = "P-" + sha256(plan_hash + now.isoformat() + secrets.token_hex(4))[:6].upper()
    return Proposal(
        plan_id=plan_id,
        created_at=now.isoformat(),
        pack_id=pack.meta.id,
        pack_hash=pack.pack_hash(),
        strategy=plan.strategy,
        params=dict(plan.params),
        weights=pack.weights(),
        plan_hash=plan_hash,
        approval_code=approval_code_for(plan_hash),
        idempotency_key=sha256(plan_id + canonical_json(plan.canonical_lines())),
        status="PROPOSED",
        lines=list(plan.lines),
        rejections=list(plan.rejections),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        expected_versions=versions,
        created_by=created_by,
        flags=list(plan.flags),
    )


def code_matches(given: str, expected: str) -> bool:
    return hmac.compare_digest(str(given).strip().upper().encode(), expected.encode())


def stale_reasons(p: Proposal, lots_now: dict[str, tuple[int, int]], zones_now: dict[str, int],
                  conditions_version: int, conditions_observed_at: str | None,
                  max_age_minutes: float | None, now: datetime) -> list[str]:
    """Why the plan can no longer be applied. lots_now: {lot_id: (version, quantity)}."""
    out = []
    need = p.units_by_lot()
    for lot_id, v in p.expected_versions["lots"].items():
        cur = lots_now.get(lot_id)
        if cur is None:
            out.append(f"lot {lot_id} no longer exists")
        elif cur[0] != v:
            out.append(f"lot {lot_id} changed (version {v} -> {cur[0]})")
        elif cur[1] < need.get(lot_id, 0):
            out.append(f"lot {lot_id} has {cur[1]} units, plan needs {need[lot_id]}")
    for zone_id, v in p.expected_versions["zones"].items():
        cur_v = zones_now.get(zone_id)
        if cur_v is None:
            out.append(f"zone {zone_id} no longer exists")
        elif cur_v != v:
            out.append(f"zone {zone_id} changed (version {v} -> {cur_v})")
    if p.expected_versions.get("conditions", 0) != conditions_version:
        out.append(f"conditions changed (version {p.expected_versions.get('conditions', 0)} -> {conditions_version})")
    if max_age_minutes is not None and conditions_observed_at:
        age = (now - datetime.fromisoformat(conditions_observed_at)).total_seconds() / 60
        if age > max_age_minutes:
            out.append(f"conditions are {age:.0f} min old (limit {max_age_minutes:g})")
    return out


def audit_row(prev_hash: str, seq: int, at: str, actor: str, event_type: str,
              plan_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    event = {"seq": seq, "at": at, "actor": actor, "event_type": event_type,
             "plan_id": plan_id, "payload": payload}
    return {**event, "payload_json": canonical_json(payload), "prev_hash": prev_hash,
            "row_hash": sha256(prev_hash + canonical_json(event))}


def verify_chain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prev = "GENESIS"
    for r in rows:
        payload = r["payload"] if "payload" in r else json.loads(r["payload_json"])
        event = {"seq": r["seq"], "at": r["at"], "actor": r["actor"], "event_type": r["event_type"],
                 "plan_id": r["plan_id"], "payload": payload}
        if r["prev_hash"] != prev or r["row_hash"] != sha256(prev + canonical_json(event)):
            return {"valid": False, "rows": len(rows), "broken_at_seq": r["seq"],
                    "sentence": f"Audit chain is BROKEN at row {r['seq']} of {len(rows)}."}
        prev = r["row_hash"]
    return {"valid": True, "rows": len(rows), "broken_at_seq": None,
            "sentence": f"Audit chain is valid: {len(rows)} rows verified."}


def outbox_messages(p: Proposal, zone_names: dict[str, str], unit_word: str) -> list[dict[str, Any]]:
    """Notifications written inside the commit; a separate sender delivers them."""
    return [{"plan_id": p.plan_id, "zone_id": z, "channel": "sms_or_email",
             "message": f"Plan {p.plan_id}: {units} {unit_word} are on the way to {zone_names.get(z, z)}.",
             "status": "PENDING"}
            for z, units in sorted(p.units_by_zone().items())]
