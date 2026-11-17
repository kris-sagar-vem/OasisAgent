"""BigQuery store: same contract as InMemoryStore (tests/test_gateway_store.py scenarios).

Security rules (S3, S4):
- Values ALWAYS go through query parameters. Only identifiers are interpolated, and only via
  the T[...] (tables) and Z[...] / L[...] (columns) maps built from validated SAFE_ID names.
  tests/test_static_lints.py enforces this.
- No streaming inserts: DML inside scripts, load jobs for seeding.
- The commit is ONE multi-statement transaction. ASSERTs re-check run-once, status, approval and
  versions inside it, so a race or failure rolls everything back. STALE status is written in a
  separate statement outside the transaction (a failed transaction cannot write it).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from ..engine.gateway import (CommitResult, Proposal, audit_row, code_matches, outbox_messages,
                              stale_reasons, utcnow)
from ..engine.loader import build_problem
from ..engine.model import Problem
from ..engine.pack import SAFE_ID, Pack
from .base import Store

SYSTEM_TABLES = ("proposals", "approvals", "dispatch_log", "outbox", "audit_log", "conditions_meta")
_PROJECT_RX = re.compile(r"^[a-z][a-z0-9\-]{4,61}[a-z0-9]$")


class BigQueryStore(Store):
    kind = "bigquery"

    def __init__(self, pack: Pack, client: Any = None, project: str | None = None,
                 dataset: str | None = None, location: str | None = None) -> None:
        super().__init__(pack)
        from google.cloud import bigquery
        self.bq = bigquery
        # BQ_PROJECT wins: on Agent Runtime GOOGLE_CLOUD_PROJECT is managed by the platform.
        self.project = project or os.environ.get("BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.dataset = dataset or os.environ.get("BQ_DATASET", "")
        self.location = location or os.environ.get("BQ_LOCATION", "US")
        if not _PROJECT_RX.match(self.project) or not SAFE_ID.match(self.dataset):
            raise ValueError("Set BQ_PROJECT (or GOOGLE_CLOUD_PROJECT) and BQ_DATASET to valid identifiers for STORE=bigquery")
        self.client = client or bigquery.Client(project=self.project, location=self.location)
        ds = f"{self.project}.{self.dataset}"
        self.T = {name: f"`{ds}.{name}`" for name in SYSTEM_TABLES}
        for ent, m in pack.entities.items():
            if not SAFE_ID.match(m.table):
                raise ValueError(f"unsafe table name {m.table!r}")
            self.T[ent] = f"`{ds}.{m.table}`"
        zm, lm = pack.entities["zones"], pack.entities["lots"]
        self.Z = {f: zm.col(f) for f in ("zone_id", "name", "served", "version")}
        self.L = {f: lm.col(f) for f in ("lot_id", "quantity", "version")}
        for col in [*self.Z.values(), *self.L.values()]:
            if not SAFE_ID.match(col):
                raise ValueError(f"unsafe column name {col!r}")
        self._job_ids: list[str] = []

    # ---- low-level --------------------------------------------------------------
    def _params(self, params: dict[str, Any]) -> list[Any]:
        bq, out = self.bq, []
        for name, v in params.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "ARRAY_STRUCT":
                out.append(bq.ArrayQueryParameter(name, "STRUCT", [
                    bq.StructQueryParameter(None, *[self._scalar(k, x) for k, x in row.items()]) for row in v[1]]))
            else:
                out.append(self._scalar(name, v))
        return out

    def _scalar(self, name: str | None, v: Any) -> Any:
        bq = self.bq
        if isinstance(v, bool):
            return bq.ScalarQueryParameter(name, "BOOL", v)
        if isinstance(v, int):
            return bq.ScalarQueryParameter(name, "INT64", v)
        if isinstance(v, float):
            return bq.ScalarQueryParameter(name, "FLOAT64", v)
        return bq.ScalarQueryParameter(name, "STRING", None if v is None else str(v))

    def _run(self, sql: str, params: dict[str, Any] | None = None, track: bool = False) -> list[dict[str, Any]]:
        cfg = self.bq.QueryJobConfig(query_parameters=self._params(params or {}),
                                     labels={"app": "allocator", "pack": self.pack.meta.id.lower()})
        job = self.client.query(sql, job_config=cfg)
        rows = [dict(r.items()) for r in job.result()]
        if track:
            self._job_ids = (self._job_ids + [job.job_id])[-5:]
        return rows

    def last_job_ids(self) -> list[str]:
        return list(self._job_ids)

    def _audit_params(self, actor: str, event_type: str, plan_id: str | None,
                      payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        last = self._run(f"SELECT seq, row_hash FROM {self.T['audit_log']} ORDER BY seq DESC LIMIT 1")
        prev_seq = int(last[0]["seq"]) if last else 0
        prev_hash = last[0]["row_hash"] if last else "GENESIS"
        row = audit_row(prev_hash, prev_seq + 1, utcnow().isoformat(), actor, event_type, plan_id, payload)
        return ({"a_seq": row["seq"], "a_at": row["at"], "a_actor": actor, "a_type": event_type,
                 "a_plan": plan_id, "a_payload": row["payload_json"], "a_prev": prev_hash,
                 "a_hash": row["row_hash"], "a_prev_seq": prev_seq}, prev_seq)

    def _audit_sql(self) -> str:
        return (f"ASSERT (SELECT COALESCE(MAX(seq), 0) FROM {self.T['audit_log']}) = @a_prev_seq AS 'AUDIT_RACE';\n"
                f"INSERT INTO {self.T['audit_log']} (seq, at, actor, event_type, plan_id, payload_json, prev_hash, row_hash) "
                f"VALUES (@a_seq, @a_at, @a_actor, @a_type, @a_plan, @a_payload, @a_prev, @a_hash);\n")

    def _audited(self, body_sql: str, params: dict[str, Any], actor: str, event_type: str,
                 plan_id: str | None, payload: dict[str, Any]) -> None:
        a, _ = self._audit_params(actor, event_type, plan_id, payload)
        sql = "BEGIN TRANSACTION;\n" + body_sql + self._audit_sql() + "COMMIT TRANSACTION;"
        self._run(sql, {**params, **a})

    # ---- reads -------------------------------------------------------------------
    def load_problem(self) -> Problem:
        tables = {ent: self._run(f"SELECT * FROM {self.T[ent]}", track=True) for ent in self.pack.entities}
        meta = self._run(f"SELECT version, observed_at, source FROM {self.T['conditions_meta']} "
                         f"ORDER BY version DESC LIMIT 1")
        cm = {"version": 0, "observed_at": None, "source": "none"}
        if meta:
            obs = meta[0]["observed_at"]
            cm = {"version": int(meta[0]["version"]),
                  "observed_at": obs.isoformat() if isinstance(obs, datetime) else obs,
                  "source": meta[0]["source"]}
        return build_problem(self.pack, tables, cm)

    def available_columns(self) -> dict[str, set[str]]:
        rows = self._run(f"SELECT table_name, column_name FROM `{self.project}.{self.dataset}`.INFORMATION_SCHEMA.COLUMNS")
        by_table: dict[str, set[str]] = {}
        for r in rows:
            by_table.setdefault(r["table_name"], set()).add(r["column_name"])
        return {ent: by_table[m.table] for ent, m in self.pack.entities.items() if m.table in by_table}

    def get_proposal(self, plan_id: str) -> Proposal | None:
        rows = self._run(f"SELECT status, proposal_json FROM {self.T['proposals']} WHERE plan_id = @pid",
                         {"pid": plan_id})
        if not rows:
            return None
        p = Proposal.from_dict(json.loads(rows[0]["proposal_json"]))
        p.status = rows[0]["status"]
        return p

    def list_proposals(self, limit: int = 10) -> list[Proposal]:
        rows = self._run(f"SELECT status, proposal_json FROM {self.T['proposals']} "
                         f"ORDER BY created_at DESC LIMIT @lim", {"lim": int(limit)})
        out = []
        for r in rows:
            p = Proposal.from_dict(json.loads(r["proposal_json"]))
            p.status = r["status"]
            out.append(p)
        return out

    def audit_rows(self) -> list[dict[str, Any]]:
        return self._run(f"SELECT seq, at, actor, event_type, plan_id, payload_json, prev_hash, row_hash "
                         f"FROM {self.T['audit_log']} ORDER BY seq")

    def outbox(self) -> list[dict[str, Any]]:
        return self._run(f"SELECT plan_id, zone_id, channel, message, status FROM {self.T['outbox']} "
                         f"ORDER BY plan_id, zone_id")

    # ---- append-only writes --------------------------------------------------------
    def save_proposal(self, p: Proposal) -> None:
        body = (f"INSERT INTO {self.T['proposals']} (plan_id, created_at, pack_id, pack_hash, strategy, plan_hash, "
                f"approval_code, idempotency_key, status, created_by, proposal_json) VALUES (@pid, "
                f"TIMESTAMP(@created), @pack, @phash_pack, @strategy, @phash, @code, @ik, 'PROPOSED', @by, @json);\n")
        self._audited(body, {"pid": p.plan_id, "created": p.created_at, "pack": p.pack_id,
                             "phash_pack": p.pack_hash, "strategy": p.strategy, "phash": p.plan_hash,
                             "code": p.approval_code, "ik": p.idempotency_key, "by": p.created_by,
                             "json": json.dumps(p.to_dict(), default=str)},
                      p.created_by, "PROPOSED", p.plan_id,
                      {"plan_hash": p.plan_hash, "strategy": p.strategy, "metrics": p.metrics})

    def record_approval(self, plan_id: str, plan_hash: str, approver: str, channel: str) -> dict[str, Any]:
        p = self.get_proposal(plan_id)
        if p is None:
            return {"ok": False, "reason": "NOT_FOUND"}
        if p.plan_hash != plan_hash:
            return {"ok": False, "reason": "PLAN_HASH_MISMATCH"}
        if p.status not in ("PROPOSED", "APPROVED"):
            return {"ok": False, "reason": f"STATUS_{p.status}"}
        body = (f"INSERT INTO {self.T['approvals']} (plan_id, plan_hash, approver, decision, channel, decided_at) "
                f"VALUES (@pid, @phash, @who, 'APPROVE', @chan, CURRENT_TIMESTAMP());\n"
                f"UPDATE {self.T['proposals']} SET status = 'APPROVED' WHERE plan_id = @pid "
                f"AND status IN ('PROPOSED', 'APPROVED');\n")
        self._audited(body, {"pid": plan_id, "phash": plan_hash, "who": approver, "chan": channel},
                      approver, "APPROVED", plan_id, {"channel": channel})
        return {"ok": True, "status": "APPROVED"}

    def reject(self, plan_id: str, reason: str, actor: str) -> dict[str, Any]:
        p = self.get_proposal(plan_id)
        if p is None:
            return {"ok": False, "status": "NOT_FOUND"}
        if p.status not in ("PROPOSED", "APPROVED"):
            return {"ok": False, "status": p.status, "reason": f"cannot reject a {p.status} plan"}
        body = (f"UPDATE {self.T['proposals']} SET status = 'REJECTED' WHERE plan_id = @pid "
                f"AND status IN ('PROPOSED', 'APPROVED');\n")
        self._audited(body, {"pid": plan_id}, actor, "REJECTED", plan_id, {"reason": reason})
        return {"ok": True, "status": "REJECTED"}

    # ---- commit ------------------------------------------------------------------
    def _current_versions(self) -> tuple[dict[str, tuple[int, int]], dict[str, int], dict[str, Any]]:
        L, Z = self.L, self.Z
        lots = self._run(f"SELECT {L['lot_id']} AS id, {L['version']} AS v, {L['quantity']} AS q FROM {self.T['lots']}")
        zones = self._run(f"SELECT {Z['zone_id']} AS id, {Z['version']} AS v FROM {self.T['zones']}")
        meta = self._run(f"SELECT version, observed_at FROM {self.T['conditions_meta']} ORDER BY version DESC LIMIT 1")
        cm = {"version": int(meta[0]["version"]) if meta else 0,
              "observed_at": (meta[0]["observed_at"].isoformat() if meta and meta[0]["observed_at"] else None)}
        return ({r["id"]: (int(r["v"]), int(r["q"])) for r in lots}, {r["id"]: int(r["v"]) for r in zones}, cm)

    def commit(self, plan_id: str, approval_code: str, actor: str,
               approver: str | None = None) -> CommitResult:
        p = self.get_proposal(plan_id)
        if p is None:
            return CommitResult("NOT_FOUND", plan_id, "no such plan")
        done = self._run(f"SELECT COUNT(*) AS n FROM {self.T['dispatch_log']} WHERE idempotency_key = @ik",
                         {"ik": p.idempotency_key})
        if done and int(done[0]["n"]) > 0:
            return CommitResult("ALREADY_APPLIED", plan_id, "this plan was already committed; nothing changed")
        if p.status in ("REJECTED", "STALE"):
            return CommitResult(p.status, plan_id, f"plan is {p.status.lower()}")
        if not code_matches(approval_code, p.approval_code):
            self._audited("", {}, actor, "COMMIT_REFUSED", plan_id, {"reason": "BAD_APPROVAL_CODE"})
            return CommitResult("REJECTED", plan_id, "BAD_APPROVAL_CODE")
        if not approver:
            ok = self._run(f"SELECT COUNT(*) AS n FROM {self.T['approvals']} WHERE plan_id = @pid "
                           f"AND plan_hash = @ph AND decision = 'APPROVE'", {"pid": plan_id, "ph": p.plan_hash})
            if not ok or int(ok[0]["n"]) == 0:
                self._audited("", {}, actor, "COMMIT_REFUSED", plan_id, {"reason": "NOT_APPROVED"})
                return CommitResult("REJECTED", plan_id, "NOT_APPROVED: no human approval recorded for this plan")

        lots_now, zones_now, cm = self._current_versions()
        reasons = stale_reasons(p, lots_now, zones_now, cm["version"], cm["observed_at"],
                                self.pack.conditions.max_age_minutes, utcnow())
        if reasons:
            # Outside any transaction on purpose: a rolled-back transaction could not record this.
            self._audited(f"UPDATE {self.T['proposals']} SET status = 'STALE' WHERE plan_id = @pid;\n",
                          {"pid": plan_id}, actor, "STALE", plan_id, {"reasons": reasons})
            return CommitResult("STALE", plan_id, "data changed since the plan was made", reasons)

        before = {"lots": {k: lots_now[k][1] for k in sorted(p.units_by_lot())}}
        result = self._commit_txn(p, actor, approver, cm["version"])
        if result is not None:
            return result
        lots_after, _, _ = self._current_versions()
        return CommitResult("APPLIED", plan_id, "committed atomically", [], before,
                            {"lots": {k: lots_after[k][1] for k in sorted(p.units_by_lot())}})

    def _commit_txn(self, p: Proposal, actor: str, approver: str | None, cond_version: int) -> CommitResult | None:
        T, L, Z = self.T, self.L, self.Z
        need = p.units_by_lot()
        exp_lots = [{"id": k, "v": int(v), "need": int(need.get(k, 0))} for k, v in p.expected_versions["lots"].items()]
        exp_zones = [{"id": k, "v": int(v)} for k, v in p.expected_versions["zones"].items()]
        lot_d = [{"id": k, "units": int(u)} for k, u in need.items()]
        zone_d = [{"id": k, "units": int(u)} for k, u in p.units_by_zone().items()]
        lines = [{"lot": ln.lot_id, "zone": ln.zone_id, "units": int(ln.units)} for ln in p.lines]
        prob = self.load_problem()
        names = {z.zone_id: z.name for z in prob.zones}
        msgs = [{"zone": m["zone_id"], "chan": m["channel"], "msg": m["message"]}
                for m in outbox_messages(p, names, self.pack.word("unit", "units"))]
        sql = "BEGIN TRANSACTION;\n"
        if approver:
            sql += (f"INSERT INTO {T['approvals']} (plan_id, plan_hash, approver, decision, channel, decided_at) "
                    f"VALUES (@pid, @ph, @approver, 'APPROVE', 'adk_confirmation', CURRENT_TIMESTAMP());\n")
        sql += (
            f"ASSERT NOT EXISTS (SELECT 1 FROM {T['dispatch_log']} WHERE idempotency_key = @ik) AS 'ALREADY_APPLIED';\n"
            f"ASSERT (SELECT COUNT(*) FROM {T['proposals']} WHERE plan_id = @pid "
            f"AND status IN ('PROPOSED', 'APPROVED')) = 1 AS 'BAD_STATUS';\n"
            f"ASSERT EXISTS (SELECT 1 FROM {T['approvals']} WHERE plan_id = @pid AND plan_hash = @ph "
            f"AND decision = 'APPROVE') AS 'NOT_APPROVED';\n"
            f"ASSERT (SELECT COUNT(*) FROM {T['lots']} t JOIN UNNEST(@exp_lots) e ON t.{L['lot_id']} = e.id "
            f"WHERE t.{L['version']} = e.v AND t.{L['quantity']} >= e.need) = ARRAY_LENGTH(@exp_lots) AS 'STALE';\n"
            f"ASSERT (SELECT COUNT(*) FROM {T['zones']} t JOIN UNNEST(@exp_zones) e ON t.{Z['zone_id']} = e.id "
            f"WHERE t.{Z['version']} = e.v) = ARRAY_LENGTH(@exp_zones) AS 'STALE';\n"
            f"ASSERT (SELECT COALESCE(MAX(version), 0) FROM {T['conditions_meta']}) = @cv AS 'STALE';\n"
            f"UPDATE {T['lots']} t SET {L['quantity']} = t.{L['quantity']} - d.units, {L['version']} = t.{L['version']} + 1 "
            f"FROM UNNEST(@lot_d) d WHERE t.{L['lot_id']} = d.id;\n"
            f"UPDATE {T['zones']} t SET {Z['served']} = COALESCE(t.{Z['served']}, 0) + d.units, "
            f"{Z['version']} = t.{Z['version']} + 1 FROM UNNEST(@zone_d) d WHERE t.{Z['zone_id']} = d.id;\n"
            f"UPDATE {T['proposals']} SET status = 'COMMITTED' WHERE plan_id = @pid;\n"
            f"INSERT INTO {T['dispatch_log']} (plan_id, idempotency_key, lot_id, zone_id, units, committed_at) "
            f"SELECT @pid, @ik, x.lot, x.zone, x.units, CURRENT_TIMESTAMP() FROM UNNEST(@lines) x;\n"
            f"INSERT INTO {T['outbox']} (plan_id, zone_id, channel, message, status) "
            f"SELECT @pid, m.zone, m.chan, m.msg, 'PENDING' FROM UNNEST(@msgs) m;\n"
        )
        params: dict[str, Any] = {
            "pid": p.plan_id, "ph": p.plan_hash, "ik": p.idempotency_key, "cv": int(cond_version),
            "exp_lots": ("ARRAY_STRUCT", exp_lots), "exp_zones": ("ARRAY_STRUCT", exp_zones),
            "lot_d": ("ARRAY_STRUCT", lot_d), "zone_d": ("ARRAY_STRUCT", zone_d),
            "lines": ("ARRAY_STRUCT", lines), "msgs": ("ARRAY_STRUCT", msgs),
        }
        if approver:
            params["approver"] = approver
        a, _ = self._audit_params(actor, "COMMITTED", p.plan_id,
                                  {"idempotency_key": p.idempotency_key, "units": sum(need.values())})
        sql += self._audit_sql() + "COMMIT TRANSACTION;"
        try:
            self._run(sql, {**params, **a})
        except Exception as e:  # noqa: BLE001 - map ASSERT names / concurrent aborts to results
            msg = str(e)
            if "ALREADY_APPLIED" in msg:
                return CommitResult("ALREADY_APPLIED", p.plan_id, "this plan was already committed; nothing changed")
            if "STALE" in msg:
                return CommitResult("STALE", p.plan_id, "data changed while committing; nothing was applied",
                                    ["versions changed inside the transaction"])
            if "NOT_APPROVED" in msg or "BAD_STATUS" in msg:
                return CommitResult("REJECTED", p.plan_id, "NOT_APPROVED or status changed; nothing was applied")
            if "AUDIT_RACE" in msg or "concurrent" in msg.lower():
                return CommitResult("STALE", p.plan_id, "another write happened at the same moment; nothing was "
                                    "applied, retry the commit", ["concurrent transaction"])
            raise
        return None

    # ---- ingest -----------------------------------------------------------------
    def update_conditions(self, rows: list[dict[str, Any]], source: str, observed_at: str,
                          actor: str) -> dict[str, Any]:
        m = self.pack.entities["conditions"]
        cz, cf, cv = m.col("zone_id"), m.col("factor"), m.col("value")
        for col in (cz, cf, cv):
            if not SAFE_ID.match(col):
                raise ValueError(f"unsafe column name {col!r}")
        cur = self._run(f"SELECT COALESCE(MAX(version), 0) AS v FROM {self.T['conditions_meta']}")
        new_v = int(cur[0]["v"]) + 1
        data = [{"z": r[cz], "f": r[cf], "v": float(r[cv])} for r in rows]
        body = (f"DELETE FROM {self.T['conditions']} WHERE TRUE;\n"
                f"INSERT INTO {self.T['conditions']} ({cz}, {cf}, {cv}) SELECT x.z, x.f, x.v FROM UNNEST(@rows) x;\n"
                f"INSERT INTO {self.T['conditions_meta']} (version, observed_at, source) "
                f"VALUES (@ver, TIMESTAMP(@obs), @src);\n")
        params: dict[str, Any] = {"ver": new_v, "obs": observed_at, "src": source}
        if data:
            params["rows"] = ("ARRAY_STRUCT", data)
        else:
            body = body.replace(f"INSERT INTO {self.T['conditions']} ({cz}, {cf}, {cv}) SELECT x.z, x.f, x.v "
                                f"FROM UNNEST(@rows) x;\n", "")
        self._audited(body, params, actor, "CONDITIONS_UPDATED", None,
                      {"source": source, "rows": len(rows), "version": new_v})
        return {"version": new_v, "observed_at": observed_at, "source": source}

    def record_field_update(self, lot_id: str, delta: int, actor: str, note: str) -> dict[str, Any]:
        L = self.L
        cur = self._run(f"SELECT {L['quantity']} AS q FROM {self.T['lots']} WHERE {L['lot_id']} = @id", {"id": lot_id})
        if not cur:
            raise KeyError(f"unknown lot_id: {lot_id}")
        old = int(cur[0]["q"])
        body = (f"UPDATE {self.T['lots']} SET {L['quantity']} = GREATEST(0, {L['quantity']} + @d), "
                f"{L['version']} = {L['version']} + 1 WHERE {L['lot_id']} = @id;\n")
        self._audited(body, {"id": lot_id, "d": int(delta)}, actor, "FIELD_UPDATE", None,
                      {"lot_id": lot_id, "delta": int(delta), "note": note})
        return {"lot_id": lot_id, "before": old, "after": max(0, old + int(delta))}

    def mark_outbox_sent(self, plan_id: str) -> int:
        n = self._run(f"SELECT COUNT(*) AS n FROM {self.T['outbox']} WHERE plan_id = @pid AND status = 'PENDING'",
                      {"pid": plan_id})
        self._run(f"UPDATE {self.T['outbox']} SET status = 'SENT' WHERE plan_id = @pid AND status = 'PENDING'",
                  {"pid": plan_id})
        return int(n[0]["n"]) if n else 0

    def reset(self) -> None:
        from .bq_seed import seed
        seed(self)
