"""Operator CLI. All writes here are human actions (approve, shock, conditions) or setup.

  python -m allocator.cli <command> [args]    (PACK, STORE, APPROVAL_MODE from env)

NOTE: with STORE=memory every process has its own state, so `approve` in a separate process
cannot affect a running `adk web`. Use STORE=bigquery for out-of-band approval, or
APPROVAL_MODE=confirm for offline live demos. `demo` runs the full story in one process.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .engine.pack import load_pack, validate_pack
from .engine.service import EngineError, get_engine


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in cols))


def cmd_validate(a) -> int:
    pack = load_pack(a.pack)
    errors = validate_pack(pack)
    if not errors and a.with_store:
        errors = get_engine(pack=a.pack).validate()
    print("OK: pack is valid" if not errors else "INVALID:\n- " + "\n- ".join(errors))
    return 0 if not errors else 1


def cmd_seed(a) -> int:
    e = get_engine()
    if e.store.kind == "bigquery":
        from .stores.bq_seed import seed
        _print({"loaded": seed(e.store)})
    else:
        e.store.reset()
        print("memory store reset from seed CSVs")
    return 0


def cmd_rank(a) -> int:
    r = get_engine().rank(a.top)
    _table(r["zones"], ["rank", "zone_id", "name", "score", "open_need"])
    return 0


def cmd_compare(a) -> int:
    c = get_engine().compare()
    rows = [{"method": k, **v} for k, v in c["metrics"].items()]
    _table(rows, ["method", *c["metric_labels"].keys()])
    for h in c["headlines"]:
        print("-", h["sentence"])
    if c["flags"]:
        print("flags:", c["flags"])
    return 0


def cmd_dial(a) -> int:
    alphas = [float(x) for x in a.alphas.split(",")] if a.alphas else None
    r = get_engine().policy_dial(alphas=alphas)
    _table(r["rows"], ["alpha", "alpha_used", "vwu", "worst_off_coverage", "delivered", "spoiled"])
    print(r["sentence"])
    return 0


def cmd_propose(a) -> int:
    p = get_engine().propose(a.strategy, None, actor=f"cli:{a.by}")
    _table(p["allocations"], ["zone_id", "name", "units"])
    print(f"plan {p['plan_id']}  approval code {p['approval_code']}  total {p['total_units']}")
    for h in p["headlines"]:
        print("-", h)
    print(p["next_step"])
    return 0


def cmd_approve(a) -> int:
    _print(get_engine().approve(a.plan, a.by, channel="cli"))
    return 0


def cmd_commit(a) -> int:
    r = get_engine().commit(a.plan, a.code, actor=f"cli:{a.by}")
    print(r["sentence"])
    return 0 if r["status"] in ("APPLIED", "ALREADY_APPLIED") else 2


def cmd_reject(a) -> int:
    _print(get_engine().reject(a.plan, a.reason, actor=f"cli:{a.by}"))
    return 0


def cmd_shock(a) -> int:
    _print(get_engine().store.record_field_update(a.lot, a.delta, f"field:{a.by}", a.note))
    return 0


def cmd_conditions(a) -> int:
    e = get_engine()
    if a.file:
        rows = json.load(open(a.file))
        _print(e.ingest_conditions(rows.get("rows", rows), source=rows.get("source", "file")
                                   if isinstance(rows, dict) else "file"))
    else:
        from . import conditions
        _print(conditions.refresh(e))
    return 0


def cmd_verify(a) -> int:
    r = get_engine().verify_audit()
    print(r["sentence"])
    return 0 if r["valid"] else 3


def cmd_outbox(a) -> int:
    rows = get_engine().store.outbox()
    _table(rows, ["plan_id", "zone_id", "status", "message"]) if rows else print("outbox is empty")
    return 0


def cmd_state(a) -> int:
    s = get_engine().state()
    _table(s["zones"], ["zone_id", "name", "need", "served", "open_need"])
    _table(s["lots"], ["lot_id", "quantity", "category", "expiry_h", "handling"])
    return 0


def cmd_demo(a) -> int:
    """The whole story in one process: compare, propose, refused commit, approve, shock ->
    STALE, re-plan, approve, commit, re-commit (ALREADY_APPLIED), audit."""
    e = get_engine()

    def step(title: str) -> None:
        print(f"\n=== {title} ===")

    step("1. Same hard rules, different policies")
    cmd_compare(a)
    step("2. Propose (nothing moves yet)")
    p = e.propose(actor="demo")
    print(f"plan {p['plan_id']}  code {p['approval_code']}  units {p['total_units']}")
    step("3. Commit without a human approval -> refused")
    print(e.commit(p["plan_id"], p["approval_code"], actor="demo")["sentence"])
    step("4. Human approves; meanwhile a truck arrives short -> commit refused as STALE")
    e.approve(p["plan_id"], "demo-human")
    lot = max(e.state()["lots"], key=lambda x: x["quantity"])["lot_id"]
    print("field update:", e.store.record_field_update(lot, -10, "field:demo", "truck arrived short"))
    print(e.commit(p["plan_id"], p["approval_code"], actor="demo")["sentence"])
    step("5. Re-plan on fresh data, human approves, commit -> applied exactly once")
    p2 = e.propose(actor="demo")
    print(f"plan {p2['plan_id']}  code {p2['approval_code']}  units {p2['total_units']}")
    e.approve(p2["plan_id"], "demo-human")
    print(e.commit(p2["plan_id"], p2["approval_code"], actor="demo")["sentence"])
    print(e.commit(p2["plan_id"], p2["approval_code"], actor="demo")["sentence"])
    step("6. Audit and notifications")
    print(e.verify_audit()["sentence"])
    print(f"outbox: {len(e.store.outbox())} notifications queued")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="allocator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("validate"); s.add_argument("--pack", default=None); s.add_argument("--with-store", action="store_true")  # noqa: E702
    sub.add_parser("seed")
    sub.add_parser("reset")
    s = sub.add_parser("rank"); s.add_argument("--top", type=int, default=None)  # noqa: E702
    sub.add_parser("compare")
    s = sub.add_parser("dial"); s.add_argument("--alphas", default="")  # noqa: E702
    s = sub.add_parser("propose"); s.add_argument("--strategy", default=None); s.add_argument("--by", default="operator")  # noqa: E702
    s = sub.add_parser("approve"); s.add_argument("plan"); s.add_argument("--by", required=True)  # noqa: E702
    s = sub.add_parser("commit"); s.add_argument("plan"); s.add_argument("code"); s.add_argument("--by", default="operator")  # noqa: E702
    s = sub.add_parser("reject"); s.add_argument("plan"); s.add_argument("--reason", required=True); s.add_argument("--by", default="operator")  # noqa: E702
    s = sub.add_parser("shock"); s.add_argument("lot"); s.add_argument("delta", type=int); s.add_argument("--note", default="field update"); s.add_argument("--by", default="field")  # noqa: E702
    s = sub.add_parser("conditions"); s.add_argument("--file", default=None)  # noqa: E702
    sub.add_parser("verify-audit")
    sub.add_parser("outbox")
    sub.add_parser("state")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "validate" and a.pack is None:
        import os
        a.pack = os.environ.get("PACK", "food_equity")
    handlers = {"validate": cmd_validate, "seed": cmd_seed, "reset": cmd_seed, "rank": cmd_rank,
                "compare": cmd_compare, "dial": cmd_dial, "propose": cmd_propose, "approve": cmd_approve,
                "commit": cmd_commit, "reject": cmd_reject, "shock": cmd_shock, "conditions": cmd_conditions,
                "verify-audit": cmd_verify, "outbox": cmd_outbox, "state": cmd_state, "demo": cmd_demo}
    try:
        return handlers[a.cmd](a)
    except (EngineError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
