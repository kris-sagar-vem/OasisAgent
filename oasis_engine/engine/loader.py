"""Turn raw rows (CSV strings or BigQuery values) into a Problem, using the pack's column map."""
from __future__ import annotations

from typing import Any, Iterable

from .model import Conditions, Lot, Problem, Zone
from .pack import Pack


def _f(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)


def _i(v: Any, default: int | None) -> int | None:
    return default if v is None or v == "" else int(float(v))


def _s(v: Any, default: str) -> str:
    return default if v is None or v == "" else str(v)


def _set(v: Any, default: frozenset[str]) -> frozenset[str]:
    if v is None or v == "":
        return default
    if isinstance(v, (list, tuple, set, frozenset)):
        return frozenset(str(x).strip() for x in v if str(x).strip())
    return frozenset(x.strip() for x in str(v).split("|") if x.strip())


def _b(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def zones_from_rows(pack: Pack, rows: Iterable[dict[str, Any]]) -> list[Zone]:
    m = pack.entities["zones"]
    out = []
    for r in rows:
        g = lambda f: r.get(m.col(f))  # noqa: E731
        zid = str(g("zone_id"))
        out.append(Zone(
            zone_id=zid,
            name=_s(g("name"), zid),
            need=int(float(g("need"))),
            vulnerability=float(g("vulnerability")),
            deadline_h=_f(g("deadline_h")),
            capacity=_i(g("capacity"), None),
            handling=_set(g("handling"), frozenset({"ambient"})),
            zone_type=_s(g("zone_type"), "default"),
            lat=_f(g("lat")),
            lon=_f(g("lon")),
            request_seq=_i(g("request_seq"), 0) or 0,
            served=_i(g("served"), 0) or 0,
            version=_i(g("version"), 1) or 1,
            attrs={c: r.get(c) for c in m.extra},
        ))
    return out


def lots_from_rows(pack: Pack, rows: Iterable[dict[str, Any]]) -> list[Lot]:
    m = pack.entities["lots"]
    out = []
    for r in rows:
        g = lambda f: r.get(m.col(f))  # noqa: E731
        out.append(Lot(
            lot_id=str(g("lot_id")),
            quantity=int(float(g("quantity"))),
            category=_s(g("category"), "default"),
            expiry_h=_f(g("expiry_h")),
            handling=_s(g("handling"), "ambient"),
            origin_id=_s(g("origin_id"), "DEPOT"),
            lat=_f(g("lat")),
            lon=_f(g("lon")),
            version=_i(g("version"), 1) or 1,
            quantity_total=_i(g("quantity_total"), None),
            attrs={c: r.get(c) for c in m.extra},
        ))
    return out


def build_problem(
    pack: Pack,
    tables: dict[str, list[dict[str, Any]]],
    conditions_meta: dict[str, Any] | None = None,
) -> Problem:
    zones = sorted(zones_from_rows(pack, tables.get("zones", [])), key=lambda z: z.zone_id)
    lots = sorted(lots_from_rows(pack, tables.get("lots", [])), key=lambda lt: lt.lot_id)

    denied: set[tuple[str, str]] = set()
    if "eligibility" in pack.entities:
        m = pack.entities["eligibility"]
        for r in tables.get("eligibility", []):
            if not _b(r.get(m.col("allowed"))):
                denied.add((str(r.get(m.col("category"))), str(r.get(m.col("zone_type")))))

    travel: dict[tuple[str, str], float] = {}
    if "travel" in pack.entities:
        m = pack.entities["travel"]
        for r in tables.get("travel", []):
            travel[(str(r.get(m.col("origin_id"))), str(r.get(m.col("zone_id"))))] = float(r.get(m.col("hours")))

    values: dict[str, dict[str, float]] = {}
    if "conditions" in pack.entities:
        m = pack.entities["conditions"]
        for r in tables.get("conditions", []):
            values.setdefault(str(r.get(m.col("zone_id"))), {})[str(r.get(m.col("factor")))] = float(
                r.get(m.col("value")))
    meta = conditions_meta or {}
    conditions = Conditions(
        values=values,
        version=int(meta.get("version", 0)),
        observed_at=meta.get("observed_at"),
        source=str(meta.get("source", pack.conditions.provider)),
    )
    return Problem(pack=pack, zones=tuple(zones), lots=tuple(lots), denied=frozenset(denied),
                   travel=travel, conditions=conditions)
