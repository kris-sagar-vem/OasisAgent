"""Problem packs: configuration, not code. Load, validate and hash a pack."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .registry import CONSTRAINTS, METRICS, SCORING_TERMS, STRATEGIES

PACKS_DIR = Path(__file__).resolve().parent.parent / "packs"
SAFE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

ZONE_REQUIRED = ("zone_id", "need", "vulnerability")
ZONE_FIELDS = ZONE_REQUIRED + (
    "name", "deadline_h", "capacity", "handling", "zone_type", "lat", "lon",
    "request_seq", "served", "version",
)
LOT_REQUIRED = ("lot_id", "quantity")
LOT_FIELDS = LOT_REQUIRED + (
    "category", "expiry_h", "handling", "origin_id", "lat", "lon", "version", "quantity_total",
)
FIXED_FIELDS = {
    "eligibility": ("category", "zone_type", "allowed"),
    "travel": ("origin_id", "zone_id", "hours"),
    "conditions": ("zone_id", "factor", "value"),
}
ENTITY_FIELDS = {"zones": ZONE_FIELDS, "lots": LOT_FIELDS, **FIXED_FIELDS}
REQUIRED_FIELDS = {"zones": ZONE_REQUIRED, "lots": LOT_REQUIRED, **FIXED_FIELDS}

# Tokens that mark an individual protected attribute (S5). Aggregate indices are fine.
DEFAULT_BANNED_TOKENS = [
    "race", "ethnicity", "religion", "gender", "sex", "nationality",
    "citizenship", "immigration", "ssn",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Meta(_Strict):
    id: str
    title: str
    track: str = "generic"
    version: str = "1"
    disclaimer: str = "Synthetic data. Weights are policy choices, not facts."


class EntityMap(_Strict):
    table: str
    columns: dict[str, str] = Field(default_factory=dict)  # canonical field -> source column
    extra: list[str] = Field(default_factory=list)  # source columns copied into attrs

    def col(self, canonical: str) -> str:
        return self.columns.get(canonical, canonical)


class TermRef(_Strict):
    term: str
    weight: float
    params: dict[str, Any] = Field(default_factory=dict)


class ConstraintRef(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class StrategyCfg(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    baselines: list[str] = Field(default_factory=lambda: ["nearest_first"])


class TravelCfg(_Strict):
    mode: Literal["table", "haversine"] = "table"
    speed_kmh: float = 40.0


class ConditionsCfg(_Strict):
    provider: Literal["none", "static", "a2a"] = "none"
    max_age_minutes: float | None = None  # None = no age check at commit
    a2a_url: str | None = None


class MetricsCfg(_Strict):
    headline: str = "vwu"
    show: list[str] = Field(default_factory=lambda: [
        "delivered", "spoiled", "unmet", "vwu", "worst_off_coverage", "most_vulnerable_coverage",
    ])


class Pack(_Strict):
    meta: Meta
    entities: dict[str, EntityMap]
    scoring: list[TermRef]
    constraints: list[ConstraintRef]
    strategy: StrategyCfg
    travel: TravelCfg = Field(default_factory=TravelCfg)
    conditions: ConditionsCfg = Field(default_factory=ConditionsCfg)
    vocabulary: dict[str, str] = Field(default_factory=dict)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)
    banned_tokens: list[str] = Field(default_factory=lambda: list(DEFAULT_BANNED_TOKENS))

    _source_dir: Path | None = PrivateAttr(default=None)

    @property
    def source_dir(self) -> Path | None:
        return self._source_dir

    def weights(self) -> dict[str, float]:
        return {t.term: t.weight for t in self.scoring}

    def pack_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_weights(self, weights: dict[str, float]) -> "Pack":
        """Copy with new weights for existing terms, renormalised to sum to 1."""
        unknown = set(weights) - {t.term for t in self.scoring}
        if unknown:
            raise ValueError(f"unknown scoring terms: {sorted(unknown)}")
        merged = {t.term: float(weights.get(t.term, t.weight)) for t in self.scoring}
        if any(w < 0 for w in merged.values()):
            raise ValueError("weights must be non-negative")
        total = sum(merged.values())
        if total <= 0:
            raise ValueError("weights must not all be zero")
        scoring = [t.model_copy(update={"weight": merged[t.term] / total}) for t in self.scoring]
        new = self.model_copy(update={"scoring": scoring})
        new._source_dir = self._source_dir
        return new

    def word(self, key: str, default: str) -> str:
        return self.vocabulary.get(key, default)


def load_pack(name_or_path: str | Path) -> Pack:
    path = Path(name_or_path)
    if not path.exists():
        path = PACKS_DIR / str(name_or_path)
    if path.is_dir():
        path = path / "pack.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"pack not found: {name_or_path}")
    data = yaml.safe_load(path.read_text())
    pack = Pack.model_validate(data)
    pack._source_dir = path.parent
    return pack


def _tokens(column: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", column.lower()) if t}


def validate_pack(pack: Pack, available: dict[str, set[str]] | None = None) -> list[str]:
    """Return a list of problems; empty means valid.

    available: optional {entity_name: set of source column names} from the store, used to
    check that referenced tables and columns exist.
    """
    errors: list[str] = []
    total = sum(t.weight for t in pack.scoring)
    if abs(total - 1.0) > 1e-9:
        errors.append(f"scoring weights sum to {total:.9f}, not 1")
    for t in pack.scoring:
        if t.term not in SCORING_TERMS:
            errors.append(f"unregistered scoring term: {t.term}")
        if not 0.0 <= t.weight <= 1.0:
            errors.append(f"weight for {t.term} must be in [0, 1]")
    for c in pack.constraints:
        if c.name not in CONSTRAINTS:
            errors.append(f"unregistered constraint: {c.name}")
    for s in [pack.strategy.name, *pack.strategy.baselines]:
        if s not in STRATEGIES:
            errors.append(f"unregistered strategy: {s}")
    for m in [pack.metrics.headline, *pack.metrics.show]:
        if m not in METRICS:
            errors.append(f"unregistered metric: {m}")
    if not SAFE_ID.match(pack.meta.id):
        errors.append(f"unsafe pack id: {pack.meta.id!r}")

    for required in ("zones", "lots"):
        if required not in pack.entities:
            errors.append(f"missing entity: {required}")
    if pack.travel.mode == "table" and "travel" not in pack.entities:
        errors.append("travel.mode is 'table' but no 'travel' entity is mapped")
    if pack.conditions.provider == "static" and "conditions" not in pack.entities:
        errors.append("conditions.provider is 'static' but no 'conditions' entity is mapped")
    if pack.conditions.provider == "a2a" and not pack.conditions.a2a_url:
        errors.append("conditions.provider is 'a2a' but a2a_url is not set")
    if any(t.term == "condition" for t in pack.scoring) and pack.conditions.provider == "none":
        errors.append("scoring uses 'condition' but conditions.provider is 'none'")

    used_columns: set[str] = set()
    for name, ent in pack.entities.items():
        if name not in ENTITY_FIELDS:
            errors.append(f"unknown entity: {name}")
            continue
        if not SAFE_ID.match(ent.table):
            errors.append(f"unsafe table name: {ent.table!r}")
        for canonical, col in ent.columns.items():
            if canonical not in ENTITY_FIELDS[name]:
                errors.append(f"{name}: unknown field {canonical!r}")
            if not SAFE_ID.match(col):
                errors.append(f"{name}: unsafe column name {col!r}")
        for col in ent.extra:
            if not SAFE_ID.match(col):
                errors.append(f"{name}: unsafe extra column {col!r}")
        if name == "zones":
            used_columns |= {ent.col(f) for f in ZONE_FIELDS} | set(ent.extra)
        if available is not None:
            cols = available.get(name)
            if cols is None:
                errors.append(f"table for entity {name!r} ({ent.table}) not found")
                continue
            needed = {ent.col(f) for f in REQUIRED_FIELDS[name]} | set(ent.columns.values()) | set(ent.extra)
            for col in sorted(needed - cols):
                errors.append(f"{name}: column {col!r} not found in {ent.table}")

    for t in pack.scoring:
        col = t.params.get("column")
        if col:
            used_columns.add(str(col))
            if "zones" in pack.entities and col not in pack.entities["zones"].extra:
                errors.append(f"term {t.term}: column {col!r} must be listed in zones.extra")
    banned = {b.lower() for b in pack.banned_tokens}
    for col in sorted(used_columns):
        hit = _tokens(col) & banned
        if hit:
            errors.append(f"column {col!r} looks like a protected attribute ({', '.join(sorted(hit))}); "
                          "use an aggregate index instead (S5)")
    return errors
