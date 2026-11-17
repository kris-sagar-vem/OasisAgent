"""Domain objects. Pure data; every number shown to a user is computed from these."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pack import Pack

COLD_HANDLING = frozenset({"chilled", "frozen", "ultra_cold"})


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    need: int
    vulnerability: float
    deadline_h: float | None = None
    capacity: int | None = None
    handling: frozenset[str] = frozenset({"ambient"})
    zone_type: str = "default"
    lat: float | None = None
    lon: float | None = None
    request_seq: int = 0
    served: int = 0
    version: int = 1
    attrs: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def open_need(self) -> int:
        """Need not yet served this cycle."""
        return max(0, self.need - self.served)

    @property
    def room(self) -> int:
        """Units the zone can still receive: open need, capped by remaining capacity."""
        room = self.open_need
        if self.capacity is not None:
            room = min(room, max(0, self.capacity - self.served))
        return room


@dataclass(frozen=True)
class Lot:
    lot_id: str
    quantity: int  # remaining units
    category: str = "default"
    expiry_h: float | None = None  # None = does not expire
    handling: str = "ambient"
    origin_id: str = "DEPOT"
    lat: float | None = None
    lon: float | None = None
    version: int = 1
    quantity_total: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def perishable(self) -> bool:
        return self.expiry_h is not None


@dataclass(frozen=True)
class Conditions:
    """External conditions (for example weather) entering as data, never as instructions."""

    values: dict[str, dict[str, float]] = field(default_factory=dict, hash=False)
    version: int = 0
    observed_at: str | None = None
    source: str = "none"


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str = "OK"
    detail: str = ""


OK = Verdict(True)


@dataclass(frozen=True)
class Line:
    lot_id: str
    zone_id: str
    units: int
    sentence: str = ""


@dataclass(frozen=True)
class Rejection:
    lot_id: str  # "*" when the reason is not lot-specific (for example CAPACITY)
    zone_id: str
    code: str
    detail: str = ""


@dataclass
class Plan:
    strategy: str
    params: dict[str, Any]
    lines: list[Line]
    rejections: list[Rejection]
    flags: list[str] = field(default_factory=list)

    def by_zone(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ln in self.lines:
            out[ln.zone_id] = out.get(ln.zone_id, 0) + ln.units
        return out

    def by_lot(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ln in self.lines:
            out[ln.lot_id] = out.get(ln.lot_id, 0) + ln.units
        return out

    def canonical_lines(self) -> list[list[Any]]:
        return sorted([ln.lot_id, ln.zone_id, ln.units] for ln in self.lines)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Problem:
    pack: Pack
    zones: tuple[Zone, ...]
    lots: tuple[Lot, ...]
    denied: frozenset[tuple[str, str]] = frozenset()  # (category, zone_type) pairs not allowed
    travel: dict[tuple[str, str], float] = field(default_factory=dict, hash=False)
    conditions: Conditions = field(default_factory=Conditions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_zones", {z.zone_id: z for z in self.zones})
        object.__setattr__(self, "_lots", {lt.lot_id: lt for lt in self.lots})

    def zone(self, zone_id: str) -> Zone:
        try:
            return self._zones[zone_id]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown zone_id: {zone_id}") from None

    def lot(self, lot_id: str) -> Lot:
        try:
            return self._lots[lot_id]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown lot_id: {lot_id}") from None

    def travel_h(self, lot: Lot, zone: Zone) -> float:
        key = (lot.origin_id, zone.zone_id)
        if key in self.travel:
            return self.travel[key]
        if None in (lot.lat, lot.lon, zone.lat, zone.lon):
            raise ValueError(f"no travel time for {key} and no coordinates for an estimate")
        km = haversine_km(lot.lat, lot.lon, zone.lat, zone.lon)  # type: ignore[arg-type]
        return round(km / self.pack.travel.speed_kmh, 3)

    def versions(self) -> dict[str, Any]:
        return {
            "lots": {lt.lot_id: lt.version for lt in self.lots},
            "zones": {z.zone_id: z.version for z in self.zones},
            "conditions": self.conditions.version,
        }
