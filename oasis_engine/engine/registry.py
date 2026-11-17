"""Plug-in registries. Adding a plug-in = one decorated function + one line in pack.yaml.

Interfaces:
  scoring term : fn(zone, problem, params) -> float in [0, 1]
  constraint   : fn(zone, lot, problem, params) -> Verdict
  strategy     : fn(problem, params) -> Plan
  metric       : fn(problem, plan) -> float
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

REASON_CODES = frozenset({
    "COLD_CHAIN", "HANDLING", "EXPIRY", "DEADLINE", "ELIGIBILITY",
    "CAPACITY", "DISTANCE", "SUPPLY_EXHAUSTED",
})


@dataclass(frozen=True)
class TermSpec:
    name: str
    fn: Callable
    label: str


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable
    label: str
    better: str  # "lower" or "higher"


SCORING_TERMS: dict[str, TermSpec] = {}
CONSTRAINTS: dict[str, Callable] = {}
STRATEGIES: dict[str, Callable] = {}
METRICS: dict[str, MetricSpec] = {}


def _register(table: dict, name: str, value: object) -> None:
    if name in table:
        raise ValueError(f"duplicate plug-in name: {name}")
    table[name] = value


def scoring_term(name: str, label: str | None = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        _register(SCORING_TERMS, name, TermSpec(name, fn, label or name.replace("_", " ")))
        return fn
    return deco


def constraint(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _register(CONSTRAINTS, name, fn)
        return fn
    return deco


def strategy(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _register(STRATEGIES, name, fn)
        return fn
    return deco


def metric(name: str, label: str, better: str = "lower") -> Callable:
    if better not in ("lower", "higher"):
        raise ValueError("better must be 'lower' or 'higher'")

    def deco(fn: Callable) -> Callable:
        _register(METRICS, name, MetricSpec(name, fn, label, better))
        return fn
    return deco


_loaded = False


def load_builtins() -> None:
    """Import the built-in plug-in modules once (importing registers them)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import constraints, metrics, scoring, strategies  # noqa: F401
