"""Store implementations. Pick one with STORE=memory|bigquery."""
from __future__ import annotations

import os

from ..engine.pack import Pack
from .base import Store


def make_store(pack: Pack, kind: str | None = None) -> Store:
    kind = (kind or os.environ.get("STORE", "memory")).lower()
    if kind == "memory":
        from .memory import InMemoryStore
        return InMemoryStore(pack)
    if kind == "bigquery":
        from .bigquery import BigQueryStore
        return BigQueryStore(pack)
    raise ValueError(f"unknown STORE={kind!r}; use memory or bigquery")
