"""Conditions providers (STRETCH-3). External data enters as numbers, never as instructions.

provider: none   -> nothing to refresh
provider: static -> seed table only (refresh is a no-op; `make conditions` can ingest a file)
provider: a2a    -> ask the remote conditions agent over A2A. [Unverified] client wiring:
                    Spike 10 must confirm the RemoteA2aAgent call pattern; until then this
                    falls back to the agent's plain JSON endpoint if CONDITIONS_JSON_URL is set.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from .engine.service import Engine

log = logging.getLogger(__name__)


def refresh(engine: Engine) -> dict[str, Any]:
    cfg = engine.pack.conditions
    if cfg.provider in ("none", "static"):
        return {"refreshed": False, "provider": cfg.provider,
                "sentence": f"Conditions provider is '{cfg.provider}'; using stored conditions."}
    url = os.environ.get("CONDITIONS_JSON_URL")
    if not url:
        return {"refreshed": False, "provider": cfg.provider,
                "sentence": "A2A conditions agent not reachable (CONDITIONS_JSON_URL unset); using stored conditions."}
    try:
        zone_ids = [z.zone_id for z in engine.store.load_problem().zones]
        req = urllib.request.Request(url, data=json.dumps({"zone_ids": zone_ids}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310 - URL comes from operator config
            body = json.loads(r.read())
        meta = engine.ingest_conditions(body.get("rows", []), source=body.get("source", "a2a"),
                                        observed_at=body.get("observed_at"))
        return {"refreshed": True, "provider": cfg.provider, "version": meta["version"],
                "sentence": f"Conditions refreshed from {meta['source']} (version {meta['version']})."}
    except Exception as e:  # noqa: BLE001 - a dead feed must not stop allocation
        log.warning("conditions refresh failed: %s", e)
        return {"refreshed": False, "provider": cfg.provider,
                "sentence": "Conditions refresh failed; using stored conditions (staleness is checked at commit)."}
