"""Live BigQuery contract test (opt-in): make test-cloud. Uses a throwaway dataset.

Runs the same scenarios as the in-memory contract: refuse without approval, apply once,
ALREADY_APPLIED, STALE after a field update, audit chain valid.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_BQ") != "1", reason="set RUN_BQ=1 (make test-cloud)")


@pytest.fixture(scope="module")
def engine():
    from allocator.engine.pack import load_pack
    from allocator.engine.service import Engine
    from allocator.stores.bigquery import BigQueryStore
    from allocator.stores.bq_seed import seed
    pack = load_pack(os.environ.get("PACK", "food_equity"))
    store = BigQueryStore(pack, dataset=os.environ.get("BQ_DATASET", "allocator_test"))
    seed(store)
    return Engine(pack, store, "out_of_band")


def test_bq_contract(engine):
    assert engine.validate() == []
    p = engine.propose(actor="test")
    assert engine.commit(p["plan_id"], p["approval_code"])["status"] == "REJECTED"
    engine.approve(p["plan_id"], "tester")
    assert engine.commit(p["plan_id"], p["approval_code"])["status"] == "APPLIED"
    assert engine.commit(p["plan_id"], p["approval_code"])["status"] == "ALREADY_APPLIED"
    assert engine.verify_audit()["valid"]


def test_bq_stale(engine):
    p = engine.propose(actor="test")
    engine.approve(p["plan_id"], "tester")
    lot = max(engine.state()["lots"], key=lambda x: x["quantity"])["lot_id"]
    engine.store.record_field_update(lot, -1, "test", "shock")
    assert engine.commit(p["plan_id"], p["approval_code"])["status"] == "STALE"
    assert engine.verify_audit()["valid"]
