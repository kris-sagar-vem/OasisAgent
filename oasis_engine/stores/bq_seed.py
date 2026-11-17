"""Create the dataset, system tables and load a pack's seed CSVs (load jobs, never streaming)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..engine.gateway import utcnow

if TYPE_CHECKING:
    from .bigquery import BigQueryStore

DDL = Path(__file__).resolve().parents[2] / "sql" / "ddl.sql"


def seed(store: "BigQueryStore") -> dict[str, int]:
    bq, client = store.bq, store.client
    ds_id = f"{store.project}.{store.dataset}"
    ds = bq.Dataset(ds_id)
    ds.location = store.location
    client.create_dataset(ds, exists_ok=True)
    ddl = DDL.read_text().replace("${DATASET}", ds_id)
    client.query(ddl).result()
    for t in ("proposals", "approvals", "dispatch_log", "outbox", "audit_log", "conditions_meta"):
        client.query(f"TRUNCATE TABLE {store.T[t]}").result()
    seed_dir = Path(store.pack.source_dir or ".") / "seed"
    loaded = {}
    for ent, m in store.pack.entities.items():
        path = seed_dir / f"{m.table}.csv"
        if not path.exists():
            continue
        cfg = bq.LoadJobConfig(source_format=bq.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
                               write_disposition=bq.WriteDisposition.WRITE_TRUNCATE)
        with path.open("rb") as f:
            job = client.load_table_from_file(f, f"{ds_id}.{m.table}", job_config=cfg)
        job.result()
        loaded[m.table] = job.output_rows or 0
    if "conditions" in store.pack.entities:
        store._run(f"INSERT INTO {store.T['conditions_meta']} (version, observed_at, source) "
                   f"VALUES (1, TIMESTAMP(@obs), 'seed')", {"obs": utcnow().isoformat()})
    return loaded
