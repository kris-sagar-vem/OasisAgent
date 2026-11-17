"""Static security lints (S3, S4, plus 'no second write path'). Fast, offline, no imports of app code."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = [p for d in ("allocator", "guardian", "conditions_agent") for p in (ROOT / d).rglob("*.py")]


def _text(p: Path) -> str:
    return p.read_text()


def test_no_streaming_inserts():
    for p in SRC:
        assert "insert_rows" not in _text(p), f"{p}: streaming insert (use DML or load jobs)"


def test_no_credentials_in_code():
    rx = re.compile(r"(AIza[0-9A-Za-z_\-]{30,}|-----BEGIN (RSA )?PRIVATE KEY|\"private_key\"\s*:)")
    for p in [*SRC, *(ROOT / "allocator" / "packs").rglob("*.yaml")]:
        assert not rx.search(_text(p)), f"{p}: looks like a credential"


def test_no_value_interpolation_in_sql():
    """In SQL f-strings, every {placeholder} must be an identifier map lookup (T/Z/L/self.T/...)
    or a known-safe local, never a raw value. Values go through @params."""
    # T/Z/L values are validated against SAFE_ID when BigQueryStore is constructed, so any
    # key lookup into them is an identifier, never a value.
    allowed = re.compile(r"^(self\.|store\.)?(T|Z|L)\[(['\"][a-z_]+['\"]|[a-z_]+)\]$|"
                         r"^(cz|cf|cv|ds_id)$|^self\.(project|dataset)$")
    sql_kw = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|ASSERT|TRUNCATE)\b")
    for p in (ROOT / "allocator" / "stores").glob("*.py"):
        tree = ast.parse(_text(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                literal = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
                if not sql_kw.search(literal):
                    continue
                for v in node.values:
                    if isinstance(v, ast.FormattedValue):
                        src = ast.unparse(v.value)
                        assert allowed.match(src), f"{p}:{node.lineno}: SQL interpolates {src!r}; use @param"


def test_no_antigravity_imports():
    for p in SRC:
        assert "antigravity" not in _text(p).lower(), f"{p}: Antigravity import"


def test_single_write_path():
    """Only stores mutate state; agent tools reach them only through Engine methods."""
    tools_src = _text(ROOT / "allocator" / "tools.py")
    for forbidden in ("record_approval", "record_field_update", "update_conditions", "ingest_conditions",
                      ".approve(", "store."):
        assert forbidden not in tools_src, f"tools.py must not call {forbidden}"
    guardian_src = _text(ROOT / "guardian" / "agent.py")
    assert "store.commit" not in guardian_src and "get_engine().commit" not in guardian_src.replace("engine.commit", "")
    assert "engine.commit(" in guardian_src  # same gateway as the core agent
