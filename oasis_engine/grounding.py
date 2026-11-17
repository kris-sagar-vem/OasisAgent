"""Grounding guard (R6, S1): every number in the agent's answer must come from this turn's
tool results or the user's own message.

after_tool_callback records numbers from each tool result; after_model_callback checks the
final text. Mode via NUMBER_CHECK=flag (default: append a warning) or block (replace answer).
Pure functions are unit-tested without a model.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Iterable

# A number not glued to letters/IDs: skips Z01, P-B02243, VG7G72, L2.
_NUM = re.compile(r"(?<![A-Za-z0-9_\-.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(%?)(?![A-Za-z0-9_])")
SMALL_INT_EXEMPT = 10  # list numbering, "two options", "#3" ranks are allowed

_TURN: dict[str, set[float]] = {}
_LOCK = threading.Lock()


def extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    out = []
    for m in _NUM.finditer(text or ""):
        raw, pct = m.group(1), bool(m.group(2))
        try:
            out.append((m.group(0), float(raw.replace(",", "")), pct))
        except ValueError:
            continue
    return out


def _walk(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for _, v, _ in extract_numbers(obj):
            yield v
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk(v)


def allowed_forms(values: Iterable[float]) -> set[float]:
    """Each source number also allows its common renderings: rounded, percent, fraction."""
    out: set[float] = set()
    for v in values:
        for x in (v, v * 100, v / 100):
            out.add(round(x, 4))
            for nd in (0, 1, 2):
                out.add(round(x, nd))
    return out


def unverified_numbers(answer: str, allowed: set[float]) -> list[str]:
    bad = []
    for token, v, pct in extract_numbers(answer):
        if v.is_integer() and 0 <= v <= SMALL_INT_EXEMPT and not pct:
            continue
        cands = {round(v, 4), round(v, 2), round(v, 1), round(v, 0)}
        if pct:
            cands |= {round(v / 100, 4), round(v / 100, 2)}
        if not cands & allowed:
            bad.append(token)
    return bad


# ---- ADK callbacks ------------------------------------------------------------------
def _turn_key(ctx: Any) -> str:
    return str(getattr(ctx, "invocation_id", None) or "default")


def record_tool_numbers(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any) -> None:
    """after_tool_callback: remember every number a tool returned in this invocation."""
    nums = set(_walk(tool_response)) | set(_walk(args))
    with _LOCK:
        _TURN.setdefault(_turn_key(tool_context), set()).update(nums)
    return None


def _user_numbers(ctx: Any) -> set[float]:
    content = getattr(ctx, "user_content", None)
    texts = [getattr(p, "text", "") or "" for p in (getattr(content, "parts", None) or [])]
    return set(_walk(" ".join(texts)))


def check_answer_numbers(callback_context: Any, llm_response: Any) -> Any:
    """after_model_callback: flag or block numbers that no tool produced this turn."""
    if getattr(llm_response, "partial", False):
        return None
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts or any(getattr(p, "function_call", None) for p in parts):
        return None
    text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text.strip():
        return None
    with _LOCK:
        seen = set(_TURN.get(_turn_key(callback_context), set()))
    allowed = allowed_forms(seen | _user_numbers(callback_context))
    bad = unverified_numbers(text, allowed)
    try:
        callback_context.state["number_check"] = {"unverified": bad, "ok": not bad}
    except Exception:  # noqa: BLE001 - state may be read-only in some contexts
        pass
    if not bad:
        return None
    from google.genai import types
    mode = os.environ.get("NUMBER_CHECK", "flag").lower()
    if mode == "block":
        new_text = ("I drafted an answer containing numbers that no tool produced "
                    f"({', '.join(bad)}), so I withheld it. Ask again and I will answer only from tool results.")
    else:
        new_text = (text + f"\n\n[Number check: {', '.join(bad)} did not come from a tool in this turn; "
                    "treat as unverified.]")
    llm_response.content = types.Content(role=content.role or "model", parts=[types.Part(text=new_text)])
    return llm_response


def reset_turn(key: str | None = None) -> None:
    with _LOCK:
        if key is None:
            _TURN.clear()
        else:
            _TURN.pop(key, None)
