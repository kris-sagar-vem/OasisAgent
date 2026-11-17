"""Guardian Workflow (STRETCH-2) driven offline through ADK's InMemoryRunner. No model calls."""
from __future__ import annotations

import asyncio

import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types

from allocator.engine.service import get_engine, reset_engines
from guardian.agent import build_guardian, parse_reply


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("PACK", "food_equity")
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("APPROVAL_MODE", "out_of_band")
    reset_engines()
    yield
    reset_engines()


def _request(events):
    for ev in events:
        for p in (ev.content.parts if ev.content else []) or []:
            fc = getattr(p, "function_call", None)
            if fc and fc.name == "adk_request_input":
                return fc
    return None


async def _send(runner, sid, msg):
    return [ev async for ev in runner.run_async(user_id="u", session_id=sid, new_message=msg)]


def _reply(fc, **resp):
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=fc.id, name="adk_request_input", response=resp))])


async def _drive(decisions, shock=False):
    runner = InMemoryRunner(agent=build_guardian(), app_name="guardian")
    s = await runner.session_service.create_session(app_name="guardian", user_id="u")
    evs = await _send(runner, s.id, types.Content(role="user", parts=[types.Part(text="go")]))
    pauses = 0
    for i, decision in enumerate(decisions):
        fc = _request(evs)
        if fc is None:
            break
        pauses += 1
        plan_id = fc.args["payload"]["plan_id"]
        code = decision.get("approval_code", get_engine().get_plan(plan_id)["approval_code"])
        if shock and i == 0:
            get_engine().store.record_field_update("L01", -100, "field", "truck damaged")
        evs = await _send(runner, s.id, _reply(fc, **{**decision, "approval_code": code}))
    outputs = [ev.output for ev in evs if getattr(ev, "output", None)]
    return pauses, outputs[-1] if outputs else None


def test_guardian_happy_path():
    pauses, out = asyncio.run(_drive([{"decision": "approve", "approver": "alice"}]))
    assert pauses == 1 and out["status"] == "APPLIED" and out["attempts"] == 1
    assert [r["event_type"] for r in get_engine().store.audit_rows()] == ["PROPOSED", "APPROVED", "COMMITTED"]


def test_guardian_replans_on_stale():
    pauses, out = asyncio.run(_drive([{"decision": "approve", "approver": "alice"}] * 2, shock=True))
    assert pauses == 2 and out["status"] == "APPLIED" and out["attempts"] == 2
    events = [r["event_type"] for r in get_engine().store.audit_rows()]
    assert events == ["PROPOSED", "FIELD_UPDATE", "APPROVED", "STALE", "PROPOSED", "APPROVED", "COMMITTED"]
    assert get_engine().verify_audit()["valid"]


def test_guardian_reject_moves_nothing():
    before = get_engine().state()["lots"]
    pauses, out = asyncio.run(_drive([{"decision": "reject", "approver": "bob", "reason": "check Z08"}]))
    assert out["status"] == "REJECTED"
    assert get_engine().state()["lots"] == before


def test_guardian_wrong_code_is_refused():
    pauses, out = asyncio.run(_drive([{"decision": "approve", "approver": "eve", "approval_code": "ZZZZZZ"}]))
    assert out["status"] == "REJECTED"
    assert "COMMITTED" not in [r["event_type"] for r in get_engine().store.audit_rows()]


def test_parse_reply_text_forms():
    assert parse_reply("approve alice K3F9QX").approval_code == "K3F9QX"
    assert parse_reply('{"decision": "reject", "approver": "bob"}').decision == "reject"
    assert parse_reply("").decision == "reject"
