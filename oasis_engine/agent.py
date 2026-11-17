"""ADK entry point: `adk web` / `adk run allocator` load `root_agent` from here.

Configuration (env vars): PACK, STORE, APPROVAL_MODE, ROOT_MODEL, NUMBER_CHECK, BQ_DATASET,
MODEL_ARMOR_TEMPLATE. Swapping the problem = swapping PACK; no code change.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from . import grounding, prompts, safety, subagents, tools
from .engine.service import get_engine


def build_root_agent() -> LlmAgent:
    engine = get_engine()
    confirm = engine.approval_mode == "confirm"
    agent_tools = [FunctionTool(f) for f in tools.READ_TOOLS]
    agent_tools += [FunctionTool(tools.propose_plan), FunctionTool(tools.reject_plan)]
    # Only commit_plan uses ADK confirmation, and only in confirm mode (PREP_GUIDE 4.9).
    agent_tools.append(FunctionTool(tools.commit_plan, require_confirmation=confirm))
    agent_tools += subagents.specialist_tools()
    instruction = prompts.ROOT_INSTRUCTION.format(
        pack_title=engine.pack.meta.title,
        unit_word=engine.pack.word("unit", "units"),
    )
    return LlmAgent(
        name="incident_commander",
        model=subagents.model_name(),
        description="Allocates scarce resources fairly with human approval before anything moves.",
        instruction=instruction,
        tools=agent_tools,
        before_model_callback=safety.screen_user_input,
        after_tool_callback=grounding.record_tool_numbers,
        after_model_callback=grounding.check_answer_numbers,
    )


os.environ.setdefault("PACK", "dallas_heat")
root_agent = build_root_agent()

