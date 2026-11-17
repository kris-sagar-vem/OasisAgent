"""Specialist agents used as tools (AgentTool) by the root agent.

Rule (PREP_GUIDE 5.5): a specialist is an *agent* only when it needs model judgement over
several tools; deterministic work stays a plain tool. Neither specialist can write anything.
"""
from __future__ import annotations

import logging
import os

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from . import grounding, prompts, tools

log = logging.getLogger(__name__)


def model_name(env: str = "ROOT_MODEL") -> str:
    return os.environ.get(env) or os.environ.get("ROOT_MODEL") or "gemini-3.5-flash"


def reviewer_agent() -> LlmAgent:
    return LlmAgent(
        name="policy_reviewer",
        model=model_name("REVIEWER_MODEL"),
        description="Critiques fairness trade-offs of allocation methods. Read-only; cannot approve.",
        instruction=prompts.REVIEWER_INSTRUCTION,
        tools=[FunctionTool(f) for f in tools.REVIEWER_TOOLS],
        after_tool_callback=grounding.record_tool_numbers,
    )


def analyst_agent() -> LlmAgent | None:
    """BigQuery read-only analyst. Only built when STORE=bigquery and BQ_DATASET is set."""
    if os.environ.get("STORE", "memory").lower() != "bigquery" or not os.environ.get("BQ_DATASET"):
        return None
    try:
        import google.auth
        from google.adk.integrations.bigquery import BigQueryCredentialsConfig, BigQueryToolset
        from google.adk.integrations.bigquery.config import BigQueryToolConfig, WriteMode

        creds, project = google.auth.default()
        toolset = BigQueryToolset(
            credentials_config=BigQueryCredentialsConfig(credentials=creds),
            bigquery_tool_config=BigQueryToolConfig(
                write_mode=WriteMode.BLOCKED,
                maximum_bytes_billed=int(os.environ.get("BQ_MAX_BYTES", str(10 * 1024**3))),
                max_query_result_rows=200,
                application_name="allocator-analyst",
                default_project_id=os.environ.get("GOOGLE_CLOUD_PROJECT") or project,
                default_dataset_id=os.environ["BQ_DATASET"],
                location=os.environ.get("BQ_LOCATION"),
            ),
        )
    except Exception as e:  # noqa: BLE001 - analyst is optional; never break the root agent
        log.warning("data_analyst disabled: %s", e)
        return None
    return LlmAgent(
        name="data_analyst",
        model=model_name("ANALYST_MODEL"),
        description="Answers questions about the raw dataset with read-only BigQuery SELECT queries.",
        instruction=prompts.ANALYST_INSTRUCTION,
        tools=[toolset],
        after_tool_callback=grounding.record_tool_numbers,
    )


def specialist_tools() -> list[AgentTool]:
    out = [AgentTool(reviewer_agent())]
    analyst = analyst_agent()
    if analyst is not None:
        out.append(AgentTool(analyst))
    return out
