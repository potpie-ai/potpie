"""
Langfuse integration tools for the Prompt Tuner Agent.

Provides fetch_langfuse_trace and list_langfuse_traces tools
that query the Langfuse REST API to retrieve LLM conversation traces.

Requires env vars:
- LANGFUSE_HOST: e.g. http://langfuse.default.svc.cluster.local:3000
- LANGFUSE_PUBLIC_KEY: Langfuse project public key
- LANGFUSE_SECRET_KEY: Langfuse project secret key
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

_API_TIMEOUT = 30.0


def _langfuse_config() -> Dict[str, str]:
    return {
        "base_url": (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_API_BASE_URL")
            or os.getenv("LANGFUSE_BASE_URL")
            or ""
        )
        .strip()
        .rstrip("/"),
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
    }


def _langfuse_auth(config: Dict[str, str]) -> httpx.BasicAuth:
    return httpx.BasicAuth(
        username=config["public_key"],
        password=config["secret_key"],
    )


# ---------------------------------------------------------------------------
# fetch_langfuse_trace
# ---------------------------------------------------------------------------


class FetchLangfuseTraceInput(BaseModel):
    trace_id: str = Field(..., description="The Langfuse trace ID to fetch.")


class FetchLangfuseTraceTool:
    name = "fetch_langfuse_trace"
    description = """Fetch a complete LLM conversation trace from Langfuse by trace ID.

    Returns the full trace including:
    - The system prompt and user message
    - All tool calls with their arguments and results
    - The assistant's response
    - Latencies and token usage per observation

    Use this when the user provides a Langfuse trace ID to analyze.
    Requires LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars.
    """
    args_schema = FetchLangfuseTraceInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = FetchLangfuseTraceInput(**kwargs)
        config = _langfuse_config()
        if not config["base_url"]:
            return "LANGFUSE_HOST is not configured."

        url = f"{config['base_url']}/api/public/traces/{input_data.trace_id}"
        try:
            with httpx.Client(timeout=_API_TIMEOUT) as client:
                resp = client.get(url, auth=_langfuse_auth(config))
                resp.raise_for_status()
                trace = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Langfuse API error: HTTP {e.response.status_code} - {e.response.text[:500]}"
        except Exception as e:
            return f"Failed to fetch trace: {e}"

        # Fetch observations (tool calls, generations) for this trace
        obs_url = f"{config['base_url']}/api/public/observations?traceId={input_data.trace_id}&limit=100"
        try:
            with httpx.Client(timeout=_API_TIMEOUT) as client:
                obs_resp = client.get(obs_url, auth=_langfuse_auth(config))
                obs_resp.raise_for_status()
                observations = obs_resp.json().get("data", [])
        except Exception as e:
            observations = []
            logger.warning(
                "Failed to fetch observations for trace {}: {}", input_data.trace_id, e
            )

        return _format_trace(trace, observations)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


def _format_trace(trace: Dict[str, Any], observations: List[Dict[str, Any]]) -> str:
    lines = []

    # Trace metadata
    lines.append(f"## Trace: {trace.get('id', '?')}")
    lines.append(f"- Name: {trace.get('name', 'N/A')}")
    lines.append(f"- Created: {trace.get('timestamp', 'N/A')}")
    lines.append(f"- Tags: {trace.get('tags', [])}")
    lines.append(f"- User ID: {trace.get('userId', 'N/A')}")
    lines.append(f"- Session ID: {trace.get('sessionId', 'N/A')}")

    metadata = trace.get("metadata") or {}
    if metadata:
        lines.append(f"- Metadata: {metadata}")

    # Input/Output
    trace_input = trace.get("input")
    trace_output = trace.get("output")
    if trace_input:
        lines.append(
            f"\n### Trace Input\n```\n{_truncate(str(trace_input), 2000)}\n```"
        )
    if trace_output:
        lines.append(
            f"\n### Trace Output\n```\n{_truncate(str(trace_output), 2000)}\n```"
        )

    # Observations (tool calls, generations)
    if observations:
        observations.sort(key=lambda o: o.get("startTime", ""))

        generations = [o for o in observations if o.get("type") == "GENERATION"]
        spans = [o for o in observations if o.get("type") == "SPAN"]

        if generations:
            lines.append(f"\n### LLM Generations ({len(generations)})")
            for i, gen in enumerate(generations, 1):
                lines.append(f"\n#### Generation {i}: {gen.get('name', 'N/A')}")
                lines.append(f"- Model: {gen.get('model', 'N/A')}")
                usage = gen.get("usage") or {}
                if usage:
                    lines.append(
                        f"- Tokens: input={usage.get('input', '?')}, output={usage.get('output', '?')}, total={usage.get('total', '?')}"
                    )
                latency = gen.get("latency")
                if latency is not None:
                    lines.append(f"- Latency: {latency}ms")
                gen_input = gen.get("input")
                gen_output = gen.get("output")
                if gen_input:
                    lines.append(
                        f"- Input:\n```\n{_truncate(str(gen_input), 1500)}\n```"
                    )
                if gen_output:
                    lines.append(
                        f"- Output:\n```\n{_truncate(str(gen_output), 1500)}\n```"
                    )

        if spans:
            lines.append(f"\n### Tool Calls / Spans ({len(spans)})")
            for i, span in enumerate(spans, 1):
                lines.append(f"\n#### Span {i}: {span.get('name', 'N/A')}")
                latency = span.get("latency")
                if latency is not None:
                    lines.append(f"- Latency: {latency}ms")
                span_input = span.get("input")
                span_output = span.get("output")
                if span_input:
                    lines.append(
                        f"- Input:\n```\n{_truncate(str(span_input), 1000)}\n```"
                    )
                if span_output:
                    lines.append(
                        f"- Output:\n```\n{_truncate(str(span_output), 1000)}\n```"
                    )

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} total chars)"


# ---------------------------------------------------------------------------
# list_langfuse_traces
# ---------------------------------------------------------------------------


class ListLangfuseTracesInput(BaseModel):
    limit: int = Field(
        default=10, ge=1, le=50, description="Number of traces to return (max 50)."
    )
    user_id: Optional[str] = Field(
        default=None, description="Filter traces by Langfuse user ID."
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Filter traces by tags (e.g., agent name)."
    )
    name: Optional[str] = Field(default=None, description="Filter traces by name.")


class ListLangfuseTracesTool:
    name = "list_langfuse_traces"
    description = """List recent LLM conversation traces from Langfuse.

    Returns a summary of recent traces including trace ID, name, timestamp, tags,
    and token usage. Use this to discover trace IDs for further investigation
    with fetch_langfuse_trace.

    Supports filtering by user_id, tags, and name.
    Requires LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars.
    """
    args_schema = ListLangfuseTracesInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = ListLangfuseTracesInput(**kwargs)
        config = _langfuse_config()
        if not config["base_url"]:
            return "LANGFUSE_HOST is not configured."

        params: Dict[str, Any] = {"limit": input_data.limit}
        if input_data.user_id:
            params["userId"] = input_data.user_id
        if input_data.name:
            params["name"] = input_data.name
        if input_data.tags:
            for tag in input_data.tags:
                params.setdefault("tags", []).append(tag)

        url = f"{config['base_url']}/api/public/traces"
        try:
            with httpx.Client(timeout=_API_TIMEOUT) as client:
                resp = client.get(url, params=params, auth=_langfuse_auth(config))
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Langfuse API error: HTTP {e.response.status_code} - {e.response.text[:500]}"
        except Exception as e:
            return f"Failed to list traces: {e}"

        traces = data.get("data", [])
        if not traces:
            return "No traces found matching the given filters."

        lines = [f"## Recent Traces ({len(traces)})"]
        for i, t in enumerate(traces, 1):
            lines.append(f"\n### {i}. {t.get('name', 'N/A')}")
            lines.append(f"- Trace ID: `{t.get('id', '?')}`")
            lines.append(f"- Timestamp: {t.get('timestamp', 'N/A')}")
            lines.append(f"- Tags: {t.get('tags', [])}")
            lines.append(f"- User ID: {t.get('userId', 'N/A')}")
            usage = t.get("usage") or {}
            if usage:
                lines.append(
                    f"- Tokens: input={usage.get('input', '?')}, output={usage.get('output', '?')}"
                )

        return "\n".join(lines)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def fetch_langfuse_trace_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = FetchLangfuseTraceTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=FetchLangfuseTraceTool.name,
        description=FetchLangfuseTraceTool.description,
        args_schema=FetchLangfuseTraceInput,
    )


def list_langfuse_traces_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = ListLangfuseTracesTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=ListLangfuseTracesTool.name,
        description=ListLangfuseTracesTool.description,
        args_schema=ListLangfuseTracesInput,
    )
