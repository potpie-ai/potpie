"""
Upload parsing tool for the Prompt Tuner Agent.

Parses manually pasted or file-uploaded trace data into a structured format.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ParseUploadedTraceInput(BaseModel):
    content: str = Field(
        ...,
        description="The raw trace content to parse. Can be JSON (Langfuse export, tool call log) or plain text (pasted conversation).",
    )


class ParseUploadedTraceTool:
    name = "parse_uploaded_trace"
    description = """Parse manually provided trace data into a structured format.

    Accepts raw text or JSON content that the user pasted or uploaded.
    Attempts to parse as:
    1. Langfuse trace export (JSON with 'observations', 'input', 'output' keys)
    2. Tool call log (JSON array of tool call objects)
    3. Plain text conversation (returned as-is with formatting)

    Use this when the user pastes trace data directly or uploads a file
    instead of providing a Langfuse trace ID or message ID.
    """
    args_schema = ParseUploadedTraceInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = ParseUploadedTraceInput(**kwargs)
        content = input_data.content.strip()

        # Try JSON parsing
        try:
            data = json.loads(content)
            return self._format_json_trace(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # Plain text — return with basic formatting
        return f"## Uploaded Trace (plain text)\n\n```\n{content[:5000]}\n```"

    def _format_json_trace(self, data: Any) -> str:
        lines = ["## Uploaded Trace (JSON)"]

        if isinstance(data, dict):
            # Langfuse-style export
            if "observations" in data or "input" in data:
                if data.get("input"):
                    lines.append(f"\n### Input\n```\n{str(data['input'])[:2000]}\n```")
                if data.get("output"):
                    lines.append(f"\n### Output\n```\n{str(data['output'])[:2000]}\n```")
                for i, obs in enumerate(data.get("observations", []), 1):
                    name = obs.get("name", "unknown")
                    obs_type = obs.get("type", "?")
                    lines.append(f"\n### Observation {i}: {name} ({obs_type})")
                    if obs.get("input"):
                        lines.append(f"- Input:\n```\n{str(obs['input'])[:1000]}\n```")
                    if obs.get("output"):
                        lines.append(f"- Output:\n```\n{str(obs['output'])[:1000]}\n```")
                return "\n".join(lines)

            # Generic dict — dump formatted
            formatted = json.dumps(data, indent=2, default=str)[:5000]
            lines.append(f"\n```json\n{formatted}\n```")
            return "\n".join(lines)

        if isinstance(data, list):
            # Tool call array
            lines.append(f"\n### Tool Calls ({len(data)})")
            for i, item in enumerate(data[:20], 1):
                if isinstance(item, dict):
                    name = item.get("tool_name") or item.get("name") or "unknown"
                    lines.append(f"\n#### {i}. {name}")
                    formatted = json.dumps(item, indent=2, default=str)[:1500]
                    lines.append(f"```json\n{formatted}\n```")
                else:
                    lines.append(f"\n#### {i}.\n```\n{str(item)[:500]}\n```")
            return "\n".join(lines)

        return f"## Uploaded Trace\n\n```\n{str(data)[:5000]}\n```"

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def parse_uploaded_trace_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = ParseUploadedTraceTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=ParseUploadedTraceTool.name,
        description=ParseUploadedTraceTool.description,
        args_schema=ParseUploadedTraceInput,
    )
