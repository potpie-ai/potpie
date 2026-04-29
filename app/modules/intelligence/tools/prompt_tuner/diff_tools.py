"""
Diff proposal and apply tools for the Prompt Tuner Agent.

propose_prompt_diff: Generates a structured before/after diff.
apply_prompt_change: Applies approved changes to the CustomAgent prompt in DB.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional
from uuid import uuid4

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
from app.modules.intelligence.prompts.prompt_model import (
    Prompt,
    PromptStatusType,
    PromptType,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# propose_prompt_diff
# ---------------------------------------------------------------------------

class PromptEdit(BaseModel):
    title: str = Field(..., description="Short title for this edit (e.g., 'Add search strategy section').")
    location: str = Field(..., description="Where in the prompt this edit applies (e.g., 'After ## Available Tools section').")
    old_text: str = Field(
        default="",
        description="The original text being replaced. Empty string for new additions.",
    )
    new_text: str = Field(..., description="The replacement or new text.")
    rationale: str = Field(..., description="Why this edit fixes the issue.")


class ProposePromptDiffInput(BaseModel):
    root_cause: str = Field(..., description="Explanation of the root cause of the prompt failure.")
    edits: List[PromptEdit] = Field(..., description="List of proposed edits to the prompt.", min_length=1)
    agent_id: Optional[str] = Field(default=None, description="Custom agent ID (for context in the diff output).")


class ProposePromptDiffTool:
    name = "propose_prompt_diff"
    description = """Generate a structured before/after diff of prompt changes.

    Takes a root cause analysis and a list of proposed edits, each with:
    - title: what the edit does
    - location: where in the prompt
    - old_text: original text (empty for additions)
    - new_text: replacement text
    - rationale: why this fixes the issue

    Formats the output as a reviewable diff block for the user.
    After presenting, ask the user to approve with "yes", "no", or "apply edit 1 and 3 only".
    """
    args_schema = ProposePromptDiffInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = ProposePromptDiffInput(**kwargs)

        lines = []
        lines.append("### Root Cause")
        lines.append(input_data.root_cause)
        lines.append("")
        lines.append(f"### Proposed Changes ({len(input_data.edits)} edits)")

        for i, edit in enumerate(input_data.edits, 1):
            lines.append(f"\n--- EDIT {i}: {edit.title} ---")
            lines.append(f"Location: {edit.location}")
            lines.append("")
            if edit.old_text:
                for old_line in edit.old_text.splitlines():
                    lines.append(f"- {old_line}")
            else:
                lines.append("(new section)")
            for new_line in edit.new_text.splitlines():
                lines.append(f"+ {new_line}")
            lines.append("")
            lines.append(f"Rationale: {edit.rationale}")

        lines.append("")
        lines.append("Apply these changes? (yes / no / apply edit N and M only)")

        return "\n".join(lines)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# apply_prompt_change
# ---------------------------------------------------------------------------

class ApplyPromptChangeInput(BaseModel):
    agent_id: str = Field(..., description="The custom agent ID whose prompt to update.")
    field: str = Field(
        default="system_prompt",
        description="Which field to update: 'system_prompt', 'backstory', 'role', or 'goal'.",
    )
    new_value: str = Field(..., description="The complete new value for the field after applying edits.")


class ApplyPromptChangeTool:
    name = "apply_prompt_change"
    description = """Apply an approved prompt change to a custom agent in the database.

    Before calling this, the user MUST have approved the changes proposed by propose_prompt_diff.

    This tool:
    1. Snapshots the current prompt value into the Prompt table (for rollback)
    2. Updates the CustomAgent record with the new value
    3. Returns confirmation with the old version info

    Only call this after explicit user approval.
    """
    args_schema = ApplyPromptChangeInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = ApplyPromptChangeInput(**kwargs)

        agent = (
            self.sql_db.query(CustomAgent)
            .filter(
                CustomAgent.id == input_data.agent_id,
                (CustomAgent.user_id == self.user_id) | (CustomAgent.visibility == "public"),
            )
            .first()
        )
        if not agent:
            return f"Custom agent not found or not accessible: {input_data.agent_id}"

        allowed_fields = {"system_prompt", "backstory", "role", "goal"}
        if input_data.field not in allowed_fields:
            return f"Invalid field: {input_data.field}. Must be one of: {allowed_fields}"

        old_value = getattr(agent, input_data.field) or ""

        # Snapshot the old value into Prompt table for rollback
        existing_versions = (
            self.sql_db.query(Prompt)
            .filter(
                Prompt.created_by == self.user_id,
                Prompt.type == PromptType.SYSTEM,
            )
            .order_by(Prompt.version.desc())
            .first()
        )
        next_version = (existing_versions.version + 1) if existing_versions else 1

        snapshot = Prompt(
            id=str(uuid4()),
            text=old_value,
            type=PromptType.SYSTEM,
            version=next_version,
            status=PromptStatusType.INACTIVE,
            created_by=self.user_id,
        )
        self.sql_db.add(snapshot)

        # Apply the new value
        setattr(agent, input_data.field, input_data.new_value)

        try:
            self.sql_db.commit()
        except Exception as e:
            self.sql_db.rollback()
            return f"Failed to apply prompt change: {e}"

        return (
            f"Prompt updated successfully.\n"
            f"- Agent: {agent.id}\n"
            f"- Field: {input_data.field}\n"
            f"- Previous version archived as Prompt ID: {snapshot.id} (version {next_version})\n"
            f"- To rollback, use this snapshot ID."
        )

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def propose_prompt_diff_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = ProposePromptDiffTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=ProposePromptDiffTool.name,
        description=ProposePromptDiffTool.description,
        args_schema=ProposePromptDiffInput,
    )


def apply_prompt_change_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = ApplyPromptChangeTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=ApplyPromptChangeTool.name,
        description=ApplyPromptChangeTool.description,
        args_schema=ApplyPromptChangeInput,
    )
