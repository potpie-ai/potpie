"""
Skill loading tool for the Prompt Tuner Agent.

Loads analysis playbook files from the skills directory
and returns their content for injection into the agent's reasoning context.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Skills directory relative to this project
SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "agents" / "prompt_tuner_skills"


class LoadSkillInput(BaseModel):
    skill_name: str = Field(
        ...,
        description="Name of the skill to load (without .md extension). Use 'list' to see available skills.",
    )


class LoadSkillTool:
    name = "load_skill"
    description = """Load an analysis skill (playbook) to guide prompt diagnosis.

    Skills are structured analysis workflows stored as markdown files.
    Each skill provides step-by-step instructions for a specific type of analysis.

    Use skill_name='list' to see all available skills with descriptions.
    Use a specific skill name to load its full content.

    Available skills are in the prompt_tuner_skills directory.
    """
    args_schema = LoadSkillInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = LoadSkillInput(**kwargs)

        if input_data.skill_name == "list":
            return self._list_skills()

        skill_file = (SKILLS_DIR / f"{input_data.skill_name}.md").resolve()
        if not str(skill_file).startswith(str(SKILLS_DIR.resolve())):
            return "Invalid skill name."
        if not skill_file.exists():
            available = self._list_skills()
            return f"Skill not found: '{input_data.skill_name}'\n\n{available}"

        content = skill_file.read_text(encoding="utf-8")
        logger.info("Loaded skill: {} ({} chars)", input_data.skill_name, len(content))
        return f"## Skill Loaded: {input_data.skill_name}\n\n{content}"

    def _list_skills(self) -> str:
        if not SKILLS_DIR.exists():
            return "No skills directory found."

        skill_files = sorted(SKILLS_DIR.glob("*.md"))
        if not skill_files:
            return "No skills available."

        lines = ["## Available Skills"]
        for sf in skill_files:
            content = sf.read_text(encoding="utf-8")
            # Extract description from frontmatter
            desc = ""
            if content.startswith("---"):
                end = content.find("---", 3)
                if end > 0:
                    frontmatter = content[3:end]
                    for line in frontmatter.splitlines():
                        if line.strip().startswith("description:"):
                            desc = line.split(":", 1)[1].strip().strip('"').strip("'")
                            break
            name = sf.stem
            lines.append(f"- **{name}**: {desc or '(no description)'}")

        return "\n".join(lines)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_skill_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = LoadSkillTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=LoadSkillTool.name,
        description=LoadSkillTool.description,
        args_schema=LoadSkillInput,
    )
