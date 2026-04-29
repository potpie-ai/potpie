"""
Tests for the Prompt Tuner Agent's skills infrastructure.
"""

from app.modules.intelligence.tools.prompt_tuner.skill_tools import (
    LoadSkillTool,
    SKILLS_DIR,
)


def test_skills_directory_exists():
    assert SKILLS_DIR.exists(), f"Skills directory not found: {SKILLS_DIR}"


def test_list_skills():
    tool = LoadSkillTool(None, None)
    result = tool.run(skill_name="list")
    assert "Available Skills" in result
    assert "systematic_debugging" in result


def test_load_systematic_debugging_skill():
    tool = LoadSkillTool(None, None)
    result = tool.run(skill_name="systematic_debugging")
    assert "Skill Loaded" in result
    assert "Step 1" in result
    assert "Step 2" in result
    assert "Step 3" in result
    assert "Step 4" in result
    assert "Step 5" in result
    assert "Missing Instruction" in result
    assert "Ambiguous Instruction" in result
    assert "Conflicting Instruction" in result


def test_load_unknown_skill():
    tool = LoadSkillTool(None, None)
    result = tool.run(skill_name="nonexistent_skill")
    assert "not found" in result.lower()
    assert "Available Skills" in result


def test_skill_has_frontmatter():
    skill_file = SKILLS_DIR / "systematic_debugging.md"
    content = skill_file.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "description:" in content
    assert "trigger_hint:" in content
