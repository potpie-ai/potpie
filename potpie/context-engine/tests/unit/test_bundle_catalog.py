"""Tests for bundle-derived skill catalog."""

from __future__ import annotations

from adapters.outbound.skills.agent_installer import iter_template_files
from adapters.outbound.skills.bundle_catalog import (
    catalog_by_id,
    load_bundle_skills,
    recommended_skill_ids,
)


def test_load_bundle_skills_matches_template_directories() -> None:
    skill_dirs = {
        rel_path.parent.name
        for rel_path, _ in iter_template_files()
        if rel_path.name == "SKILL.md"
    }
    loaded = load_bundle_skills()
    assert {skill.id for skill in loaded} == skill_dirs
    assert "potpie-graph" in skill_dirs
    assert len(loaded) == len(skill_dirs)


def test_catalog_fields_are_populated_from_skill_front_matter() -> None:
    catalog = catalog_by_id()
    cli = catalog["potpie-cli"]
    assert cli.title == "Potpie CLI"
    assert cli.version == "3"
    assert "Potpie CLI" in cli.description or "potpie" in cli.description.lower()

    agent_context = catalog["potpie-agent-context"]
    assert agent_context.id == "potpie-agent-context"
    assert agent_context.title == "Potpie Agent Context"


def test_recommended_skill_ids_matches_loaded_catalog() -> None:
    assert recommended_skill_ids() == tuple(skill.id for skill in load_bundle_skills())
