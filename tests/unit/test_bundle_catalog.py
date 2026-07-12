"""Tests for bundle-derived skill catalog."""

from __future__ import annotations

from potpie.skills.installer import (
    iter_template_files,
)
from potpie.skills.catalog import (
    catalog_by_id,
    load_bundle_skills,
    recommended_skill_ids,
)
from potpie.skills.resource_provider import ROOT_TEMPLATE_RESOURCES

TEMPLATE_RESOURCES = ROOT_TEMPLATE_RESOURCES


def _iter_template_files():
    return iter_template_files(template_resources=TEMPLATE_RESOURCES)


def _load_bundle_skills():
    return load_bundle_skills(template_resources=TEMPLATE_RESOURCES)


def _catalog_by_id():
    return catalog_by_id(template_resources=TEMPLATE_RESOURCES)


def _recommended_skill_ids():
    return recommended_skill_ids(template_resources=TEMPLATE_RESOURCES)


def test_load_bundle_skills_matches_template_directories() -> None:
    skill_dirs = {
        rel_path.parent.name
        for rel_path, _ in _iter_template_files()
        if rel_path.name == "SKILL.md"
    }
    loaded = _load_bundle_skills()
    assert {skill.id for skill in loaded} == skill_dirs
    assert "potpie-graph" in skill_dirs
    assert len(loaded) == len(skill_dirs)


def test_catalog_fields_are_populated_from_skill_front_matter() -> None:
    catalog = _catalog_by_id()
    cli = catalog["potpie-cli"]
    assert cli.title == "Potpie CLI"
    assert cli.version == "2"
    assert "Potpie CLI" in cli.description or "potpie" in cli.description.lower()

    graph = catalog["potpie-graph"]
    assert graph.id == "potpie-graph"
    assert graph.title == "Potpie Graph Workbench"


def test_recommended_skill_ids_matches_loaded_catalog() -> None:
    assert _recommended_skill_ids() == tuple(
        skill.id for skill in _load_bundle_skills()
    )


def test_bundle_catalog_uses_root_resources_by_default() -> None:
    assert load_bundle_skills() == _load_bundle_skills()
