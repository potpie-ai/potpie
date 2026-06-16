"""Contract tests for the harness-led repo baseline skill.

``potpie-repo-baseline`` is the Stage 2 procedure of the harness-led repo
ingestion plan: the harness reads selected authored sources and writes
semantic mutations; Potpie registers, reads, validates, and stores. These
tests pin the skill's frontmatter, required sections, source-priority order,
ontology references, and the anti-scanning boundary so a future edit cannot
quietly reintroduce a CLI scanner or drop the write requirements.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import adapters.inbound.cli as _clipkg
from domain.graph_contract import TRUTH_CLASSES
from domain.ontology import CANONICAL_EDGE_TYPES, CANONICAL_LABELS

pytestmark = pytest.mark.unit

TEMPLATES = Path(_clipkg.__file__).resolve().parent / "templates"
AGENT_SKILL = (
    TEMPLATES / "agent_bundle" / ".agents" / "skills" / "potpie-repo-baseline" / "SKILL.md"
)
PLUGIN_SKILL = TEMPLATES / "claude_plugin" / "skills" / "potpie-repo-baseline" / "SKILL.md"

_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _frontmatter_and_body(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("---\n"), f"{path} missing frontmatter"
    end = raw.find("\n---\n", 4)
    fm: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            fm[key.strip()] = value.strip().strip('"')
    return fm, raw[end + 5 :]


def test_skill_exists_in_agent_and_plugin_bundles() -> None:
    assert AGENT_SKILL.is_file()
    assert PLUGIN_SKILL.is_file()
    assert AGENT_SKILL.read_text() == PLUGIN_SKILL.read_text(), (
        "agent_bundle and claude_plugin copies of potpie-repo-baseline diverged"
    )


def test_frontmatter_is_recommended_and_named() -> None:
    fm, _ = _frontmatter_and_body(AGENT_SKILL)
    assert fm["name"] == "potpie-repo-baseline"
    assert fm.get("recommended", "").lower() == "true"
    assert "harness" in fm["description"].lower()
    assert "never scans" in fm["description"].lower()


def test_required_sections_present() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    for heading in (
        "Procedure",
        "Source priority order",
        "Baseline memory families",
        "Mutation requirements",
        "Anti-patterns",
    ):
        assert re.search(rf"(?m)^##\s+{re.escape(heading)}\s*$", body), (
            f"missing required section: ## {heading}"
        )


def test_procedure_covers_plan_steps() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    for marker in (
        "pot info",
        "source add repo",
        "graph catalog",
        "graph describe",
        "graph search-entities",
        "graph propose",
        "plan_id",
    ):
        assert marker in body, f"procedure missing step marker: {marker!r}"


def test_source_priority_starts_with_authored_docs() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    section = body.split("## Source priority order", 1)[1]
    first_item = section.strip().splitlines()[0]
    assert "README" in first_item


def test_referenced_entities_and_predicates_are_canonical() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    for label in ("Repository", "Service", "Feature", "Environment", "DataStore",
                  "APIContract", "Dependency", "Preference"):
        assert f"`{label}`" in body, f"skill does not mention entity `{label}`"
        assert label in CANONICAL_LABELS
    for edge in ("PROVIDES", "IMPLEMENTED_IN", "DEFINED_IN", "DEPENDS_ON", "USES",
                 "EXPOSES", "DEPLOYED_TO", "POLICY_APPLIES_TO"):
        assert f"`{edge}`" in body, f"skill does not mention predicate `{edge}`"
        assert edge in CANONICAL_EDGE_TYPES


def test_example_mutation_is_valid_json_with_required_fields() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    blocks = [json.loads(b) for b in _JSON_BLOCK_RE.findall(body)]
    assert blocks, "expected a JSON mutation example"
    op = blocks[0]["operations"][0]
    assert op["truth"] in TRUTH_CLASSES
    assert op["predicate"] in CANONICAL_EDGE_TYPES
    assert op["subject"]["summary"] and op["subject"]["description"]
    assert op["object"]["summary"] and op["object"]["description"]
    assert op["evidence"], "example claim must carry evidence refs"


def test_skill_forbids_scanning_and_requires_write_metadata() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    lowered = body.lower()
    assert "code scanner" in lowered or "deterministic" in lowered
    assert "do not walk the file tree" in lowered
    assert "retrieval-grade" in lowered
    assert "summary" in lowered and "evidence" in lowered and "truth" in lowered


def test_skill_separates_baseline_from_change_history() -> None:
    _, body = _frontmatter_and_body(AGENT_SKILL)
    assert "change-history" in body
    assert "potpie-change-timeline" in body
