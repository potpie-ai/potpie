"""Installed agent templates/skills match the Graph V2 responsibility split.

These pin the *content* contract of the shipped templates so a future edit cannot
reintroduce a stale include name, drop the graph surface, or forget the
retrieval-grade-description / nudge guidance. The Python recipe catalog is pinned
separately by ``test_agent_surface_contract``; this covers the markdown harness
instructions that humans and agents actually read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import potpie.skills.resources as _skill_resources
from potpie_context_engine.domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    CONTEXT_RECORD_TYPES,
)

pytestmark = pytest.mark.unit

TEMPLATES = Path(_skill_resources.__file__).resolve().parent / "templates"
MD_FILES = sorted(TEMPLATES.rglob("*.md"))

# Stale include names from the pre-V1.5 templates. Underscored → unambiguous, so a
# bare-substring scan over the markdown has no false positives in prose.
STALE_INCLUDE_TOKENS = (
    "feature_map",
    "service_map",
    "prior_fixes",
    "source_status",
    "repo_map",
    "local_workflows",
    "agent_instructions",
    "diagnostic_signals",
)

STALE_PUBLIC_VIEW_TOKENS = (
    "bugs.prior_occurrences",
    "preferences.active_preferences",
    "features.provided",
    "ownership.owner_context",
    "docs.reference_context",
)
# NB: ``recent_changes`` is intentionally absent — it collides with the legitimate
# view name ``recent_changes.timeline``. The JSON-include allowlist check below
# still rejects ``recent_changes`` if it ever reappears as an include value.

_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
_RECORD_ENUM_RE = re.compile(r"^[a-z_]+(?:\|[a-z_]+){3,}$", re.MULTILINE)


def _iter_json_blocks(text: str):
    for raw in _JSON_BLOCK_RE.findall(text):
        try:
            yield json.loads(raw)
        except (ValueError, TypeError):
            continue  # illustrative non-JSON fence; not a recipe


def test_templates_exist() -> None:
    names = {p.name for p in MD_FILES}
    assert {"AGENTS.md", "CLAUDE.md"} <= names
    assert any("potpie-graph" in p.as_posix() for p in MD_FILES)
    agent_skill_ids = {
        p.parent.name
        for p in MD_FILES
        if p.name == "SKILL.md" and "agent_bundle/.agents/skills" in p.as_posix()
    }
    assert {
        "potpie-project-preferences",
        "potpie-infra-architecture",
        "potpie-change-timeline",
        "potpie-debug-memory",
        "potpie-source-ingestion",
        "potpie-repo-baseline",
    } <= agent_skill_ids


def test_no_stale_include_names_anywhere() -> None:
    for path in MD_FILES:
        text = path.read_text(encoding="utf-8")
        hits = [tok for tok in STALE_INCLUDE_TOKENS if tok in text]
        assert not hits, (
            f"{path.relative_to(TEMPLATES)} still advertises stale includes: {hits}"
        )


def test_every_json_recipe_include_is_supported() -> None:
    checked = 0
    for path in MD_FILES:
        for block in _iter_json_blocks(path.read_text(encoding="utf-8")):
            if not isinstance(block, dict) or "include" not in block:
                continue
            include = block["include"]
            assert isinstance(include, list) and include, f"{path.name}: empty include"
            unknown = set(include) - CONTEXT_INCLUDE_VALUES
            assert not unknown, f"{path.name} recipe has unknown includes: {unknown}"
            checked += 1
    assert checked >= 3, "expected several JSON recipe blocks across templates"


def test_record_type_enums_are_supported() -> None:
    found_enum = False
    for path in MD_FILES:
        text = path.read_text(encoding="utf-8")
        for enum_line in _RECORD_ENUM_RE.findall(text):
            tokens = set(enum_line.split("|"))
            # Only treat it as a record-type enum if it overlaps the real vocabulary.
            if not (tokens & CONTEXT_RECORD_TYPES):
                continue
            found_enum = True
            unknown = tokens - CONTEXT_RECORD_TYPES
            assert not unknown, f"{path.name} lists unknown record types: {unknown}"
    assert found_enum, "expected a record_type enum in the templates"


def _read(name_fragment: str) -> str:
    matches = [p for p in MD_FILES if name_fragment in p.as_posix()]
    assert matches, f"no template matching {name_fragment!r}"
    return "\n".join(p.read_text(encoding="utf-8") for p in matches)


def test_agents_md_advertises_graph_surface() -> None:
    text = _read("agent_bundle/AGENTS.md")
    for verb in (
        "graph status",
        "graph catalog --task",
        "graph describe",
        "graph read --subgraph",
        "graph search-entities",
        "graph propose",
        "graph commit",
        "graph history",
    ):
        assert verb in text, f"AGENTS.md missing `{verb}`"


def test_templates_do_not_advertise_v1_write_workflow() -> None:
    forbidden = (
        "graph mutate --file",
        "graph mutate --dry-run",
        "--dry-run",
    )
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        hits = [tok for tok in forbidden if tok in text]
        assert not hits, f"{rel} still advertises V1 write workflow: {hits}"


def test_graph_commit_examples_use_verified_gate() -> None:
    checked = 0
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        for line in path.read_text(encoding="utf-8").splitlines():
            if "potpie --json graph commit" not in line:
                continue
            checked += 1
            assert "--verify" in line, (
                f"{rel} has graph commit example without --verify: {line}"
            )
    assert checked >= 8, "expected verified commit examples across templates"


def test_source_ingestion_uses_hard_verification_gate() -> None:
    text = _read("potpie-source-ingestion/SKILL.md")
    assert "graph commit --verify" in text
    assert "reads back committed claims and checks quality" in text
    assert "Read back the graph and run quality checks" not in text


def test_templates_are_text_first_for_agent_reads() -> None:
    forbidden = (
        "potpie --json graph read",
        "potpie --json graph search-entities",
    )
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        hits = [tok for tok in forbidden if tok in text]
        assert not hits, f"{rel} uses JSON for routine agent reads: {hits}"


def test_timeline_read_examples_are_bounded() -> None:
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        for block in _BASH_BLOCK_RE.findall(text):
            if "graph read" not in block or "--view timeline" not in block:
                continue
            assert "--format table" in block or "--format jsonl" in block, (
                f"{rel} timeline read lacks table/jsonl format"
            )


def test_templates_use_canonical_v2_view_syntax() -> None:
    command_view_arg = re.compile(r"--view\s+[a-z_]+\.[a-z_]+")
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        stale_views = [tok for tok in STALE_PUBLIC_VIEW_TOKENS if tok in text]
        assert not stale_views, f"{rel} still uses obsolete public views: {stale_views}"
        bad_args = command_view_arg.findall(text)
        assert not bad_args, f"{rel} uses fully-qualified --view args: {bad_args}"


def test_context_record_is_labeled_mcp_compatibility() -> None:
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES).as_posix()
        text = path.read_text(encoding="utf-8")
        if "context_record" not in text:
            continue
        lowered_lines = text.lower().splitlines()
        for idx, line in enumerate(lowered_lines):
            if "context_record" not in line:
                continue
            window = " ".join(lowered_lines[max(0, idx - 5) : idx + 2])
            assert "mcp" in window or "compatib" in window, (
                f"{rel} mentions context_record without MCP compatibility framing"
            )


def test_recommended_skills_teach_v2_workflow() -> None:
    text = _read("potpie-graph/SKILL.md")
    for token in (
        "graph status",
        "graph catalog --task",
        "--profile read",
        "graph describe",
        "graph search-entities",
        "graph read --subgraph",
        "graph propose",
        "graph commit",
        "--verify",
        "graph history",
        "graph inbox",
    ):
        assert token in text, f"potpie-graph missing V2 workflow token: {token}"


def test_graph_skill_present_in_each_harness_bundle() -> None:
    graph_skills = [
        p for p in MD_FILES if p.name == "SKILL.md" and "potpie-graph" in p.as_posix()
    ]
    bundles = {p.relative_to(TEMPLATES).parts[0] for p in graph_skills}
    assert {"agent_bundle", "claude_bundle", "claude_plugin"} <= bundles


def test_templates_require_retrieval_grade_descriptions() -> None:
    text = _read("potpie-graph/SKILL.md")
    assert "retrieval" in text.lower()
    assert "description" in text.lower()
    # The skill must say the description is for search, not display.
    assert "for search" in text.lower() or "not display" in text.lower()


def test_templates_document_nudge_handling() -> None:
    text = _read("potpie-graph/SKILL.md")
    assert "inject_context" in text
    assert "instruction" in text
    # A write instruction is a prompt to decide, never an auto-write.
    assert "auto-write" in text.lower() or "prompt to decide" in text.lower()


def test_context_tools_kept_as_compatibility_wrappers() -> None:
    assert "context_resolve" in _read("agent_bundle/AGENTS.md")
    assert "context_resolve" in _read("claude_bundle/CLAUDE.md")


# The Stage 6 core skills: every one must carry the harness-led boundary in
# its body — the harness reads/decides/writes, Potpie validates/stores, no
# scanner mutates the graph.
_CORE_SKILLS = (
    "potpie-source-ingestion",
    "potpie-repo-baseline",
    "potpie-cli",
    "potpie-graph",
    "potpie-project-preferences",
    "potpie-infra-architecture",
    "potpie-change-timeline",
    "potpie-debug-memory",
)

_HARNESS_LED_MARKERS = (
    "harness-led",
    "harness is the intelligence",
    "you are the intelligence",
    "interpreted by the harness",
    "harness must read",
    "the harness reads",
    "capture is harness-led",
    "memory is harness-led",
    "ingestion is harness-led",
)


@pytest.mark.parametrize("skill_id", _CORE_SKILLS)
def test_core_skills_state_harness_led_boundary(skill_id: str) -> None:
    # Collapse whitespace so markers match across markdown line wraps.
    text = " ".join(_read(f"{skill_id}/SKILL.md").lower().split())
    assert any(marker in text for marker in _HARNESS_LED_MARKERS), (
        f"{skill_id} never states that ingestion/decisions are harness-led"
    )
    assert "scan" in text, (
        f"{skill_id} should explicitly rule out scanner-driven graph updates"
    )


@pytest.mark.parametrize(
    "skill_id",
    (
        "potpie-source-ingestion",
        "potpie-repo-baseline",
        "potpie-graph",
        "potpie-project-preferences",
        "potpie-infra-architecture",
        "potpie-debug-memory",
    ),
)
def test_writing_skills_require_descriptions_evidence_and_truth(skill_id: str) -> None:
    text = _read(f"{skill_id}/SKILL.md").lower()
    assert "description" in text, f"{skill_id} missing description guidance"
    assert "evidence" in text or "source_refs" in text, (
        f"{skill_id} missing evidence guidance"
    )
    assert "truth" in text, f"{skill_id} missing truth-class guidance"
    assert "summary" in text or "retrieval" in text, (
        f"{skill_id} missing summary/retrieval guidance"
    )


def test_feature_ontology_reaches_skills() -> None:
    for fragment in (
        "potpie-repo-baseline/SKILL.md",
        "potpie-source-ingestion/SKILL.md",
        "potpie-graph/SKILL.md",
    ):
        text = _read(fragment)
        assert "PROVIDES" in text and "Feature" in text, (
            f"{fragment} does not teach the Feature/PROVIDES ontology"
        )


def test_source_ingestion_requires_deep_harness_workflow() -> None:
    text = _read("potpie-source-ingestion/SKILL.md")
    lowered = text.lower()
    for token in (
        "todo/checklist",
        "phase 0: scope and preflight",
        "phase 1: todo plan",
        "phase 2: parallel discovery",
        "phase 3: local repo inspection targets",
        "phase 4: hosted/github hydration",
        "phase 5: evidence matrix",
        "phase 6: identity resolution",
        "phase 7: write",
        "phase 8: verify and quality gate",
    ):
        assert token in lowered, (
            f"source ingestion missing deep workflow token: {token}"
        )


def test_source_ingestion_ships_subagent_handoffs_and_github_checklist() -> None:
    text = _read("potpie-source-ingestion/SKILL.md")
    lowered = text.lower()
    for token in (
        "subagent prompts",
        "docs/product",
        "local architecture",
        "runtime/deploy",
        "api/data/integrations",
        "github history",
        "repository metadata",
        "recent merged prs",
        "open/high-signal issues",
        "ci/workflows",
    ):
        assert token in lowered, (
            f"source ingestion missing subagent/GitHub token: {token}"
        )


def test_source_ingestion_clarifies_local_inspection_not_scanner_writes() -> None:
    text = _read("potpie-source-ingestion/SKILL.md").lower()
    collapsed = " ".join(text.split())
    assert "local inspection is required" in collapsed
    assert "scanner-driven graph updates are forbidden" in collapsed
    assert "rg --files" in text
    assert "do not infer durable facts from filenames alone" in text


def test_templates_do_not_advertise_local_ingest_or_scan_commands() -> None:
    forbidden = (
        "potpie ingest",
        "--scan",
        "ingest scan",
        "ledger pull --apply",
    )
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        hits = [tok for tok in forbidden if tok in text]
        assert not hits, f"{rel} advertises removed local ingest/scan commands: {hits}"


def test_hosted_integration_ingestion_is_agent_led() -> None:
    for fragment in (
        "potpie-source-ingestion/SKILL.md",
        "potpie-change-timeline/SKILL.md",
        "potpie-graph/SKILL.md",
        "AGENTS.md",
        "CLAUDE.md",
    ):
        text = " ".join(_read(fragment).lower().split())
        assert "agent's integration tools/connectors" in text, (
            f"{fragment} does not direct agents to hydrate hosted sources through integrations"
        )
        assert "do not use potpie cli queue ingestion" in text, (
            f"{fragment} does not rule out Potpie CLI queue ingestion"
        )
        assert "graph propose" in text and "graph commit" in text, (
            f"{fragment} does not route hosted ingestion back through graph plans"
        )


def test_removed_connector_queue_commands_are_not_advertised() -> None:
    removed_commands = (
        "potpie pot linear-team ingest",
        "potpie pot linear-team diff-sync",
        "potpie pot jira-project ingest",
        "potpie pot jira-project diff-sync",
    )
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        lowered = path.read_text(encoding="utf-8").lower()
        for token in removed_commands:
            assert token not in lowered, (
                f"{rel} still advertises removed connector queue command `{token}`"
            )
