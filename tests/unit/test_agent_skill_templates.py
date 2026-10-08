"""Installed agent templates/skills match the graph workbench responsibility split.

These pin the *content* contract of the shipped templates so a future edit cannot
reintroduce a stale include name, drop the graph surface, or forget the
retrieval-grade-description / nudge guidance. The Python recipe catalog is pinned
separately by ``test_agent_surface_contract``; this covers the markdown harness
instructions that humans and agents actually read.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

import potpie.cli as _clipkg
from potpie.skills.installer import _validate_potpie_command_tokens
from potpie_context_engine.core.agent_context_port import CONTEXT_RECORD_TYPES

pytestmark = pytest.mark.unit

TEMPLATES = Path(_clipkg.__file__).resolve().parent / "templates"
MD_FILES = sorted(TEMPLATES.rglob("*.md"))

# Stale include names from earlier templates. Underscored → unambiguous, so a
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

_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
_RECORD_ENUM_RE = re.compile(r"^[a-z_]+(?:\|[a-z_]+){3,}$", re.MULTILINE)


def test_templates_exist() -> None:
    names = {p.name for p in MD_FILES}
    assert "POTPIE.md" in names
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
    text = _read("potpie-graph/SKILL.md")
    for verb in (
        "graph status",
        "graph catalog",
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


def test_templates_do_not_recommend_removed_potpie_mcp_tools() -> None:
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES).as_posix()
        text = path.read_text(encoding="utf-8")
        removed_tools = {
            tool
            for tool in (
                "context_resolve",
                "context_search",
                "context_record",
                "context_status",
                "potpie-mcp",
            )
            if tool in text
        }
        assert removed_tools == set(), (
            f"{rel} recommends removed Potpie MCP tools: {sorted(removed_tools)}"
        )


def test_recommended_skills_teach_v2_workflow() -> None:
    text = _read("potpie-graph/SKILL.md")
    for token in (
        "graph status",
        "graph catalog",
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
    assert graph_skills == [
        TEMPLATES / "agent_bundle/.agents/skills/potpie-graph/SKILL.md"
    ]


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


def test_agent_instructions_use_the_cli_graph_surface() -> None:
    assert "potpie graph read" in _read("routing/POTPIE.md")
    skill_instructions = (
        "potpie-change-timeline/SKILL.md",
        "potpie-debug-memory/SKILL.md",
        "potpie-graph/SKILL.md",
        "potpie-infra-architecture/SKILL.md",
        "potpie-project-preferences/SKILL.md",
        "potpie-repo-baseline/SKILL.md",
        "potpie-source-ingestion/SKILL.md",
    )
    for path in skill_instructions:
        assert "potpie graph read" in _read(path), path


# The skills do not prescribe contract discovery before reads and writes,
# teach the one-call verbs, and state the read shapes this CLI actually has.
_USE_CASE_SKILLS = (
    "potpie-project-preferences",
    "potpie-infra-architecture",
    "potpie-change-timeline",
    "potpie-debug-memory",
)


def _bash_lines(text: str) -> list[str]:
    lines: list[str] = []
    for block in _BASH_BLOCK_RE.findall(text):
        lines.extend(
            line.strip()
            for line in block.splitlines()
            if line.strip().startswith("potpie")
        )
    return lines


def test_templates_do_not_prescribe_contract_discovery_before_work() -> None:
    """`describe --examples` has no mutation example and `catalog --task` is a
    no-op; neither belongs in a prescribed command block. The health quintet
    (`pot info`, `source list`, `graph status`) is one `potpie status`."""
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        assert "graph catalog --task" not in text, (
            f"{rel} still passes the ignored --task"
        )
        # potpie-cli is the command reference: it lists `pot info` and
        # `source list` as commands, not as a health check to run first.
        reference_skill = rel.parent.name == "potpie-cli"
        for line in _bash_lines(text):
            assert "--examples" not in line, (
                f"{rel} prescribes describe --examples: {line}"
            )
            if reference_skill:
                continue
            assert "pot info" not in line, f"{rel} prescribes pot info: {line}"
            assert "potpie --json source list" not in line, (
                f"{rel} prescribes source list as a health check: {line}"
            )


def test_use_case_skills_teach_resolve_first_and_record_for_one_learning() -> None:
    for skill_id in _USE_CASE_SKILLS:
        text = _read(f"{skill_id}/SKILL.md")
        assert "potpie resolve" in text, f"{skill_id} never mentions potpie resolve"
        assert "mutation-template" in text, (
            f"{skill_id} never names the payload template"
        )
    for fragment in (
        "potpie-project-preferences/SKILL.md",
        "potpie-debug-memory/SKILL.md",
        "potpie-graph/SKILL.md",
        "potpie-cli/SKILL.md",
    ):
        assert "potpie record" in _read(fragment), (
            f"{fragment} never mentions potpie record"
        )
    # resolve does not infer its intent, so the skills pass it for failures.
    assert "--intent debugging" in _read("potpie-graph/SKILL.md")
    assert "--intent debugging" in _read("potpie-debug-memory/SKILL.md")


def test_record_is_taught_only_for_the_types_it_accepts() -> None:
    """``potpie record`` takes ``--type``/``--summary``/``--scope`` only.

    A structured type (decision, preference, bug pattern, verification) needs
    fields the command cannot carry and is refused, so the skills route those
    through ``graph mutation-template`` and a plan instead.
    """
    structured = {"decision", "preference", "policy", "bug_pattern", "verification"}
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        for line in _bash_lines(path.read_text(encoding="utf-8")):
            if not line.startswith("potpie record"):
                continue
            tokens = shlex.split(line)
            record_type = tokens[tokens.index("--type") + 1]
            assert record_type not in structured, (
                f"{rel} teaches a one-call record for a structured type: {line}"
            )
            assert "--detail" not in tokens, f"{rel} passes --detail to record: {line}"


def test_templates_state_the_measured_read_shapes() -> None:
    """Preferences are read by scope (a task-shaped --query drops them);
    a neighborhood --environment filter must keep the unqualified edges;
    decisions anchor on services; the repo key has one spelling."""
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        for line in _bash_lines(text):
            if "preferences_for_scope" in line:
                assert "--query" not in line, (
                    f"{rel} passes --query to preferences: {line}"
                )
            if "service_neighborhood" in line and "--environment" in line:
                assert "include_unqualified_environment:true" in line, (
                    f"{rel} filters by environment without keeping unqualified edges: {line}"
                )
            if "active_decisions" in line and "--scope" in line:
                assert "--scope service:" in line, (
                    f"{rel} scopes decisions by repo: {line}"
                )
        for stale in ("repo:<owner-repo>", "repo:<owner/repo>", "repo:acme/x"):
            assert stale not in text, f"{rel} spells the repo key as {stale}"


def test_templates_do_not_carry_known_wrong_lines() -> None:
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES)
        text = path.read_text(encoding="utf-8")
        assert "potpie login <api-key>" not in text, f"{rel}: login takes --api-key"
        assert "sections_created" not in text, (
            f"{rel}: the import report key is sections_added"
        )
        assert '"graph_contract_version"' not in text, (
            f"{rel}: payload examples must omit graph_contract_version"
        )


# The core skills: every one must carry the harness-led boundary in
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


def test_every_skill_has_one_canonical_source() -> None:
    """Every harness installs skills from ``agent_bundle``; nothing else copies one."""
    skill_files = sorted(TEMPLATES.rglob("SKILL.md"))
    assert skill_files
    for path in skill_files:
        assert "agent_bundle/.agents/skills" in path.as_posix(), path


def test_inline_potpie_commands_exist_on_this_cli() -> None:
    """Prose cites commands too, and an agent runs what the prose names.

    The installer validates commands inside ``bash`` fences; this applies the
    same check to every inline `` `potpie …` `` span, so a skill cannot teach a
    command group or flag this CLI does not have.
    """
    span = re.compile(r"`(potpie [^`]+)`")
    errors: list[str] = []
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES).as_posix()
        flat = " ".join(path.read_text(encoding="utf-8").split())
        for command in span.findall(flat):
            error = _validate_potpie_command_tokens(shlex.split(command))
            if error:
                errors.append(f"{rel}: {error}")
    assert errors == []


def test_templates_do_not_prescribe_a_threshold_the_views_ignore() -> None:
    """``--query-threshold`` is a floor only ``preferences_for_scope`` and the
    timeline apply; ``prior_occurrences`` and ``document_context`` rank their
    pool and ignore it. A skill that passes the flag on those views teaches a
    no-op, and the agent then trusts a full list as evidence.
    """
    for path in MD_FILES:
        rel = path.relative_to(TEMPLATES).as_posix()
        text = path.read_text(encoding="utf-8")
        for line in _bash_lines(text):
            if "--query-threshold" in line:
                assert "preferences_for_scope" in line or "--view timeline" in line, (
                    f"{rel} passes --query-threshold to a view that ignores it: {line}"
                )
    graph = _read("potpie-graph/SKILL.md")
    assert "--direction out|in|both" in graph
    assert "graph neighborhood --entity" in graph
    flat_debug = " ".join(_read("potpie-debug-memory/SKILL.md").split())
    assert "`--query-threshold` does nothing here" in flat_debug
    # A timeline --query filters on this CLI, so the skill must say so rather
    # than promise a re-rank that never empties the window.
    flat_timeline = " ".join(_read("potpie-change-timeline/SKILL.md").split())
    assert "A `--query` filters the window" in flat_timeline
    assert "never empties" not in flat_timeline


def test_templates_state_pot_resolution_and_score_semantics() -> None:
    """A no-``--pot`` command resolves the pot from the repo registration, and
    the ``*`` in ``pot list`` loses to it; the resolve header's ``confidence``
    is not a verdict. Both were misread in testing — an agent
    in an unregistered checkout saw ``items=0`` on the wrong pot and every
    correct answer on a small pot arrived under ``confidence=low``.
    """
    graph = " ".join(_read("potpie-graph/SKILL.md").split())
    assert "resolves the pot from the repo you are in" in graph
    assert "not a verdict" in graph
    cli = (TEMPLATES / "agent_bundle/.agents/skills/potpie-cli/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "not a verdict" in cli
    assert "--include docs" in cli
    # Timeline reads keep the fact whole: compact rows cut it at ~120 chars.
    timeline = _read("potpie-change-timeline/SKILL.md")
    for line in _bash_lines(timeline):
        if "--view timeline" in line:
            assert "--detail full" in line, line
