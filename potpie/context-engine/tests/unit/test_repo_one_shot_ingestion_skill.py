"""Contract test for the repo one-shot ingestion skill markdown.

The skill at ``potpie_context_engine/domain/playbooks/repo_one_shot_ingestion.md`` is read by
Claude Code and (later) embedded into the internal reconciliation agent
prompt. It documents tool signatures, entity labels, edge types, and stable
key formats — every one of those is a real promise about the surrounding
code. This test pins each promise so the skill cannot silently drift away
from the ontology, the GitHub adapter, or the deep agent's tool surface.

Failures here mean either: (a) the skill names something that no longer
exists, or (b) the code moved and the skill was not updated.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.connectors.github.api_client import (
    PyGithubSourceControl,
)
from potpie_context_engine.adapters.outbound.reconciliation.llm_plan_schema import (
    LlmReconciliationPlan,
)
from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
    PydanticDeepReconciliationAgent,
)
from potpie_context_engine.domain.identity import (
    IdentityClass,
    _EXTERNAL_ID_SAFE_RE,
    _SLUG_BODY_RE,
    get_identity,
    mint_entity_key,
)
from potpie_context_engine.domain.ontology import EDGE_TYPES, ENTITY_TYPES

pytestmark = pytest.mark.unit


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "potpie_context_engine"
    / "domain"
    / "playbooks"
    / "repo_one_shot_ingestion.md"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_skill() -> tuple[dict[str, str], str]:
    """Return (frontmatter, body) for the skill markdown."""
    raw = SKILL_PATH.read_text(encoding="utf-8")
    assert raw.startswith("---\n"), "skill must open with a YAML frontmatter fence"
    end = raw.find("\n---\n", 4)
    assert end > 0, "skill frontmatter is not terminated"
    fm_block = raw[4:end]
    body = raw[end + 5 :]
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm, body


def _section(body: str, heading: str) -> str:
    """Extract the text under a `## heading` until the next `## ` heading."""
    pattern = re.compile(
        rf"(?m)^##\s+{re.escape(heading)}\s*$(.+?)(?=^##\s|\Z)",
        re.DOTALL,
    )
    m = pattern.search(body)
    assert m, f"section ## {heading!r} not found in skill"
    return m.group(1)


def _all_entity_labels_in(text: str) -> set[str]:
    """Canonical entity labels referenced as inline code, e.g. ``Repository``."""
    return {
        token
        for token in re.findall(r"`([A-Z][A-Za-z]+)`", text)
        if token in ENTITY_TYPES
    }


def _all_edge_types_in(text: str) -> set[str]:
    """Canonical edge types referenced as inline code, e.g. ``TOUCHED``."""
    return {
        token for token in re.findall(r"`([A-Z_]{3,})`", text) if token in EDGE_TYPES
    }


# ---------------------------------------------------------------------------
# Frontmatter + structure
# ---------------------------------------------------------------------------


def test_skill_file_exists() -> None:
    assert SKILL_PATH.is_file(), f"skill missing at {SKILL_PATH}"


def test_frontmatter_has_required_keys() -> None:
    fm, _ = _read_skill()
    required = {
        "name",
        "description",
        "source_system",
        "event_type",
        "action",
        "enables_planner",
    }
    missing = required - fm.keys()
    assert not missing, f"frontmatter missing keys: {missing}"
    assert fm["source_system"] == "github"
    assert fm["event_type"] == "repository"
    assert fm["action"] == "one_shot_ingest"
    assert fm["enables_planner"].lower() == "true"


def test_all_required_sections_present() -> None:
    _, body = _read_skill()
    for heading in (
        "When to invoke",
        "Inputs",
        "Tools assumed available",
        "Procedure",
        "Mutations (per item)",
        "Bounds and budget",
        "Anti-patterns",
        "Single-event contract",
    ):
        assert re.search(rf"(?m)^##\s+{re.escape(heading)}\s*$", body), (
            f"missing required section: ## {heading}"
        )


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_apply_graph_mutations_signature_matches_real_tool() -> None:
    """Skill documents apply_graph_mutations(plan, event_id, summary)."""
    _, body = _read_skill()
    sig = inspect.signature(
        PydanticDeepReconciliationAgent._build_mutation_tools.__wrapped__
        if hasattr(PydanticDeepReconciliationAgent._build_mutation_tools, "__wrapped__")
        else PydanticDeepReconciliationAgent._build_mutation_tools
    )
    del sig  # we inspect via source instead — the closure is what matters
    source = inspect.getsource(PydanticDeepReconciliationAgent._build_mutation_tools)
    # Real signature in the source:
    assert (
        "async def apply_graph_mutations(\n"
        "            plan: dict[str, Any],\n"
        "            event_id: str,\n"
        "            summary: str,\n" in source
    ), "apply_graph_mutations signature changed; update the skill"
    # Skill documents the same arg order:
    assert "apply_graph_mutations(plan, event_id, summary)" in body, (
        "skill must document the real (plan, event_id, summary) signature"
    )


def test_llm_reconciliation_plan_fields_documented() -> None:
    """Skill enumerates every required LlmReconciliationPlan field."""
    _, body = _read_skill()
    plan_fields = set(LlmReconciliationPlan.model_fields.keys())
    # Required fields in the plan dict the agent passes to apply_graph_mutations.
    for fld in (
        "summary",
        "entity_upserts",
        "edge_upserts",
        "edge_deletes",
        "invalidations",
        "evidence",
        "confidence",
        "warnings",
    ):
        assert fld in plan_fields, (
            f"LlmReconciliationPlan no longer has field {fld!r}; "
            f"adjust the test and skill"
        )
        assert f"`{fld}`" in body, (
            f"skill must mention LlmReconciliationPlan field `{fld}`"
        )


def test_completion_tools_documented() -> None:
    _, body = _read_skill()
    assert "mark_event_processed(event_id, summary)" in body
    assert "finish_batch(summary)" in body


def test_todo_tool_names_match_pydantic_deep_toolset() -> None:
    _, body = _read_skill()
    assert "read_todos" in body
    assert "write_todos" in body
    assert "update_todo_status" in body
    assert "mark_todo_done" not in body


def test_github_tool_names_match_adapter() -> None:
    """Every `github_*` tool the skill references must exist on the read port."""
    _, body = _read_skill()
    referenced = set(re.findall(r"`(github_[a-z_]+)\b", body))
    # The PyGithub adapter is the source of truth for available methods.
    adapter_methods = {
        f"github_{name}"
        for name, _ in inspect.getmembers(
            PyGithubSourceControl, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    unknown = referenced - adapter_methods
    assert not unknown, (
        f"skill names github_* tools that have no adapter method: {unknown}"
    )


def test_skill_documents_include_diff_files_contract() -> None:
    """include_diff=true must match PyGithubSourceControl.get_pull_request."""
    _, body = _read_skill()
    assert "include_diff=true" in body
    proc = _section(body, "Procedure")
    assert "LAST RESORT" in proc
    for fld in ("filename", "status", "additions", "deletions", "patch"):
        assert fld in proc, f"Procedure must document diff file field {fld!r}"
    touched = body[body.find("Touched services") : body.find("## Source-priority")]
    assert "filename" in touched, "Touched services must reference filename"


def test_skill_uses_real_pr_response_fields() -> None:
    """The skill must reference the real field names in get_pull_request."""
    _, body = _read_skill()
    real_fields = {
        "head_branch",
        "base_branch",
        "merged_at",
        "title",
        "body",
        "author",
        "url",
    }
    # Each field the skill explicitly relies on must exist in adapter output.
    for fld in real_fields:
        assert f"`{fld}`" in body or fld in body, (
            f"skill should mention real PR field {fld!r}"
        )
    # Fields the adapter does NOT produce may only appear in a "do not
    # assume" guard. Flag any positive recommendation.
    for fld in ("head_ref", "merge_commit_sha", "changed_files"):
        for m in re.finditer(rf"`{re.escape(fld)}`", body):
            window = body[max(0, m.start() - 200) : m.end() + 50]
            assert (
                "do NOT assume" in window
                or "Do NOT assume" in window
                or "NOT in this response" in window
                or "does NOT return" in window
                or "do not return" in window.lower()
            ), (
                f"skill references {fld!r} without flagging it as missing "
                f"from the adapter response"
            )


def test_skill_uses_real_issue_response_fields() -> None:
    """The issue path must rely only on fields returned by get_issue."""
    _, body = _read_skill()
    issue_section = _section(body, "Procedure")
    issue_idx = issue_section.find("Issue items")
    assert issue_idx >= 0, "Procedure must have an Issue items subsection"
    issue_section = issue_section[issue_idx:]
    for fld in (
        "title",
        "body",
        "state",
        "author",
        "labels",
        "created_at",
        "updated_at",
        "url",
    ):
        assert fld in issue_section, (
            f"issue procedure should mention real issue field {fld!r}"
        )


# ---------------------------------------------------------------------------
# Ontology references
# ---------------------------------------------------------------------------


def test_all_referenced_entity_labels_are_canonical() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    labels = _all_entity_labels_in(section)
    # Sanity: skill should reference at least these.
    expected = {
        "Repository",
        "Person",
        "Activity",
        "Period",
        "BugPattern",
        "Fix",
        "Decision",
    }
    missing = expected - labels
    assert not missing, f"skill Mutations section missing expected labels: {missing}"


def test_all_referenced_edge_types_are_canonical() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    edges = _all_edge_types_in(section)
    expected = {
        "PERFORMED",
        "AUTHORED",
        "TOUCHED",
        "IN_PERIOD",
        "RESOLVED",
        "REPRODUCES",
        "DECIDED",
        "AFFECTS",
    }
    missing = expected - edges
    assert not missing, f"skill missing expected edge types: {missing}"


@pytest.mark.parametrize(
    "edge_type, from_label, to_label",
    [
        ("PERFORMED", "Person", "Activity"),
        ("AUTHORED", "Person", "Activity"),
        ("TOUCHED", "Activity", "Repository"),
        ("IN_PERIOD", "Activity", "Period"),
        ("RESOLVED", "Fix", "BugPattern"),
        ("REPRODUCES", "BugPattern", "Repository"),
        ("DECIDED", "Decision", "Repository"),
        ("AFFECTS", "Decision", "Repository"),
    ],
)
def test_skill_edge_endpoints_match_ontology(
    edge_type: str, from_label: str, to_label: str
) -> None:
    """Each edge the skill emits must be allowed by EDGE_TYPES.allowed_pairs."""
    spec = EDGE_TYPES[edge_type]
    assert spec.allows([from_label], [to_label]), (
        f"{edge_type}: {from_label} → {to_label} is NOT in allowed_pairs "
        f"{spec.allowed_pairs}"
    )


# ---------------------------------------------------------------------------
# Stable key formats
# ---------------------------------------------------------------------------


def test_repository_key_is_slug_format_not_url_format() -> None:
    """Repository is SLUG_ALIAS — `repo:github.com/owner/repo` would be invalid."""
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    # Skill must NOT positively recommend the URL-style key. Mentions inside
    # an explicit "INVALID" / "do NOT" guard are fine (they teach the rule).
    for m in re.finditer(r"repo:github\.com/", section):
        window = section[max(0, m.start() - 200) : m.end() + 200]
        assert "INVALID" in window or "do NOT" in window or "Do NOT" in window, (
            "Repository is SLUG_ALIAS; `repo:github.com/<owner>/<repo>` is "
            "invalid (slashes/dots fail _SLUG_BODY_RE). Skill mentions it "
            "outside a negative-example guard. Use `repo:<owner>-<repo>`."
        )
    # And must mention the slug form somewhere in the Mutations section.
    assert "repo:<owner>-<repo>" in section or "repo:acme-api" in section, (
        "skill should show the real slug Repository key format"
    )
    # Sanity: the slug form actually mints cleanly.
    minted = mint_entity_key(get_identity("Repository"), name="acme/api")
    assert minted == "repo:acme-api"


def test_fix_and_decision_keys_are_content_hash() -> None:
    """Fix and Decision are CONTENT_HASH — keys must be `<prefix>:<12-hex>`."""
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    # Skill must NOT recommend PR-number-encoded keys.
    assert "fix:github:pr:" not in section, (
        "Fix is CONTENT_HASH; key body must be a hash, not `github:pr:<owner>/<repo>:<n>`"
    )
    assert "decision:github:pr:" not in section, (
        "Decision is CONTENT_HASH; key body must be a hash, not PR-encoded"
    )
    # Skill should explicitly show the content-hash form.
    assert "fix:<12-hex" in section, "skill should show fix:<12-hex-sha256> form"
    assert "decision:<12-hex" in section, (
        "skill should show decision:<12-hex-sha256> form"
    )


def test_period_key_matches_production_builder() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    assert "timeline:period:daily:<pot>:<yyyy-mm-dd>" in section, (
        "Period key must match the ontology identity_policy for Period"
    )


def test_activity_key_external_id_form() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    # The example key from the skill must pass _EXTERNAL_ID_SAFE_RE for each
    # post-prefix segment.
    example = "activity:github:pr:acme/api:1042"
    assert example.startswith("activity:")
    segments = example.split(":")[1:]
    for seg in segments:
        assert _EXTERNAL_ID_SAFE_RE.match(seg), (
            f"Activity key segment {seg!r} is not external-id safe"
        )
    # And the skill must mention the canonical form.
    assert "activity:github:pr:<owner>/<repo>:<n>" in section


def test_person_key_is_slug_safe_for_typical_github_handles() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    assert "person:<handle" in section
    # Typical GitHub handles slugify cleanly.
    for handle in ("alice", "bob-smith", "user123"):
        minted = mint_entity_key(get_identity("Person"), name=handle)
        assert minted == f"person:{handle}"


def test_bug_pattern_key_segments_are_slugs() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    assert "bug_pattern:<repo-slug>:<symptom-slug>" in section, (
        "skill should clarify BugPattern key uses slug segments"
    )
    # Sanity: each segment of an example key matches the slug body regex.
    example = "bug_pattern:acme-api:db-timeout"
    parts = example.split(":")
    assert parts[0] == "bug_pattern"
    for seg in parts[1:]:
        assert _SLUG_BODY_RE.match(seg), f"BugPattern segment {seg!r} not slug-valid"


# ---------------------------------------------------------------------------
# Identity-class invariants — guard against ontology silently flipping a class.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected_class, expected_prefix",
    [
        ("Repository", IdentityClass.SLUG_ALIAS, "repo"),
        ("Person", IdentityClass.SLUG_ALIAS, "person"),
        ("Activity", IdentityClass.EXTERNAL_ID, "activity"),
        ("Period", IdentityClass.SLUG_ALIAS, "period"),
        ("BugPattern", IdentityClass.SLUG_ALIAS, "bug_pattern"),
        ("Fix", IdentityClass.CONTENT_HASH, "fix"),
        ("Decision", IdentityClass.CONTENT_HASH, "decision"),
    ],
)
def test_ontology_identity_classes_unchanged(
    label: str, expected_class: IdentityClass, expected_prefix: str
) -> None:
    spec = ENTITY_TYPES[label]
    assert spec.identity_class is expected_class, (
        f"{label} identity_class changed; re-validate the skill's key guidance"
    )
    assert spec.key_prefix == expected_prefix


# ---------------------------------------------------------------------------
# Anti-pattern guarantees
# ---------------------------------------------------------------------------


def test_skill_explicitly_forbids_repository_added_reemit() -> None:
    _, body = _read_skill()
    anti = _section(body, "Anti-patterns")
    assert "repository.added" in anti.lower() or "repository.added" in anti
    assert "do not" in anti.lower(), "Anti-patterns section should be imperative"


def test_skill_caps_list_pagination_to_two_calls_one_per_kind() -> None:
    """Two list calls total — one for PRs, one for issues. No pagination."""
    _, body = _read_skill()
    bounds = _section(body, "Bounds and budget")
    # Either "Two" or explicit per-tool "one" each is acceptable phrasing.
    assert (
        "Two" in bounds
        or "two" in bounds
        or (
            "one `github_list_pull_requests`" in bounds
            and "one `github_list_issues`" in bounds
        )
    ), "Bounds must state two list calls (one per kind)"
    assert "github_list_pull_requests" in bounds
    assert "github_list_issues" in bounds
    assert "No pagination" in bounds or "no pagination" in bounds.lower()


# ---------------------------------------------------------------------------
# EventPlaybook registration (slice 2)
# ---------------------------------------------------------------------------


def test_one_shot_ingest_playbook_is_registered() -> None:
    """The internal agent must be able to resolve the one-shot event-kind."""
    from potpie_context_engine.domain.event_playbooks import (
        find_playbook,
        is_default_playbook,
    )

    pb = find_playbook("github", "repository", "one_shot_ingest")
    assert not is_default_playbook(pb), (
        "(github, repository, one_shot_ingest) fell through to the default "
        "playbook — registration missing"
    )
    assert pb.source_system == "github"
    assert pb.event_type == "repository"
    assert pb.action == "one_shot_ingest"
    assert pb.enables_planner is True
    assert pb.max_tool_calls >= 100, (
        "one-shot ingestion needs a generous tool-call budget"
    )


def test_one_shot_playbook_extract_is_the_markdown_body() -> None:
    """The skill markdown body must be embedded verbatim in the playbook."""
    from potpie_context_engine.domain.event_playbooks import find_playbook

    pb = find_playbook("github", "repository", "one_shot_ingest")
    _, body = _read_skill()
    # Sanity: a few distinctive markers from the markdown must appear in the
    # rendered playbook so they end up in the agent prompt.
    for marker in (
        "# Repo change-history ingestion (one-shot)",
        "## Mutations (per item)",
        "## Anti-patterns",
        "Source-priority rationale",
        "Single-event contract",
        "activity:github:pr:<owner>/<repo>:<n>",
    ):
        assert marker in pb.extract, (
            f"playbook extract missing skill marker: {marker!r}"
        )
    # And it must NOT contain the YAML frontmatter — the loader strips it.
    assert not pb.extract.startswith("---"), (
        "playbook extract should strip the markdown frontmatter"
    )
    # Body length should be roughly the same as the markdown body (the loader
    # only strips frontmatter, no other transformation).
    assert abs(len(pb.extract) - len(body.lstrip())) < 50, (
        "playbook extract length diverges from skill body — loader transform?"
    )


def test_one_shot_playbook_tool_hints_reference_real_tools() -> None:
    from potpie_context_engine.domain.event_playbooks import find_playbook

    pb = find_playbook("github", "repository", "one_shot_ingest")
    # Every hinted github_* tool must exist on the adapter.
    adapter_methods = {
        f"github_{name}"
        for name, _ in inspect.getmembers(
            PyGithubSourceControl, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    for hint in pb.tool_hints:
        if hint.startswith("github_"):
            assert hint in adapter_methods, f"tool_hint {hint!r} not in adapter methods"


def test_one_shot_playbook_renders_into_agent_prompt() -> None:
    """render_playbooks_section must include the one-shot skill body."""
    from potpie_context_engine.domain.event_playbooks import (
        find_playbook,
        render_playbooks_section,
    )

    pb = find_playbook("github", "repository", "one_shot_ingest")
    rendered = render_playbooks_section([pb])
    assert "github / repository / one_shot_ingest" in rendered
    assert "## Mutations (per item)" in rendered
    assert "Anti-patterns" in rendered


def test_skill_documents_source_priority_order() -> None:
    """Code is last-resort; the skill must enforce that priority."""
    _, body = _read_skill()
    proc = _section(body, "Procedure")
    # Priority list appears in order. We assert the ordering by relative index.
    for earlier, later in [
        ("Commit messages", "Branch name"),
        ("Branch name", "PR title"),
        ("PR title", "PR description"),
        ("PR description", "Code diff"),
    ]:
        i_e = proc.find(earlier)
        i_l = proc.find(later)
        assert i_e >= 0 and i_l >= 0, (
            f"Procedure missing source-priority label: {earlier!r} / {later!r}"
        )
        assert i_e < i_l, (
            f"source priority broken: {earlier!r} must come before {later!r}"
        )
    assert "LAST RESORT" in proc, "Procedure must call code-reading LAST RESORT"


# ---------------------------------------------------------------------------
# Issue ingestion path (expansion: PRs + issues)
# ---------------------------------------------------------------------------


def test_skill_covers_both_prs_and_issues() -> None:
    """Frontmatter + body must announce the two-kind scope."""
    fm, body = _read_skill()
    assert "issues" in fm["description"].lower()
    assert (
        "pull request" in fm["description"].lower() or "pr" in fm["description"].lower()
    )
    assert "github_list_issues" in body
    assert "github_get_issue" in body
    assert "activity:github:issue:" in body
    assert "activity:github:pr:" in body


def test_skill_phase_1_calls_both_list_tools_with_limit_count() -> None:
    """count is read from event.payload.count and passed as limit on BOTH lists."""
    _, body = _read_skill()
    assert "event.payload.count" in body, (
        "skill must tell the agent to read count from event.payload.count"
    )
    assert "github_list_pull_requests(repo, limit=count)" in body
    assert "github_list_issues(repo, limit=count)" in body


def test_phase_0_initializes_two_todos_one_per_kind() -> None:
    _, body = _read_skill()
    proc = _section(body, "Procedure")
    # Both enumeration todos must be initialized in Phase 0.
    assert "Enumerate" in proc and "merged PRs" in proc
    assert "Enumerate" in proc and "issues" in proc


def test_issue_activity_key_form_documented() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    assert "activity:github:issue:<owner>/<repo>:<n>" in section, (
        "issue Activity key must be activity:github:issue:<owner>/<repo>:<n>"
    )


def test_issue_verb_class_uses_state() -> None:
    _, body = _read_skill()
    section = _section(body, "Mutations (per item)")
    # verb_class value pattern (not a tool name).
    assert "github_issue_<state>" in section or "github_issue_open" in section
    assert "github_issue_closed" in section or "github_issue_<state>" in section


def test_no_fix_emitted_from_issues() -> None:
    """Fix is reserved for merged PRs; issues never emit Fix."""
    _, body = _read_skill()
    anti = _section(body, "Anti-patterns")
    assert "Do NOT emit `Fix` for an issue" in anti, (
        "Anti-patterns must explicitly forbid Fix from issue filings"
    )
    section = _section(body, "Mutations (per item)")
    # Inside the per-issue conditional block, must call this out too.
    issue_block_start = section.find("### Per issue — conditionally emit")
    assert issue_block_start >= 0, "Per issue — conditionally emit block missing"
    issue_block = section[issue_block_start:]
    assert "Do NOT emit `Fix`" in issue_block, (
        "Per-issue conditional must say Do NOT emit Fix"
    )


def test_playbook_tool_hints_include_issue_tools() -> None:
    """Playbook registration must expose the issue-side tools."""
    from potpie_context_engine.domain.event_playbooks import find_playbook

    pb = find_playbook("github", "repository", "one_shot_ingest")
    assert "github_list_issues" in pb.tool_hints
    assert "github_get_issue" in pb.tool_hints


def test_skill_documents_issue_source_priority_order() -> None:
    """For issues: labels > state > title > body."""
    _, body = _read_skill()
    proc = _section(body, "Procedure")
    # Anchor on the issue items section to scope the search.
    issue_idx = proc.find("Issue items")
    assert issue_idx >= 0, "Procedure must have an Issue items subsection"
    issue_section = proc[issue_idx:]
    for earlier, later in [
        ("Labels", "State"),
        ("State", "Title"),
        ("Title", "Body"),
    ]:
        i_e = issue_section.find(earlier)
        i_l = issue_section.find(later)
        assert i_e >= 0 and i_l >= 0, (
            f"Issue procedure missing priority label: {earlier!r}/{later!r}"
        )
        assert i_e < i_l, (
            f"issue source priority broken: {earlier!r} must precede {later!r}"
        )


def test_discussions_explicitly_out_of_scope() -> None:
    """GitHub Discussions have no connector tool; skill must call this out."""
    _, body = _read_skill()
    assert "Discussions" in body
    anti = _section(body, "Anti-patterns")
    assert "Discussions" in anti and "unsupported" in anti.lower()


def test_issue_anti_pattern_no_invented_issue_comments() -> None:
    """There's no separate issue-comments tool; skill must forbid inventing them."""
    _, body = _read_skill()
    anti = _section(body, "Anti-patterns")
    assert "issue comments" in anti.lower() or "issue-comments" in anti.lower()
