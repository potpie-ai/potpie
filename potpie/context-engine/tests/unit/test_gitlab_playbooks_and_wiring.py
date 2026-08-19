"""GitLab playbooks are the tool allowlist, and the wiring must honour them.

``PydanticDeepReconciliationAgent._enforce_playbook_tool_allowlist`` treats
each playbook's ``tool_hints`` as a hard allowlist for the external tool
surface. A ``gitlab_*`` tool that no playbook names is silently dropped at
runtime, and a hint naming a tool the adapter cannot serve fails only when
the agent calls it. Both directions are pinned here.
"""

from __future__ import annotations

import inspect
import re

from potpie_context_core.event_playbooks import find_playbook
from potpie_context_engine.adapters.outbound.connectors.gitlab.agent_tools import (
    build_gitlab_tools,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabRestSourceControl,
)
from potpie_context_engine.bootstrap.http_projects import ExplicitPotResolution
from potpie_context_engine.bootstrap.ingestion_server import (
    build_source_connector_registry,
)

_GITLAB_EVENT_KINDS = [
    ("gitlab", "repository", "added"),
    ("gitlab", "repository", "one_shot_ingest"),
    ("gitlab", "merge_request", "merged"),
    ("gitlab", "issue", "opened"),
]


class _State:
    pot_id = "pot-1"


def _adapter_tool_names() -> set[str]:
    return {
        f"gitlab_{name}"
        for name, _ in inspect.getmembers(
            GitLabRestSourceControl, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }


def _built_tool_names() -> set[str]:
    builder = build_gitlab_tools(
        lambda _p: object(), allowed_projects_for_pot=lambda _pot: {"acme/api"}
    )
    return {t.name for t in builder(_State())}


# ----------------------------------------------------------------------
# Playbook registration
# ----------------------------------------------------------------------
def test_every_gitlab_event_kind_resolves_to_its_own_playbook():
    for kind in _GITLAB_EVENT_KINDS:
        pb = find_playbook(*kind)
        assert (pb.source_system, pb.event_type, pb.action) == kind


def test_gitlab_events_do_not_fall_through_to_the_github_playbooks():
    pb = find_playbook("gitlab", "merge_request", "merged")
    assert "github_" not in " ".join(pb.tool_hints)
    assert "github_" not in pb.extract


def test_backfill_playbooks_enable_the_planner():
    for action in ("added", "one_shot_ingest"):
        assert find_playbook("gitlab", "repository", action).enables_planner is True


def test_live_event_playbooks_leave_the_planner_off():
    assert find_playbook("gitlab", "merge_request", "merged").enables_planner is False
    assert find_playbook("gitlab", "issue", "opened").enables_planner is False


# ----------------------------------------------------------------------
# Allowlist ↔ tool surface agreement
# ----------------------------------------------------------------------
def test_every_hinted_gitlab_tool_exists_on_the_adapter():
    adapter = _adapter_tool_names()
    for kind in _GITLAB_EVENT_KINDS:
        for hint in find_playbook(*kind).tool_hints:
            if hint.startswith("gitlab_"):
                assert hint in adapter, (
                    f"{kind}: tool_hint {hint!r} has no adapter method"
                )


def test_backfill_playbooks_allow_the_whole_gitlab_tool_surface():
    """A dropped tool would silently cripple backfill instead of erroring."""
    built = _built_tool_names()
    for action in ("added", "one_shot_ingest"):
        hints = set(find_playbook("gitlab", "repository", action).tool_hints)
        assert built <= hints, f"{action} playbook omits {sorted(built - hints)}"


def test_merged_playbook_allows_every_review_history_tool():
    hints = set(find_playbook("gitlab", "merge_request", "merged").tool_hints)
    assert {
        "gitlab_get_merge_request",
        "gitlab_get_merge_request_commits",
        "gitlab_get_merge_request_discussions",
        "gitlab_get_merge_request_notes",
        "gitlab_get_merge_request_approvals",
        "gitlab_get_merge_request_closes_issues",
    } <= hints


def test_issue_playbook_allows_every_task_tracking_tool():
    hints = set(find_playbook("gitlab", "issue", "opened").tool_hints)
    assert {
        "gitlab_get_issue",
        "gitlab_get_issue_notes",
        "gitlab_get_issue_links",
    } <= hints


# ----------------------------------------------------------------------
# One-shot skill body
# ----------------------------------------------------------------------
def _skill_body() -> str:
    return find_playbook("gitlab", "repository", "one_shot_ingest").extract


def test_skill_body_loaded_not_a_missing_file_stub():
    body = _skill_body()
    assert "is unavailable" not in body
    assert body.startswith("# GitLab project change-history ingestion")


def test_skill_names_only_tools_the_adapter_serves():
    referenced = set(re.findall(r"`(gitlab_[a-z_]+)\b", _skill_body()))
    unknown = referenced - _adapter_tool_names()
    assert not unknown, f"skill names gitlab_* tools with no adapter method: {unknown}"


def test_skill_documents_the_gitlab_activity_key_format():
    body = _skill_body()
    assert "activity:gitlab:mr:<group>/<project>:<iid>" in body
    assert "activity:gitlab:issue:<group>/<project>:<iid>" in body
    # A GitHub-shaped key here would silently fork the graph identity.
    assert "activity:github:" not in body


def test_skill_documents_the_include_diff_file_contract():
    body = _skill_body()
    assert "include_diff=true" in body
    for field in ("filename", "status", "patch"):
        assert field in body
    # GitLab returns no per-file line counts. The skill must say so
    # explicitly — silence would let the agent reach for GitHub's
    # additions/deletions and find nothing.
    assert "`additions`/`deletions` — do not expect them" in body


def test_skill_warns_about_self_managed_hosts_and_iid_collisions():
    body = _skill_body()
    assert "Do NOT assume `gitlab.com`" in body
    assert "`!7` and `#7` are different" in body


def test_skill_uses_real_merge_request_response_fields():
    body = _skill_body()
    for field in ("head_branch", "base_branch", "merged_at", "merge_status", "draft"):
        assert f"`{field}`" in body


# ----------------------------------------------------------------------
# Container wiring
# ----------------------------------------------------------------------
def _pots():
    return ExplicitPotResolution({"pot-1": "gitlab:acme/platform/api"})


def test_registry_registers_gitlab_when_a_token_is_present():
    registry = build_source_connector_registry(pots=_pots(), gitlab_token="glpat-x")
    connector = registry.get("gitlab")
    assert connector is not None
    assert registry.find_for_webhook("gitlab") is connector


def test_registry_omits_gitlab_without_a_token():
    registry = build_source_connector_registry(pots=_pots(), github_token="ghp_x")
    assert registry.get("gitlab") is None


def test_both_code_hosts_can_be_registered_together():
    registry = build_source_connector_registry(
        pots=_pots(), github_token="ghp_x", gitlab_token="glpat-x"
    )
    assert registry.get("github") is not None
    assert registry.get("gitlab") is not None


def test_gitlab_connector_reports_the_configured_self_managed_host():
    registry = build_source_connector_registry(
        pots=_pots(),
        gitlab_token="glpat-x",
        gitlab_url="https://gitlab.corp.example",
    )
    connector = registry.get("gitlab")
    assert connector._instance_host == "gitlab.corp.example"


def test_registry_still_serves_a_manifest_with_no_code_host_token():
    registry = build_source_connector_registry(pots=_pots())
    assert registry.get("notion") is not None
    assert registry.aggregated_capabilities()
