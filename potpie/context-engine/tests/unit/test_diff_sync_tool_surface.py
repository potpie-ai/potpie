"""Guard: every diff-sync playbook tool_hint maps to a real, buildable tool.

This is the test that would have caught the original gap — diff-sync playbooks
listing ``read_sync_history`` / ``jira_search_issues`` / ``linear_list_*`` in
their ``tool_hints`` while no implementation existed. It binds the skill's
declared tool surface to the actual builders + the engine-core tool catalog, so
a future rename or a removed builder fails loudly instead of shipping a
playbook that references a tool the agent can never call.
"""

from __future__ import annotations

import pytest

from adapters.outbound.agent_tools.sync_history import build_sync_history_tools
from adapters.outbound.connectors.jira.agent_tools import build_jira_tools
from adapters.outbound.connectors.linear.agent_tools import build_linear_tools
from adapters.outbound.reconciliation.context_graph_tools import READ_TOOL_INCLUDE
from domain.event_playbooks import find_playbook

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _FullLinearFetcher:
    def get_issue(self, issue_id, *, pot_id=None):
        return {"id": issue_id}

    def list_issues(self, **kw):
        return []

    def list_projects(self, **kw):
        return []

    def get_project(self, project_id, *, pot_id=None):
        return {"id": project_id}

    def list_documents(self, **kw):
        return []

    def get_document(self, document_id, *, pot_id=None):
        return {"id": document_id}


class _FullJiraFetcher:
    def get_issue(self, issue_key, *, pot_id=None):
        return {"key": issue_key}

    def search_issues(self, jql, *, pot_id=None, limit=None):
        return []

    def get_issue_changelog(self, issue_key, *, pot_id=None):
        return []

    def bulk_fetch_changelogs(self, issue_keys, *, pot_id=None):
        return {}


class _FakeStore:
    def read(self, **kw):
        return []

    def append(self, **kw):
        return {"path": "x", "written": True}


def _buildable_tool_names() -> set[str]:
    state = _State("pot-1")
    names: set[str] = set()
    for builder in (
        build_linear_tools(_FullLinearFetcher()),
        build_jira_tools(_FullJiraFetcher()),
        build_sync_history_tools(_FakeStore()),
    ):
        names.update(t.name for t in builder(state))
    # Engine-core tools the agent always has (read + mutation + control), plus
    # the deep agent's built-in planner/todo tools.
    names.update(READ_TOOL_INCLUDE)  # context_search, context_timeline, ...
    names.update(
        {
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
            "read_todos",
            "write_todos",
            "update_todo_status",
        }
    )
    return names


@pytest.mark.parametrize(
    ("source_system", "event_type"),
    [("linear", "linear_team"), ("jira", "jira_project")],
)
def test_every_diff_sync_tool_hint_is_buildable(
    source_system: str, event_type: str
) -> None:
    buildable = _buildable_tool_names()
    pb = find_playbook(source_system, event_type, "diff_sync")

    missing = [hint for hint in pb.tool_hints if hint not in buildable]
    assert not missing, (
        f"{source_system}/{event_type}/diff_sync lists tool_hints with no "
        f"implementation: {missing}"
    )


def test_sync_history_tools_present_for_both_diff_syncs() -> None:
    for source_system, event_type in (
        ("linear", "linear_team"),
        ("jira", "jira_project"),
    ):
        pb = find_playbook(source_system, event_type, "diff_sync")
        assert "read_sync_history" in pb.tool_hints
        assert "write_sync_history" in pb.tool_hints
