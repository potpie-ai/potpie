"""C-5: GitLab agent tools must reject projects not attached to the pot.

The reconciliation agent is prompt-injectable; ``project`` is model-
supplied and ``source_for_project`` authenticates with a shared instance
credential. These tests pin the server-side allowlist binding so a
hijacked agent cannot exfiltrate a foreign private project.
"""

from __future__ import annotations

from potpie_context_engine.adapters.outbound.connectors.gitlab.agent_tools import (
    build_gitlab_tools,
)


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _FakePort:
    def get_merge_request(self, project, iid, include_diff=False):
        return {"project": project, "iid": iid, "ok": True}

    def get_issue(self, project, iid):
        return {"project": project, "iid": iid, "ok": True}

    def list_issues(self, project, *, state="all", limit=None):
        return []


def _tools(allowed, *, pot_id="pot-1", calls=None):
    def source_for_project(project):
        if calls is not None:
            calls.append(project)
        return _FakePort()

    builder = build_gitlab_tools(
        source_for_project,
        allowed_projects_for_pot=(lambda _pid: set(allowed))
        if allowed is not None
        else None,
    )
    tools = builder(_State(pot_id))
    return {t.name: t.function for t in tools}


def test_allowed_project_passes_through():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, calls=calls)
    out = tools["gitlab_get_merge_request"]("acme/widgets", 7)
    assert out == {"project": "acme/widgets", "iid": 7, "ok": True}
    assert calls == ["acme/widgets"]


def test_nested_subgroup_project_allowed():
    calls: list[str] = []
    tools = _tools({"acme/platform/widgets"}, calls=calls)
    out = tools["gitlab_get_merge_request"]("acme/platform/widgets", 2)
    assert out["ok"] is True
    assert calls == ["acme/platform/widgets"]


def test_foreign_project_blocked_before_source_for_project():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, calls=calls)
    out = tools["gitlab_get_merge_request"]("victim/private", 1, True)
    assert out == {"error": "unknown_project", "project": "victim/private"}
    assert calls == []  # shared-credential resolver never reached


def test_sibling_project_under_allowed_group_still_blocked():
    """An allowlisted project must not authorize its whole namespace."""
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, calls=calls)
    out = tools["gitlab_get_issue"]("acme/secrets", 1)
    assert out == {"error": "unknown_project", "project": "acme/secrets"}
    assert calls == []


def test_case_and_slash_insensitive_match():
    tools = _tools({"acme/widgets"})
    out = tools["gitlab_get_issue"]("/ACME/Widgets/", 3)
    assert out.get("error") != "unknown_project"


def test_fail_closed_when_allowlist_unwired():
    calls: list[str] = []
    tools = _tools(None, calls=calls)  # allowed_projects_for_pot not provided
    out = tools["gitlab_list_issues"]("acme/widgets")
    assert out == {"error": "unknown_project", "project": "acme/widgets"}
    assert calls == []


def test_fail_closed_when_pot_id_missing():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, pot_id=None, calls=calls)
    out = tools["gitlab_get_merge_request"]("acme/widgets", 1)
    assert out == {"error": "unknown_project", "project": "acme/widgets"}
    assert calls == []


def test_fail_closed_when_allowlist_resolution_raises():
    calls: list[str] = []

    def source_for_project(project):
        calls.append(project)
        return _FakePort()

    def boom(_pot_id):
        raise RuntimeError("pot lookup exploded")

    builder = build_gitlab_tools(source_for_project, allowed_projects_for_pot=boom)
    tools = {t.name: t.function for t in builder(_State("pot-1"))}
    out = tools["gitlab_get_merge_request"]("acme/widgets", 1)
    assert out == {"error": "unknown_project", "project": "acme/widgets"}
    assert calls == []
