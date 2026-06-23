"""C-5: GitHub agent tools must reject repos not attached to the pot.

The reconciliation agent is prompt-injectable; ``repo_name`` is model-
supplied and ``source_for_repo`` authenticates with a shared org
credential. These tests pin the server-side allowlist binding so a
hijacked agent cannot exfiltrate a foreign private repo.
"""

from __future__ import annotations

from potpie.context_engine.adapters.outbound.connectors.github.agent_tools import build_github_tools


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _FakePort:
    def get_pull_request(self, repo_name, pr_number, include_diff=False):
        return {"repo_name": repo_name, "pr_number": pr_number, "ok": True}


def _tools(allowed, *, pot_id="pot-1", calls=None):
    def source_for_repo(repo_name):
        if calls is not None:
            calls.append(repo_name)
        return _FakePort()

    builder = build_github_tools(
        source_for_repo,
        allowed_repos_for_pot=(lambda _pid: set(allowed))
        if allowed is not None
        else None,
    )
    tools = builder(_State(pot_id))
    return {t.name: t.function for t in tools}


def test_allowed_repo_passes_through():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, calls=calls)
    out = tools["github_get_pull_request"]("acme/widgets", 7)
    assert out == {"repo_name": "acme/widgets", "pr_number": 7, "ok": True}
    assert calls == ["acme/widgets"]


def test_foreign_repo_blocked_before_source_for_repo():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, calls=calls)
    out = tools["github_get_pull_request"]("victim/private", 1, True)
    assert out == {"error": "unknown_repo", "repo_name": "victim/private"}
    assert calls == []  # shared-credential resolver never reached


def test_case_insensitive_match():
    tools = _tools({"acme/widgets"})
    out = tools["github_get_issue"]("ACME/Widgets", 3)
    assert out.get("error") != "unknown_repo"


def test_fail_closed_when_allowlist_unwired():
    calls: list[str] = []
    tools = _tools(None, calls=calls)  # allowed_repos_for_pot not provided
    out = tools["github_list_issues"]("acme/widgets")
    assert out == {"error": "unknown_repo", "repo_name": "acme/widgets"}
    assert calls == []


def test_fail_closed_when_pot_id_missing():
    calls: list[str] = []
    tools = _tools({"acme/widgets"}, pot_id=None, calls=calls)
    out = tools["github_get_pull_request"]("acme/widgets", 1)
    assert out == {"error": "unknown_repo", "repo_name": "acme/widgets"}
    assert calls == []


def test_tool_schema_signature_preserved():
    """functools.wraps must keep the model-facing arg schema intact."""
    builder = build_github_tools(
        lambda r: _FakePort(),
        allowed_repos_for_pot=lambda _p: {"acme/widgets"},
    )
    objs = {t.name: t for t in builder(_State("pot-1"))}
    params = objs["github_get_pull_request"].function_schema.json_schema["properties"]
    assert "repo_name" in params
    assert "pr_number" in params
