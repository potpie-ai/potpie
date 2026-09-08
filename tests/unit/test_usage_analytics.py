from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

import potpie.cli.telemetry.product_analytics as product_analytics
from potpie.cli.auth import auth_commands, github_commands
from potpie.cli.commands import _common, bootstrap, pots, query, skills
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.product_analytics import ProductAnalyticsEvent
from potpie.cli.telemetry.usage_events import (
    capture_usage_command_succeeded,
    usage_feature,
)
from potpie.pots.contracts import PotInfo
from potpie.skills.contracts import SkillOperationResult


@dataclass
class _FakeSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class _FakeGitHubStore:
    def get_provider_credentials(self, provider: str) -> dict[str, str]:
        assert provider == "github"
        return {"access_token": "gh_token"}


@pytest.fixture()
def fake_sink(monkeypatch: pytest.MonkeyPatch) -> _FakeSink:
    sink = _FakeSink()
    monkeypatch.setattr(product_analytics, "_sink", sink)
    monkeypatch.setattr(
        "potpie.cli.telemetry.product_analytics.current_telemetry_context",
        _telemetry_context,
    )
    return sink


@contextmanager
def _cli_json_mode() -> Iterator[None]:
    previous = _common.is_json()
    _common.set_json(True)
    try:
        yield
    finally:
        _common.set_json(previous)


def _telemetry_context() -> TelemetryContext:
    return TelemetryContext(
        anonymous_install_id="install_123",
        invocation_id="invoke_456",
        daemon_session_id="daemon_789",
        environment="staging",
        command="resolve",
        subcommand=None,
        output_mode="json",
        cli_version="0.1.0",
        python_version="3.13.0",
        os="darwin",
        arch="arm64",
    )


def test_usage_event_uses_identity_without_onboarding_fields(
    fake_sink: _FakeSink,
) -> None:
    capture_usage_command_succeeded(
        command="resolve",
        result_kind="context_result",
        item_count=3,
    )

    assert len(fake_sink.events) == 1
    event = fake_sink.events[0]
    assert event.name == "cli_usage_command_succeeded"
    assert event.distinct_id == "install_123"
    assert event.properties["anonymous_install_id"] == "install_123"
    assert event.properties["environment"] == "staging"
    assert event.properties["command"] == "resolve"
    assert event.properties["result_kind"] == "context_result"
    assert event.properties["item_count"] == 3
    assert event.properties["feature"] == "context"
    assert "setup_run_id" not in event.properties
    assert "onboarding_phase" not in event.properties
    assert "entrypoint" not in event.properties


def test_usage_event_accepts_extra_low_cardinality_properties(
    fake_sink: _FakeSink,
) -> None:
    capture_usage_command_succeeded(
        command="graph.read",
        result_kind="graph_command",
        properties={
            "command": "callsite",
            "result_kind": "callsite_kind",
            "command_family": "read",
            "duration_ms": 12.5,
            "subgraph": "recent_changes",
            "view": "recent_changes.timeline",
        },
    )

    event = fake_sink.events[0]
    assert event.name == "cli_usage_command_succeeded"
    assert event.properties["command"] == "graph.read"
    assert event.properties["result_kind"] == "graph_command"
    assert event.properties["command_family"] == "read"
    assert event.properties["duration_ms"] == 12.5
    assert event.properties["subgraph"] == "recent_changes"
    assert event.properties["view"] == "recent_changes.timeline"
    assert event.properties["feature"] == "graph"


def test_context_activation_also_records_usage(fake_sink: _FakeSink) -> None:
    query._capture_context_activation(command="search", item_count=2)

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_first_use_command_succeeded",
        "cli_onboarding_first_context_result_returned",
        "cli_usage_command_succeeded",
    ]
    assert fake_sink.events[-1].properties["command"] == "search"
    assert fake_sink.events[-1].properties["result_kind"] == "context_result"
    assert fake_sink.events[-1].properties["item_count"] == 2
    assert fake_sink.events[-1].properties["feature"] == "context"


def test_host_status_activation_does_not_record_usage(fake_sink: _FakeSink) -> None:
    bootstrap._capture_host_status_activation()

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_first_use_command_succeeded"
    ]


def test_github_repos_records_usage_after_success(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(github_commands, "get_store", lambda: _FakeGitHubStore())
    monkeypatch.setattr(
        github_commands, "ensure_runtime_environment_loaded", lambda: None
    )
    monkeypatch.setattr(
        github_commands,
        "list_user_owned_repositories",
        lambda token: [{"full_name": "potpie/example"}, {"full_name": "potpie/docs"}],
    )

    with _cli_json_mode():
        github_commands.github_repos_impl()

    assert [event.name for event in fake_sink.events] == ["cli_usage_command_succeeded"]
    assert fake_sink.events[0].properties["command"] == "github repos"
    assert fake_sink.events[0].properties["provider"] == "github"
    assert fake_sink.events[0].properties["result_kind"] == "provider_list"
    assert fake_sink.events[0].properties["item_count"] == 2
    assert fake_sink.events[0].properties["feature"] == "integrations"


@pytest.mark.parametrize(
    ("command", "runner_name", "fetch_name"),
    [
        ("linear ls", "linear_ls", "fetch_linear_workspaces"),
        ("jira ls", "jira_ls", "fetch_jira_projects"),
        (
            "confluence ls",
            "confluence_ls",
            "fetch_confluence_spaces_sample",
        ),
    ],
)
def test_provider_list_commands_record_usage_after_success(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    runner_name: str,
    fetch_name: str,
) -> None:
    monkeypatch.setattr(
        auth_commands, "ensure_runtime_environment_loaded", lambda: None
    )
    monkeypatch.setattr(
        auth_commands,
        fetch_name,
        lambda limit: [{"key": "ONE"}, {"key": "TWO"}],
    )

    runner = getattr(auth_commands, runner_name)
    with _cli_json_mode():
        runner(limit=2)

    assert [event.name for event in fake_sink.events] == ["cli_usage_command_succeeded"]
    event = fake_sink.events[0]
    provider = command.split()[0]
    assert event.properties["command"] == command
    assert event.properties["provider"] == provider
    assert event.properties["result_kind"] == "provider_list"
    assert event.properties["item_count"] == 2
    assert event.properties["feature"] == "integrations"


@pytest.mark.parametrize(
    ("product", "provider"),
    [("linear", "linear"), ("jira", "jira"), ("wiki", "confluence")],
)
def test_provider_select_result_records_usage_after_success(
    fake_sink: _FakeSink,
    product: str,
    provider: str,
) -> None:
    with _cli_json_mode():
        auth_commands._run_product_use_result(
            {
                "product": product,
                "workspace_key": "ENG",
                "workspace_name": "Engineering",
                "items": [{"id": "one"}, {"id": "two"}],
            },
            product_label=provider.title(),
        )

    assert [event.name for event in fake_sink.events] == ["cli_usage_command_succeeded"]
    assert fake_sink.events[0].properties["command"] == f"{provider} select"
    assert fake_sink.events[0].properties["provider"] == provider
    assert fake_sink.events[0].properties["result_kind"] == "provider_selection"
    assert fake_sink.events[0].properties["item_count"] == 2
    assert fake_sink.events[0].properties["feature"] == "integrations"


@pytest.mark.parametrize(
    ("command", "result_kind", "feature"),
    [
        ("resolve", "context_result", "context"),
        ("search", "context_result", "context"),
        ("record", "record_result", "context"),
        ("graph.read", "graph_command", "graph"),
        ("github repos", "provider_list", "integrations"),
        ("linear select", "provider_selection", "integrations"),
        ("skills install", "skills_result", "skills"),
        ("skills update", "skills_result", "skills"),
        ("skills remove", "skills_result", "skills"),
        ("ui", "ui_session", "ui"),
        ("pot create", "pot_result", "pots_sources"),
        ("pot use", "pot_result", "pots_sources"),
        ("source add", "source_result", "pots_sources"),
        ("doctor", "status_result", None),
    ],
)
def test_usage_feature_is_a_closed_product_surface(
    command: str,
    result_kind: str,
    feature: str | None,
) -> None:
    assert usage_feature(command=command, result_kind=result_kind) == feature


def test_usage_feature_ignores_callsite_override(fake_sink: _FakeSink) -> None:
    capture_usage_command_succeeded(
        command="resolve",
        result_kind="context_result",
        properties={"feature": "invented"},
    )
    assert fake_sink.events[0].properties["feature"] == "context"


def test_unknown_usage_command_omits_feature(fake_sink: _FakeSink) -> None:
    capture_usage_command_succeeded(command="doctor", result_kind="status_result")
    assert "feature" not in fake_sink.events[0].properties


def test_skills_remove_records_usage(fake_sink: _FakeSink) -> None:
    class _Skills:
        def remove(self, **kwargs: object) -> SkillOperationResult:
            return SkillOperationResult(
                agent="codex",
                operation="remove",
                changed=("potpie-graph",),
            )

    _common.set_runtime(type("Host", (), {"skills": _Skills()})())
    try:
        with _cli_json_mode():
            result = CliRunner().invoke(skills.skills_app, ["remove", "--all"])
        assert result.exit_code == 0, result.output
    finally:
        _common.set_runtime(None)

    usage = [
        event
        for event in fake_sink.events
        if event.name == "cli_usage_command_succeeded"
    ]
    assert len(usage) == 1
    assert usage[0].properties["command"] == "skills remove"
    assert usage[0].properties["feature"] == "skills"
    assert usage[0].properties["item_count"] == 1


def test_pot_create_records_usage(fake_sink: _FakeSink) -> None:
    class _Pots:
        def create_pot(
            self, *, name: str, repo: str | None = None, use: bool = False
        ) -> PotInfo:
            return PotInfo(pot_id="pot-new", name=name, active=use)

        def list_sources(self, *, pot_id: str) -> list[object]:
            return []

    _common.set_runtime(type("Host", (), {"pots": _Pots()})())
    try:
        with _cli_json_mode():
            result = CliRunner().invoke(pots.pot_app, ["create", "demo"])
        assert result.exit_code == 0, result.output
    finally:
        _common.set_runtime(None)

    usage = [
        event
        for event in fake_sink.events
        if event.name == "cli_usage_command_succeeded"
    ]
    assert len(usage) == 1
    assert usage[0].properties["command"] == "pot create"
    assert usage[0].properties["feature"] == "pots_sources"
    assert "demo" not in repr(usage[0].properties)
    assert "pot-new" not in repr(usage[0].properties)
