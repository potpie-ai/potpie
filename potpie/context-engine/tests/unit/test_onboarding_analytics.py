from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from context_engine.adapters.inbound.cli.auth import auth_commands
from context_engine.adapters.inbound.cli.telemetry.onboarding_events import (
    CliSetupAnalyticsObserver,
    begin_setup_run,
    capture_activation_succeeded,
    capture_github_prompt_outcome,
    capture_github_prompt_shown,
    capture_setup_completed,
    capture_setup_started,
    current_setup_run_id,
    onboarding_entrypoint,
    repo_location_kind,
)
from context_engine.adapters.inbound.cli.telemetry.product_analytics import (
    ProductAnalyticsEvent,
    set_product_analytics_sink,
)
from context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from context_engine.bootstrap.host_wiring import build_host_shell
from context_engine.domain.lifecycle import SetupPlan


@dataclass
class _FakeSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)


@pytest.fixture()
def fake_sink(monkeypatch: pytest.MonkeyPatch) -> _FakeSink:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "context_engine.adapters.inbound.cli.telemetry.product_analytics.current_telemetry_context",
        lambda: _telemetry_context(),
    )
    return sink


def _telemetry_context():
    from context_engine.adapters.inbound.cli.telemetry.context import TelemetryContext

    return TelemetryContext(
        anonymous_install_id="install_123",
        invocation_id="invoke_456",
        daemon_session_id="daemon_789",
        environment="staging",
        command="setup",
        subcommand=None,
        output_mode="json",
        cli_version="0.1.0",
        python_version="3.13.0",
        os="darwin",
        arch="arm64",
    )


def test_repo_location_kind_does_not_expose_paths() -> None:
    assert repo_location_kind(None) == "none"
    assert repo_location_kind(".") == "current_directory"
    assert repo_location_kind("/Users/dsantra/private/repo") == "explicit_path"


def test_setup_events_share_one_setup_run_id(fake_sink: _FakeSink) -> None:
    plan = SetupPlan(repo=".", agent="claude", scan=True, assume_yes=True)

    setup_run_id = begin_setup_run()
    capture_setup_started(plan, interactive=False, json_output=True)
    capture_setup_completed(
        plan=plan,
        ok=True,
        duration_ms=12,
        hard_failed_step=None,
        soft_warning_count=1,
    )

    assert current_setup_run_id() == setup_run_id
    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_setup_started",
        "cli_onboarding_setup_completed",
    ]
    assert fake_sink.events[0].properties["setup_run_id"] == setup_run_id
    assert fake_sink.events[0].properties["repo_location_kind"] == "current_directory"
    assert fake_sink.events[0].properties["scan_requested"] is True
    assert fake_sink.events[1].properties["duration_ms"] == 12
    assert "repo" not in fake_sink.events[0].properties


def test_setup_observer_emits_step_timing(
    fake_sink: _FakeSink, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    host = build_host_shell(backend=InMemoryGraphBackend())
    host.setup.set_observer(CliSetupAnalyticsObserver())
    begin_setup_run()

    report = host.setup.run(SetupPlan(repo=".", agent="claude", defer_default_pot=True))

    assert report.ok is True
    names = [event.name for event in fake_sink.events]
    assert "cli_onboarding_setup_step_started" in names
    assert "cli_onboarding_setup_step_completed" in names
    source_events = [
        event
        for event in fake_sink.events
        if event.properties.get("step") == "source"
        and event.name == "cli_onboarding_setup_step_completed"
    ]
    assert source_events[0].properties["step_state"] == "skipped"
    duration_ms = source_events[0].properties["duration_ms"]
    assert isinstance(duration_ms, int)
    assert duration_ms >= 0


def test_github_prompt_events_record_prompt_outcome(fake_sink: _FakeSink) -> None:
    begin_setup_run()

    capture_github_prompt_shown(default_answer=True)
    capture_github_prompt_outcome("accepted", duration_ms=25)

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_github_prompt_shown",
        "cli_onboarding_github_prompt_accepted",
    ]
    assert fake_sink.events[1].properties["duration_ms"] == 25


def test_github_prompt_unknown_outcome_falls_back_to_aborted(
    fake_sink: _FakeSink,
) -> None:
    begin_setup_run()

    capture_github_prompt_outcome("unexpected", duration_ms=25)

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_github_prompt_aborted"
    ]
    assert fake_sink.events[0].properties["duration_ms"] == 25


def test_activation_event_marks_context_results(fake_sink: _FakeSink) -> None:
    begin_setup_run()

    capture_activation_succeeded(
        command="resolve",
        result_kind="context_result",
        item_count=3,
    )

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_first_use_command_succeeded",
        "cli_onboarding_first_context_result_returned",
    ]
    assert fake_sink.events[0].properties["command"] == "resolve"
    assert fake_sink.events[1].properties["item_count"] == 3


def test_direct_linear_login_records_integration_funnel(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []

    def _fake_linear_login(*, force: bool = False, add: bool = False) -> None:
        calls.append((force, add))

    monkeypatch.setattr(auth_commands, "_run_linear_oauth_flow", _fake_linear_login)

    auth_commands.linear_login(force=True, add=True)

    assert calls == [(True, True)]
    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_integration_auth_started",
        "cli_onboarding_integration_auth_completed",
    ]
    assert fake_sink.events[0].properties["provider"] == "linear"
    assert fake_sink.events[0].properties["entrypoint"] == "direct_integration_auth"


def test_integration_login_records_unexpected_failure(
    fake_sink: _FakeSink,
) -> None:
    class _IntegrationFailure(RuntimeError):
        pass

    def _run() -> None:
        raise _IntegrationFailure("boom")

    with pytest.raises(_IntegrationFailure):
        auth_commands._run_tracked_integration_login(
            "linear",
            entrypoint="direct_integration_auth",
            runner=_run,
        )

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_integration_auth_started",
        "cli_onboarding_integration_auth_failed",
    ]
    assert fake_sink.events[1].properties["provider"] == "linear"
    assert fake_sink.events[1].properties["failure_kind"] == "_IntegrationFailure"


def test_setup_picker_login_preserves_entrypoint(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth_commands,
        "_run_linear_oauth_flow",
        lambda *, force=False, add=False: None,
    )

    with onboarding_entrypoint("post_setup_integration_picker"):
        auth_commands.run_integration_login("linear")

    assert fake_sink.events[0].properties["provider"] == "linear"
    assert (
        fake_sink.events[0].properties["entrypoint"] == "post_setup_integration_picker"
    )


def test_setup_atlassian_login_runs_shared_jira_flow(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_atlassian_login(product: str, **kwargs: object) -> None:
        calls.append({"product": product, **kwargs})

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(auth_commands, "_flags", lambda: (False, False))
    monkeypatch.setattr(
        auth_commands,
        "run_atlassian_api_token_auth",
        _fake_atlassian_login,
    )

    with onboarding_entrypoint("post_setup_integration_picker"):
        auth_commands.run_integration_login("atlassian", force=True)

    assert calls == [
        {"product": "atlassian", "force": True, "as_json": False, "verbose": False}
    ]
    assert fake_sink.events[0].properties["provider"] == "atlassian"
    assert (
        fake_sink.events[0].properties["entrypoint"] == "post_setup_integration_picker"
    )


def test_direct_atlassian_login_records_funnel_without_credentials(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_atlassian_login(product: str, **kwargs: object) -> None:
        calls.append({"product": product, **kwargs})

    monkeypatch.setattr(auth_commands, "load_cli_env", lambda: None)
    monkeypatch.setattr(
        auth_commands,
        "run_atlassian_api_token_auth",
        _fake_atlassian_login,
    )

    auth_commands.jira_login(
        force=True,
        email="developer@example.com",
        api_token="secret-token",
        site_subdomain="potpie",
    )

    assert calls[0]["product"] == "jira"
    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_integration_auth_started",
        "cli_onboarding_integration_auth_completed",
    ]
    for event in fake_sink.events:
        assert event.properties["provider"] == "jira"
        assert "email" not in event.properties
        assert "api_token" not in event.properties
        assert "site_subdomain" not in event.properties
