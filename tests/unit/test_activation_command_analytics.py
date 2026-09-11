from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import potpie.cli.telemetry.product_analytics as product_analytics
from potpie.cli.commands import _common, query
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.product_analytics import ProductAnalyticsEvent
from potpie_context_engine.outcomes import DomainError

runner = CliRunner()


@dataclass
class _FakeSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)


@pytest.fixture()
def fake_sink(monkeypatch: pytest.MonkeyPatch) -> _FakeSink:
    sink = _FakeSink()
    monkeypatch.setattr(product_analytics, "_sink", sink)
    monkeypatch.setattr(
        product_analytics,
        "current_telemetry_context",
        lambda: TelemetryContext(
            anonymous_install_id="install_123",
            invocation_id="invoke_456",
            daemon_session_id=None,
            environment="staging",
            command="resolve",
            subcommand=None,
            output_mode="text",
            cli_version="0.1.0",
            python_version="3.13.0",
            os="darwin",
            arch="arm64",
        ),
    )
    return sink


def _query_app() -> typer.Typer:
    app = typer.Typer()
    query.register(app)
    return app


def _envelope(*, item_count: int, confidence: str):
    private_payload = {
        "fact": "private returned context from /Users/example/private-repo"
    }
    return SimpleNamespace(
        pot_id="private-pot-name",
        intent="feature",
        overall_confidence=confidence,
        items=[
            SimpleNamespace(include="architecture", score=0.9, payload=private_payload)
            for _ in range(item_count)
        ],
        coverage=[],
        unsupported_includes=[],
    )


@pytest.mark.parametrize(
    ("command", "argument", "item_count", "confidence", "result_kind"),
    [
        ("resolve", "private task text", 2, "high", "non_empty"),
        ("resolve", "private task text", 0, "low", "empty"),
        ("search", "private search query", 1, "medium", "non_empty"),
        ("search", "private search query", 0, "unknown", "empty"),
    ],
)
def test_context_commands_emit_canonical_activation_events(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    argument: str,
    item_count: int,
    confidence: str,
    result_kind: str,
) -> None:
    envelope = _envelope(item_count=item_count, confidence=confidence)
    client = SimpleNamespace(
        resolve=lambda _request: envelope,
        search=lambda _request: envelope,
    )
    monkeypatch.setattr(query, "get_engine_client", lambda _pot: client)
    monkeypatch.setattr(query, "run_engine_operation", lambda result: result)

    result = runner.invoke(_query_app(), [command, argument])

    assert result.exit_code == 0, result.stdout
    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_activation_command_outcome",
        "cli_onboarding_context_result_returned",
        "cli_usage_command_succeeded",
    ]
    outcome = fake_sink.events[0]
    assert outcome.properties["command"] == command
    assert outcome.properties["outcome"] == "succeeded"
    assert outcome.properties["result_kind"] == "context_result"
    assert isinstance(outcome.properties["duration_ms"], int)
    context_result = fake_sink.events[1]
    assert context_result.properties["result_kind"] == result_kind
    assert context_result.properties["item_count"] == item_count
    assert context_result.properties["confidence"] == confidence

    captured_properties = repr([event.properties for event in fake_sink.events])
    assert argument not in captured_properties
    assert "private-pot-name" not in captured_properties
    assert "private-repo" not in captured_properties


def test_non_activation_command_emits_no_activation_events(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = SimpleNamespace(
        status="recorded",
        record_id="private-record-id",
        mutations_applied=1,
    )
    client = SimpleNamespace(record=lambda _request: receipt)
    monkeypatch.setattr(query, "get_engine_client", lambda _pot: client)
    monkeypatch.setattr(query, "run_engine_operation", lambda result: result)

    result = runner.invoke(
        _query_app(),
        ["record", "--type", "fix", "--summary", "private summary"],
    )

    assert result.exit_code == 0, result.stdout
    assert [event.name for event in fake_sink.events] == ["cli_usage_command_succeeded"]


def test_expected_command_failure_emits_only_bounded_outcome(
    fake_sink: _FakeSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_error = "private query and /Users/example/private-repo"

    def _raise_expected_failure(_pot):
        raise _common.EngineClientError(
            DomainError(code="private_error_code", message=private_error)
        )

    monkeypatch.setattr(query, "get_engine_client", _raise_expected_failure)

    result = runner.invoke(_query_app(), ["search", "private search query"])

    assert result.exit_code == 1
    assert len(fake_sink.events) == 1
    event = fake_sink.events[0]
    assert event.name == "cli_onboarding_activation_command_outcome"
    assert event.properties["outcome"] == "expected_failed"
    assert event.properties["failure_category"] == "validation"
    captured_properties = repr(event.properties)
    assert private_error not in captured_properties
    assert "private_error_code" not in captured_properties
