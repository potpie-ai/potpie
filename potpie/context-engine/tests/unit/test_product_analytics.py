from __future__ import annotations

from dataclasses import dataclass, field

from adapters.inbound.cli.telemetry import product_analytics
from adapters.inbound.cli.telemetry.context import TelemetryContext
from adapters.inbound.cli.telemetry.product_analytics import (
    ProductAnalyticsEvent,
    ProductAnalyticsSettings,
    PostHogSink,
    capture_event,
    configure_product_analytics,
    set_product_analytics_sink,
)
from adapters.inbound.cli.telemetry.settings import load_product_analytics_settings


@dataclass
class _FakeSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)


def _telemetry_context() -> TelemetryContext:
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


def test_product_analytics_settings_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_POSTHOG_API_KEY", raising=False)
    monkeypatch.delenv("POTPIE_TELEMETRY_DISABLED", raising=False)

    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key is None


def test_product_analytics_settings_respect_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key == "phc_test"


def test_capture_event_uses_existing_telemetry_identity(monkeypatch) -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.product_analytics.current_telemetry_context",
        _telemetry_context,
    )

    capture_event(
        "cli_onboarding_setup_started",
        {
            "anonymous_install_id": "callsite_install",
            "environment": "callsite_env",
            "invocation_id": "callsite_invocation",
            "repo_provided": True,
        },
    )

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.name == "cli_onboarding_setup_started"
    assert event.distinct_id == "install_123"
    assert event.properties["anonymous_install_id"] == "install_123"
    assert event.properties["invocation_id"] == "invoke_456"
    assert event.properties["daemon_session_id"] == "daemon_789"
    assert event.properties["environment"] == "staging"
    assert event.properties["repo_provided"] is True


def test_capture_event_is_noop_without_telemetry_context(monkeypatch) -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.product_analytics.current_telemetry_context",
        lambda: None,
    )

    capture_event("cli_onboarding_setup_started", {"repo_provided": True})

    assert sink.events == []


def test_configure_product_analytics_uses_noop_when_disabled(monkeypatch) -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.product_analytics.current_telemetry_context",
        _telemetry_context,
    )

    configure_product_analytics(
        ProductAnalyticsSettings(
            enabled=False, api_key=None, host="https://us.i.posthog.com"
        )
    )
    capture_event("cli_onboarding_setup_started", {"repo_provided": True})

    assert sink.events == []


def test_posthog_sink_payload_excludes_secrets(monkeypatch) -> None:
    @dataclass
    class _PostCall:
        url: str
        payload: dict[str, object]

    calls: list[_PostCall] = []

    def _send(url: str, payload: dict[str, object]) -> None:
        calls.append(_PostCall(url=url, payload=payload))

    monkeypatch.setattr(product_analytics, "_send_product_analytics_payload", _send)
    sink = PostHogSink(
        ProductAnalyticsSettings(
            enabled=True,
            api_key="phc_test",
            host="https://us.i.posthog.com",
        )
    )

    sink.capture(
        ProductAnalyticsEvent(
            name="cli_onboarding_setup_started",
            distinct_id="install_123",
            properties={"repo_location_kind": "explicit_path"},
        )
    )

    assert calls[0].url == "https://us.i.posthog.com/capture/"
    payload = calls[0].payload
    assert payload["api_key"] == "phc_test"
    assert payload["event"] == "cli_onboarding_setup_started"
    assert payload["distinct_id"] == "install_123"
    properties = payload["properties"]
    assert isinstance(properties, dict)
    assert properties == {"repo_location_kind": "explicit_path"}


def test_posthog_sink_schedules_delivery_in_background(monkeypatch) -> None:
    @dataclass
    class _ScheduledThread:
        target_name: str
        url: str
        daemon: bool
        name: str
        started: bool = False

    scheduled: list[_ScheduledThread] = []

    class _Thread:
        def __init__(self, *, target, kwargs, daemon: bool, name: str) -> None:
            scheduled.append(
                _ScheduledThread(
                    target_name=target.__name__,
                    url=kwargs["url"],
                    daemon=daemon,
                    name=name,
                )
            )

        def start(self) -> None:
            scheduled[0].started = True

    monkeypatch.setattr(product_analytics.threading, "Thread", _Thread)

    product_analytics._send_product_analytics_payload(
        "https://us.i.posthog.com/capture/",
        {
            "api_key": "phc_test",
            "event": "cli_onboarding_setup_started",
            "distinct_id": "install_123",
            "properties": {"repo_location_kind": "explicit_path"},
        },
    )

    assert scheduled == [
        _ScheduledThread(
            target_name="_post_product_analytics_payload",
            url="https://us.i.posthog.com/capture/",
            daemon=True,
            name="potpie-product-analytics",
            started=True,
        )
    ]
