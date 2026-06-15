from __future__ import annotations

from dataclasses import dataclass, field

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

    capture_event("cli_onboarding_setup_started", {"repo_provided": True})

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


def test_configure_product_analytics_uses_noop_when_disabled() -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)

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

    class _Client:
        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, json: dict[str, object]) -> None:
            calls.append(_PostCall(url=url, payload=json))

    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.product_analytics.httpx.Client",
        lambda **_kwargs: _Client(),
    )
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
