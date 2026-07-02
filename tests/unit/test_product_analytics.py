from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from potpie.cli.telemetry import product_analytics
from potpie.cli.telemetry import settings as cli_telemetry_settings
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.preferences import (
    TelemetryPreferences,
    save_preferences,
)
from potpie.cli.telemetry.product_analytics import (
    PostHogSink,
    ProductAnalyticsEvent,
    ProductAnalyticsSettings,
    capture_event,
    configure_product_analytics,
    set_product_analytics_sink,
)
from potpie.cli.telemetry.settings import load_product_analytics_settings
from potpie.runtime import settings as runtime_settings

_PRODUCT_ANALYTICS_ENV_NAMES = (
    "POTPIE_ENVIRONMENT",
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
)


@pytest.fixture(autouse=True)
def _clear_product_analytics_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in _PRODUCT_ANALYTICS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.setattr(runtime_settings, "load_distribution_defaults", lambda: {})


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
    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key is None


def test_product_analytics_settings_respect_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key == "phc_test"


def test_product_analytics_settings_respect_persisted_telemetry_disable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    save_preferences(TelemetryPreferences(enabled=False))

    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key == "phc_test"


def test_product_analytics_settings_persisted_enable_preserves_existing_gates(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    save_preferences(TelemetryPreferences(enabled=True))

    settings = load_product_analytics_settings()

    assert settings.enabled is True


def test_product_analytics_settings_use_distribution_defaults_without_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        cli_telemetry_settings,
        "_load_product_analytics_defaults",
        lambda: {
            "posthog_api_key": "phc_dist",
            "posthog_host": "https://dist.invalid",
        },
    )

    settings = load_product_analytics_settings()

    assert settings.enabled is True
    assert settings.api_key == "phc_dist"
    assert settings.host == "https://dist.invalid"


def test_product_analytics_runtime_env_overrides_distribution_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_telemetry_settings,
        "_load_product_analytics_defaults",
        lambda: {
            "posthog_api_key": "phc_dist",
            "posthog_host": "https://dist.invalid",
        },
    )
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_runtime")
    monkeypatch.setenv("POTPIE_POSTHOG_HOST", "https://runtime.invalid")

    settings = load_product_analytics_settings()

    assert settings.enabled is True
    assert settings.api_key == "phc_runtime"
    assert settings.host == "https://runtime.invalid"


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("POTPIE_TELEMETRY_DISABLED", "1"),
        ("POTPIE_PRODUCT_ANALYTICS_ENABLED", "0"),
    ],
)
def test_product_analytics_runtime_opt_out_overrides_distribution_enablement(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
) -> None:
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_runtime")
    monkeypatch.setenv(env_name, env_value)

    settings = load_product_analytics_settings()

    assert settings.enabled is False


def test_capture_event_uses_existing_telemetry_identity(monkeypatch) -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "potpie.cli.telemetry.product_analytics.current_telemetry_context",
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
        "potpie.cli.telemetry.product_analytics.current_telemetry_context",
        lambda: None,
    )

    capture_event("cli_onboarding_setup_started", {"repo_provided": True})

    assert sink.events == []


def test_configure_product_analytics_uses_noop_when_disabled(monkeypatch) -> None:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "potpie.cli.telemetry.product_analytics.current_telemetry_context",
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


def test_product_analytics_dispatcher_flushes_queued_events(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    thread_names: list[str] = []
    daemon_flags: list[bool] = []
    thread_ids: list[int | None] = []
    release_worker = product_analytics.threading.Event()

    def _post(
        *,
        url: str,
        payload: dict[str, str | dict[str, str]],
    ) -> None:
        worker_thread = product_analytics.threading.current_thread()
        thread_names.append(worker_thread.name)
        daemon_flags.append(worker_thread.daemon)
        thread_ids.append(worker_thread.ident)
        calls.append((url, str(payload["event"])))
        if len(calls) == 1:
            release_worker.wait(timeout=1.0)

    monkeypatch.setattr(product_analytics, "_post_product_analytics_payload", _post)

    product_analytics._send_product_analytics_payload(
        "https://us.i.posthog.com/capture/",
        {
            "api_key": "phc_test",
            "event": "cli_onboarding_setup_completed",
            "distinct_id": "install_123",
            "properties": {"repo_location_kind": "explicit_path"},
        },
    )
    product_analytics._send_product_analytics_payload(
        "https://us.i.posthog.com/capture/",
        {
            "api_key": "phc_test",
            "event": "cli_onboarding_integration_auth_failed",
            "distinct_id": "install_123",
            "properties": {"provider": "github"},
        },
    )
    release_worker.set()

    product_analytics._flush_product_analytics_dispatcher()

    assert calls == [
        (
            "https://us.i.posthog.com/capture/",
            "cli_onboarding_setup_completed",
        ),
        (
            "https://us.i.posthog.com/capture/",
            "cli_onboarding_integration_auth_failed",
        ),
    ]
    assert thread_names == ["potpie-product-analytics", "potpie-product-analytics"]
    assert daemon_flags == [False, False]
    assert len(set(thread_ids)) == 1


def test_product_analytics_dispatcher_flush_uses_bounded_drain(monkeypatch) -> None:
    dispatcher = product_analytics._ProductAnalyticsDispatcher()
    dispatcher._queue.put_nowait(
        product_analytics._QueuedProductAnalyticsPayload(
            url="https://us.i.posthog.com/capture/",
            payload={
                "api_key": "phc_test",
                "event": "cli_onboarding_setup_completed",
                "distinct_id": "install_123",
                "properties": {},
            },
        )
    )
    monkeypatch.setattr(
        dispatcher._queue,
        "join",
        lambda: (_ for _ in ()).throw(AssertionError("unbounded queue.join()")),
    )
    monkeypatch.setattr(product_analytics, "_DISPATCH_WORKER_JOIN_TIMEOUT_SECONDS", 0.0)

    dispatcher.flush()
