from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from potpie.cli.telemetry import product_analytics
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.product_analytics import (
    ProductAnalyticsEvent,
    ProductAnalyticsSettings,
    PostHogSink,
    capture_event,
    configure_product_analytics,
    set_product_analytics_sink,
)
from potpie.cli.telemetry.settings import load_product_analytics_settings
from potpie_context_engine.bootstrap import runtime_settings

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
    from potpie.cli.telemetry.preferences import (
        TelemetryPreferences,
        save_preferences,
    )

    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    save_preferences(TelemetryPreferences(enabled=False))

    settings = load_product_analytics_settings()

    assert settings.enabled is False
    assert settings.api_key == "phc_test"


def test_product_analytics_settings_persisted_enable_preserves_existing_gates(
    monkeypatch,
) -> None:
    from potpie.cli.telemetry.preferences import (
        TelemetryPreferences,
        save_preferences,
    )

    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    save_preferences(TelemetryPreferences(enabled=True))

    settings = load_product_analytics_settings()

    assert settings.enabled is True


def test_product_analytics_settings_use_distribution_defaults_without_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        runtime_settings,
        "load_distribution_defaults",
        lambda: {
            "environment": "prod_oss",
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
        runtime_settings,
        "load_distribution_defaults",
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
    spooled: list[dict[str, object]] = []
    monkeypatch.setattr(
        product_analytics, "_spool_product_analytics_event", spooled.append
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

    # The key never leaves settings: the flusher adds it per batch from the
    # same settings, so it is neither in the event nor on disk.
    payload = spooled[0]
    assert "api_key" not in payload
    assert payload["event"] == "cli_onboarding_setup_started"
    assert payload["distinct_id"] == "install_123"
    properties = payload["properties"]
    assert isinstance(properties, dict)
    assert properties == {"repo_location_kind": "explicit_path"}


def test_posthog_sink_spools_nothing_when_disabled(monkeypatch) -> None:
    spooled: list[dict[str, object]] = []
    monkeypatch.setattr(
        product_analytics, "_spool_product_analytics_event", spooled.append
    )
    sink = PostHogSink(ProductAnalyticsSettings(enabled=False, api_key=None, host="h"))

    sink.capture(ProductAnalyticsEvent(name="e", distinct_id="i", properties={}))

    assert spooled == []


def test_product_analytics_posts_reuse_one_http_client(monkeypatch) -> None:
    created: list[object] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            created.append(self)
            self.posts: list[str] = []

        def post(self, url: str, *, json: object) -> None:
            self.posts.append(url)

        def close(self) -> None:
            return None

    monkeypatch.setattr(product_analytics.httpx, "Client", _Client)
    monkeypatch.setattr(product_analytics, "_http_client", None)

    product_analytics._post_product_analytics_batch(
        url="https://us.i.posthog.com/batch/", payload={"api_key": "k", "batch": []}
    )
    product_analytics._post_product_analytics_batch(
        url="https://us.i.posthog.com/batch/", payload={"api_key": "k", "batch": []}
    )
    product_analytics._close_http_client()

    assert len(created) == 1
    assert product_analytics._http_client is None
