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
    @dataclass
    class _SendCall:
        url: str
        api_key: str
        payload: dict[str, object]

    calls: list[_SendCall] = []

    def _send(url: str, *, api_key: str, payload: dict[str, object]) -> None:
        calls.append(_SendCall(url=url, api_key=api_key, payload=payload))

    monkeypatch.setattr(product_analytics, "_send_product_analytics_event", _send)
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

    # One batch endpoint per host; the key rides on the batch, not the event.
    assert calls[0].url == "https://us.i.posthog.com/batch/"
    assert calls[0].api_key == "phc_test"
    payload = calls[0].payload
    assert "api_key" not in payload
    assert payload["event"] == "cli_onboarding_setup_started"
    assert payload["distinct_id"] == "install_123"
    properties = payload["properties"]
    assert isinstance(properties, dict)
    assert properties == {"repo_location_kind": "explicit_path"}


def test_product_analytics_dispatcher_batches_queued_events(monkeypatch) -> None:
    """Every event a process captured leaves in one POST on one daemon thread.

    The per-event shape this replaced opened a fresh TLS connection for each
    event and held the interpreter open (non-daemon worker) until the last one
    was answered — ``search`` paid ~2.4 s for three events after it had already
    printed its result.
    """
    posts: list[tuple[str, dict[str, object]]] = []
    thread_names: list[str] = []
    daemon_flags: list[bool] = []
    release_worker = product_analytics.threading.Event()

    def _post(*, url: str, payload: dict[str, object]) -> None:
        worker_thread = product_analytics.threading.current_thread()
        thread_names.append(worker_thread.name)
        daemon_flags.append(worker_thread.daemon)
        posts.append((url, payload))

    monkeypatch.setattr(product_analytics, "_post_product_analytics_batch", _post)
    dispatcher = product_analytics._ProductAnalyticsDispatcher()
    # Hold the worker until both events are queued so the batch is deterministic.
    original_run = dispatcher._run

    def _gated_run() -> None:
        release_worker.wait(timeout=1.0)
        original_run()

    monkeypatch.setattr(dispatcher, "_run", _gated_run)

    dispatcher.dispatch(
        url="https://us.i.posthog.com/batch/",
        api_key="phc_test",
        payload={
            "event": "cli_onboarding_setup_completed",
            "distinct_id": "install_123",
            "properties": {"repo_location_kind": "explicit_path"},
        },
    )
    dispatcher.dispatch(
        url="https://us.i.posthog.com/batch/",
        api_key="phc_test",
        payload={
            "event": "cli_onboarding_integration_auth_failed",
            "distinct_id": "install_123",
            "properties": {"provider": "github"},
        },
    )
    release_worker.set()

    dispatcher.flush()

    assert len(posts) == 1
    url, payload = posts[0]
    assert url == "https://us.i.posthog.com/batch/"
    assert payload["api_key"] == "phc_test"
    assert [event["event"] for event in payload["batch"]] == [
        "cli_onboarding_setup_completed",
        "cli_onboarding_integration_auth_failed",
    ]
    assert thread_names == ["potpie-product-analytics"]
    assert daemon_flags == [True]


def test_product_analytics_batches_split_by_destination(monkeypatch) -> None:
    posts: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        product_analytics,
        "_post_product_analytics_batch",
        lambda *, url, payload: posts.append((url, payload)),
    )
    dispatcher = product_analytics._ProductAnalyticsDispatcher()
    release_worker = product_analytics.threading.Event()
    original_run = dispatcher._run
    monkeypatch.setattr(
        dispatcher, "_run", lambda: (release_worker.wait(timeout=1.0), original_run())
    )
    dispatcher.dispatch(
        url="https://us.i.posthog.com/batch/",
        api_key="phc_a",
        payload={"event": "one", "distinct_id": "i", "properties": {}},
    )
    dispatcher.dispatch(
        url="https://eu.i.posthog.com/batch/",
        api_key="phc_b",
        payload={"event": "two", "distinct_id": "i", "properties": {}},
    )
    release_worker.set()

    dispatcher.flush()

    assert sorted(url for url, _ in posts) == [
        "https://eu.i.posthog.com/batch/",
        "https://us.i.posthog.com/batch/",
    ]
    assert {payload["api_key"] for _, payload in posts} == {"phc_a", "phc_b"}


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


def test_product_analytics_dispatcher_flush_uses_bounded_drain(monkeypatch) -> None:
    dispatcher = product_analytics._ProductAnalyticsDispatcher()
    dispatcher._queue.put_nowait(
        product_analytics._QueuedProductAnalyticsEvent(
            url="https://us.i.posthog.com/batch/",
            api_key="phc_test",
            payload={
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
