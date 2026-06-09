# ruff: noqa: S101
from __future__ import annotations

import logging
import sys

from sentry_sdk.transport import Transport

from observability import ObservabilityConfig, configure, get_logger, log_context
from observability.config import SentryConfig
from observability.sinks.sentry_sink import SentrySink


class MemoryTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def capture_event(self, event: dict) -> None:
        self.events.append(event)

    def capture_envelope(self, envelope) -> None:
        event = envelope.get_event()
        if event is not None:
            self.events.append(event)


def test_sentry_sink_uses_no_network_transport_and_allowlists_fields() -> None:
    transport = MemoryTransport()
    configure(
        ObservabilityConfig(
            service_name="potpie-cli",
            env="staging",
            sinks=["sentry"],
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                environment="staging",
                release="potpie-cli@test",
                transport=transport,
            ),
        )
    )
    logger = get_logger("tests.sentry")

    with log_context(
        anonymous_install_id="install-1",
        invocation_id="invoke-1",
        daemon_session_id="daemon-1",
    ):
        try:
            raise RuntimeError("secret token=ghp_123456789012345678901234567890123456")
        except RuntimeError:
            logger.exception(
                "unexpected failure for deeptendu@example.com",
                extra={
                    "command": "query",
                    "repo_name": "secret-repo",
                    "error.code": "unexpected_cli_error",
                    "error.kind": "unexpected",
                    "is_expected": "false",
                },
            )

    assert len(transport.events) == 1
    event = transport.events[0]
    assert event["tags"]["command"] == "query"
    assert event["tags"]["error.code"] == "unexpected_cli_error"
    assert "repo_name" not in event["tags"]
    assert event["contexts"]["telemetry"]["anonymous_install_id"] == "install-1"
    assert "server_name" not in event
    exception = event["exception"]["values"][0]
    assert "value" not in exception
    blob = repr(event)
    assert "deeptendu@example.com" not in blob
    assert "ghp_" not in blob
    assert "secret-repo" not in blob


def test_sentry_sink_does_not_attach_sensitive_logging_breadcrumbs() -> None:
    transport = MemoryTransport()
    configure(
        ObservabilityConfig(
            service_name="potpie-cli",
            env="staging",
            sinks=["sentry"],
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                environment="staging",
                release="potpie-cli@breadcrumbs",
                transport=transport,
            ),
        )
    )
    logger = get_logger("tests.sentry.breadcrumbs")

    logger.warning(
        "Authorization: Basic abcdefghijklmnop "
        "sk-123456789012345678901234 "
        "AKIAABCDEFGHIJKLMNOP "
        "xoxb-1234567890-secret"
    )
    try:
        raise RuntimeError("unexpected")
    except RuntimeError:
        logger.exception(
            "unexpected failure",
            extra={
                "error.code": "unexpected_cli_error",
                "error.kind": "unexpected",
                "is_expected": "false",
            },
        )

    assert len(transport.events) == 1
    blob = repr(transport.events[0])
    assert "Basic abcdefghijklmnop" not in blob
    assert "sk-123456789012345678901234" not in blob
    assert "AKIAABCDEFGHIJKLMNOP" not in blob
    assert "xoxb-1234567890-secret" not in blob


def test_sentry_sink_captures_error_log_with_safe_scrubbed_message() -> None:
    transport = MemoryTransport()
    configure(
        ObservabilityConfig(
            service_name="potpie-cli",
            env="staging",
            sinks=["sentry"],
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                environment="staging",
                release="potpie-cli@plain-log",
                transport=transport,
            ),
        )
    )
    logger = get_logger("tests.sentry.plain")

    logger.error(
        "database adapter failed for retryable status",
        extra={
            "error.code": "plain_log_error",
            "error.kind": "unexpected",
            "is_expected": "false",
        },
    )

    assert len(transport.events) == 1
    event = transport.events[0]
    assert event["message"] == "potpie.plain_log_error"
    assert (
        event["extra"]["log_message"] == "database adapter failed for retryable status"
    )
    assert "logentry" not in event


def test_sentry_sink_drops_sensitive_plain_log_message() -> None:
    transport = MemoryTransport()
    configure(
        ObservabilityConfig(
            service_name="potpie-cli",
            env="staging",
            sinks=["sentry"],
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                environment="staging",
                release="potpie-cli@sensitive-plain-log",
                transport=transport,
            ),
        )
    )
    logger = get_logger("tests.sentry.sensitive_plain")

    logger.error(
        "agent prompt: summarize confidential roadmap for Project Zephyr",
        extra={
            "error.code": "plain_log_error",
            "error.kind": "unexpected",
            "is_expected": "false",
        },
    )

    assert len(transport.events) == 1
    event = transport.events[0]
    assert event["message"] == "potpie.plain_log_error"
    assert "log_message" not in event.get("extra", {})
    assert "logentry" not in event
    blob = repr(event)
    assert "confidential roadmap" not in blob
    assert "Project Zephyr" not in blob


def test_disabled_sentry_builds_no_handler() -> None:
    cfg = ObservabilityConfig(
        sentry=SentryConfig(enabled=False, dsn=None),
    )

    assert SentrySink().build_handler(cfg) is None


def test_sentry_sdk_failures_are_non_fatal(monkeypatch) -> None:
    def fail_flush(*, timeout: float) -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr("sentry_sdk.flush", fail_flush)

    SentrySink().shutdown(ObservabilityConfig())


def test_sentry_init_disables_source_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture_init(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("sentry_sdk.init", capture_init)

    SentrySink().setup(
        ObservabilityConfig(
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                release="potpie-cli@source-context",
            )
        )
    )

    assert captured["include_source_context"] is False


def test_sentry_init_failure_is_silent(monkeypatch, capsys) -> None:
    def fail_init(**kwargs) -> None:
        raise RuntimeError("init failed")

    monkeypatch.setattr("sentry_sdk.init", fail_init)

    SentrySink().setup(
        ObservabilityConfig(
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
                release="potpie-cli@init-failure",
            )
        )
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_sentry_capture_failure_is_silent(monkeypatch, capsys) -> None:
    handler = SentrySink().build_handler(
        ObservabilityConfig(
            sentry=SentryConfig(
                enabled=True,
                dsn="https://public@example.com/1",
            )
        )
    )
    assert handler is not None
    monkeypatch.setattr(
        "sentry_sdk.capture_exception",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            "tests.sentry",
            logging.ERROR,
            __file__,
            1,
            "message",
            None,
            sys.exc_info(),
        )
        handler.emit(record)

    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
