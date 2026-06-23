from __future__ import annotations

import logging
from io import StringIO

from observability import configure, get_logger, log_context
from observability.config import ObservabilityConfig
from observability.profiles import standalone


def test_structured_logger_redacts_message_and_fields(monkeypatch):
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()
    stream = StringIO()
    bearer_secret = "".join(["abc", "123"])
    token_secret = "-".join(["raw", "token"])
    password_secret = "-".join(["raw", "password"])
    monkeypatch.setattr("sys.stderr", stream)
    monkeypatch.setenv("SERVICE_NAME", "potpie-test")
    try:
        configure(ObservabilityConfig(redact=True))
        logger = get_logger("test.observability")

        with log_context(request_id="req-1"):
            logger.info(
                f"sending Bearer {bearer_secret} token=%s",
                token_secret,
                **{"token": token_secret},
                metadata={"password": password_secret, "safe": "ok"},
            )

        record = stream.getvalue()
    finally:
        root.handlers[:] = old_handlers

    forbidden_values = [bearer_secret, token_secret, password_secret]
    leaked_values = [value for value in forbidden_values if value in record]
    if leaked_values:
        raise AssertionError(f"sensitive values leaked into log output: {leaked_values}")
    if "***REDACTED***" not in record:
        raise AssertionError(f"redaction marker missing from log output: {record}")
    if "req-1" not in record:
        raise AssertionError(f"context field missing from log output: {record}")


def test_configure_formatter_allows_plain_stdlib_logs():
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        configure(ObservabilityConfig(redact=True))
        logging.getLogger("plain").info("plain message")
    finally:
        root.handlers[:] = old_handlers


def test_standalone_preserves_service_name(monkeypatch):
    monkeypatch.setenv("SERVICE_NAME", "standalone-worker")

    if standalone().service_name != "standalone-worker":
        raise AssertionError("standalone profile did not preserve SERVICE_NAME")
