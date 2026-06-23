from __future__ import annotations

import logging

from integrations.logging import get_logger, log_context


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_log_context_is_emitted_in_obs_fields():
    logger = logging.getLogger("test.integrations.logging")
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    handler = _CaptureHandler()
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        structured = get_logger(logger.name).bind(component="oauth")
        with log_context(request_id="req-123"):
            structured.info("connected", extra={"obs_fields": {"provider": "linear"}})
    finally:
        logger.handlers[:] = old_handlers
        logger.propagate = old_propagate

    if len(handler.records) != 1:
        raise AssertionError(f"expected one log record, got {len(handler.records)}")

    expected_fields = {
        "request_id": "req-123",
        "component": "oauth",
        "provider": "linear",
    }
    if handler.records[0].obs_fields != expected_fields:
        raise AssertionError(f"unexpected obs_fields: {handler.records[0].obs_fields!r}")
