import pytest

from app.modules.intelligence.tracing import logfire_tracer


def test_logfire_trace_metadata_does_not_mask_inner_exception(monkeypatch):
    class Baggage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(logfire_tracer, "_LOGFIRE_INITIALIZED", True)
    monkeypatch.setattr(logfire_tracer.logfire, "set_baggage", lambda **_: Baggage())

    with pytest.raises(ValueError, match="original failure"):
        with logfire_tracer.logfire_trace_metadata(request_id="req-1"):
            raise ValueError("original failure")
