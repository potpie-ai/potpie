from __future__ import annotations

from typing import Any

from potpie_context_engine.adapters.inbound.cli.telemetry.sentry_privacy import (
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)


def test_before_send_removes_sensitive_event_payload() -> None:
    event: dict[str, Any] = {
        "request": {"headers": {"authorization": "secret"}, "data": "body"},
        "headers": {"authorization": "secret"},
        "env": {"TOKEN": "secret"},
        "extra": {"argv": ["potpie", "secret"]},
        "server_name": "host.local",
        "exception": {
            "values": [
                {
                    "value": "sdk exploded with token=secret",
                    "stacktrace": {
                        "frames": [
                            {
                                "abs_path": "/Users/person/repo/file.py",
                                "context_line": "token = 'secret'",
                                "filename": "/Users/person/repo/file.py",
                                "post_context": ["print(token)"],
                                "pre_context": ["token = load_token()"],
                                "vars": {"token": "secret"},
                            }
                        ]
                    },
                }
            ]
        },
        "tags": {"service": "potpie-cli"},
    }

    scrubbed = scrub_sentry_event(event, {})

    assert scrubbed is event
    assert "request" not in event
    assert "headers" not in event
    assert "env" not in event
    assert "extra" not in event
    assert "server_name" not in event
    exception = event["exception"]
    assert isinstance(exception, dict)
    values = exception["values"]
    assert isinstance(values, list)
    first_value = values[0]
    assert isinstance(first_value, dict)
    assert "value" not in first_value
    stacktrace = first_value["stacktrace"]
    assert isinstance(stacktrace, dict)
    frames = stacktrace["frames"]
    assert isinstance(frames, list)
    frame = frames[0]
    assert isinstance(frame, dict)
    assert "abs_path" not in frame
    assert "context_line" not in frame
    assert "post_context" not in frame
    assert "pre_context" not in frame
    assert "vars" not in frame
    assert frame["filename"] == "file.py"


def test_before_breadcrumb_drops_risky_breadcrumbs() -> None:
    assert scrub_sentry_breadcrumb({"category": "subprocess", "data": {}}, {}) is None
    assert scrub_sentry_breadcrumb({"category": "http", "data": {}}, {}) is None


def test_before_breadcrumb_strips_data_from_kept_breadcrumb() -> None:
    breadcrumb: dict[str, object] = {
        "category": "ui",
        "message": "clicked",
        "data": {"path": "/secret"},
    }

    scrubbed = scrub_sentry_breadcrumb(breadcrumb, {})

    assert scrubbed == {"category": "ui"}
