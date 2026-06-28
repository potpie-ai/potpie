"""Privacy scrubbers for Potpie Sentry events and metrics."""

from __future__ import annotations

from pathlib import PurePath

SentryEvent = dict[str, object]

_DROP_EVENT_KEYS = {
    "cookies",
    "env",
    "extra",
    "headers",
    "modules",
    "request",
    "server_name",
}
_DROP_BREADCRUMB_CATEGORIES = {"http", "subprocess", "urllib3"}
_DROP_FRAME_KEYS = {
    "abs_path",
    "context_line",
    "post_context",
    "pre_context",
    "value",
    "vars",
}


def scrub_sentry_event(event: SentryEvent, _hint: SentryEvent) -> SentryEvent | None:
    for key in _DROP_EVENT_KEYS:
        _ = event.pop(key, None)
    _scrub_nested(event.get("exception"))
    return event


def scrub_sentry_breadcrumb(
    breadcrumb: SentryEvent,
    _hint: SentryEvent,
) -> SentryEvent | None:
    category = breadcrumb.get("category")
    if isinstance(category, str) and category in _DROP_BREADCRUMB_CATEGORIES:
        return None
    _ = breadcrumb.pop("data", None)
    _ = breadcrumb.pop("message", None)
    return breadcrumb


def _scrub_nested(value: object) -> None:
    if isinstance(value, dict):
        mapping: dict[object, object] = value
        for key in _DROP_FRAME_KEYS:
            _ = mapping.pop(key, None)
        filename = mapping.get("filename")
        if isinstance(filename, str):
            mapping["filename"] = PurePath(filename).name
        for child in mapping.values():
            _scrub_nested(child)
        return
    if isinstance(value, list):
        for child in value:
            _scrub_nested(child)
