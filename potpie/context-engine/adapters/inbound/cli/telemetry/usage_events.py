from __future__ import annotations

from .product_analytics import AnalyticsValue, capture_event


def capture_usage_command_succeeded(
    *,
    command: str,
    result_kind: str,
    item_count: int | None = None,
    provider: str | None = None,
    properties: dict[str, AnalyticsValue] | None = None,
) -> None:
    props: dict[str, AnalyticsValue] = dict(properties or {})
    props["command"] = command
    props["result_kind"] = result_kind
    if item_count is not None:
        props["item_count"] = item_count
    if provider is not None:
        props["provider"] = provider
    capture_event("cli_usage_command_succeeded", props)


__all__ = ["capture_usage_command_succeeded"]
