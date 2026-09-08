from __future__ import annotations

from .product_analytics import AnalyticsValue, capture_event

_CONTEXT_COMMANDS: frozenset[str] = frozenset({"resolve", "search", "record"})
_PROVIDER_RESULT_KINDS: frozenset[str] = frozenset(
    {"provider_list", "provider_selection"}
)
_SKILLS_COMMANDS: frozenset[str] = frozenset(
    {"skills install", "skills update", "skills remove"}
)
_POTS_SOURCES_COMMANDS: frozenset[str] = frozenset(
    {"pot create", "pot use", "source add"}
)


def usage_feature(*, command: str, result_kind: str) -> str | None:
    """Map a usage event onto a closed Q13 product surface, or None."""
    if result_kind == "ui_session" or command == "ui":
        return "ui"
    if result_kind == "graph_command" or command.startswith("graph."):
        return "graph"
    if command in _CONTEXT_COMMANDS:
        return "context"
    if result_kind in _PROVIDER_RESULT_KINDS:
        return "integrations"
    if command in _SKILLS_COMMANDS:
        return "skills"
    if command in _POTS_SOURCES_COMMANDS:
        return "pots_sources"
    return None


def capture_usage_command_succeeded(
    *,
    command: str,
    result_kind: str,
    item_count: int | None = None,
    provider: str | None = None,
    properties: dict[str, AnalyticsValue] | None = None,
) -> None:
    props: dict[str, AnalyticsValue] = dict(properties or {})
    props.pop("feature", None)
    props["command"] = command
    props["result_kind"] = result_kind
    if item_count is not None:
        props["item_count"] = item_count
    if provider is not None:
        props["provider"] = provider
    feature = usage_feature(command=command, result_kind=result_kind)
    if feature is not None:
        props["feature"] = feature
    capture_event("cli_usage_command_succeeded", props)


__all__ = ["capture_usage_command_succeeded", "usage_feature"]
