"""Timeline reader: per-actor / per-feature pulse over a time window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from adapters.outbound.graphiti.query_helpers import get_timeline
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import ContextGraphQuery


_DEFAULT_TIMELINE_WINDOW = timedelta(days=7)


class TimelineReader:
    FAMILY = "timeline"

    def __init__(self, *, structural: StructuralReadPort) -> None:
        self._structural = structural

    def family(self) -> str:
        return self.FAMILY

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self.FAMILY,
            description="Per-actor / per-feature pulse over a time window.",
            intents=frozenset({"planning", "operations", "review"}),
            requires_scope=frozenset(),
            cost=ReaderCost(label="cheap", estimated_ms=150),
            backend="structural",
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        since, until = _resolve_window(request)
        scope = request.scope
        bundle = get_timeline(
            self._structural,
            request.pot_id,
            since_iso=since.isoformat(),
            until_iso=until.isoformat(),
            limit=request.limit,
            user=(scope.user or "").strip() or None,
            feature=((scope.features[0] or "").strip() if scope.features else None),
            file_path=(scope.file_path or "").strip() or None,
            branch=(scope.branch or "").strip() or None,
            verbs=[v for v in (request.verbs or []) if v and v.strip()] or None,
        )
        activities = bundle.get("activities") if isinstance(bundle, dict) else None
        count = len(activities) if isinstance(activities, list) else None
        return ReaderResult(family=self.FAMILY, result=bundle, count=count)


def _resolve_window(request: ContextGraphQuery) -> tuple[datetime, datetime]:
    until = request.until or request.as_of or datetime.now(timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    since = request.since
    if since is None:
        delta = _parse_window(request.window) or _DEFAULT_TIMELINE_WINDOW
        since = until - delta
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    return since, until


def _parse_window(raw: str | None) -> timedelta | None:
    if not raw or not raw.strip():
        return None
    value = raw.strip().lower()
    try:
        if value.endswith("d"):
            return timedelta(days=int(value[:-1] or "0"))
        if value.endswith("h"):
            return timedelta(hours=int(value[:-1] or "0"))
        if value.endswith("m"):
            return timedelta(minutes=int(value[:-1] or "0"))
        if value.endswith("w"):
            return timedelta(weeks=int(value[:-1] or "0"))
    except ValueError:
        return None
    return None


__all__ = ["TimelineReader"]
