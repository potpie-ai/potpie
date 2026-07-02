"""Read and timeline formatting helpers for graph CLI commands."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import typer

from potpie.cli.commands._common import emit
from potpie.cli.commands.graph_common import (
    _parse_duration,
    _parse_instant,
    _parse_sort_dt,
    _str,
    _string_list,
)


def _emit_read(
    result: Any,
    *,
    format_: str,
    sort: str,
    dedupe: str,
    event_limit: int | None = None,
    human_prefix: str | None = None,
    warnings: tuple[str, ...] = (),
) -> None:
    normalized_format = _effective_read_format(result, format_)
    if normalized_format == "jsonl":
        rows = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        if not rows:
            rows = _raw_item_rows(result)
        for row in rows:
            typer.echo(json.dumps(row, default=str))
        return
    payload = _read_payload(
        result,
        format_=normalized_format,
        sort=sort,
        dedupe=dedupe,
        event_limit=event_limit,
    )
    emit(
        payload,
        human=_with_read_context(
            _read_human(
                result,
                format_=normalized_format,
                sort=sort,
                dedupe=dedupe,
                event_limit=event_limit,
            ),
            human_prefix=human_prefix,
            warnings=warnings,
        ),
    )


def _with_read_context(
    human: str, *, human_prefix: str | None, warnings: tuple[str, ...]
) -> str:
    lines: list[str] = []
    if human_prefix:
        lines.append(human_prefix)
    lines.append(human)
    lines.extend(f"! {warning}" for warning in warnings)
    return "\n".join(lines)


def _read_payload(
    result: Any,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> dict:
    payload = result.to_dict()
    if format_ in ("events", "table"):
        events = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        if payload.get("detail") != "full":
            payload.pop("items", None)
        payload["read_shape"] = "events"
        payload["events"] = events
        payload["event_count"] = len(events)
        payload["freshness"] = _timeline_freshness(events)
    return payload


def _read_human(
    result: Any,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> str:
    if format_ in ("events", "table"):
        return _timeline_human(
            result, sort=sort, dedupe=dedupe, event_limit=event_limit
        )
    payload = result.to_dict()
    items = payload.get("items", [])
    lines = [
        f"view={payload.get('view')} backed={payload.get('backed')} "
        f"items={len(items)} quality={payload.get('quality', {}).get('status')}"
    ]
    for item in items[:10]:
        fact = item.get("summary") or item.get("entity_key") or ""
        lines.append(f"  • [{item.get('entity_type') or '?'}] {fact}")
    return "\n".join(lines)


def _raw_item_rows(result: Any) -> list[dict[str, Any]]:
    return list(result.to_dict().get("items", []))


def _effective_read_format(result: Any, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "raw", "events", "table", "jsonl"}:
        raise ValueError("--format must be one of: auto, raw, events, table, jsonl")
    if value == "auto":
        return "events" if _is_timeline_view(result.to_dict().get("view")) else "raw"
    return value


def _effective_requested_format(*, subgraph: str, view: str, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return "events" if _is_timeline_view(f"{subgraph}.{view}") else "raw"
    return value


def _service_limit_for_read(
    *, subgraph: str, view: str, format_: str, requested_limit: int
) -> int:
    if _is_timeline_view(f"{subgraph}.{view}") and format_ in {
        "events",
        "table",
        "jsonl",
    }:
        return min(max(requested_limit * 8, 40), 200)
    return requested_limit


def _is_timeline_view(view: str | None) -> bool:
    return str(view or "").strip() == "recent_changes.timeline"


def _timeline_events(
    result: Any,
    *,
    sort: str = "auto",
    dedupe: str = "auto",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    payload = result.to_dict()
    if not _is_timeline_view(payload.get("view")):
        return []
    dedupe_mode = _normalize_dedupe(dedupe)
    by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    items = getattr(result, "items", None)
    if items is None:
        items = payload.get("items", [])
    for item in items:
        for event in _events_from_item(item):
            key = _event_dedupe_key(event, mode=dedupe_mode)
            if key is not None and key in by_key:
                existing = by_key[key]
                if float(event.get("score") or 0.0) > float(
                    existing.get("score") or 0.0
                ):
                    existing.update(event)
                continue
            ordered.append(event)
            if key is not None:
                by_key[key] = event
    events = _sort_events(ordered, sort=sort)
    return events[:limit] if limit is not None and limit >= 0 else events


def _events_from_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = dict(item)
    relations = payload.get("relations")
    item_score = float(payload.get("score") or 0.0)
    if isinstance(relations, list):
        return [
            _event_from_relation(rel, item_score=item_score)
            for rel in relations
            if isinstance(rel, Mapping) and _relation_has_timeline_fact(rel)
        ]
    claim_value = payload.get("claim")
    claim = claim_value if isinstance(claim_value, Mapping) else {}
    if claim.get("source_refs") or payload.get("source_refs") or payload.get("summary"):
        flat_payload = {**dict(claim), **payload}
        return [_event_from_flat_payload(flat_payload, item_score=item_score)]
    return []


def _relation_has_timeline_fact(rel: Mapping[str, Any]) -> bool:
    return bool(
        rel.get("fact") or rel.get("source_refs") or _activity_key_from_relation(rel)
    )


def _event_from_relation(
    rel: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    fact = _str(rel.get("fact"))
    source_refs = _string_list(rel.get("source_refs"))
    related_entity_value = rel.get("related_entity")
    related_entity = (
        related_entity_value if isinstance(related_entity_value, Mapping) else {}
    )
    return {
        "activity_key": _activity_key_from_relation(rel),
        "occurred_at": _event_occurred_at(rel, fact=fact),
        "source_refs": source_refs,
        "fact": fact,
        "predicate": _str(rel.get("predicate")),
        "actor_key": _actor_key_from_relation(rel),
        "target_key": _target_key_from_relation(rel),
        "related_key": _str(rel.get("related_key")),
        "related_name": _str(related_entity.get("name")),
        "truth": _str(rel.get("truth")),
        "evidence_strength": _str(rel.get("evidence_strength")),
        "source_system": _str(rel.get("source_system")),
        "score": float(rel.get("score") or item_score or 0.0),
    }


def _event_from_flat_payload(
    payload: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    fact = _str(payload.get("fact") or payload.get("summary"))
    return {
        "activity_key": _str(payload.get("activity_key")),
        "occurred_at": _event_occurred_at(payload, fact=fact),
        "source_refs": _string_list(payload.get("source_refs")),
        "fact": fact,
        "predicate": _str(payload.get("predicate")),
        "actor_key": None,
        "target_key": _str(payload.get("object_key")),
        "related_key": _str(payload.get("object_key")),
        "related_name": None,
        "truth": _str(payload.get("truth")),
        "evidence_strength": _str(payload.get("evidence_strength")),
        "source_system": _str(payload.get("source_system")),
        "score": float(item_score or 0.0),
    }


def _activity_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    from_key = _str(rel.get("from_key"))
    to_key = _str(rel.get("to_key"))
    if predicate in {"TOUCHED", "MENTIONS"}:
        return from_key
    if predicate in {"PERFORMED", "AUTHORED"}:
        return to_key
    return from_key if from_key.startswith("activity:") else to_key


def _actor_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    if predicate in {"PERFORMED", "AUTHORED"}:
        return _str(rel.get("from_key")) or None
    return None


def _target_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    if predicate in {"TOUCHED", "MENTIONS"}:
        return _str(rel.get("to_key")) or None
    return None


def _event_occurred_at(
    payload: Mapping[str, Any], *, fact: str | None = None
) -> str | None:
    props_value = payload.get("properties")
    props = props_value if isinstance(props_value, Mapping) else {}
    for value in (
        payload.get("occurred_at"),
        props.get("occurred_at"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fact:
        m = re.search(r"\bon (\d{4}-\d{2}-\d{2})\b", fact)
        if m:
            return m.group(1)
    value = payload.get("valid_at")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_dedupe(value: str) -> str:
    mode = (value or "auto").strip().lower()
    if mode == "auto":
        return "source_ref"
    if mode not in {"none", "source_ref", "activity"}:
        raise ValueError("--dedupe must be one of: auto, none, source_ref, activity")
    return mode


def _event_dedupe_key(event: Mapping[str, Any], *, mode: str) -> str | None:
    if mode == "none":
        return None
    if mode == "activity":
        return _str(event.get("activity_key")) or None
    refs = _string_list(event.get("source_refs"))
    if refs:
        return "source_ref:" + "|".join(sorted(refs))
    return _str(event.get("activity_key")) or _str(event.get("fact")) or None


def _sort_events(events: list[dict[str, Any]], *, sort: str) -> list[dict[str, Any]]:
    mode = (sort or "auto").strip().lower()
    if mode == "auto":
        mode = "occurred_at"
    if mode not in {"score", "occurred_at"}:
        raise ValueError("--sort must be one of: auto, score, occurred_at")
    if mode == "score":
        return sorted(events, key=lambda e: float(e.get("score") or 0.0), reverse=True)
    return sorted(
        events,
        key=lambda e: (
            _parse_sort_dt(e.get("occurred_at")),
            float(e.get("score") or 0.0),
        ),
        reverse=True,
    )


def _timeline_freshness(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dates = [
        occurred_at
        for e in events
        if isinstance((occurred_at := e.get("occurred_at")), str) and occurred_at
    ]
    return {
        "latest_event_at": max(dates) if dates else None,
        "source_refs_count": len(
            {ref for e in events for ref in _string_list(e.get("source_refs"))}
        ),
        "local_worktree_included": False,
        "note": "Timeline reads recorded graph events for the whole pot/project across repo sources; uncommitted local changes are not included unless recorded.",
    }


def _timeline_human(
    result: Any, *, sort: str, dedupe: str, event_limit: int | None = None
) -> str:
    events = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
    payload = result.to_dict()
    lines = [
        f"view={payload.get('view')} events={len(events)} "
        f"quality={payload.get('quality', {}).get('status')}",
        "scope=project-wide pot timeline across registered repo sources; local uncommitted worktree is not included",
    ]
    for event in events[:20]:
        refs = ", ".join(_string_list(event.get("source_refs"))) or "no-source-ref"
        when = event.get("occurred_at") or "unknown-date"
        fact = event.get("fact") or event.get("activity_key") or "(no fact)"
        lines.append(f"  • {when} [{refs}] {fact}")
    return "\n".join(lines)


def _resolve_time_bounds(
    *, since: str | None, until: str | None, window: str | None
) -> tuple[datetime | None, datetime | None]:
    until_dt = _parse_instant(until) if until else None
    since_dt = _parse_instant(since) if since else None
    if since_dt is not None:
        return since_dt, until_dt
    if window:
        end = until_dt or datetime.now(timezone.utc)
        return end - _parse_duration(window), until_dt
    return None, until_dt
