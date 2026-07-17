"""Human-readable presentation for ``graph read`` and ``timeline recent``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from domain.ports.services.graph_service import (
    normalize_read_detail,
    normalize_read_relations,
    read_item_for_detail,
)

_FACT_TRUNCATE_LEN = 120
_DEFAULT_ITEM_LIMIT = 10


@dataclass(frozen=True, slots=True)
class ReadPresentationContext:
    view: str
    detail: str
    relations: str
    format_mode: str
    sort: str
    dedupe: str
    event_limit: int | None


def prepare_items(result) -> list[dict[str, Any]]:
    detail = normalize_read_detail(getattr(result, "detail", None))
    relations = normalize_read_relations(getattr(result, "relations", None))
    raw_items = getattr(result, "items", None) or ()
    return [
        read_item_for_detail(item, detail=detail, relations=relations)
        for item in raw_items
    ]


def items_by_key(items: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _str(item.get("entity_key"))
        if key:
            indexed[key] = dict(item)
    return indexed


def build_presentation_context(
    result,
    *,
    format_: str,
    sort: str,
    dedupe: str,
    event_limit: int | None,
) -> ReadPresentationContext:
    return ReadPresentationContext(
        view=_str(getattr(result, "view", None)),
        detail=normalize_read_detail(getattr(result, "detail", None)),
        relations=normalize_read_relations(getattr(result, "relations", None)),
        format_mode=format_,
        sort=sort,
        dedupe=dedupe,
        event_limit=event_limit,
    )


def render_timeline_events(
    result,
    events: list[Mapping[str, Any]],
    shaped_items: list[dict[str, Any]],
    ctx: ReadPresentationContext,
) -> str:
    payload = {
        "view": getattr(result, "view", None),
        "unsupported": getattr(result, "unsupported", ()),
    }
    quality = getattr(result, "quality", {}) or {}
    lines = _timeline_header_lines(payload, len(events), quality)
    if not events:
        lines.append("(no events)")
        return "\n".join(lines)

    item_index = items_by_key(shaped_items)
    limit = _effective_limit(ctx.event_limit, default=20)
    for event in events[:limit]:
        lines.extend(
            _timeline_event_bullet_lines(event, item_index=item_index, ctx=ctx)
        )
    return "\n".join(lines)


def render_timeline_table(
    result,
    events: list[Mapping[str, Any]],
    shaped_items: list[dict[str, Any]],
    ctx: ReadPresentationContext,
) -> str:
    payload = {
        "view": getattr(result, "view", None),
        "unsupported": getattr(result, "unsupported", ()),
    }
    quality = getattr(result, "quality", {}) or {}
    lines = _timeline_header_lines(payload, len(events), quality)
    if not events:
        lines.append("(no events)")
        return "\n".join(lines)

    item_index = items_by_key(shaped_items)
    limit = _effective_limit(ctx.event_limit, default=20)
    display_events = events[:limit]

    if ctx.detail == "full":
        headers = [
            "occurred_at",
            "source_ref",
            "activity",
            "fact",
            "score",
            "truth",
            "predicate",
            "target_key",
            "actor_key",
            "relations",
        ]
    else:
        headers = [
            "occurred_at",
            "source_ref",
            "activity",
            "fact",
            "score",
            "relations",
        ]
    lines.append(" | ".join(headers))
    lines.append(" | ".join("---" for _ in headers))

    for event in display_events:
        parent = _parent_item_for_event(event, item_index)
        relations_cell = _relations_cell(parent, ctx)
        fact = _display_fact(event.get("fact") or event.get("activity_key"), ctx)
        row = [
            _escape_table_cell(event.get("occurred_at") or "unknown-date"),
            _escape_table_cell(
                ", ".join(_string_list(event.get("source_refs"))) or "-"
            ),
            _escape_table_cell(event.get("activity_key") or "-"),
            _escape_table_cell(fact),
            _escape_table_cell(event.get("score")),
        ]
        if ctx.detail == "full":
            row.extend(
                [
                    _escape_table_cell(event.get("truth") or "-"),
                    _escape_table_cell(event.get("predicate") or "-"),
                    _escape_table_cell(event.get("target_key") or "-"),
                    _escape_table_cell(event.get("actor_key") or "-"),
                ]
            )
        row.append(_escape_table_cell(relations_cell))
        lines.append(" | ".join(str(cell) for cell in row))

    if ctx.relations == "full":
        seen_activities: set[str] = set()
        for event in display_events:
            activity_key = _str(event.get("activity_key"))
            if not activity_key or activity_key in seen_activities:
                continue
            seen_activities.add(activity_key)
            parent = _parent_item_for_event(event, item_index)
            if parent and parent.get("relations"):
                lines.append("")
                lines.append(f"relations for {activity_key}")
                lines.append("predicate | from | to | fact | truth")
                lines.append("--- | --- | --- | --- | ---")
                for rel in parent.get("relations", ()):
                    if not isinstance(rel, Mapping):
                        continue
                    lines.append(
                        " | ".join(
                            _escape_table_cell(value)
                            for value in (
                                rel.get("predicate") or "-",
                                rel.get("from_key") or "-",
                                rel.get("to_key") or "-",
                                _display_fact(rel.get("fact"), ctx),
                                rel.get("truth") or "-",
                            )
                        )
                    )
    return "\n".join(lines)


def render_items_bullets(
    result,
    items: list[dict[str, Any]],
    ctx: ReadPresentationContext,
) -> str:
    payload = {
        "view": getattr(result, "view", None),
        "backed": getattr(result, "backed", None),
        "unsupported": getattr(result, "unsupported", ()),
    }
    quality = getattr(result, "quality", {}) or {}
    lines = _items_header_lines(payload, len(items), quality)
    if not items:
        lines.append("(no rows)")
        return "\n".join(lines)

    limit = _effective_limit(ctx.event_limit, default=_DEFAULT_ITEM_LIMIT)
    for item in items[:limit]:
        lines.extend(_item_bullet_lines(item, ctx))
    return "\n".join(lines)


def render_items_table(
    items: list[dict[str, Any]],
    ctx: ReadPresentationContext,
    *,
    result=None,
) -> str:
    lines: list[str] = []
    if result is not None:
        payload = result.to_dict()
        quality = payload.get("quality", {})
        lines.extend(_items_header_lines(payload, len(items), quality))

    if ctx.detail == "full":
        headers = [
            "score",
            "type",
            "entity_key",
            "summary",
            "truth",
            "coverage",
            "relations",
        ]
    else:
        headers = ["score", "type", "entity_key", "summary", "relations"]
    lines.append(" | ".join(headers))
    lines.append(" | ".join("---" for _ in headers))

    if not items:
        lines.append("(no rows)")
        return "\n".join(lines)

    limit = _effective_limit(ctx.event_limit, default=_DEFAULT_ITEM_LIMIT)
    for item in items[:limit]:
        entity_key = _item_entity_key(item)
        row = [
            _escape_table_cell(item.get("score")),
            _escape_table_cell(item.get("entity_type") or "-"),
            _escape_table_cell(entity_key or "-"),
            _escape_table_cell(_display_fact(item.get("summary") or entity_key, ctx)),
        ]
        if ctx.detail == "full":
            row.extend(
                [
                    _escape_table_cell(item.get("truth") or "-"),
                    _escape_table_cell(item.get("coverage_status") or "-"),
                ]
            )
        row.append(_escape_table_cell(_relations_cell(item, ctx)))
        lines.append(" | ".join(str(cell) for cell in row))

    if ctx.relations == "full":
        for item in items[:limit]:
            relation_lines = _format_relations_full_lines(item, indent="  ")
            if relation_lines:
                entity_key = _item_entity_key(item) or "item"
                lines.append("")
                lines.append(f"relations for {entity_key}")
                lines.extend(relation_lines)
    return "\n".join(lines)


def _timeline_header_lines(
    payload: Mapping[str, Any],
    event_count: int,
    quality: Mapping[str, Any],
) -> list[str]:
    lines = [
        f"view={payload.get('view')} events={event_count} "
        f"quality={quality.get('status')}",
        "scope=applied graph-read scope across registered repo sources; "
        "local uncommitted worktree is not included",
    ]
    if quality.get("status") == "unsupported":
        reason = quality.get("reason") or "unsupported_filter"
        names = ", ".join(
            str(item.get("name"))
            for item in payload.get("unsupported", ())
            if item.get("name")
        )
        lines.append(f"unsupported_filter={names or reason}")
    return lines


def _items_header_lines(
    payload: Mapping[str, Any],
    item_count: int,
    quality: Mapping[str, Any],
) -> list[str]:
    lines = [
        f"view={payload.get('view')} backed={payload.get('backed')} "
        f"items={item_count} quality={quality.get('status')}"
    ]
    if quality.get("status") == "unsupported":
        reason = quality.get("reason") or "unsupported_filter"
        names = ", ".join(
            str(item.get("name"))
            for item in payload.get("unsupported", ())
            if item.get("name")
        )
        lines.append(f"unsupported_filter={names or reason}")
    return lines


def _timeline_event_bullet_lines(
    event: Mapping[str, Any],
    *,
    item_index: dict[str, dict[str, Any]],
    ctx: ReadPresentationContext,
) -> list[str]:
    refs = ", ".join(_string_list(event.get("source_refs"))) or "no-source-ref"
    when = event.get("occurred_at") or "unknown-date"
    fact = _display_fact(
        event.get("fact") or event.get("activity_key") or "(no fact)", ctx
    )
    score = event.get("score")
    suffix_parts: list[str] = []
    if score is not None:
        suffix_parts.append(f"score={score}")
    if ctx.detail == "full":
        for key in (
            "truth",
            "predicate",
            "actor_key",
            "target_key",
            "evidence_strength",
        ):
            value = event.get(key)
            if value:
                suffix_parts.append(f"{key}={value}")
    suffix = f"  {'  '.join(suffix_parts)}" if suffix_parts else ""
    lines = [f"  • {when} [{refs}] {fact}{suffix}"]

    parent = _parent_item_for_event(event, item_index)
    if ctx.relations == "summary" and parent:
        summary = _format_relations_summary(parent)
        if summary:
            lines.append(f"    relations: {summary}")
    elif ctx.relations == "full" and parent:
        lines.extend(_format_relations_full_lines(parent, indent="    "))
    return lines


def _item_bullet_lines(
    item: Mapping[str, Any], ctx: ReadPresentationContext
) -> list[str]:
    entity_type = item.get("entity_type") or "?"
    entity_key = _item_entity_key(item)
    summary = _display_fact(item.get("summary") or entity_key or "", ctx)
    meta_parts: list[str] = []
    if item.get("score") is not None:
        meta_parts.append(f"score={item.get('score')}")
    if ctx.detail == "full":
        for key in ("truth", "coverage_status"):
            value = item.get(key)
            if value:
                meta_parts.append(f"{key}={value}")
    meta = f"  {'  '.join(meta_parts)}" if meta_parts else ""
    lines = [f"  • [{entity_type}] {entity_key or summary}{meta}"]
    if entity_key and summary and summary != entity_key:
        lines.append(f"    {summary}")
    refs = _string_list(item.get("source_refs"))
    if refs:
        lines.append(f"    refs: {', '.join(refs)}")
    claim = item.get("claim")
    if ctx.detail == "full" and isinstance(claim, Mapping):
        claim_parts = [
            f"{key}={claim.get(key)}"
            for key in ("predicate", "subject_key", "object_key", "claim_key")
            if claim.get(key) is not None
        ]
        if claim_parts:
            lines.append(f"    claim: {' '.join(claim_parts)}")
    breakdown = item.get("breakdown")
    if ctx.detail == "full" and isinstance(breakdown, Mapping) and breakdown:
        breakdown_text = " ".join(
            f"{key}={value}" for key, value in sorted(breakdown.items())
        )
        lines.append(f"    breakdown: {breakdown_text}")
    if ctx.relations == "summary":
        summary_line = _format_relations_summary(item)
        if summary_line:
            lines.append(f"    relations: {summary_line}")
    elif ctx.relations == "full":
        lines.extend(_format_relations_full_lines(item, indent="    "))
    return lines


def _parent_item_for_event(
    event: Mapping[str, Any],
    item_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    activity_key = _str(event.get("activity_key"))
    if activity_key and activity_key in item_index:
        return item_index[activity_key]
    return None


def _relations_cell(
    item: Mapping[str, Any] | None, ctx: ReadPresentationContext
) -> str:
    if item is None:
        return "-"
    if ctx.relations == "full" and item.get("relations"):
        count = len(item.get("relations", ()))
        predicates = sorted(
            {
                _str(rel.get("predicate"))
                for rel in item.get("relations", ())
                if isinstance(rel, Mapping) and rel.get("predicate")
            }
        )
        if predicates:
            return f"{count} [{', '.join(predicates)}]"
        return str(count) if count else "-"
    return _format_relations_summary(item) or "-"


def _format_relations_summary(item: Mapping[str, Any]) -> str:
    count = item.get("relation_count")
    if count is None and item.get("relations"):
        count = len(item.get("relations", ()))
    if not count:
        return ""
    predicates = item.get("relation_predicates")
    if predicates is None and item.get("relations"):
        predicates = sorted(
            {
                _str(rel.get("predicate"))
                for rel in item.get("relations", ())
                if isinstance(rel, Mapping) and rel.get("predicate")
            }
        )
    predicate_text = ""
    if predicates:
        predicate_text = f" [{', '.join(str(p) for p in predicates)}]"
    related = item.get("related_keys")
    if related is None and item.get("relations"):
        related = _related_keys_from_relations(item.get("relations", ()))
    related_text = ""
    if related:
        related_text = f" → {', '.join(str(key) for key in related[:5])}"
    return f"{count}{predicate_text}{related_text}"


def _format_relations_full_lines(item: Mapping[str, Any], *, indent: str) -> list[str]:
    relations = item.get("relations")
    if not isinstance(relations, list) or not relations:
        return []
    lines: list[str] = []
    for rel in relations:
        if not isinstance(rel, Mapping):
            continue
        predicate = _str(rel.get("predicate")) or "RELATED"
        from_key = _str(rel.get("from_key"))
        to_key = _str(rel.get("to_key"))
        direction = f"{from_key} → {to_key}" if from_key or to_key else ""
        lines.append(f"{indent}↳ {predicate} {direction}".rstrip())
        fact = rel.get("fact")
        if fact:
            lines.append(f"{indent}  fact: {fact}")
        refs = _string_list(rel.get("source_refs"))
        if refs:
            lines.append(f"{indent}  refs: {', '.join(refs)}")
        truth = rel.get("truth")
        if truth:
            lines.append(f"{indent}  truth: {truth}")
    return lines


def _related_keys_from_relations(
    relations: list[Mapping[str, Any]],
) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for rel in relations:
        for candidate in (
            rel.get("related_key"),
            rel.get("to_key"),
            rel.get("from_key"),
        ):
            key = _str(candidate)
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
            if len(keys) >= 10:
                return keys
    return keys


def _item_entity_key(item: Mapping[str, Any]) -> str:
    key = _str(item.get("entity_key"))
    if key:
        return key
    claim = item.get("claim")
    if isinstance(claim, Mapping):
        return _str(claim.get("subject_key") or claim.get("object_key"))
    return ""


def _display_fact(value: Any, ctx: ReadPresentationContext) -> str:
    text = _str(value) if value is not None else ""
    if ctx.detail == "compact":
        return _truncate(text, _FACT_TRUNCATE_LEN)
    return text


def _escape_table_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _effective_limit(limit: int | None, *, default: int) -> int:
    if limit is None or limit < 0:
        return default
    return limit


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v]
    return []
