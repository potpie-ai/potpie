"""Query + memory commands: ``resolve`` / ``search`` / ``record``.

All three operations route through the context-bound ``EngineClient``.
These three (plus ``status``) are the four-tool agent contract; new use cases
become new ``--intent`` / ``--include`` / ``--type`` values, never new commands.
"""

from __future__ import annotations

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_engine_client,
    get_root_runtime,
    resolve_pot_id,
    run_engine_operation,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_activation_succeeded,
)
from potpie.cli.telemetry.usage_events import (
    capture_usage_command_succeeded,
)
from potpie_context_engine.requests import (
    RecordRequest as EngineRecordRequest,
    ResolveRequest as EngineResolveRequest,
)
from potpie_context_engine.core.context_records import (
    REQUIRED_RECORD_DETAILS,
    record_detail_choices,
    required_record_details,
)
from potpie_context_engine.requests import SearchRequest as EngineSearchRequest


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


def register(root: typer.Typer) -> None:
    @root.command()
    def resolve(
        task: str = typer.Argument(..., help="The task to pull context for."),
        intent: str = typer.Option("feature", "--intent"),
        include: str = typer.Option(
            None, "--include", help="Comma-separated include families."
        ),
        mode: str = typer.Option(
            "fast", "--mode", help="fast | balanced | verify | deep"
        ),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_resolve — a bounded context wrap for a task."""
        with contract():
            client = get_engine_client(resolve_pot_id(get_root_runtime(), pot))
            env = run_engine_operation(
                client.resolve(
                    EngineResolveRequest(
                        task=task,
                        intent=intent,
                        include=_split(include),
                        mode=mode,
                    )
                )
            )
            _capture_context_activation(command="resolve", item_count=len(env.items))
            emit(_envelope_payload(env), human=_envelope_human(env))

    @root.command()
    def search(
        query: str = typer.Argument(..., help="A known phrase or entity to look up."),
        include: str = typer.Option(None, "--include"),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_search — narrow follow-up lookup."""
        with contract():
            client = get_engine_client(resolve_pot_id(get_root_runtime(), pot))
            env = run_engine_operation(
                client.search(EngineSearchRequest(query=query, include=_split(include)))
            )
            _capture_context_activation(command="search", item_count=len(env.items))
            emit(_envelope_payload(env), human=_envelope_human(env))

    @root.command()
    def record(
        type: str = typer.Option(
            ...,
            "--type",
            help=(
                "Record type. Structured types and their required --detail keys: "
                + _RECORD_TYPE_HELP
            ),
        ),
        summary: str = typer.Option(..., "--summary"),
        scope: str = typer.Option(
            None, "--scope", help="key:value scope, e.g. service:inventory-svc"
        ),
        detail: list[str] | None = typer.Option(
            None,
            "--detail",
            help=(
                "Repeatable key=value payload field, e.g. "
                "--detail rationale='cheaper to operate'. Required for the "
                "structured record types (see --type)."
            ),
        ),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_record — write a durable project learning."""
        with contract():
            details = _parse_details(detail)
            _require_record_details(type, details)
            receipt = run_engine_operation(
                get_engine_client(resolve_pot_id(get_root_runtime(), pot)).record(
                    EngineRecordRequest(
                        record_type=type,
                        summary=summary,
                        details=details,
                        scope=_parse_scope(scope),
                    )
                )
            )
            capture_usage_command_succeeded(
                command="record",
                result_kind="record_result",
                item_count=receipt.mutations_applied,
            )
            emit(
                {
                    "status": receipt.status,
                    "record_id": receipt.record_id,
                    "mutations_applied": receipt.mutations_applied,
                },
                human=f"{receipt.status}: {receipt.record_id} ({receipt.mutations_applied} mutations)",
            )


def _record_type_help() -> str:
    """Advertise only what ``record`` can actually write, with its flags."""

    structured = "; ".join(
        f"{record_type} (needs "
        + ", ".join(f"--detail {field}=…" for field in fields)
        + ")"
        for record_type, fields in sorted(REQUIRED_RECORD_DETAILS.items())
    )
    return f"free-form types (fix, investigation, workflow, …) need no --detail; {structured}"


_RECORD_TYPE_HELP = _record_type_help()


def _parse_details(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or ():
        key, sep, value = str(pair).partition("=")
        if not sep or not key.strip():
            fail(
                code="validation_error",
                message=f"--detail expects key=value, got {pair!r}",
                next_action="pass a field such as --detail rationale='<why>'",
            )
        out[key.strip()] = value.strip()
    return out


def _require_record_details(record_type: str, details: dict[str, str]) -> None:
    """Fail with the flag to type, before the write reaches the engine."""

    for field in required_record_details(record_type):
        choices = record_detail_choices(record_type, field)
        value = details.get(field)
        if not value:
            hint = f"one of {', '.join(choices)}" if choices else "<value>"
            fail(
                code="validation_error",
                message=f"record type '{record_type}' requires a '{field}' detail",
                next_action=f"add --detail {field}={hint}",
            )
        if choices and value not in choices:
            fail(
                code="validation_error",
                message=(
                    f"record type '{record_type}' detail '{field}' must be one of "
                    f"{', '.join(choices)}; got {value!r}"
                ),
                next_action=f"add --detail {field}={choices[0]}",
            )


def _parse_scope(scope: str | None) -> dict[str, str]:
    if not scope:
        return {}
    out: dict[str, str] = {}
    for pair in scope.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _envelope_payload(env) -> dict[str, object]:
    return {
        "pot_id": env.pot_id,
        "intent": env.intent,
        "overall_confidence": env.overall_confidence,
        "items": [
            {"include": i.include, "score": i.score, "payload": dict(i.payload)}
            for i in env.items
        ],
        "coverage": [
            # graph_view is the canonical workbench name serving this include
            # family — the pointer to follow when moving to `graph read`.
            {"include": c.include, "status": c.status, "graph_view": c.graph_view}
            for c in env.coverage
        ],
        "unsupported_includes": [
            {"name": u.name, "reason": u.reason} for u in env.unsupported_includes
        ],
    }


# Payload keys that can carry an item's human-readable line, most specific
# first. Reader families disagree on which one they populate, so a renderer
# that only knows two of them prints blank bullets for the rest.
_ITEM_TEXT_KEYS: tuple[str, ...] = (
    "fact",
    "summary",
    "description",
    "label",
    "section_title",
    "snippet",
    "title",
    "name",
    "text",
)


def _item_text(item) -> str:
    payload = dict(item.payload)
    for key in _ITEM_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    # Nothing readable: fall back to an identifier so the bullet still points
    # somewhere instead of rendering as an empty line.
    for key in ("fetch", "source_ref", "claim_key", "subject_key"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(getattr(item, "candidate_key", "") or "(no summary in payload)")


def _envelope_human(env) -> str:
    lines = [
        f"pot={env.pot_id} intent={env.intent} confidence={env.overall_confidence} items={len(env.items)}"
    ]
    # Identical items are already merged upstream; anything that still renders
    # the same text is a *different* record worded identically, so name what
    # distinguishes it rather than printing the same line twice.
    seen: dict[str, int] = {}
    for item in env.items[:10]:
        text = _item_text(item)
        seen[text] = seen.get(text, 0) + 1
        suffix = ""
        if seen[text] > 1:
            subject = dict(item.payload).get("subject_key") or getattr(
                item, "candidate_key", ""
            )
            suffix = f"  (distinct record: {subject})" if subject else ""
        lines.append(f"  • [{item.include}] {text}{suffix}")
    for unsup in env.unsupported_includes:
        lines.append(f"  ! {unsup.name}: {unsup.reason}")
    return "\n".join(lines)


__all__ = ["register"]


def _capture_context_activation(*, command: str, item_count: int) -> None:
    capture_activation_succeeded(
        command=command,
        result_kind="context_result",
        item_count=item_count,
    )
    capture_usage_command_succeeded(
        command=command,
        result_kind="context_result",
        item_count=item_count,
    )
