"""Query + memory commands: ``resolve`` / ``search`` / ``record``.

Routes ``CLI -> HostShell.agent_context -> GraphService -> readers/mutation``.
These three (plus ``status``) are the four-tool agent contract; new use cases
become new ``--intent`` / ``--include`` / ``--type`` values, never new commands.
"""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import (
    contract,
    emit,
    get_host,
    resolve_pot_id,
)
from adapters.inbound.cli.telemetry.onboarding_events import (
    capture_activation_succeeded,
)
from domain.ports.agent_context import RecordRequest, ResolveRequest, SearchRequest


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
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            env = host.agent_context.resolve(
                ResolveRequest(
                    pot_id=pot_id,
                    task=task,
                    intent=intent,
                    include=_split(include),
                    mode=mode,
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
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            env = host.agent_context.search(
                SearchRequest(pot_id=pot_id, query=query, include=_split(include))
            )
            _capture_context_activation(command="search", item_count=len(env.items))
            emit(_envelope_payload(env), human=_envelope_human(env))

    @root.command()
    def record(
        type: str = typer.Option(
            ..., "--type", help="Record type (fix, decision, preference, …)."
        ),
        summary: str = typer.Option(..., "--summary"),
        scope: str = typer.Option(
            None, "--scope", help="key:value scope, e.g. service:inventory-svc"
        ),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_record — write a durable project learning."""
        with contract():
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            receipt = host.agent_context.record(
                RecordRequest(
                    pot_id=pot_id,
                    record_type=type,
                    summary=summary,
                    scope=_parse_scope(scope),
                )
            )
            emit(
                {
                    "status": receipt.status,
                    "record_id": receipt.record_id,
                    "mutations_applied": receipt.mutations_applied,
                },
                human=f"{receipt.status}: {receipt.record_id} ({receipt.mutations_applied} mutations)",
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
        "coverage": [{"include": c.include, "status": c.status} for c in env.coverage],
        "unsupported_includes": [
            {"name": u.name, "reason": u.reason} for u in env.unsupported_includes
        ],
    }


def _envelope_human(env) -> str:
    lines = [
        f"pot={env.pot_id} intent={env.intent} confidence={env.overall_confidence} items={len(env.items)}"
    ]
    for item in env.items[:10]:
        fact = dict(item.payload).get("fact") or dict(item.payload).get("summary") or ""
        lines.append(f"  • [{item.include}] {fact}")
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
