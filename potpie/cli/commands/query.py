"""Query + memory commands: ``resolve`` / ``search`` / ``record``.

Routes ``CLI -> HostShell.agent_context -> GraphService -> readers/mutation``.
These three (plus ``status``) are the four-tool agent contract; new use cases
become new ``--intent`` / ``--include`` / ``--type`` values, never new commands.
"""

from __future__ import annotations

import typer

from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    contract,
    emit,
    fail,
    get_host,
    parse_scope_pairs,
    require_text,
    resolve_pot_id,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_activation_succeeded,
)
from potpie.cli.telemetry.usage_events import (
    capture_usage_command_succeeded,
)
from potpie_context_core.agent_context_port import (
    CONTEXT_INTENTS,
    READER_BACKED_INCLUDES,
)
from potpie_context_core.context_records import REQUIRED_DETAIL_KEYS
from potpie_context_core.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_core.source_references import RESOLVE_MODES

# Spelled out in --help because the values are not guessable: an agent with no
# list in front of it reaches for the subgraph names it saw in `graph catalog`
# and gets an unsupported_include back. Derived, so the help cannot drift from
# what the orchestrator actually answers.
_INCLUDE_HELP = "Comma-separated include families: " + ", ".join(
    sorted(READER_BACKED_INCLUDES - {"raw_graph"})
)
_INTENT_HELP = "One of: " + ", ".join(sorted(CONTEXT_INTENTS))
_RESOLVE_INTENT_HELP = (
    "Task intent; inferred from the task text when omitted. " + _INTENT_HELP
)
_MODE_HELP = "Retrieval depth. One of: " + ", ".join(sorted(RESOLVE_MODES))
# Derived from the same table the validator is held to, so the help cannot
# promise a type that then refuses for want of a field it never named.
_REQUIRED_DETAILS_HELP = "; ".join(
    f"{record_type}: {', '.join(keys) or 'none'}"
    for record_type, keys in REQUIRED_DETAIL_KEYS.items()
)
_TYPE_HELP = (
    "Record type (fix, decision, preference, …). Required --detail per type — "
    f"{_REQUIRED_DETAILS_HELP}."
)
_DETAIL_HELP = (
    "Structured field for --type, as key=value (repeatable; repeat a key to "
    f"build a list). Required per type — {_REQUIRED_DETAILS_HELP}; e.g. "
    "`--type decision --detail rationale=...`."
)


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _require_choice(
    value: str, *, argument: str, allowed: frozenset[str], example: str
) -> str:
    """The canonical spelling of ``value``, refusing anything outside ``allowed``.

    Both vocabularies this guards *normalize* an unknown value rather than
    reject it, and both normalizations are silent-wrong-answer bugs at a
    keyboard: ``--mode blanced`` becomes ``fast`` and reports a shallower read
    as though it were the deep one that was asked for, and ``--intent debuging``
    becomes ``unknown``, which queries a different set of reader families
    entirely. The service keeps normalizing — it is the compatibility rule for
    the managed HTTP surface, where the caller is another program — but a typo
    typed at this CLI has no compatible meaning to preserve.
    """
    cleaned = (value or "").strip().lower()
    if cleaned in allowed:
        return cleaned
    fail(
        code="validation_error",
        message=f"unknown {argument} {value!r}.",
        detail={"argument": argument, "allowed": sorted(allowed)},
        next_action=f"use one of: {', '.join(sorted(allowed))} — e.g. {example}",
    )


def _parse_detail_pairs(pairs: list[str] | None) -> dict[str, object]:
    """``--detail key=value`` entries → the ``details`` payload of a record.

    Repeating a key builds a list, which is how the list-shaped fields
    (``alternatives_rejected``, ``affects_refs``, ``fix_steps``) are reachable
    from a shell. Split on the *first* ``=`` only: a rationale or a prescription
    routinely contains one, and so does every URL.

    Raised as ``ValueError`` so the shared ``contract()`` renders it as
    ``validation_error``, the same way ``--scope`` does.
    """
    out: dict[str, object] = {}
    for raw in pairs or ():
        entry = raw.strip()
        if not entry:
            continue
        key, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(
                f"invalid --detail entry {entry!r}; expected key=value pairs"
            )
        key = key.strip()
        if not key:
            raise ValueError(
                f"invalid --detail entry {entry!r}; detail keys must not be empty"
            )
        value = value.strip()
        if not value:
            raise ValueError(
                f"invalid --detail entry {entry!r}; detail values must not be empty"
            )
        existing = out.get(key)
        if existing is None:
            out[key] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            out[key] = [existing, value]
    return out


def register(root: typer.Typer) -> None:
    @root.command()
    def resolve(
        task: str = typer.Argument(..., help="The task to pull context for."),
        intent: str = typer.Option(None, "--intent", help=_RESOLVE_INTENT_HELP),
        include: str = typer.Option(None, "--include", help=_INCLUDE_HELP),
        mode: str = typer.Option("fast", "--mode", help=_MODE_HELP),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_resolve — a bounded context wrap for a task."""
        with contract():
            task = require_text(
                task,
                argument="task",
                example="potpie resolve 'add rate limiting to the API'",
            )
            # Unset stays unset: the service infers the intent from the task
            # text and reports ``intent_source`` in the envelope. Defaulting to
            # ``feature`` here hid ``prior_bugs`` and ``timeline`` from every
            # "why is X broken" that did not also spell out ``--intent``.
            if intent is not None:
                intent = _require_choice(
                    intent,
                    argument="--intent",
                    allowed=CONTEXT_INTENTS,
                    example="--intent debugging",
                )
            mode = _require_choice(
                mode,
                argument="--mode",
                allowed=RESOLVE_MODES,
                example="--mode balanced",
            )
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
            emit(env.to_dict(), human=_envelope_human(env))

    @root.command()
    def search(
        query: str = typer.Argument(..., help="A known phrase or entity to look up."),
        include: str = typer.Option(None, "--include", help=_INCLUDE_HELP),
        intent: str = typer.Option(
            None,
            "--intent",
            help=f"Narrow the search to one intent's families. {_INTENT_HELP}",
        ),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_search — narrow follow-up lookup."""
        with contract():
            query = require_text(
                query, argument="query", example="potpie search 'rate limiter'"
            )
            # Unset stays unset — normalization is the service's job — but a
            # value the caller *did* type is held to the same vocabulary
            # ``resolve`` holds it to; the silent downgrade to ``unknown``
            # changes which reader families answer.
            if intent is not None:
                intent = _require_choice(
                    intent,
                    argument="--intent",
                    allowed=CONTEXT_INTENTS,
                    example="--intent docs",
                )
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            env = host.agent_context.search(
                SearchRequest(
                    pot_id=pot_id,
                    query=query,
                    include=_split(include),
                    intent=intent,
                )
            )
            _capture_context_activation(command="search", item_count=len(env.items))
            emit(env.to_dict(), human=_envelope_human(env))

    @root.command()
    def record(
        type: str = typer.Option(..., "--type", help=_TYPE_HELP),
        summary: str = typer.Option(..., "--summary"),
        detail: list[str] = typer.Option(None, "--detail", help=_DETAIL_HELP),
        scope: str = typer.Option(
            None, "--scope", help="key:value scope, e.g. service:inventory-svc"
        ),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """context_record — write a durable project learning.

        ``--detail`` is what makes the structured record types reachable. Two of
        the three types this command's own ``--type`` help advertises validate a
        field that lives nowhere else: ``decision`` requires ``rationale`` and
        ``preference`` requires ``policy_kind``, so before this flag existed the
        two headline uses were impossible to execute from the CLI at all — every
        attempt came back as a validation error naming a field with no way to
        supply it.
        """
        with contract():
            # Refused before the pot is even resolved: this command *writes*, and
            # a blank type or summary is a durable row nothing can retrieve.
            type = require_text(type, argument="--type", example="--type fix")
            summary = require_text(
                summary,
                argument="--summary",
                example="--summary 'retries need a jittered backoff'",
            )
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            receipt = host.agent_context.record(
                RecordRequest(
                    pot_id=pot_id,
                    record_type=type,
                    summary=summary,
                    details=_parse_detail_pairs(detail),
                    scope=parse_scope_pairs(scope),
                )
            )
            if receipt.accepted:
                # Gated on the receipt for the same reason the exit code is: a
                # "command succeeded" event for a refused write is the same lie
                # as exit 0, told to the usage funnel instead of the caller.
                capture_usage_command_succeeded(
                    command="record",
                    result_kind="record_result",
                    item_count=receipt.mutations_applied,
                )
            # ``accepted`` and ``detail`` are the two fields that say whether the
            # write actually landed, and dropping them made a *refused* record
            # indistinguishable from a stored one: the graph service answers
            # `status="rejected"` with the reason attached, and this printed
            # "rejected: <id> (0 mutations)" at exit 0 — a receipt, for nothing.
            # `ok` mirrors `accepted` so a consumer can branch on the same key
            # the error envelope uses without knowing which shape it got.
            emit(
                {
                    "ok": receipt.accepted,
                    "accepted": receipt.accepted,
                    "status": receipt.status,
                    "record_id": receipt.record_id,
                    "mutations_applied": receipt.mutations_applied,
                    "detail": receipt.detail,
                },
                human=_record_human(receipt),
            )
            if not receipt.accepted:
                raise typer.Exit(code=EXIT_VALIDATION)


def _record_human(receipt) -> str:
    line = (
        f"{receipt.status}: {receipt.record_id} ({receipt.mutations_applied} mutations)"
    )
    return f"{line}\n  ! {receipt.detail}" if receipt.detail else line


# NOTE: both read commands emit ``AgentEnvelope.to_dict()`` rather than a
# payload assembled here. The hand-rolled one this replaced silently dropped six
# fields the envelope carries — ``candidate_key``, ``coverage_status``,
# ``breakdown``, ``candidate_pool``, ``as_of`` and ``metadata`` — and the first
# is the one an agent dedupes on, so repeated calls could not tell the same
# evidence item from a new one. Serialisation belongs to the shape, once.


# Human lines shown before the ``+N more`` footer takes over.
_HUMAN_ITEM_LIMIT = 10


def _envelope_human(env) -> str:
    """The envelope as lines an agent can act on without ``--json``.

    Each claim line carries the triple — ``subject PREDICATE object`` — and
    then the fact. The previous rendering printed ``payload.fact`` alone, which
    for most claims is the *evidence note* ("README: 'a Postgres catalog'",
    "src/acme/checkout.py") rather than anything the claim says, so half the
    lines stated no fact at all and the agent re-ran with ``--json`` to learn
    the keys the JSON had carried all along.

    Two more things the old form hid: the ``[:10]`` cut was silent, so a
    21-item envelope looked like a 10-item one; and the same claim reached by
    two families printed twice. Now the cut is announced and a claim is one
    line naming every family that found it.
    """
    intent = env.intent
    if dict(env.metadata or {}).get("intent_source") == "inferred":
        intent = f"{intent} (inferred)"
    lines = [
        f"pot={env.pot_id} intent={intent} confidence={env.overall_confidence} items={len(env.items)}"
    ]
    rows = _dedupe_items(env.items)
    shown = 0
    hidden = 0
    for includes, item in rows:
        body = _item_body(item)
        if body is None:
            continue
        if shown >= _HUMAN_ITEM_LIMIT:
            hidden += 1
            continue
        lines.append(f"  • [{', '.join(includes)}] {body}")
        shown += 1
    if hidden:
        lines.append(f"  … +{hidden} more (use --json)")
    for unsup in env.unsupported_includes:
        lines.append(f"  ! {unsup.name}: {unsup.reason}")
    return "\n".join(lines)


def _dedupe_items(items) -> list[tuple[list[str], object]]:
    """Items in envelope order, one entry per claim, with every include that
    reached it. Keyed on ``claim_key`` (falling back to ``candidate_key``),
    which is what an agent dedupes on across calls too."""
    seen: dict[str, int] = {}
    out: list[tuple[list[str], object]] = []
    for item in items:
        payload = dict(item.payload)
        key = str(payload.get("claim_key") or item.candidate_key or "")
        if key and key in seen:
            includes = out[seen[key]][0]
            if item.include not in includes:
                includes.append(item.include)
            continue
        if key:
            seen[key] = len(out)
        out.append(([item.include], item))
    return out


def _item_body(item) -> str | None:
    """One line of substance for an item, or ``None`` when it has none."""
    payload = dict(item.payload)
    score = f"{float(item.score):.2f}"
    if payload.get("kind") == "resource_chunk":
        where = "/".join(p for p in (payload.get("doc"), payload.get("section")) if p)
        label = payload.get("label") or payload.get("section_title") or ""
        head = " ".join(p for p in (where, payload.get("resource_id")) if p)
        if not head and not label:
            return None
        return f"{head} — {label} ({score})" if label else f"{head} ({score})"
    triple = _triple(payload)
    fact = _collapse_repeats(
        payload.get("fact") or payload.get("description") or payload.get("summary")
    )
    if not triple and not fact:
        return None
    truth = payload.get("truth")
    tail = f"({truth}, {score})" if truth else f"({score})"
    if triple and fact:
        return f"{triple} · {fact} {tail}"
    return f"{triple or fact} {tail}"


def _triple(payload: dict) -> str | None:
    subject = payload.get("subject_key")
    predicate = payload.get("predicate")
    obj = payload.get("object_key")
    if not (subject and predicate and obj):
        return None
    return f"{subject} {predicate} {obj}"


def _collapse_repeats(text) -> str | None:
    """``"X • X"`` → ``"X"``. The record bridge builds a description from the
    summary and the symptom, and a bug pattern recorded with the same text for
    both printed it twice."""
    if not isinstance(text, str) or not text.strip():
        return None
    parts = [p.strip() for p in text.split(" • ")]
    kept: list[str] = []
    for part in parts:
        if part and part not in kept:
            kept.append(part)
    return " • ".join(kept)


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
