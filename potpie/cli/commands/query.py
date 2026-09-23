"""Query + memory commands: ``resolve`` / ``search`` / ``record``.

Routes ``CLI -> HostShell.agent_context -> GraphService -> readers/mutation``.
These three (plus ``status``) are the four-tool agent contract; new use cases
become new ``--intent`` / ``--include`` / ``--type`` values, never new commands.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

import typer
from potpie_context_core.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    CONTEXT_INTENTS,
    READER_BACKED_INCLUDES,
)
from potpie_context_core.agent_envelope import bound_agent_envelope
from potpie_context_core.context_records import (
    REQUIRED_DETAIL_KEYS,
    has_structured_schema,
)
from potpie_context_core.ontology import PUBLIC_RECORD_TYPES
from potpie_context_core.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_core.ports.graph_service import GraphCatalogRequest
from potpie_context_core.source_references import (
    RESOLVE_MODES,
    evidence_review_warnings,
)
from potpie_context_core.vocabulary import close_candidates

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

# Spelled out in --help because the values are not guessable: an agent with no
# list in front of it reaches for the subgraph names it saw in `graph catalog`
# and gets an unsupported_include back. Derived, so the help cannot drift from
# what the orchestrator actually answers.
_INCLUDE_HELP = (
    "Comma-separated include families: "
    + ", ".join(sorted(READER_BACKED_INCLUDES - {"raw_graph"}))
    + ". docs searches summaries and document text; resources searches text only."
    + " Enabled extensions may advertise additional families in graph catalog."
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


def _record_type_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(structured, free_form)`` public record types, from the catalog.

    The help used to hand-write three examples and an ellipsis over a
    fifteen-type vocabulary, so an agent could not discover ``policy`` or
    ``runbook_note`` from the CLI at all. Deriving both groups from
    ``RECORD_TYPES`` keeps the advertised set exactly the accepted set.
    """
    public = sorted(PUBLIC_RECORD_TYPES)
    structured = tuple(rt for rt in public if has_structured_schema(rt))
    free_form = tuple(rt for rt in public if not has_structured_schema(rt))
    return structured, free_form


_STRUCTURED_TYPES, _FREE_FORM_TYPES = _record_type_groups()
_STRUCTURED_TYPES_HELP = "; ".join(
    f"{record_type} (needs {', '.join(REQUIRED_DETAIL_KEYS.get(record_type, ())) or 'no detail'})"
    for record_type in _STRUCTURED_TYPES
)
_TYPE_HELP = (
    "Record type. Structured, retrievable by their own reader — "
    f"{_STRUCTURED_TYPES_HELP}. Free-form notes (summary plus any --detail): "
    f"{', '.join(_FREE_FORM_TYPES)}."
)
_DETAIL_HELP = (
    "Structured field for --type, as key=value (repeatable; repeat a key to "
    f"build a list). Required per type — {_REQUIRED_DETAILS_HELP}; e.g. "
    "`--type decision --detail rationale=...`."
)


def _split(value: str | None, *, host, pot_id: str) -> tuple[str, ...]:
    if not value:
        return ()
    values = tuple(v.strip() for v in value.split(",") if v.strip())
    allowed = set(CONTEXT_INCLUDE_VALUES)
    if set(values) - allowed:
        # Extensions are selected by the serving host, including managed hosts
        # whose settings need not match this client. Only extension/unknown
        # includes need discovery; ordinary queries keep their existing cost.
        catalog = host.graph.catalog(GraphCatalogRequest(pot_id=pot_id))
        allowed.update(
            include
            for view in catalog.views
            if view.get("backed") and isinstance(include := view.get("v1_include"), str)
        )
    unknown = sorted(set(values) - allowed)
    if unknown:
        fail(
            code="validation_error",
            message=f"Unknown include families: {', '.join(unknown)}",
            detail={"argument": "--include", "allowed": sorted(allowed)},
            next_action="Use --include docs for documents, or see --help for valid families.",
        )
    return values


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
        limit: int = typer.Option(12, "--limit", help="Maximum total results across all families."),
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
            if limit < 1:
                raise ValueError("--limit must be >= 1")
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            env = host.agent_context.resolve(
                ResolveRequest(
                    pot_id=pot_id,
                    task=task,
                    intent=intent,
                    include=_split(include, host=host, pot_id=pot_id),
                    mode=mode,
                    max_items=limit,
                )
            )
            env = bound_agent_envelope(_cap_legacy_host_envelope(env, limit))
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
        limit: int = typer.Option(12, "--limit", help="Maximum total results across all families."),
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
            if limit < 1:
                raise ValueError("--limit must be >= 1")
            host = get_host()
            pot_id = resolve_pot_id(host, pot)
            env = host.agent_context.search(
                SearchRequest(
                    pot_id=pot_id,
                    query=query,
                    include=_split(include, host=host, pot_id=pot_id),
                    intent=intent,
                    max_items=limit,
                )
            )
            env = bound_agent_envelope(_cap_legacy_host_envelope(env, limit))
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

        Structured types (fix, bug_pattern, decision, preference, policy,
        verification) validate their required ``--detail`` keys and surface
        through a dedicated reader; free-form types store the summary plus any
        details as a note. ``--type`` help lists every accepted value.
        """
        with contract():
            # Refused before the pot is even resolved: this command *writes*, and
            # a blank type or summary is a durable row nothing can retrieve.
            type = require_text(type, argument="--type", example="--type fix")
            type = _require_record_type(type)
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


def _require_record_type(value: str) -> str:
    """The canonical record type, or a refusal that shows how to fix it.

    Case is folded quietly (``Fix`` is ``fix``; the canonical spelling is in
    the receipt). Anything else is refused *here*, before a pot is resolved
    or a host is asked, with a bounded list of valid types and one corrected
    command carrying the required ``--detail`` keys — nothing is written.
    """
    cleaned = value.strip().lower()
    if cleaned in PUBLIC_RECORD_TYPES:
        return cleaned
    candidates = list(close_candidates(cleaned, PUBLIC_RECORD_TYPES))
    suggested = candidates[0] if candidates else "fix"
    details = " ".join(
        f"--detail {key}=<{key}>" for key in REQUIRED_DETAIL_KEYS.get(suggested, ())
    )
    template = f"potpie record --type {suggested} --summary '<summary>'"
    if details:
        template += f" {details}"
    fail(
        code="validation_error",
        message=f"unknown --type {value!r}.",
        detail={
            "argument": "--type",
            "requested": value,
            "candidates": candidates,
            "structured_types": list(_STRUCTURED_TYPES),
            "free_form_types": list(_FREE_FORM_TYPES),
            "corrected_template": template,
        },
        next_action=f"use one of: {', '.join(candidates)} — e.g. {template}",
    )


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


def _cap_legacy_host_envelope(env, limit: int):
    """Honor a CLI budget when an installed daemon predates total slicing."""
    if len(env.items) <= limit:
        return env
    shown = env.items[:limit]
    metadata = dict(env.metadata or {})
    families = tuple(metadata.get("searched_families") or (row.include for row in env.coverage))
    available = {family: sum(item.include == family for item in env.items) for family in families}
    returned = {family: sum(item.include == family for item in shown) for family in families}
    omitted = {family: available[family] - returned[family] for family in families}
    more_by_family = dict(metadata.get("more_results_by_family") or {})
    more_by_family.update({family: bool(omitted[family]) or more_by_family.get(family, False)
                           for family in families})
    return replace(env, items=shown, metadata={
        **metadata,
        "total_result_budget": limit,
        "returned_by_family": returned,
        "omitted_by_total_budget": omitted,
        "families_with_candidates_omitted": [
            family for family in families if available[family] and not returned[family]
        ],
        "more_results_available": True,
        "more_results_by_family": more_by_family,
    })


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
    metadata = dict(env.metadata or {})
    searched = metadata.get("searched_families") or ()
    if searched:
        lines.append(f"searched={', '.join(searched)} match={metadata.get('match_status', 'unknown')}")
    omitted = metadata.get("omitted_by_total_budget") or {}
    if any(omitted.values()):
        detail = ", ".join(f"{family} {count}" for family, count in omitted.items() if count)
        lines.append(f"  … total limit {metadata['total_result_budget']} omitted {sum(omitted.values())} results ({detail})")
    byte_omitted = metadata.get("omitted_by_output_budget") or {}
    if any(byte_omitted.values()) or metadata.get("omitted_fields_by_candidate"):
        lines.append(
            f"  … output budget {metadata.get('output_budget_bytes')} bytes; "
            f"omitted results={sum(byte_omitted.values())}; "
            f"bounded fields={len(metadata.get('omitted_fields_by_candidate') or {})}"
        )
    if metadata.get("more_results_available"):
        lines.append("  … more results available; raise --limit or narrow the query or family")
    if metadata.get("match_status") == "ambiguous_exact_match":
        repos = metadata.get("matching_repositories") or ()
        lines.append(f"  ! exact ID exists in multiple repositories: {', '.join(repos)}")
    elif metadata.get("match_status") == "no_exact_match":
        exact = metadata.get("exact_identifier") or {}
        lines.append(f"  ! no exact match for {exact.get('display', 'the requested ID')}")
    for family, metadata in env.metadata.get("readers", {}).items():
        lines.extend(
            f"  ! [{family}] {warning}" for warning in metadata.get("warnings", ())
        )
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
        lines.extend(_human_detail_lines(item.payload))
        lines.extend(
            f"    ! {warning}" for warning in evidence_review_warnings(item.payload)
        )
        shown += 1
    if hidden:
        lines.append(f"  … +{hidden} more (use --json)")
    for unsup in env.unsupported_includes:
        lines.append(f"  ! {unsup.name}: {unsup.reason}")
    return "\n".join(lines)


def _human_detail_lines(payload) -> list[str]:
    """Render the same bounded answer fields carried by JSON envelopes."""
    details = payload.get("details")
    if not isinstance(details, Mapping):
        return []
    lines: list[str] = []
    for key, value in details.items():
        if value is None or value == "" or value == []:
            continue
        rendered = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (Mapping, list, tuple))
            else str(value)
        )
        lines.append(f"    {key}: {rendered}")
    if details.get("omitted"):
        lines.append(
            "    fetch_more: use the item's source refs or an exact graph neighborhood read for complete stored details"
        )
    follow_ups = payload.get("follow_up_commands")
    if isinstance(follow_ups, Mapping):
        for name, command in follow_ups.items():
            if command:
                lines.append(f"    {name}: {command}")
    return lines


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
