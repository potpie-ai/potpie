"""The single read trunk (rebuild plan P8 + P9).

One orchestrator is the entire read path. It does exactly four things:

1. expand ``intent`` → include families (``includes_for_request``),
2. route each include to its P9 reader (the ``_ROUTING`` table),
3. run each reader over the canonical claim store (``ClaimQueryPort``),
4. assemble one ranked :class:`AgentEnvelope` (P8 shape).

Includes that resolve to no reader are surfaced as
``unsupported_include`` (reason ``not_implemented``) rather than silently
returning nothing — the plan's anti-phantom-vocabulary rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping

from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.coding_preferences import (
    CodingPreferencesReader,
)
from potpie_context_engine.application.readers.decisions import DecisionsReader
from potpie_context_engine.application.readers.docs import DocsReader
from potpie_context_engine.application.readers.features import FeaturesReader
from potpie_context_engine.application.readers.infra_topology import InfraTopologyReader
from potpie_context_engine.application.readers.owners import OwnersReader
from potpie_context_engine.application.readers.prior_bugs import PriorBugsReader
from potpie_context_engine.application.readers.raw_graph import RawGraphReader
from potpie_context_engine.application.readers.resources import ResourcesReader
from potpie_context_engine.application.readers.timeline_reader import TimelineReader
from potpie_context_engine.application.services.envelope_builder import (
    EnvelopeBuilder,
    IncludeResult,
)
from potpie_context_core.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    includes_for_request,
    normalize_context_intent,
)
from potpie_context_core.agent_envelope import AgentEnvelope, UnsupportedInclude, bound_agent_envelope
from potpie_context_core.cli_commands import (
    graph_neighborhood_command,
    resource_get_command,
)
from potpie_context_core.definition import GraphReaderSpec
from potpie_context_core.ports.claim_query import ClaimQueryPort
from potpie_context_core.ports.resource_index import ResourceIndexPort
from potpie_context_engine.domain.ranking import RankingService
from potpie_context_engine.application.search_identity import (
    exact_identity,
    record_identity_matches,
    repositories_in,
)


# Protocol-free reader alias: every P9 reader exposes ``read(ReadRequest)``.
_ReaderT = Any


@dataclass(slots=True)
class ReadOrchestrator:
    """Build P9 readers over a claim store and resolve a task into an envelope."""

    claim_query: ClaimQueryPort
    ranker: RankingService = field(default_factory=RankingService)
    builder: EnvelopeBuilder = field(default_factory=EnvelopeBuilder)
    reader_registry: Mapping[str, GraphReaderSpec] = field(default_factory=dict)
    resource_store: Any = None
    resource_index: ResourceIndexPort | None = None
    """Backs ``resources``. ``None`` wires the fail-closed ``none`` profile.

    Optional because a graph runtime can be composed with no document store at
    all (an ingestion pipeline, a test), and mandatory-by-substitution because
    the include family is always registered: the substitute answers zero hits
    with ``match_mode="disabled"`` instead of the family vanishing from the
    contract, which is what keeps the coherence check meaningful."""

    _routing: dict[str, _ReaderT] = field(init=False)

    def __post_init__(self) -> None:
        cq, rk = self.claim_query, self.ranker
        # include family → reader. One reader can back multiple includes.
        self._routing = {
            "coding_preferences": CodingPreferencesReader(claim_query=cq, ranker=rk),
            "features": FeaturesReader(claim_query=cq, ranker=rk),
            "infra_topology": InfraTopologyReader(claim_query=cq, ranker=rk),
            "timeline": TimelineReader(claim_query=cq, ranker=rk),
            "prior_bugs": PriorBugsReader(claim_query=cq, ranker=rk),
            "decisions": DecisionsReader(claim_query=cq, ranker=rk),
            "owners": OwnersReader(claim_query=cq, ranker=rk),
            "docs": DocsReader(claim_query=cq, ranker=rk),
            # The one reader that does not read the claim store: document
            # payloads come from the resource index. ``docs`` answers "what did
            # we say about this document", ``resources`` answers "which passage
            # says it" — a fact no section summary mentions is reachable only
            # through the second.
            "resources": ResourcesReader(
                index=self.resource_index or _disabled_resource_index(), ranker=rk
            ),
            # Visualization read: the whole canonical partition (all RELATES_TO,
            # incl. generic RELATED_TO) for the graph explorer — not a UC slice.
            "raw_graph": RawGraphReader(claim_query=cq, ranker=rk),
        }
        for include, spec in self.reader_registry.items():
            if spec.factory is None:
                continue
            self._routing[include] = _build_reader(
                spec.factory,
                claim_query=cq,
                ranker=rk,
                resource_store=self.resource_store,
            )
        self.builder.additional_includes = frozenset(self.reader_registry)

    @property
    def backed_includes(self) -> frozenset[str]:
        return frozenset(self._routing)

    def resolve(
        self,
        *,
        pot_id: str,
        intent: str | None = None,
        query: str | None = None,
        scope: Mapping[str, Any] | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        as_of: datetime | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_items: int = 12,
        detail: str = "compact",
        freshness_preference: str = "balanced",
        include_invalidated: bool = False,
        source_refs: tuple[str, ...] = (),
        query_threshold: float | None = None,
        depth: int | None = None,
        direction: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentEnvelope:
        intent = normalize_context_intent(intent)
        resolved = includes_for_request(intent, include or [], exclude or [])
        req = ReadRequest(
            pot_id=pot_id,
            scope=dict(scope or {}),
            query=query,
            intent=intent,
            as_of=as_of,
            since=since,
            until=until,
            max_items=max_items,
            detail=detail,
            freshness_preference=freshness_preference,
            include_invalidated=include_invalidated,
            source_refs=tuple(source_refs),
            query_threshold=query_threshold,
            depth=depth,
            direction=direction,
        )
        results: list[IncludeResult] = []
        extra_unsupported: list[UnsupportedInclude] = []
        for inc in resolved:
            reader = self._routing.get(inc)
            if reader is not None:
                response = reader.read(req)
                response = replace(response, meta={
                    "result_limit": req.max_items,
                    "returned": len(response.items),
                    "page_full": len(response.items) >= req.max_items,
                    "candidate_pool_unit": "reader_candidates",
                    **dict(response.meta),
                    "completeness": (
                        "truncated" if response.meta.get("ranking_omitted", 0) or response.meta.get("truncated")
                        else response.meta.get("completeness", "unknown")
                    ),
                })
                results.append(IncludeResult(include=inc, response=response))
            elif inc in CONTEXT_INCLUDE_VALUES:
                # In the vocab but no reader yet → honest not-implemented.
                extra_unsupported.append(
                    UnsupportedInclude(name=inc, reason="not_implemented")
                )
            # else: not in the vocab at all → the EnvelopeBuilder flags it
            # ``unknown_include`` from ``requested_includes``.

        envelope = self.builder.build(
            pot_id=pot_id,
            intent=intent,
            results=results,
            requested_includes=include or None,
            extra_unsupported=extra_unsupported,
            as_of=as_of,
            metadata=metadata,
        )
        return _describe_search_result(
            envelope,
            query=query,
            searched_families=resolved,
            max_items=max_items,
        )


__all__ = ["ReadOrchestrator"]


def _describe_search_result(
    envelope: AgentEnvelope,
    *,
    query: str | None,
    searched_families: list[str],
    max_items: int,
) -> AgentEnvelope:
    """Add the W8 honesty/continuation contract without changing reader data."""
    identity = exact_identity(query)
    items = list(envelope.items)
    exact_items = [
        item
        for item in items
        if identity is not None
        and record_identity_matches(
            identity,
            canonical_key=(
                item.payload.get("subject_key")
                or item.payload.get("entity_key")
                or item.candidate_key
            ),
            explicit_identity=(
                item.payload.get("source_ref"),
                item.payload.get("source_refs"),
                item.payload.get("properties"),
            ),
            description=(
                item.payload.get("fact"),
                item.payload.get("description"),
                item.payload.get("summary"),
            ),
        )
    ]

    if identity is not None:
        # A made-up ticket must not be presented as found merely because the
        # similarity backend can always fill a page.
        identity_families = {"timeline", "docs", "resources"}
        if dict(envelope.metadata).get("search"):
            items = exact_items
        else:
            # Resolve can be a combined question. Exactness governs the
            # identity-bearing families without erasing an independently
            # requested decision/preference/architecture answer.
            items = [
                item
                for item in items
                if item.include not in identity_families or item in exact_items
            ]
        repos = sorted({repo for item in items for repo in repositories_in(item.payload)})
        if not items:
            match_status = "no_exact_match"
        elif len(repos) > 1:
            match_status = "ambiguous_exact_match"
        else:
            match_status = "exact_match"
    else:
        repos = []
        match_status = "possible_matches" if items else "no_matches"

    items = [_with_follow_ups(item, envelope.pot_id, identity) for item in items]
    available_by_family = {
        family: sum(item.include == family for item in items)
        for family in searched_families
    }
    items = items[: max(max_items, 0)]
    pool_by_family = {report.include: report.candidate_pool for report in envelope.coverage}
    returned_by_family = {
        family: sum(item.include == family for item in items)
        for family in searched_families
    }
    omitted_by_total_budget = {
        family: available_by_family[family] - returned_by_family[family]
        for family in searched_families
    }
    more_by_family = {
        family: bool(omitted_by_total_budget[family]) or (
            pool_by_family.get(family, 0) > available_by_family[family]
            and available_by_family[family] >= max_items
        )
        for family in searched_families
    }
    meta = {
        **dict(envelope.metadata),
        "searched_families": list(searched_families),
        "match_status": match_status,
        "total_result_budget": max_items,
        "returned_by_family": returned_by_family,
        "omitted_by_total_budget": omitted_by_total_budget,
        "families_with_candidates_omitted": [
            family for family in searched_families
            if available_by_family[family] and not returned_by_family[family]
        ],
        "more_results_available": any(more_by_family.values()),
        "more_results_by_family": more_by_family,
    }
    if identity is not None:
        meta["exact_identifier"] = {
            "kind": identity.kind,
            "value": identity.value,
            "display": identity.display,
        }
        meta["exact_match_count"] = len(exact_items)
    if repos:
        meta["matching_repositories"] = repos
        if len(repos) > 1:
            meta["disambiguation"] = "Repeat the lookup with an explicit repository scope."
    return bound_agent_envelope(replace(envelope, items=tuple(items), metadata=meta))


def _with_follow_ups(item, pot_id: str, identity):
    payload = dict(item.payload)
    commands = dict(payload.get("follow_up_commands") or {})
    entity_key = payload.get("subject_key") or payload.get("entity_key")
    if isinstance(entity_key, str) and entity_key:
        command_name = "named_record" if identity is not None else "entity_context"
        commands[command_name] = graph_neighborhood_command(
            entity_key, pot_id=pot_id, depth=1, limit=50, detail="full"
        )
    chunk_ids = payload.get("chunk_ids") or ()
    resource_id = payload.get("resource_id")
    chunk_id = next((x for x in chunk_ids if isinstance(x, str)), None)
    if not chunk_id:
        chunk_id = next(
            (
                ref
                for ref in payload.get("source_refs") or ()
                if isinstance(ref, str) and ref.startswith("potpie://res/")
            ),
            None,
        )
    if not chunk_id and isinstance(resource_id, str):
        chunk_id = resource_id
    if chunk_id:
        commands["source_passage"] = resource_get_command(
            chunk_id, pot_id=pot_id, with_neighbors=True
        )
    payload["follow_up_commands"] = commands
    return replace(item, payload=payload)


def _disabled_resource_index() -> ResourceIndexPort:
    """The stand-in used when no index was wired.

    Imported lazily so ``application`` keeps depending on ``domain`` and
    ``ports`` rather than on an outbound adapter at module load — the same
    reason the graph readers never import a backend."""
    from potpie_context_engine.adapters.outbound.resources.index._unimplemented import (
        NullResourceIndex,
    )

    return NullResourceIndex(
        detail="no resource index is wired on this host, so document payloads "
        "are not searchable"
    )


def _build_reader(
    factory: Any,
    *,
    claim_query: ClaimQueryPort,
    ranker: RankingService,
    resource_store=None,
):
    if not callable(factory):
        if not callable(getattr(factory, "read", None)):
            raise TypeError("graph reader must be callable or expose read(request)")
        return factory
    import inspect

    if "resource_store" in inspect.signature(factory).parameters:
        return factory(
            claim_query=claim_query, ranker=ranker, resource_store=resource_store
        )
    try:
        reader = factory(claim_query=claim_query, ranker=ranker)
    except TypeError:
        try:
            reader = factory(claim_query=claim_query)
        except TypeError:
            reader = factory(claim_query)
    if not callable(getattr(reader, "read", None)):
        raise TypeError("graph reader factory must return an object with read(request)")
    return reader
