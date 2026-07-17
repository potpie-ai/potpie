"""``DefaultGraphService`` — data plane over a ``GraphBackend``.

This is the real seam, with a thin body behind it. ``resolve``/``search`` run the
existing one-read-trunk (:class:`ReadOrchestrator`) over the backend's
``claim_query`` port. Graph Surface Lite (V1.5) adds ``catalog`` / ``read`` /
``search_entities`` / ``mutate``:

- ``catalog`` derives the contract from the ontology, the view map, and the
  contract constants — no docs needed.
- ``read`` maps a V2-style ``view`` onto a V1 include and routes through the
  read trunk, stamping graph-contract metadata + ``subgraph_versions``.
- ``search_entities`` projects claim rows into entity candidates for identity
  resolution before a write.
- ``mutate`` validates → risk-classifies → lowers → dry-runs or applies semantic
  mutations through the one write door.

``record`` (the V1 compatibility write) is rewired through the *same* semantic
mutation path (Step 8): it converts the structured record into a
``SemanticMutationRequest`` and calls ``self.mutate`` — there is no private
direct-lowering path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from potpie_context_engine.application.services.read_orchestrator import ReadOrchestrator
from potpie_context_core.record_to_semantic import record_to_semantic_request
from potpie_context_core.semantic_mutation_lowering import lower_semantic_request
from potpie_context_core.semantic_mutation_validator import validate_semantic_request
from potpie_context_core.agent_context_port import (
    build_context_record_source_id,
    normalize_record_type,
)
from potpie_context_core.agent_envelope import AgentEnvelope, EvidenceItem
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.graph_contract import (
    APPLICABLE_MUTATION_OPS,
    DEFERRED_OPS,
    GRAPH_CONTRACT_VERSION,
    ONTOLOGY_VERSION,
    REVIEW_REQUIRED_OPS,
    SOURCE_AUTHORITIES,
    TRUTH_CLASSES,
)
from potpie_context_core.graph_entity_summary import normalize_entity_properties
from potpie_context_core.graph_views import (
    GRAPH_VIEWS,
    UnknownGraphViewError,
    include_guess_guidance,
    view_spec,
    views_for_catalog,
)
from potpie_context_core.graph_workbench_ontology import (
    ViewContract,
    describe_contract,
    ontology_contract,
)
from potpie_context_core.ontology import EDGE_TYPES, ENTITY_TYPES, canonical_entity_labels
from potpie_context_core.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.ports.graph_service import (
    DataPlaneStatus,
    GraphCatalogRequest,
    GraphCatalogResult,
    GraphDescribeRequest,
    GraphEntityCandidate,
    GraphEntitySearchRequest,
    GraphEntitySearchResult,
    GraphReadRequest,
    GraphReadResult,
    normalize_read_detail,
    normalize_read_relations,
)
from potpie_context_core.semantic_mutations import (
    SemanticMutationRequest,
    SemanticMutationResult,
)

_COMMANDS = ("catalog", "read", "search-entities", "mutate")
_KEY_PREFIX_TO_LABEL: dict[str, str] = {
    spec.key_prefix: label for label, spec in ENTITY_TYPES.items()
}


@dataclass(slots=True)
class DefaultGraphService:
    """Data-plane service backed by a swappable ``GraphBackend``."""

    backend: GraphBackend
    _orchestrator: ReadOrchestrator = field(init=False)

    def __post_init__(self) -> None:
        # One read trunk over the backend's canonical claim store.
        self._orchestrator = ReadOrchestrator(claim_query=self.backend.claim_query)

    @property
    def backed_includes(self) -> frozenset[str]:
        return self._orchestrator.backed_includes

    # --- reads --------------------------------------------------------------
    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent=request.intent,
            query=request.task,
            scope=dict(request.scope),
            include=list(request.include) or None,
            exclude=list(request.exclude) or None,
            as_of=request.as_of,
            since=request.since,
            until=request.until,
            max_items=request.max_items,
            freshness_preference=request.freshness_preference,
            include_invalidated=request.include_invalidated,
            metadata={
                "mode": request.mode,
                "source_policy": request.source_policy,
                **dict(request.metadata),
            },
        )

    def search(self, request: SearchRequest) -> AgentEnvelope:
        return self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent="unknown",
            query=request.query,
            scope=dict(request.scope),
            include=list(request.include) or None,
            max_items=request.max_items,
            metadata={"mode": request.mode, "search": True},
        )

    # --- writes -------------------------------------------------------------
    def record(self, request: RecordRequest) -> RecordReceipt:
        """Rewired through the semantic mutation path (Step 8)."""
        record_type = normalize_record_type(request.record_type)
        source_id = build_context_record_source_id(
            record_type=record_type,
            summary=request.summary,
            scope=dict(request.scope),
            source_refs=list(request.source_refs),
            idempotency_key=request.idempotency_key,
        )
        sem_request = record_to_semantic_request(
            request, record_type=record_type, source_id=source_id
        )
        result = self.mutate(sem_request)

        accepted = result.status in ("applied", "validated")
        status = {
            "applied": "recorded",
            "validated": "recorded",
            "review_required": "review_required",
            "rejected": "rejected",
            "error": "rejected",
        }.get(result.status, "rejected")
        detail = result.detail
        if not detail and result.issues:
            detail = "; ".join(i.message for i in result.issues if i.is_error) or None
        return RecordReceipt(
            pot_id=request.pot_id,
            record_type=record_type,
            accepted=accepted and result.ok,
            record_id=source_id,
            status=status,
            mutations_applied=result.operations_applied,
            detail=detail,
            metadata={
                "graph_contract_version": result.graph_contract_version,
                "ontology_version": result.ontology_version,
                "mutation_id": result.mutation_id,
                "claim_keys": list(result.claim_keys),
                "subgraph": result.subgraphs[0] if result.subgraphs else None,
                "subgraphs": list(result.subgraphs),
                "truth": sem_request.operations[0].truth,
                "risk": result.risk,
                "auto_committed": result.auto_committed,
            },
        )

    # --- Graph Surface Lite -------------------------------------------------
    def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult:
        # ``task`` is accepted but ignored in V1.5 (V2 turns it into a ranker).
        views = views_for_catalog()
        if request.subgraph:
            subgraph = request.subgraph.strip()
            known_subgraphs = sorted({str(v["subgraph"]) for v in views})
            views = [v for v in views if v["subgraph"] == subgraph]
            if not views:
                raise _unknown_subgraph_error(subgraph, known_subgraphs)
        return GraphCatalogResult(
            graph_contract_version=GRAPH_CONTRACT_VERSION,
            ontology_version=ONTOLOGY_VERSION,
            commands=_COMMANDS,
            truth_classes=TRUTH_CLASSES,
            mutation_operations=APPLICABLE_MUTATION_OPS,
            review_required_operations=REVIEW_REQUIRED_OPS,
            deferred_operations=DEFERRED_OPS,
            views=tuple(views),
            entity_types=tuple(_catalog_entity_types()),
            predicates=tuple(_catalog_predicates()),
            match_mode=self._match_mode(),
            source_authorities=tuple(sorted(SOURCE_AUTHORITIES)),
        )

    def describe(self, request: GraphDescribeRequest) -> dict[str, Any]:
        # Service-routed (not CLI-local) so the answer always reflects this
        # build's ontology and errors cross the RPC boundary like every other
        # graph command.
        return describe_contract(
            subgraph=request.subgraph,
            view=request.view,
            include_examples=request.include_examples,
        )

    def read(self, request: GraphReadRequest) -> GraphReadResult:
        detail = normalize_read_detail(request.detail)
        relations = normalize_read_relations(request.relations)
        view_name = _qualified_view_name(request.subgraph, request.view)
        contract = ontology_contract().view(view_name)
        spec = view_spec(view_name)
        if contract is None or spec is None:
            raise _unknown_view_error(
                view_name, subgraph=request.subgraph, view=request.view
            )

        unsupported = _unsupported_read_filters(request, contract)
        missing = _missing_required_read_scope(request, contract)
        if unsupported or missing:
            missing_item = {
                "name": contract.name,
                "reason": "missing_required_scope",
                "detail": {
                    "required_scope": list(contract.required_scope),
                    "required_any_scope": list(contract.required_any_scope),
                },
            }
            unsupported_items = (
                tuple([*unsupported, missing_item]) if missing else unsupported
            )
            quality_reason = (
                "missing_required_scope" if missing else "unsupported_filter"
            )
            return GraphReadResult(
                view=contract.name,
                subgraph=contract.subgraph,
                ok=not missing,
                status="missing_required_scope" if missing else None,
                message=(
                    _missing_required_scope_message(contract) if missing else None
                ),
                items=(),
                coverage=(
                    {
                        "view": spec.name,
                        "status": "unsupported" if unsupported or missing else "empty",
                        "candidate_pool": 0,
                    },
                ),
                freshness=_read_freshness(None, backend_freshness={}),
                quality={
                    "status": "unsupported" if unsupported or missing else "empty",
                    "reason": quality_reason,
                },
                source_refs=(),
                match_mode=self._match_mode(),
                backed=spec.backed,
                read_shape=contract.result_shape,
                inline_relations=spec.inline_relations,
                graph_contract_version=GRAPH_CONTRACT_VERSION,
                ontology_version=ONTOLOGY_VERSION,
                subgraph_versions=self._subgraph_versions(request.pot_id),
                unsupported=unsupported_items,
                detail=detail,
                relations=relations,
            )

        scope = dict(request.scope)
        if request.environment:
            scope["environment"] = request.environment
        env = self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent=None,
            query=request.query,
            scope=scope,
            include=[spec.v1_include],
            as_of=request.as_of,
            since=request.since,
            until=request.until,
            max_items=request.limit,
            freshness_preference=request.freshness_preference,
            include_invalidated=request.include_invalidated,
            source_refs=request.source_refs,
            query_threshold=request.query_threshold,
            depth=request.depth,
            direction=request.direction,
        )
        enriched = dataclasses.replace(
            env,
            metadata={
                **dict(env.metadata),
                "graph_contract_version": GRAPH_CONTRACT_VERSION,
                "ontology_version": ONTOLOGY_VERSION,
                "view": spec.name,
                "subgraph": spec.subgraph,
                "backed": spec.backed,
                "match_mode": self._match_mode(),
                "subgraph_versions": self._subgraph_versions(request.pot_id),
                "inline_relations": list(spec.inline_relations),
            },
        )
        if spec.inline_relations:
            enriched = _assemble_inline_relation_items(
                enriched,
                claim_query=self.backend.claim_query,
                inline_relations=spec.inline_relations,
            )
        return _read_result_from_envelope(
            enriched,
            contract=contract,
            match_mode=self._match_mode(),
            subgraph_versions=self._subgraph_versions(request.pot_id),
            backend_freshness=_safe(
                lambda: dict(self.backend.analytics.freshness(request.pot_id)), {}
            ),
            backend_quality=_safe(
                lambda: dict(self.backend.analytics.quality(request.pot_id)), {}
            ),
            detail=detail,
            relations=relations,
        )

    def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult:
        cq = self.backend.claim_query
        predicate_in = (request.predicate,) if request.predicate else ()
        rows = cq.find_claims(
            ClaimQueryFilter(
                pot_id=request.pot_id,
                predicate_in=predicate_in,
                source_ref_in=request.source_refs,
                source_system_in=(request.source_system,)
                if request.source_system
                else (),
                fact_query=request.query or None,
                valid_at_after=request.since,
                valid_at_before=request.until,
                limit=max(request.limit * 10, 100),
            )
        )
        rows = [row for row in rows if _matches_search_filters(row, request)]
        if request.environment:
            rows = [r for r in rows if _row_env(r) == request.environment.lower()]

        # Aggregate per entity: best score + supporting claims.
        agg: dict[str, dict] = {}
        for row in rows:
            sim = row.properties.get("semantic_similarity")
            score = float(sim) if isinstance(sim, (int, float)) else 0.0
            for key in (row.subject_key, row.object_key):
                bucket = agg.setdefault(key, {"score": 0.0, "claims": []})
                bucket["score"] = max(bucket["score"], score)
                bucket["claims"].append(row)

        labels_map = cq.entity_labels(pot_id=request.pot_id, entity_keys=list(agg))
        entity_props = getattr(cq, "entity_properties", None)

        candidates: list[GraphEntityCandidate] = []
        for key, bucket in agg.items():
            labels = _display_labels_for_entity(key, labels_map.get(key, ()))
            if request.type and request.type not in labels:
                continue
            props = (
                entity_props(pot_id=request.pot_id, entity_key=key)
                if callable(entity_props)
                else {}
            )
            props = normalize_entity_properties(props, entity_key=key)
            if request.external_id and not (
                _matches_external_id(key, props, request.external_id)
                or any(
                    _row_matches_source_ref(row, request.external_id)
                    for row in bucket["claims"]
                )
            ):
                continue
            candidates.append(
                GraphEntityCandidate(
                    key=key,
                    labels=labels,
                    name=props.get("name") or _humanize(key),
                    summary=props.get("summary"),
                    description=props.get("description"),
                    score=bucket["score"],
                    supporting_claims=tuple(
                        _claim_brief(c)
                        for c in bucket["claims"][
                            : max(int(request.supporting_claims or 0), 0)
                        ]
                    ),
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return GraphEntitySearchResult(
            entities=tuple(candidates[: request.limit]),
            match_mode=self._match_mode(),
            graph_contract_version=GRAPH_CONTRACT_VERSION,
            ontology_version=ONTOLOGY_VERSION,
            subgraph_versions=self._subgraph_versions(request.pot_id),
        )

    def mutate(self, request: SemanticMutationRequest) -> SemanticMutationResult:
        plan = validate_semantic_request(request)
        if plan.decision == "rejected":
            return SemanticMutationResult(
                ok=False,
                status="rejected",
                risk=plan.risk,
                pot_id=request.pot_id,
                operations_accepted=len(plan.accepted_ops),
                issues=plan.issues,
                detail="; ".join(i.message for i in plan.errors) or None,
            )

        # Lower the accepted ops (for preview counts + apply).
        lower_semantic_request(request, plan)
        preview = _batch_counts(plan)
        claim_keys = tuple(k for op in plan.accepted_ops for k in op.claim_keys)
        subgraphs = tuple(
            sorted({op.subgraph for op in plan.accepted_ops if op.subgraph})
        )

        if request.dry_run:
            return SemanticMutationResult(
                ok=True,
                status="validated",
                risk=plan.risk,
                pot_id=request.pot_id,
                would_apply=(plan.decision == "apply"),
                operations_accepted=len(plan.accepted_ops),
                preview=preview,
                claim_keys=claim_keys,
                subgraphs=subgraphs,
                warnings=_warnings(plan),
                issues=plan.issues,
            )

        if plan.decision == "review_required":
            return SemanticMutationResult(
                ok=True,
                status="review_required",
                risk=plan.risk,
                pot_id=request.pot_id,
                auto_committed=False,
                operations_accepted=len(plan.accepted_ops),
                operations_applied=0,
                claim_keys=claim_keys,
                subgraphs=subgraphs,
                warnings=_warnings(plan),
                issues=plan.issues,
                detail="operations require review and have no auto-apply path in V1.5",
            )

        # decision == "apply"
        if plan.batch is None or not (
            plan.batch.entity_upserts
            or plan.batch.edge_upserts
            or plan.batch.invalidations
        ):
            return SemanticMutationResult(
                ok=True,
                status="applied",
                risk=plan.risk,
                pot_id=request.pot_id,
                auto_committed=True,
                operations_accepted=len(plan.accepted_ops),
                operations_applied=0,
                mutations_applied=preview,
                warnings=_warnings(plan),
                issues=plan.issues,
            )

        result = self.backend.mutation.apply(
            plan.batch,
            expected_pot_id=request.pot_id,
            provenance_context=plan.provenance,
        )
        summary = result.mutation_summary
        return SemanticMutationResult(
            ok=result.ok,
            status="applied" if result.ok else "error",
            risk=plan.risk,
            pot_id=request.pot_id,
            auto_committed=result.ok,
            mutation_id=result.mutation_id,
            operations_accepted=len(plan.accepted_ops),
            operations_applied=len(plan.accepted_ops) if result.ok else 0,
            mutations_applied={
                "entity_upserts": summary.entity_upserts_applied,
                "edge_upserts": summary.edge_upserts_applied,
                "invalidations": summary.invalidations_applied,
            },
            claim_keys=claim_keys,
            subgraphs=subgraphs,
            warnings=_warnings(plan),
            issues=plan.issues,
            detail=result.error,
        )

    # --- status -------------------------------------------------------------
    def data_plane_status(self, pot_id: str) -> DataPlaneStatus:
        readiness = self.backend.mutation.readiness(pot_id)
        counts = _safe(lambda: dict(self.backend.analytics.counts(pot_id)), {})
        freshness = _safe(lambda: dict(self.backend.analytics.freshness(pot_id)), {})
        quality = _safe(lambda: dict(self.backend.analytics.quality(pot_id)), {})
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile=self.backend.profile,
            backend_ready=readiness.ready,
            reader_backed_includes=tuple(sorted(self._orchestrator.backed_includes)),
            counts=counts,
            freshness=freshness,
            quality=quality,
            match_mode=self._match_mode(),
            detail=readiness.detail,
        )

    # --- internals ----------------------------------------------------------
    def _match_mode(self) -> str:
        mode = getattr(self.backend, "match_mode", None)
        if mode:
            return mode
        return getattr(self.backend.claim_query, "match_mode", "lexical")

    def _subgraph_versions(self, pot_id: str) -> dict[str, int]:
        # V1.5 stub: a single monotonic counter (claim count) is enough for V2's
        # optimistic concurrency to be additive later.
        counts = _safe(lambda: dict(self.backend.analytics.counts(pot_id)), {})
        return {"_global": int(counts.get("claims", 0))}


# ---------------------------------------------------------------------------
# Catalog projection
# ---------------------------------------------------------------------------


def _catalog_entity_types() -> list[dict]:
    out: list[dict] = []
    for label, spec in ENTITY_TYPES.items():
        if not spec.public:
            continue
        out.append(
            {
                "label": label,
                "key_prefix": spec.key_prefix,
                "identity_policy": spec.identity_policy,
                "category": spec.category,
                "scope": spec.scope,
            }
        )
    return out


def _catalog_predicates() -> list[dict]:
    out: list[dict] = []
    for name, spec in EDGE_TYPES.items():
        if not spec.public:
            continue
        out.append(
            {
                "name": name,
                "category": spec.category,
                "allowed_pairs": [list(pair) for pair in spec.allowed_pairs],
                "singleton": spec.singleton,
            }
        )
    return out


def _qualified_view_name(subgraph: str, view: str) -> str:
    subgraph_value = (subgraph or "").strip()
    view_value = (view or "").strip()
    if not subgraph_value:
        raise ValueError("--subgraph is required")
    if not view_value:
        raise ValueError("--view is required")
    if "." in view_value:
        raise ValueError(
            "graph read now requires --subgraph <name> --view <view>; "
            f"got fully-qualified view {view_value!r}"
        )
    return f"{subgraph_value}.{view_value}"


def _guidance_suffix(guidance: Mapping[str, Any] | None) -> str:
    if not guidance:
        return ""
    if guidance.get("matched_include"):
        return (
            f" The context include family {guidance['matched_include']!r} is "
            f"served by view {guidance['view']!r}; try `{guidance['read_command']}`."
        )
    return f" Did you mean {guidance['view']!r}? Try `{guidance['read_command']}`."


def _unknown_view_error(
    view_name: str, *, subgraph: str, view: str
) -> UnknownGraphViewError:
    known = ", ".join(sorted(GRAPH_VIEWS))
    guidance = include_guess_guidance(subgraph, view)
    message = (
        f"unknown graph view {view_name!r}. Known views: {known}."
        + _guidance_suffix(guidance)
    )
    return UnknownGraphViewError(
        message,
        did_you_mean=guidance,
        recommended_next_action=guidance["read_command"] if guidance else None,
    )


def _unknown_subgraph_error(
    subgraph: str, known_subgraphs: Sequence[str]
) -> UnknownGraphViewError:
    # catalog is the first command agents run; an include family typed where
    # a subgraph is expected deserves the same migration guidance as read.
    guidance = include_guess_guidance(subgraph, None)
    message = (
        f"unknown graph subgraph {subgraph!r}. "
        f"Known subgraphs: {', '.join(known_subgraphs)}." + _guidance_suffix(guidance)
    )
    return UnknownGraphViewError(
        message,
        did_you_mean=guidance,
        recommended_next_action=guidance["read_command"] if guidance else None,
    )


def _unsupported_read_filters(
    request: GraphReadRequest, contract: ViewContract
) -> tuple[dict[str, Any], ...]:
    supported = set(contract.supported_filters)
    requested = _requested_read_filters(request)
    unsupported = sorted(name for name in requested if name not in supported)
    return tuple(
        {
            "name": name,
            "reason": "unsupported_filter",
            "detail": {
                "view": contract.name,
                "supported_filters": sorted(supported),
            },
        }
        for name in unsupported
    )


def _requested_read_filters(request: GraphReadRequest) -> set[str]:
    requested = {
        key
        for key, value in dict(request.scope).items()
        if value not in (None, "", [], ())
    }
    if request.query:
        requested.add("query")
    if request.since:
        requested.add("since")
    if request.until:
        requested.add("until")
    if request.depth is not None:
        requested.add("depth")
    if request.direction:
        requested.add("direction")
    if request.environment:
        requested.add("environment")
    if request.source_refs:
        requested.add("source_ref")
    return requested


def _missing_required_read_scope(
    request: GraphReadRequest, contract: ViewContract
) -> bool:
    return any(
        not _read_scope_has(request, key) for key in contract.required_scope
    ) or (
        bool(contract.required_any_scope)
        and not any(
            _read_scope_has(request, key) for key in contract.required_any_scope
        )
    )


def _missing_required_scope_message(contract: ViewContract) -> str:
    requirements: list[str] = []
    if contract.required_scope:
        requirements.append("all of " + ", ".join(contract.required_scope))
    if contract.required_any_scope:
        requirements.append("one of " + ", ".join(contract.required_any_scope))
    required = "; ".join(requirements) or "a required scope"
    return f"graph read view {contract.name!r} requires {required}"


def _read_scope_has(request: GraphReadRequest, key: str) -> bool:
    if key == "query":
        return bool(request.query and request.query.strip())
    if key == "environment" and request.environment:
        return True
    scope = dict(request.scope)
    if key == "source_ref" and request.source_refs:
        return True
    if key == "scope":
        return any(value not in (None, "", [], ()) for value in scope.values())
    aliases = {
        "path": ("path", "file_path"),
        "file_path": ("file_path", "path"),
        "service": ("service", "services"),
        "repo": ("repo", "repo_name"),
        "feature": ("feature", "features"),
        "anchor_entity_key": ("anchor_entity_key", "entity", "entity_key"),
        "environment": ("environment", "env"),
        "source_ref": ("source_ref", "source_refs"),
    }.get(key, (key,))
    return any(scope.get(alias) not in (None, "", [], ()) for alias in aliases)


def _read_result_from_envelope(
    env: AgentEnvelope,
    *,
    contract: ViewContract,
    match_mode: str,
    subgraph_versions: Mapping[str, int],
    backend_freshness: Mapping[str, Any],
    backend_quality: Mapping[str, Any],
    detail: str,
    relations: str,
) -> GraphReadResult:
    meta = dict(env.metadata)
    items = tuple(_normalize_read_item(item) for item in env.items)
    source_refs = _source_refs_from_items(items)
    return GraphReadResult(
        view=contract.name,
        subgraph=contract.subgraph,
        items=items,
        coverage=tuple(
            _coverage_dict(report, view_name=contract.name) for report in env.coverage
        ),
        freshness=_read_freshness(env, backend_freshness=backend_freshness),
        quality=_read_quality(env, backend_quality=backend_quality),
        source_refs=source_refs,
        match_mode=match_mode,
        backed=bool(meta.get("backed", contract.backed)),
        read_shape=str(meta.get("read_shape") or contract.result_shape),
        inline_relations=tuple(
            meta.get("inline_relations") or contract.inline_relations
        ),
        inline_relation_count=int(meta.get("inline_relation_count") or 0),
        graph_contract_version=GRAPH_CONTRACT_VERSION,
        ontology_version=ONTOLOGY_VERSION,
        subgraph_versions=subgraph_versions,
        unsupported=tuple(
            {"name": item.name, "reason": item.reason}
            for item in env.unsupported_includes
        ),
        as_of=env.as_of,
        detail=detail,
        relations=relations,
    )


def _coverage_dict(report, *, view_name: str) -> dict[str, Any]:
    # The workbench speaks view vocabulary in and out; the include family the
    # read trunk routed through stays internal (a graph read routes exactly
    # one include, so coverage rows all belong to the requested view).
    return {
        "view": view_name,
        "status": report.status,
        "candidate_pool": report.candidate_pool,
    }


def _normalize_read_item(item: EvidenceItem) -> dict[str, Any]:
    payload = dict(item.payload)
    entity = payload.get("entity") if isinstance(payload.get("entity"), Mapping) else {}
    relations_raw = (
        payload.get("relations") if isinstance(payload.get("relations"), list) else []
    )
    relations = tuple(
        _normalize_read_relation(rel)
        for rel in relations_raw
        if isinstance(rel, Mapping)
    )
    if entity:
        entity_key = _str_or_none(entity.get("key")) or item.candidate_key
        labels = tuple(str(label) for label in entity.get("labels") or () if label)
        source_refs = _source_refs_from_relations(relations)
        return {
            "entity_key": entity_key,
            "entity_type": _entity_type_for_key(entity_key, labels),
            "score": item.score,
            "summary": _first_text(
                entity.get("summary"),
                entity.get("description"),
                entity.get("name"),
                entity_key,
            ),
            "status": _status_from_payload(payload),
            "relations": [dict(rel) for rel in relations],
            "source_refs": list(source_refs),
            "truth": _first_truth(relations) or _str_or_none(payload.get("truth")),
            "coverage_status": item.coverage_status,
            "breakdown": dict(item.breakdown),
        }

    source_refs = _string_tuple(payload.get("source_refs"))
    subject_key = _str_or_none(payload.get("subject_key"))
    object_key = _str_or_none(payload.get("object_key"))
    entity_key = subject_key or object_key or item.candidate_key
    return {
        "entity_key": entity_key,
        "entity_type": _entity_type_for_key(entity_key, ()),
        "score": item.score,
        "summary": _first_text(
            payload.get("description"),
            payload.get("fact"),
            payload.get("summary"),
            payload.get("name"),
            item.candidate_key,
        ),
        "status": _status_from_payload(payload),
        "claim": {
            "claim_key": payload.get("claim_key"),
            "predicate": payload.get("predicate"),
            "subject_key": subject_key,
            "object_key": object_key,
            "environment": payload.get("environment"),
            "valid_at": payload.get("valid_at"),
            "valid_until": payload.get("valid_until"),
            "observed_at": payload.get("observed_at"),
            "evidence_strength": payload.get("evidence_strength"),
        },
        "relations": [],
        "source_refs": list(source_refs),
        "truth": _str_or_none(payload.get("truth")),
        "coverage_status": item.coverage_status,
        "breakdown": dict(item.breakdown),
    }


def _normalize_read_relation(rel: Mapping[str, Any]) -> dict[str, Any]:
    claim = rel.get("claim") if isinstance(rel.get("claim"), Mapping) else {}
    return {
        "type": rel.get("predicate"),
        "predicate": rel.get("predicate"),
        "direction": rel.get("direction"),
        "from": rel.get("from_key"),
        "to": rel.get("to_key"),
        "from_key": rel.get("from_key"),
        "to_key": rel.get("to_key"),
        "related_key": rel.get("related_key"),
        "fact": rel.get("fact"),
        "source_refs": list(_string_tuple(rel.get("source_refs"))),
        "truth": rel.get("truth"),
        "environment": rel.get("environment"),
        "valid_at": rel.get("valid_at"),
        "valid_until": rel.get("valid_until"),
        "observed_at": rel.get("observed_at"),
        "properties": dict(rel.get("properties") or {}),
        "claim_key": claim.get("candidate_key"),
        "score": claim.get("score"),
    }


def _read_freshness(
    env: AgentEnvelope | None, *, backend_freshness: Mapping[str, Any]
) -> dict[str, Any]:
    items = tuple(_normalize_read_item(item) for item in env.items) if env else ()
    source_refs = _source_refs_from_items(items)
    latest = _latest_timestamp_from_items(items)
    return {
        "as_of": env.as_of.isoformat() if env and env.as_of else None,
        "latest_observed_at": latest,
        "source_refs_count": len(source_refs),
        "backend": dict(backend_freshness),
    }


def _read_quality(
    env: AgentEnvelope, *, backend_quality: Mapping[str, Any]
) -> dict[str, Any]:
    statuses = [report.status for report in env.coverage]
    return {
        "status": "ok" if statuses and all(s != "empty" for s in statuses) else "watch",
        "coverage_statuses": statuses,
        "confidence": env.overall_confidence,
        "backend": dict(backend_quality),
    }


def _source_refs_from_items(items: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        for ref in _string_tuple(item.get("source_refs")):
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
        relations = item.get("relations")
        if isinstance(relations, list):
            for rel in relations:
                if isinstance(rel, Mapping):
                    for ref in _string_tuple(rel.get("source_refs")):
                        if ref not in seen:
                            refs.append(ref)
                            seen.add(ref)
    return tuple(refs)


def _source_refs_from_relations(
    relations: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for rel in relations:
        for ref in _string_tuple(rel.get("source_refs")):
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return tuple(refs)


def _latest_timestamp_from_items(items: tuple[Mapping[str, Any], ...]) -> str | None:
    values: list[str] = []
    for item in items:
        claim = item.get("claim") if isinstance(item.get("claim"), Mapping) else {}
        for key in ("observed_at", "valid_at"):
            value = claim.get(key)
            if isinstance(value, str) and value:
                values.append(value)
        relations = item.get("relations")
        if isinstance(relations, list):
            for rel in relations:
                if not isinstance(rel, Mapping):
                    continue
                for key in ("observed_at", "valid_at"):
                    value = rel.get(key)
                    if isinstance(value, str) and value:
                        values.append(value)
    return max(values) if values else None


def _entity_type_for_key(entity_key: str, labels: tuple[str, ...]) -> str | None:
    prefix = entity_key.partition(":")[0]
    if prefix in _KEY_PREFIX_TO_LABEL:
        return _KEY_PREFIX_TO_LABEL[prefix]
    return next((label for label in labels if label != "Entity"), None)


def _status_from_payload(payload: Mapping[str, Any]) -> str | None:
    for value in (
        payload.get("status"),
        (payload.get("properties") or {}).get("status")
        if isinstance(payload.get("properties"), Mapping)
        else None,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_truth(relations: tuple[Mapping[str, Any], ...]) -> str | None:
    for rel in relations:
        truth = _str_or_none(rel.get("truth"))
        if truth:
            return truth
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def _matches_search_filters(row: ClaimRow, request: GraphEntitySearchRequest) -> bool:
    if request.subgraph and row.subgraph != request.subgraph:
        return False
    if request.truth and row.truth != request.truth:
        return False
    if request.source_family and _row_source_family(row) != request.source_family:
        return False
    if request.source_refs and not any(
        _row_matches_source_ref(row, ref) for ref in request.source_refs
    ):
        return False
    if request.scope and not _row_matches_scope(row, request.scope):
        return False
    return True


def _row_matches_scope(row: ClaimRow, scope: Mapping[str, Any]) -> bool:
    needles = tuple(_scope_needles(scope))
    if not needles:
        return True
    haystack = " ".join(
        str(value or "")
        for value in (
            row.subject_key,
            row.object_key,
            row.fact,
            row.description,
            row.subgraph,
            row.source_ref,
            " ".join(row.source_refs),
            row.source_system,
            row.environment,
        )
    ).lower()
    return all(needle in haystack for needle in needles)


def _scope_needles(scope: Mapping[str, Any]) -> list[str]:
    needles: list[str] = []
    for key, value in scope.items():
        if value in (None, "", [], ()):
            continue
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if not isinstance(item, str) or not item.strip():
                continue
            raw = item.strip().lower()
            if ":" in raw:
                needles.append(raw)
                continue
            prefix = {
                "service": "service",
                "services": "service",
                "repo": "repo",
                "repo_name": "repo",
                "feature": "feature",
                "features": "feature",
                "environment": "environment",
            }.get(key)
            needles.append(f"{prefix}:{raw}" if prefix else raw)
    return needles


def _matches_external_id(
    entity_key: str,
    properties: Mapping[str, Any],
    external_id: str,
) -> bool:
    wanted = external_id.strip().lower()
    if not wanted:
        return True
    for key in ("external_id", "externalId", "provider_id", "source_id"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip().lower() == wanted:
            return True
    for key in ("external_ids", "alternate_external_ids", "source_ids"):
        value = properties.get(key)
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, str) and item.strip().lower() == wanted for item in value
        ):
            return True
    # External-id identities keep the provider id in the canonical key body.
    return entity_key.lower().endswith(f":{wanted}")


def _row_matches_source_ref(row: ClaimRow, source_ref: str) -> bool:
    wanted = source_ref.strip().lower()
    if not wanted:
        return True
    refs = []
    if row.source_ref:
        refs.append(row.source_ref)
    refs.extend(row.source_refs)
    for item in row.evidence:
        if not isinstance(item, Mapping):
            continue
        ref = item.get("source_ref")
        if isinstance(ref, str):
            refs.append(ref)
    return any(ref.strip().lower() == wanted for ref in refs if ref)


def _batch_counts(plan) -> dict[str, int]:
    batch = plan.batch
    if batch is None:
        return {"entity_upserts": 0, "edge_upserts": 0, "invalidations": 0}
    return {
        "entity_upserts": len(batch.entity_upserts),
        "edge_upserts": len(batch.edge_upserts),
        "invalidations": len(batch.invalidations),
    }


def _warnings(plan) -> tuple[str, ...]:
    return tuple(i.message for i in plan.issues if not i.is_error)


def _row_env(row: ClaimRow) -> str | None:
    env = row.environment
    return env.lower() if isinstance(env, str) else None


def _row_source_family(row: ClaimRow) -> str | None:
    if row.source_system:
        return row.source_system.strip().split(":", 1)[0].lower()
    refs = row.source_refs or ((row.source_ref,) if row.source_ref else ())
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip().split(":", 1)[0].lower()
    return None


def _claim_brief(row: ClaimRow) -> dict:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "fact": row.fact,
        "environment": _row_env(row),
        "source_refs": list(
            row.source_refs or ((row.source_ref,) if row.source_ref else ())
        ),
    }


def _humanize(key: str) -> str:
    body = key.split(":", 1)[1] if ":" in key else key
    return body.replace("-", " ").replace("_", " ")


def _assemble_inline_relation_items(
    env: AgentEnvelope,
    *,
    claim_query,
    inline_relations: tuple[str, ...],
) -> AgentEnvelope:
    """Project flat ranked claim items into entity payloads with relations."""
    allowed = {p.upper() for p in inline_relations}
    relation_groups: dict[str, list[dict[str, Any]]] = {}
    relation_keys: dict[str, set[tuple[Any, ...]]] = {}
    best_item: dict[str, EvidenceItem] = {}

    for item in env.items:
        payload = dict(item.payload)
        predicate = _str_or_none(payload.get("predicate"))
        subject_key = _str_or_none(payload.get("subject_key"))
        object_key = _str_or_none(payload.get("object_key"))
        if (
            predicate is None
            or subject_key is None
            or object_key is None
            or predicate.upper() not in allowed
        ):
            continue

        out_rel = _relation_payload(
            item,
            payload=payload,
            predicate=predicate,
            from_key=subject_key,
            to_key=object_key,
            direction="out",
            related_key=object_key,
        )
        in_rel = _relation_payload(
            item,
            payload=payload,
            predicate=predicate,
            from_key=subject_key,
            to_key=object_key,
            direction="in",
            related_key=subject_key,
        )
        _append_relation(
            relation_groups,
            relation_keys,
            subject_key,
            out_rel,
            _relation_dedupe_key(
                payload, predicate=predicate, from_key=subject_key, to_key=object_key
            ),
        )
        _append_relation(
            relation_groups,
            relation_keys,
            object_key,
            in_rel,
            _relation_dedupe_key(
                payload, predicate=predicate, from_key=subject_key, to_key=object_key
            ),
        )
        _keep_best(best_item, subject_key, item)
        _keep_best(best_item, object_key, item)

    if not relation_groups:
        return dataclasses.replace(
            env,
            metadata={
                **dict(env.metadata),
                "read_shape": "flat_claims",
                "inline_relation_assembly": "no_relation_payloads",
            },
        )

    entity_keys = tuple(relation_groups)
    labels = _safe_entity_labels(
        claim_query, pot_id=env.pot_id, entity_keys=entity_keys
    )
    entity_props = getattr(claim_query, "entity_properties", None)
    props_by_key = {
        entity_key: normalize_entity_properties(
            (
                entity_props(pot_id=env.pot_id, entity_key=entity_key)
                if callable(entity_props)
                else {}
            ),
            entity_key=entity_key,
        )
        for entity_key in entity_keys
    }

    items: list[EvidenceItem] = []
    for entity_key, relations in relation_groups.items():
        top = best_item[entity_key]
        props = props_by_key.get(entity_key, {})
        items.append(
            EvidenceItem(
                include=top.include,
                candidate_key=entity_key,
                score=top.score,
                coverage_status=top.coverage_status,
                breakdown=dict(top.breakdown),
                payload={
                    "entity": {
                        "key": entity_key,
                        "labels": list(labels.get(entity_key, ())),
                        "name": props.get("name") or _humanize(entity_key),
                        "summary": props.get("summary"),
                        "description": props.get("description"),
                    },
                    "relations": [
                        {
                            **rel,
                            "related_entity": {
                                "key": rel["related_key"],
                                "labels": list(labels.get(rel["related_key"], ())),
                                "name": props_by_key.get(rel["related_key"], {}).get(
                                    "name"
                                )
                                or _humanize(rel["related_key"]),
                                "summary": props_by_key.get(rel["related_key"], {}).get(
                                    "summary"
                                ),
                                "description": props_by_key.get(
                                    rel["related_key"], {}
                                ).get("description"),
                            },
                        }
                        for rel in relations
                    ],
                    "relation_count": len(relations),
                },
            )
        )

    items.sort(key=lambda item: item.score, reverse=True)
    return dataclasses.replace(
        env,
        items=tuple(items),
        metadata={
            **dict(env.metadata),
            "read_shape": "entity_relations",
            "inline_relation_count": sum(len(v) for v in relation_groups.values()),
        },
    )


def _relation_payload(
    item: EvidenceItem,
    *,
    payload: Mapping[str, Any],
    predicate: str,
    from_key: str,
    to_key: str,
    direction: str,
    related_key: str,
) -> dict[str, Any]:
    return {
        "predicate": predicate,
        "direction": direction,
        "from_key": from_key,
        "to_key": to_key,
        "related_key": related_key,
        "fact": payload.get("fact"),
        "source_refs": list(payload.get("source_refs") or []),
        "source_system": payload.get("source_system"),
        "truth": payload.get("truth"),
        "environment": payload.get("environment"),
        "valid_at": payload.get("valid_at"),
        "valid_until": payload.get("valid_until"),
        "observed_at": payload.get("observed_at"),
        "properties": dict(payload.get("properties") or {}),
        "evidence_strength": payload.get("evidence_strength"),
        "claim": {
            "candidate_key": item.candidate_key,
            "score": item.score,
            "coverage_status": item.coverage_status,
            "breakdown": dict(item.breakdown),
        },
    }


def _append_relation(
    relation_groups: dict[str, list[dict[str, Any]]],
    relation_keys: dict[str, set[tuple[Any, ...]]],
    entity_key: str,
    relation: dict[str, Any],
    dedupe_key: tuple[Any, ...],
) -> None:
    seen = relation_keys.setdefault(entity_key, set())
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    relation_groups.setdefault(entity_key, []).append(relation)


def _relation_dedupe_key(
    payload: Mapping[str, Any], *, predicate: str, from_key: str, to_key: str
) -> tuple[Any, ...]:
    claim_key = _str_or_none(payload.get("claim_key"))
    if claim_key:
        return ("claim", claim_key)
    source_refs = tuple(
        sorted(ref for ref in payload.get("source_refs") or () if isinstance(ref, str))
    )
    return ("triple", predicate.upper(), from_key, to_key, source_refs)


def _keep_best(
    best_item: dict[str, EvidenceItem], entity_key: str, item: EvidenceItem
) -> None:
    previous = best_item.get(entity_key)
    if previous is None or item.score > previous.score:
        best_item[entity_key] = item


def _safe_entity_labels(claim_query, *, pot_id: str, entity_keys: tuple[str, ...]):
    try:
        raw = dict(claim_query.entity_labels(pot_id=pot_id, entity_keys=entity_keys))
    except CapabilityNotImplemented:
        raw = {}
    return {
        key: _display_labels_for_entity(key, raw.get(key, ())) for key in entity_keys
    }


def _display_labels_for_entity(
    entity_key: str, labels: tuple[str, ...]
) -> tuple[str, ...]:
    prefix = entity_key.partition(":")[0]
    prefix_label = _KEY_PREFIX_TO_LABEL.get(prefix)
    if prefix_label:
        return (prefix_label,)
    canonical = tuple(
        label for label in canonical_entity_labels(labels) if label != "Entity"
    )
    if canonical:
        return canonical
    return tuple(label for label in labels if label != "Entity")


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _safe(fn, default):
    """Run a backend analytics call, swallowing not-implemented projections."""
    try:
        return fn()
    except CapabilityNotImplemented:
        return default


__all__ = ["DefaultGraphService"]
