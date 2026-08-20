"""Temporary finite adapter from typed engine operations to the shipped host.

This module exists only while root callers and the reflective daemon migrate in
the same PR. It converts the current HostShell DTOs and expected exceptions at
one named boundary; callers never dynamically dispatch over HostShell.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from potpie.cli.repo_location import repo_identity_key
from potpie.runtime.clients import ClientOutcome, LegacyEngineClientAdapter
from potpie.runtime.operations import ENGINE_OPERATION_CATALOG, EngineOperation
from potpie.runtime.protocol import EngineOperationRequest
from potpie.runtime.resource_manager import (
    AuthenticatedActor,
    AuthorizationScope,
    AuthorizationError,
    CompositionFingerprint,
    ContextResourceManager,
    ContextSelector,
    ResourceComposition,
    SelectionError,
)
from potpie_context_engine import (
    ContextIdentity,
    DependencyError,
    DomainError,
    EngineConfig,
    EngineDependencies,
    Failure,
    Outcome,
    Success,
)
from potpie_context_engine.core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie_context_engine.core.ports.agent_context import (
    RecordRequest as HostRecordRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    ResolveRequest as HostResolveRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    SearchRequest as HostSearchRequest,
)
from potpie_context_engine.core.ports.graph_service import (
    GraphCatalogRequest,
    GraphDescribeRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.requests import (
    CatalogRequest,
    CommitRequest,
    DataPlaneStatusRequest,
    DescribeRequest,
    EngineRequest,
    ExportSnapshotRequest,
    HistoryRequest,
    ImportSnapshotRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxCloseRequest,
    InboxListRequest,
    InboxMarkAppliedRequest,
    InboxMarkRejectedRequest,
    InboxShowRequest,
    InspectRequest,
    MutateRequest,
    NeighborhoodRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RecordRequest,
    RepairRequest,
    ResolveRequest,
    SearchEntitiesRequest,
    SearchRequest,
)


class HostContextSelectorResolver:
    """Resolve exact, active, and repository selectors through Potpie services."""

    def __init__(self, host: Any) -> None:
        self._host = host

    async def resolve(
        self, selector: ContextSelector
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        return await asyncio.to_thread(self._resolve, selector)

    def _resolve(
        self, selector: ContextSelector
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        try:
            pots = tuple(self._host.pots.list_pots())
            active = self._host.pots.active_pot()
        except Exception:
            return Failure(
                SelectionError(
                    code="context_selection_unavailable",
                    message="Potpie could not read the local context catalog.",
                    recommended_next_action="check daemon readiness with 'potpie doctor'",
                    retry_posture="safe",
                )
            )

        if selector.kind == "explicit":
            for pot in pots:
                if selector.value in {pot.pot_id, pot.name}:
                    return Success(ContextIdentity(pot.pot_id))
            return Failure(
                SelectionError(
                    code="pot_not_found",
                    message=f"No pot matching '{selector.value}'.",
                    recommended_next_action="run 'potpie pot list'",
                )
            )

        if selector.kind == "active":
            if active is not None:
                return Success(ContextIdentity(active.pot_id))
            return Failure(_no_active_pot_error())

        repo = selector.value or ""
        try:
            default_pot_id = self._host.pots.repo_default(repo=repo)
        except Exception:
            default_pot_id = None
        known_ids = {pot.pot_id for pot in pots}
        if default_pot_id and default_pot_id in known_ids:
            return Success(ContextIdentity(str(default_pot_id)))

        matches = []
        for pot in pots:
            try:
                sources = self._host.pots.list_sources(pot_id=pot.pot_id)
            except Exception:  # noqa: S112 - one unreadable pot must not mask others.
                continue
            if any(
                source.kind == "repo"
                and any(
                    repo_identity_key(ref) == repo
                    for ref in (source.name, source.location)
                    if ref
                )
                for source in sources
            ):
                matches.append(pot)

        if len(matches) == 1:
            return Success(ContextIdentity(matches[0].pot_id))
        if len(matches) > 1:
            if active is not None and any(
                active.pot_id == pot.pot_id for pot in matches
            ):
                return Success(ContextIdentity(active.pot_id))
            names = ", ".join(f"{pot.name} ({pot.pot_id})" for pot in matches)
            return Failure(
                SelectionError(
                    code="ambiguous_pot",
                    message=f"Current repo is registered in multiple pots: {names}.",
                    recommended_next_action=(
                        "pick one with '--pot <id-or-name>' or set it active with "
                        "'potpie pot use <id-or-name>'"
                    ),
                )
            )
        if active is not None:
            return Success(ContextIdentity(active.pot_id))
        return Failure(_no_active_pot_error())


class LocalCliAuthenticator:
    async def authenticate(self, authentication: object):
        del authentication
        return Success(AuthenticatedActor(actor_id="local-cli"))


class LocalCliAuthorizer:
    async def authorize(
        self,
        actor: AuthenticatedActor,
        operation: str,
        context: ContextIdentity,
    ):
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
                attributes={"trust_boundary": "local_user"},
            )
        )


class HostShellEngineOperations:
    """Finite engine-owned operations backed by the current HostShell services."""

    def __init__(self, host: Any) -> None:
        self._host = host

    async def resolve(
        self, context: ContextIdentity, request: ResolveRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.agent_context.resolve(
                HostResolveRequest(
                    pot_id=context.value,
                    task=_optional_str(payload, "task"),
                    intent=_optional_str(payload, "intent"),
                    include=_tuple_of_str(payload, "include"),
                    exclude=_tuple_of_str(payload, "exclude"),
                    scope=_mapping(payload, "scope"),
                    mode=str(payload.get("mode") or "fast"),
                    source_policy=str(
                        payload.get("source_policy") or "references_only"
                    ),
                    max_items=int(payload.get("max_items") or 12),
                    as_of=payload.get("as_of"),  # type: ignore[arg-type]
                    since=payload.get("since"),  # type: ignore[arg-type]
                    until=payload.get("until"),  # type: ignore[arg-type]
                    include_invalidated=bool(payload.get("include_invalidated", False)),
                    freshness_preference=str(
                        payload.get("freshness_preference") or "balanced"
                    ),
                    metadata=_mapping(payload, "metadata"),
                )
            )
        )

    async def search(
        self, context: ContextIdentity, request: SearchRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.agent_context.search(
                HostSearchRequest(
                    pot_id=context.value,
                    query=_required_str(payload, "query"),
                    include=_tuple_of_str(payload, "include"),
                    scope=_mapping(payload, "scope"),
                    mode=str(payload.get("mode") or "fast"),
                    source_policy=str(
                        payload.get("source_policy") or "references_only"
                    ),
                    max_items=int(payload.get("max_items") or 12),
                    metadata=_mapping(payload, "metadata"),
                )
            )
        )

    async def record(
        self, context: ContextIdentity, request: RecordRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.agent_context.record(
                HostRecordRequest(
                    pot_id=context.value,
                    record_type=_required_str(payload, "record_type"),
                    summary=_required_str(payload, "summary"),
                    details=_mapping(payload, "details"),
                    scope=_mapping(payload, "scope"),
                    source_refs=_tuple_of_str(payload, "source_refs"),
                    idempotency_key=_optional_str(payload, "idempotency_key"),
                    metadata=_mapping(payload, "metadata"),
                )
            )
        )

    async def data_plane_status(
        self, context: ContextIdentity, request: DataPlaneStatusRequest
    ) -> Outcome[object]:
        del request
        return await self._call(
            lambda: self._host.graph.data_plane_status(context.value)
        )

    async def catalog(
        self, context: ContextIdentity, request: CatalogRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph.catalog(
                GraphCatalogRequest(
                    pot_id=context.value,
                    task=_optional_str(payload, "task"),
                    subgraph=_optional_str(payload, "subgraph"),
                )
            )
        )

    async def describe(
        self, context: ContextIdentity, request: DescribeRequest
    ) -> Outcome[object]:
        del context
        payload = request.payload
        return await self._call(
            lambda: self._host.graph.describe(
                GraphDescribeRequest(
                    subgraph=_required_str(payload, "subgraph"),
                    view=_optional_str(payload, "view"),
                    include_examples=bool(payload.get("include_examples", False)),
                )
            )
        )

    async def read(
        self, context: ContextIdentity, request: ReadRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph.read(
                GraphReadRequest(
                    pot_id=context.value,
                    subgraph=_required_str(payload, "subgraph"),
                    view=_required_str(payload, "view"),
                    query=_optional_str(payload, "query"),
                    scope=_mapping(payload, "scope"),
                    limit=int(payload.get("limit") or 12),
                    as_of=payload.get("as_of"),  # type: ignore[arg-type]
                    since=payload.get("since"),  # type: ignore[arg-type]
                    until=payload.get("until"),  # type: ignore[arg-type]
                    include_invalidated=bool(payload.get("include_invalidated", False)),
                    freshness_preference=str(
                        payload.get("freshness_preference") or "balanced"
                    ),
                    depth=payload.get("depth"),  # type: ignore[arg-type]
                    direction=_optional_str(payload, "direction"),
                    environment=_optional_str(payload, "environment"),
                    source_refs=_tuple_of_str(payload, "source_refs"),
                    detail=str(payload.get("detail") or "compact"),
                    relations=str(payload.get("relations") or "summary"),
                    query_threshold=float(payload.get("query_threshold") or 0.70),
                )
            )
        )

    async def search_entities(
        self, context: ContextIdentity, request: SearchEntitiesRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph.search_entities(
                GraphEntitySearchRequest(
                    pot_id=context.value,
                    query=_required_str(payload, "query"),
                    type=_optional_str(payload, "type"),
                    predicate=_optional_str(payload, "predicate"),
                    subgraph=_optional_str(payload, "subgraph"),
                    scope=_mapping(payload, "scope"),
                    truth=_optional_str(payload, "truth"),
                    source_system=_optional_str(payload, "source_system"),
                    source_family=_optional_str(payload, "source_family"),
                    since=payload.get("since"),  # type: ignore[arg-type]
                    until=payload.get("until"),  # type: ignore[arg-type]
                    environment=_optional_str(payload, "environment"),
                    external_id=_optional_str(payload, "external_id"),
                    source_refs=_tuple_of_str(payload, "source_refs"),
                    limit=int(payload.get("limit") or 10),
                    supporting_claims=int(payload.get("supporting_claims") or 0),
                )
            )
        )

    async def mutate(
        self, context: ContextIdentity, request: MutateRequest
    ) -> Outcome[object]:
        mutation = _mapping(request.payload, "mutation")
        return await self._call(
            lambda: self._host.graph.mutate(
                SemanticMutationRequest.parse(mutation, pot_id=context.value)
            )
        )

    async def neighborhood(
        self, context: ContextIdentity, request: NeighborhoodRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._inspection_neighborhood(context=context, payload=payload)
        )

    async def inspect(
        self, context: ContextIdentity, request: InspectRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._inspection_neighborhood(context=context, payload=payload)
        )

    async def export_snapshot(
        self, context: ContextIdentity, request: ExportSnapshotRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._host.backend.snapshot.export(
                pot_id=context.value,
                destination=_required_str(request.payload, "destination"),
            )
        )

    async def import_snapshot(
        self, context: ContextIdentity, request: ImportSnapshotRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._host.backend.snapshot.import_(
                pot_id=context.value,
                source=_required_str(request.payload, "source"),
            )
        )

    async def repair(
        self, context: ContextIdentity, request: RepairRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._host.backend.analytics.repair(
                context.value,
                targets=_tuple_of_str(request.payload, "targets"),
            )
        )

    async def propose(
        self, context: ContextIdentity, request: ProposeRequest
    ) -> Outcome[object]:
        mutation = _mapping(request.payload, "mutation")
        ttl_seconds = request.payload.get("ttl_seconds")
        return await self._call(
            lambda: self._host.graph_workbench.propose(
                mutation,
                pot_id=context.value,
                ttl_seconds=int(ttl_seconds) if ttl_seconds is not None else None,
            )
        )

    async def commit(
        self, context: ContextIdentity, request: CommitRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.commit(
                _required_str(payload, "plan_id"),
                pot_id=context.value,
                approved_by=_optional_str(payload, "approved_by"),
                verify=bool(payload.get("verify", False)),
            )
        )

    async def history(
        self, context: ContextIdentity, request: HistoryRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.history(
                pot_id=context.value,
                entity_key=_optional_str(payload, "entity_key"),
                claim_key=_optional_str(payload, "claim_key"),
                subgraph=_optional_str(payload, "subgraph"),
                plan_id=_optional_str(payload, "plan_id"),
                mutation_id=_optional_str(payload, "mutation_id"),
                since=payload.get("since"),  # type: ignore[arg-type]
                until=payload.get("until"),  # type: ignore[arg-type]
                limit=int(payload.get("limit") or 50),
            )
        )

    async def quality(
        self, context: ContextIdentity, request: QualityRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.quality(
                pot_id=context.value,
                report=_required_str(payload, "report"),
                subgraph=_optional_str(payload, "subgraph"),
                limit=int(payload.get("limit") or 50),
                confidence_threshold=float(
                    payload.get("confidence_threshold") or 0.5
                ),
            )
        )

    async def inbox_add(
        self, context: ContextIdentity, request: InboxAddRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_add(
                pot_id=context.value,
                summary=_required_str(payload, "summary"),
                details=_optional_str(payload, "details"),
                evidence=_tuple_of_str(payload, "evidence"),
                source_refs=_tuple_of_str(payload, "source_refs"),
                suspected_subgraphs=_tuple_of_str(payload, "suspected_subgraphs"),
                created_by=_mapping(payload, "created_by"),
            )
        )

    async def inbox_list(
        self, context: ContextIdentity, request: InboxListRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_list(
                pot_id=context.value,
                status=_tuple_of_str(payload, "status"),
                claimed_by=_optional_str(payload, "claimed_by"),
                suspected_subgraph=_optional_str(payload, "suspected_subgraph"),
                source_ref=_optional_str(payload, "source_ref"),
                since=payload.get("since"),  # type: ignore[arg-type]
                until=payload.get("until"),  # type: ignore[arg-type]
                limit=int(payload.get("limit") or 50),
            )
        )

    async def inbox_show(
        self, context: ContextIdentity, request: InboxShowRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._host.graph_workbench.inbox_show(
                pot_id=context.value,
                item_id=_required_str(request.payload, "item_id"),
            )
        )

    async def inbox_claim(
        self, context: ContextIdentity, request: InboxClaimRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_claim(
                pot_id=context.value,
                item_id=_required_str(payload, "item_id"),
                claimed_by=_required_str(payload, "claimed_by"),
            )
        )

    async def inbox_mark_applied(
        self, context: ContextIdentity, request: InboxMarkAppliedRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_mark_applied(
                pot_id=context.value,
                item_id=_required_str(payload, "item_id"),
                closed_by=_required_str(payload, "closed_by"),
                linked_plan_id=_optional_str(payload, "linked_plan_id"),
                linked_mutation_id=_optional_str(payload, "linked_mutation_id"),
            )
        )

    async def inbox_mark_rejected(
        self, context: ContextIdentity, request: InboxMarkRejectedRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_mark_rejected(
                pot_id=context.value,
                item_id=_required_str(payload, "item_id"),
                closed_by=_required_str(payload, "closed_by"),
                rejection_reason=_required_str(payload, "rejection_reason"),
            )
        )

    async def inbox_close(
        self, context: ContextIdentity, request: InboxCloseRequest
    ) -> Outcome[object]:
        payload = request.payload
        return await self._call(
            lambda: self._host.graph_workbench.inbox_close(
                pot_id=context.value,
                item_id=_required_str(payload, "item_id"),
                closed_by=_required_str(payload, "closed_by"),
                linked_plan_id=_optional_str(payload, "linked_plan_id"),
                linked_mutation_id=_optional_str(payload, "linked_mutation_id"),
                rejection_reason=_optional_str(payload, "rejection_reason"),
            )
        )

    async def invoke(
        self,
        operation: EngineOperation,
        context: ContextIdentity,
        request: EngineRequest,
    ) -> ClientOutcome:
        handlers: dict[
            EngineOperation,
            Callable[[ContextIdentity, Any], Awaitable[Outcome[object]]],
        ] = {
            EngineOperation.RESOLVE: self.resolve,
            EngineOperation.SEARCH: self.search,
            EngineOperation.RECORD: self.record,
            EngineOperation.DATA_PLANE_STATUS: self.data_plane_status,
            EngineOperation.CATALOG: self.catalog,
            EngineOperation.DESCRIBE: self.describe,
            EngineOperation.READ: self.read,
            EngineOperation.SEARCH_ENTITIES: self.search_entities,
            EngineOperation.MUTATE: self.mutate,
            EngineOperation.NEIGHBORHOOD: self.neighborhood,
            EngineOperation.INSPECT: self.inspect,
            EngineOperation.EXPORT_SNAPSHOT: self.export_snapshot,
            EngineOperation.IMPORT_SNAPSHOT: self.import_snapshot,
            EngineOperation.REPAIR: self.repair,
            EngineOperation.PROPOSE: self.propose,
            EngineOperation.COMMIT: self.commit,
            EngineOperation.HISTORY: self.history,
            EngineOperation.QUALITY: self.quality,
            EngineOperation.INBOX_ADD: self.inbox_add,
            EngineOperation.INBOX_LIST: self.inbox_list,
            EngineOperation.INBOX_SHOW: self.inbox_show,
            EngineOperation.INBOX_CLAIM: self.inbox_claim,
            EngineOperation.INBOX_MARK_APPLIED: self.inbox_mark_applied,
            EngineOperation.INBOX_MARK_REJECTED: self.inbox_mark_rejected,
            EngineOperation.INBOX_CLOSE: self.inbox_close,
        }
        handler = handlers.get(operation)
        if handler is None:
            return Failure(
                DomainError(
                    code="legacy_operation_not_migrated",
                    message=f"{operation.value} has not reached the typed adapter",
                )
            )
        return await handler(context, request)

    def _inspection_neighborhood(
        self, *, context: ContextIdentity, payload: Mapping[str, object]
    ) -> object:
        capabilities = self._host.backend.capabilities()
        if not bool(getattr(capabilities, "inspection", False)):
            profile = getattr(
                capabilities,
                "profile",
                getattr(self._host.backend, "profile", "unknown"),
            )
            raise CapabilityNotImplemented(
                f"graph.{profile}.inspection.neighborhood",
                detail=(
                    "graph neighborhood is not supported by the active "
                    f"'{profile}' backend"
                ),
                recommended_next_action=(
                    "run 'potpie backend status' to inspect capabilities, or switch "
                    "to a backend that implements inspection"
                ),
            )
        return self._host.backend.inspection.neighborhood(
            pot_id=context.value,
            entity_key=_required_str(payload, "entity_key"),
            depth=int(payload.get("depth") or 2),
            direction=str(payload.get("direction") or "both"),
            predicates=_tuple_of_str(payload, "predicates"),
            limit=int(payload.get("limit") or 50),
        )

    async def _call(self, call: Callable[[], object]) -> Outcome[object]:
        try:
            return Success(await asyncio.to_thread(call))
        except CapabilityNotImplemented as exc:
            return Failure(
                DomainError(
                    code="not_implemented",
                    message=str(exc),
                    details={"detail": exc.detail} if exc.detail else {},
                    recommended_next_action=exc.recommended_next_action,
                )
            )
        except PotNotFound as exc:
            return Failure(DomainError(code="pot_not_found", message=str(exc)))
        except ValueError as exc:
            return Failure(
                DomainError(
                    code="validation_error",
                    message=str(exc),
                    details={
                        "detail": getattr(exc, "detail", None),
                    },
                    recommended_next_action=getattr(
                        exc, "recommended_next_action", None
                    ),
                )
            )
        except ContextEngineDisabled as exc:
            return Failure(
                DependencyError(
                    code="unavailable",
                    message=str(exc),
                    recommended_next_action=(
                        "check backend/daemon readiness with 'potpie doctor'"
                    ),
                    retry_posture="safe",
                )
            )
        except Exception as exc:
            return Failure(
                DependencyError(
                    code="legacy_host_operation_failed",
                    message="the current host failed a typed engine operation",
                    details={"error_type": type(exc).__name__},
                    recommended_next_action="inspect runtime logs",
                )
            )


class HostContextResourceComposer:
    def __init__(self, host: Any) -> None:
        self._host = host

    async def fingerprint(self, context: ContextIdentity):
        del context
        profile = str(getattr(self._host.backend, "profile", "unknown"))
        return Success(CompositionFingerprint(f"host-shell-read-v1:{profile}"))

    async def compose(
        self, context: ContextIdentity, fingerprint: CompositionFingerprint
    ):
        operations = HostShellEngineOperations(self._host)
        return Success(
            ResourceComposition(
                fingerprint=fingerprint,
                config=EngineConfig(values={"backend_profile": fingerprint.value}),
                dependencies=EngineDependencies(
                    context=operations,
                    graph=operations,
                    workbench=operations,
                    ingestion=operations,
                    nudge=operations,
                ),
            )
        )


def build_local_resource_manager(host: Any) -> ContextResourceManager:
    return ContextResourceManager(
        resolver=HostContextSelectorResolver(host),
        authenticator=LocalCliAuthenticator(),
        authorizer=LocalCliAuthorizer(),
        composer=HostContextResourceComposer(host),
    )


def build_legacy_engine_client(
    *, host: Any, selector: ContextSelector
) -> LegacyEngineClientAdapter:
    resolver = HostContextSelectorResolver(host)
    authenticator = LocalCliAuthenticator()
    authorizer = LocalCliAuthorizer()
    operations = HostShellEngineOperations(host)

    async def invoke(request: EngineOperationRequest) -> ClientOutcome:
        selection = await resolver.resolve(request.selector)
        if isinstance(selection, Failure):
            return selection
        authenticated = await authenticator.authenticate(authentication=None)
        if isinstance(authenticated, Failure):
            return authenticated
        authorized = await authorizer.authorize(
            authenticated.value,
            request.operation.value,
            selection.value,
        )
        if isinstance(authorized, Failure):
            return authorized
        destructive_failure = _validate_destructive_intent(request)
        if destructive_failure is not None:
            return Failure(destructive_failure)
        return await operations.invoke(
            request.operation,
            selection.value,
            request.payload,
        )

    return LegacyEngineClientAdapter(selector=selector, invoker=invoke)


def _validate_destructive_intent(
    request: EngineOperationRequest,
) -> AuthorizationError | None:
    if not ENGINE_OPERATION_CATALOG[request.operation].destructive:
        return None
    intent = request.destructive_intent
    if (
        intent is None
        or not intent.confirmed
        or intent.operation != request.operation.value
        or intent.selector != request.selector
        or intent.request_id != request.request_id
    ):
        return AuthorizationError(
            code="destructive_intent_invalid",
            message="destructive operation confirmation does not match the request",
        )
    return None


def _required_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value


def _optional_str(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    return str(value) if value is not None else None


def _tuple_of_str(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)  # type: ignore[union-attr]


def _mapping(payload: Mapping[str, object], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    return value if isinstance(value, Mapping) else {}


def _no_active_pot_error() -> SelectionError:
    return SelectionError(
        code="no_active_pot",
        message=(
            "No active pot, and the current repo is not registered as a source "
            "in any pot."
        ),
        recommended_next_action=(
            "run 'potpie setup', or create a pot with 'potpie pot create <name> "
            "--use' and register this repo with 'potpie source add repo .'"
        ),
    )


__all__ = [
    "HostContextResourceComposer",
    "HostContextSelectorResolver",
    "HostShellEngineOperations",
    "LocalCliAuthenticator",
    "LocalCliAuthorizer",
    "build_legacy_engine_client",
    "build_local_resource_manager",
]
