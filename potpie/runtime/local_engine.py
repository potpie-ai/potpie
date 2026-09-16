"""Local typed-engine composition over explicit Potpie runtime services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from potpie.cli.repo_location import repo_identity_key
from potpie.runtime.clients import ClientOutcome
from potpie.runtime.operations import EngineOperation
from potpie.runtime.resource_manager import (
    AuthenticatedActor,
    AuthorizationScope,
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
    RecordRequest as AgentRecordRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    ResolveRequest as AgentResolveRequest,
)
from potpie_context_engine.core.ports.agent_context import (
    SearchRequest as AgentSearchRequest,
)
from potpie_context_engine.core.ports.graph_service import (
    GraphCatalogRequest,
    GraphDescribeRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.domain.nudge import GraphNudgeRequest
from potpie_context_engine.domain.ingestion_event_models import (
    IngestionSubmissionRequest,
)
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
    NudgeRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RecordRequest,
    RepairRequest,
    ResetContextRequest,
    ResolveRequest,
    SearchEntitiesRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
    ProcessingStatusRequest,
)
from potpie_context_engine.results import DescribeResult, ResetContextResult


@dataclass(frozen=True, slots=True)
class LocalEngineServices:
    """Concrete engine-owned services used by the local typed boundary."""

    pots: Any
    agent_context: Any
    graph: Any
    graph_workbench: Any
    backend: Any
    nudge: Any
    ingestion: Any | None = None
    ingestion_events: Any | None = None
    resources: Any | None = None


class LocalContextSelectorResolver:
    """Resolve exact, active, and repository selectors through Potpie services."""

    def __init__(self, services: Any) -> None:
        self._services = services

    async def resolve(
        self, selector: ContextSelector
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        return await asyncio.to_thread(self._resolve, selector)

    def _resolve(
        self, selector: ContextSelector
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        try:
            pots = tuple(self._services.pots.list_pots())
            active = self._services.pots.active_pot()
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
            default_pot_id = self._services.pots.repo_default(repo=repo)
        except Exception:
            default_pot_id = None
        known_ids = {pot.pot_id for pot in pots}
        if default_pot_id and default_pot_id in known_ids:
            return Success(ContextIdentity(str(default_pot_id)))

        matches = []
        for pot in pots:
            try:
                sources = self._services.pots.list_sources(pot_id=pot.pot_id)
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


class LocalEngineOperations:
    """Finite engine-owned operations backed by explicit local services."""

    def __init__(self, services: Any) -> None:
        self._services = services

    async def resolve(
        self, context: ContextIdentity, request: ResolveRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.agent_context.resolve(
                AgentResolveRequest(
                    pot_id=context.value,
                    task=request.task,
                    intent=request.intent,
                    include=request.include,
                    exclude=request.exclude,
                    scope=request.scope,
                    mode=request.mode,
                    source_policy=request.source_policy,
                    max_items=request.max_items,
                    as_of=request.as_of,
                    since=request.since,
                    until=request.until,
                    include_invalidated=request.include_invalidated,
                    freshness_preference=request.freshness_preference,
                    metadata=request.metadata,
                )
            )
        )

    async def search(
        self, context: ContextIdentity, request: SearchRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.agent_context.search(
                AgentSearchRequest(
                    pot_id=context.value,
                    query=_required_value(request.query, "query"),
                    include=request.include,
                    scope=request.scope,
                    mode=request.mode,
                    source_policy=request.source_policy,
                    max_items=request.max_items,
                    metadata=request.metadata,
                )
            )
        )

    async def record(
        self, context: ContextIdentity, request: RecordRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.agent_context.record(
                AgentRecordRequest(
                    pot_id=context.value,
                    record_type=_required_value(request.record_type, "record_type"),
                    summary=_required_value(request.summary, "summary"),
                    details=request.details,
                    scope=request.scope,
                    source_refs=request.source_refs,
                    idempotency_key=request.idempotency_key,
                    metadata=request.metadata,
                )
            )
        )

    async def data_plane_status(
        self, context: ContextIdentity, request: DataPlaneStatusRequest
    ) -> Outcome[object]:
        del request
        return await self._call(
            lambda: self._services.graph.data_plane_status(context.value)
        )

    async def catalog(
        self, context: ContextIdentity, request: CatalogRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph.catalog(
                GraphCatalogRequest(
                    pot_id=context.value,
                    task=request.task,
                    subgraph=request.subgraph,
                )
            )
        )

    async def describe(
        self, context: ContextIdentity, request: DescribeRequest
    ) -> Outcome[object]:
        del context
        return await self._call(
            lambda: DescribeResult(
                self._services.graph.describe(
                    GraphDescribeRequest(
                        subgraph=_required_value(request.subgraph, "subgraph"),
                        view=request.view,
                        include_examples=request.include_examples,
                    )
                )
            )
        )

    async def read(
        self, context: ContextIdentity, request: ReadRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph.read(
                GraphReadRequest(
                    pot_id=context.value,
                    subgraph=_required_value(request.subgraph, "subgraph"),
                    view=_required_value(request.view, "view"),
                    query=request.query,
                    scope=request.scope,
                    limit=request.limit,
                    as_of=request.as_of,
                    since=request.since,
                    until=request.until,
                    include_invalidated=request.include_invalidated,
                    freshness_preference=request.freshness_preference,
                    depth=request.depth,
                    direction=request.direction,
                    environment=request.environment,
                    source_refs=request.source_refs,
                    detail=request.detail,
                    relations=request.relations,
                    query_threshold=request.query_threshold,
                )
            )
        )

    async def search_entities(
        self, context: ContextIdentity, request: SearchEntitiesRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph.search_entities(
                GraphEntitySearchRequest(
                    pot_id=context.value,
                    query=_required_value(request.query, "query"),
                    type=request.type,
                    predicate=request.predicate,
                    subgraph=request.subgraph,
                    scope=request.scope,
                    truth=request.truth,
                    source_system=request.source_system,
                    source_family=request.source_family,
                    since=request.since,
                    until=request.until,
                    environment=request.environment,
                    external_id=request.external_id,
                    source_refs=request.source_refs,
                    limit=request.limit,
                    supporting_claims=request.supporting_claims,
                )
            )
        )

    async def mutate(
        self, context: ContextIdentity, request: MutateRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph.mutate(
                SemanticMutationRequest.parse(request.mutation, pot_id=context.value)
            )
        )

    async def neighborhood(
        self, context: ContextIdentity, request: NeighborhoodRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._inspection_neighborhood(context=context, request=request)
        )

    async def inspect(
        self, context: ContextIdentity, request: InspectRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._inspection_neighborhood(context=context, request=request)
        )

    async def export_snapshot(
        self, context: ContextIdentity, request: ExportSnapshotRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._snapshot_port("export").export(
                pot_id=context.value,
                destination=_required_value(request.destination, "destination"),
            )
        )

    async def import_snapshot(
        self, context: ContextIdentity, request: ImportSnapshotRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._snapshot_port("import_").import_(
                pot_id=context.value,
                source=_required_value(request.source, "source"),
            )
        )

    async def repair(
        self, context: ContextIdentity, request: RepairRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.backend.analytics.repair(
                context.value,
                targets=request.targets,
            )
        )

    async def reset_context(
        self, context: ContextIdentity, request: ResetContextRequest
    ) -> Outcome[object]:
        del request

        def reset() -> ResetContextResult:
            result = self._services.backend.mutation.reset_pot(context.value)
            return ResetContextResult(
                context_id=context.value,
                reset=bool(result.get("ok", True)),
            )

        return await self._call(reset)

    async def propose(
        self, context: ContextIdentity, request: ProposeRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.propose(
                request.mutation,
                pot_id=context.value,
                ttl_seconds=request.ttl_seconds,
            )
        )

    async def commit(
        self, context: ContextIdentity, request: CommitRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.commit(
                _required_value(request.plan_id, "plan_id"),
                pot_id=context.value,
                approved_by=request.approved_by,
                verify=request.verify,
            )
        )

    async def history(
        self, context: ContextIdentity, request: HistoryRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.history(
                pot_id=context.value,
                entity_key=request.entity_key,
                claim_key=request.claim_key,
                subgraph=request.subgraph,
                plan_id=request.plan_id,
                mutation_id=request.mutation_id,
                since=request.since,
                until=request.until,
                limit=request.limit,
            )
        )

    async def quality(
        self, context: ContextIdentity, request: QualityRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.quality(
                pot_id=context.value,
                report=_required_value(request.report, "report"),
                subgraph=request.subgraph,
                limit=request.limit,
                confidence_threshold=request.confidence_threshold,
            )
        )

    async def inbox_add(
        self, context: ContextIdentity, request: InboxAddRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_add(
                pot_id=context.value,
                summary=_required_value(request.summary, "summary"),
                details=request.details,
                evidence=request.evidence,
                source_refs=request.source_refs,
                suspected_subgraphs=request.suspected_subgraphs,
                created_by=request.created_by,
            )
        )

    async def inbox_list(
        self, context: ContextIdentity, request: InboxListRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_list(
                pot_id=context.value,
                status=request.status,
                claimed_by=request.claimed_by,
                suspected_subgraph=request.suspected_subgraph,
                source_ref=request.source_ref,
                since=request.since,
                until=request.until,
                limit=request.limit,
            )
        )

    async def inbox_show(
        self, context: ContextIdentity, request: InboxShowRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_show(
                pot_id=context.value,
                item_id=_required_value(request.item_id, "item_id"),
            )
        )

    async def inbox_claim(
        self, context: ContextIdentity, request: InboxClaimRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_claim(
                pot_id=context.value,
                item_id=_required_value(request.item_id, "item_id"),
                claimed_by=_required_value(request.claimed_by, "claimed_by"),
            )
        )

    async def inbox_mark_applied(
        self, context: ContextIdentity, request: InboxMarkAppliedRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_mark_applied(
                pot_id=context.value,
                item_id=_required_value(request.item_id, "item_id"),
                closed_by=_required_value(request.closed_by, "closed_by"),
                linked_plan_id=request.linked_plan_id,
                linked_mutation_id=request.linked_mutation_id,
            )
        )

    async def inbox_mark_rejected(
        self, context: ContextIdentity, request: InboxMarkRejectedRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_mark_rejected(
                pot_id=context.value,
                item_id=_required_value(request.item_id, "item_id"),
                closed_by=_required_value(request.closed_by, "closed_by"),
                rejection_reason=_required_value(
                    request.rejection_reason, "rejection_reason"
                ),
            )
        )

    async def inbox_close(
        self, context: ContextIdentity, request: InboxCloseRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.inbox_close(
                pot_id=context.value,
                item_id=_required_value(request.item_id, "item_id"),
                closed_by=_required_value(request.closed_by, "closed_by"),
                linked_plan_id=request.linked_plan_id,
                linked_mutation_id=request.linked_mutation_id,
                rejection_reason=request.rejection_reason,
            )
        )

    async def submit_event(
        self, context: ContextIdentity, request: SubmitEventRequest
    ) -> Outcome[object]:
        def submit() -> object:
            submission = IngestionSubmissionRequest(
                pot_id=context.value,
                ingestion_kind=request.ingestion_kind,
                source_channel=request.source_channel,
                source_system=_required_value(request.source_system, "source_system"),
                event_type=_required_value(request.event_type, "event_type"),
                action=_required_value(request.action, "action"),
                source_id=_required_value(request.source_id, "source_id"),
                payload=dict(request.payload),
                metadata=dict(request.metadata),
                idempotency_key=request.idempotency_key,
                dedup_key=request.dedup_key,
                event_id=request.event_id,
                provider=request.provider,
                provider_host=request.provider_host,
                repo_name=request.repo_name,
                source_event_id=request.source_event_id,
                artifact_refs=request.artifact_refs,
                occurred_at=request.occurred_at,
                actor=request.actor,
            )
            return self._ingestion_submission().submit(
                submission,
                wait=request.wait,
                timeout_seconds=request.timeout_seconds,
            )

        return await self._call(submit)

    async def submit_artifact(
        self, context: ContextIdentity, request: SubmitArtifactRequest
    ) -> Outcome[object]:
        def submit() -> object:
            source_system = _required_value(request.source_system, "source_system")
            artifact_type = _required_value(request.artifact_type, "artifact_type")
            artifact_id = _required_value(request.artifact_id, "artifact_id")
            source_ref = request.source_ref or (
                f"{source_system}:{artifact_type}:{artifact_id}"
            )
            submission = IngestionSubmissionRequest(
                pot_id=context.value,
                ingestion_kind="artifact_evidence",
                source_channel=request.source_channel,
                source_system=source_system,
                event_type="artifact",
                action=artifact_type,
                source_id=artifact_id,
                payload={"artifact": dict(request.artifact), "source_ref": source_ref},
                metadata=dict(request.metadata),
                idempotency_key=request.idempotency_key,
                provider=request.provider,
                provider_host=request.provider_host,
                repo_name=request.repo_name,
                artifact_refs=(source_ref,),
                occurred_at=request.occurred_at,
                actor=request.actor,
            )
            return self._ingestion_submission().submit(
                submission,
                wait=request.wait,
                timeout_seconds=request.timeout_seconds,
            )

        return await self._call(submit)

    async def processing_status(
        self, context: ContextIdentity, request: ProcessingStatusRequest
    ) -> Outcome[object]:
        event_id = _required_value(request.event_id, "event_id")
        try:
            event = await asyncio.to_thread(
                self._ingestion_event_store().get_event, event_id
            )
        except Exception as exc:
            return await self._dependency_failure("processing_status", exc)
        if event is None or event.pot_id != context.value:
            return Failure(
                DomainError(
                    code="processing_status_not_found",
                    message="the requested evidence-processing event was not found",
                    details={"event_id": event_id},
                )
            )
        return Success(event)

    async def nudge(
        self, context: ContextIdentity, request: NudgeRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.nudge.nudge(
                GraphNudgeRequest(
                    pot_id=context.value,
                    event=_required_value(request.event, "event"),
                    session_id=_required_value(request.session_id, "session_id"),
                    scope=request.scope,
                    path=request.path,
                    query=request.query,
                    limit=request.limit,
                )
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
            EngineOperation.RESET_CONTEXT: self.reset_context,
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
            EngineOperation.SUBMIT_EVENT: self.submit_event,
            EngineOperation.SUBMIT_ARTIFACT: self.submit_artifact,
            EngineOperation.PROCESSING_STATUS: self.processing_status,
            EngineOperation.NUDGE: self.nudge,
        }
        handler = handlers.get(operation)
        if handler is None:
            return Failure(
                DomainError(
                    code="operation_not_supported",
                    message=f"{operation.value} is not supported by the local engine",
                )
            )
        return await handler(context, request)

    def _inspection_neighborhood(
        self,
        *,
        context: ContextIdentity,
        request: NeighborhoodRequest | InspectRequest,
    ) -> object:
        capabilities = self._services.backend.capabilities()
        if not bool(getattr(capabilities, "inspection", False)):
            profile = getattr(
                capabilities,
                "profile",
                getattr(self._services.backend, "profile", "unknown"),
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
        return self._services.backend.inspection.neighborhood(
            pot_id=context.value,
            entity_key=_required_value(request.entity_key, "entity_key"),
            depth=request.depth,
            direction=request.direction,
            predicates=request.predicates,
            limit=request.limit,
        )

    def _snapshot_port(self, method: str) -> object:
        capabilities = self._services.backend.capabilities()
        if not bool(getattr(capabilities, "snapshot", False)):
            profile = getattr(
                capabilities,
                "profile",
                getattr(self._services.backend, "profile", "unknown"),
            )
            raise CapabilityNotImplemented(
                f"graph.{profile}.snapshot.{method}",
                detail=(
                    "snapshot operations are not supported by the executing "
                    f"'{profile}' backend"
                ),
                recommended_next_action=(
                    "inspect the selected runtime backend or switch to one that "
                    "implements snapshot operations"
                ),
            )
        return self._services.backend.snapshot

    def _ingestion_submission(self) -> Any:
        service = self._services.ingestion
        if service is None:
            raise ContextEngineDisabled("evidence submission is not composed")
        return service

    def _ingestion_event_store(self) -> Any:
        store = self._services.ingestion_events
        if store is None:
            raise ContextEngineDisabled("evidence status storage is not composed")
        return store

    async def _dependency_failure(
        self, operation: str, exc: Exception
    ) -> Failure[DependencyError]:
        return Failure(
            DependencyError(
                code="local_engine_operation_failed",
                message="the local engine operation failed",
                details={"operation": operation, "error_type": type(exc).__name__},
                recommended_next_action="inspect runtime logs",
            )
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
                    code="local_engine_operation_failed",
                    message="the local engine operation failed",
                    details={"error_type": type(exc).__name__},
                    recommended_next_action="inspect runtime logs",
                )
            )


class LocalGraphMetadataOperationHandler:
    """Shared context-free graph metadata execution for every local transport."""

    def __init__(self, services: LocalEngineServices) -> None:
        self._operations = LocalEngineOperations(services)

    async def handle(
        self, operation: EngineOperation, request: EngineRequest
    ) -> ClientOutcome:
        if operation is not EngineOperation.DESCRIBE:
            return Failure(
                DomainError(
                    code="unsupported_context_free_operation",
                    message=f"{operation.value} is not a context-free operation",
                )
            )
        return await self._operations.describe(
            ContextIdentity("context-free"), cast(DescribeRequest, request)
        )


class LocalContextResourceComposer:
    def __init__(self, services: Any) -> None:
        self._services = services

    async def fingerprint(self, context: ContextIdentity):
        del context
        profile = str(getattr(self._services.backend, "profile", "unknown"))
        return Success(CompositionFingerprint(f"local-engine-v1:{profile}"))

    async def compose(
        self, context: ContextIdentity, fingerprint: CompositionFingerprint
    ):
        operations = LocalEngineOperations(self._services)
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


def build_local_resource_manager(services: Any) -> ContextResourceManager:
    return ContextResourceManager(
        resolver=LocalContextSelectorResolver(services),
        authenticator=LocalCliAuthenticator(),
        authorizer=LocalCliAuthorizer(),
        composer=LocalContextResourceComposer(services),
    )


def _required_value(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value


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
    "LocalEngineServices",
    "LocalContextResourceComposer",
    "LocalContextSelectorResolver",
    "LocalEngineOperations",
    "LocalCliAuthenticator",
    "LocalCliAuthorizer",
    "build_local_resource_manager",
]
