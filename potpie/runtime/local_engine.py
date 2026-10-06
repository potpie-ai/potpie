"""Local typed-engine composition over explicit Potpie runtime services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from potpie.cli.repo_location import repo_identity_key
from potpie.pots.resolution import (
    ARCHIVED_POT_NEXT_ACTION,
    archived_pot_message,
    is_archived,
    match_pot_ref,
    repo_source_index,
)
from potpie.runtime.clients import ClientOutcome
from potpie.runtime.commit_access import commit_grant
from potpie.runtime.operations import EngineOperation
from potpie.runtime.resource_manager import (
    AuthenticatedActor,
    AuthorizationError,
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
from potpie_context_engine.application.services.resource_journal import (
    JOURNAL_CAPTURE_ACTIVE,
    refuse_while_journaling,
)
from potpie_context_engine.core.commit_service import CommitAccessDenied
from potpie_context_engine.core.graph_journal import JournalError
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
from potpie_context_engine.core.ports.resource_index import ResourceIndexError
from potpie_context_engine.core.ports.resource_store import (
    ResourceBatchResult,
    ResourceStoreError,
)
from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.domain.nudge import GraphNudgeRequest
from potpie_context_engine.domain.ingestion_event_models import (
    IngestionSubmissionRequest,
)
from potpie_context_engine.requests import (
    ApplyPreviewRequest,
    CatalogRequest,
    CommitRequest,
    CommitShowRequest,
    CommitStatusRequest,
    CommitsRequest,
    DataPlaneStatusRequest,
    DescribeRequest,
    DisableRollbackRequest,
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
    JournalStatusRequest,
    MutateRequest,
    NeighborhoodRequest,
    NudgeRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RebuildCommitsRequest,
    RecordRequest,
    RepairRequest,
    ResetContextRequest,
    ResolveRequest,
    ResourceGetRequest,
    ResourceImportRequest,
    ResourceIndexBuildRequest,
    ResourceIndexRebuildRequest,
    ResourceIndexStatusRequest,
    ResourceListRequest,
    ResourceRmRequest,
    ResourceStatusRequest,
    RevertPreviewRequest,
    RollbackPreviewRequest,
    SearchEntitiesRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
    ProcessingStatusRequest,
    VerifyCommitRequest,
)
from potpie_context_engine.results import (
    DescribeResult,
    GraphJournalResult,
    ResetContextResult,
    ResourceIndexRebuildResult,
    ResourceListResult,
)


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
    """The ``ResourceFacade`` (document store + index), or ``None`` if absent."""


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
            live, archived = match_pot_ref(pots, selector.value or "")
            if live is not None:
                return Success(ContextIdentity(live.pot_id))
            # An exact id always names its pot, archived or not: ids are never
            # reused, and clearing a retired pot's graph state needs to reach it.
            # What may run against it is decided by ``LocalCliAuthorizer``. A
            # name never resolves to an archived pot, because a live pot may
            # reuse that name.
            if archived is not None and archived.pot_id == selector.value:
                return Success(ContextIdentity(archived.pot_id))
            if archived is not None:
                return Failure(
                    SelectionError(
                        code="pot_archived",
                        message=archived_pot_message(archived),
                        recommended_next_action=ARCHIVED_POT_NEXT_ACTION,
                    )
                )
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
        live_ids = {pot.pot_id for pot in pots if not is_archived(pot)}
        if default_pot_id and default_pot_id in live_ids:
            return Success(ContextIdentity(str(default_pot_id)))

        # One repo→pot index read, not one ``list_sources`` call per pot.
        # Matching stays here; one entry per pot, in pot order.
        try:
            index = repo_source_index(self._services.pots)
        except Exception:
            index = []
        matches: dict[str, str] = {}
        for row in index:
            if row.pot_id in matches:
                continue
            if any(
                repo_identity_key(ref) == repo
                for ref in (row.name, row.location)
                if ref
            ):
                matches[row.pot_id] = row.pot_name

        if len(matches) == 1:
            return Success(ContextIdentity(next(iter(matches))))
        if len(matches) > 1:
            if active is not None and active.pot_id in matches:
                return Success(ContextIdentity(active.pot_id))
            names = ", ".join(f"{name} ({pot_id})" for pot_id, name in matches.items())
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
    """Authorize local operations; an archived pot admits only ``reset_context``.

    Selection resolves an archived pot by its exact id so its graph state can
    still be cleared (``pot archive`` on an already-archived pot). Every other
    operation against it is refused here, so nothing reads from or writes into
    a retired pot through the typed boundary.
    """

    def __init__(self, services: Any | None = None) -> None:
        self._services = services

    async def authorize(
        self,
        actor: AuthenticatedActor,
        operation: str,
        context: ContextIdentity,
    ):
        if operation != EngineOperation.RESET_CONTEXT:
            archived = await asyncio.to_thread(self._archived_pot, context.value)
            if archived is not None:
                return Failure(
                    AuthorizationError(
                        code="pot_archived",
                        message=archived_pot_message(archived),
                        recommended_next_action=ARCHIVED_POT_NEXT_ACTION,
                    )
                )
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
                attributes={"trust_boundary": "local_user"},
            )
        )

    def _archived_pot(self, pot_id: str) -> Any | None:
        pots = getattr(self._services, "pots", None)
        if pots is None:
            return None
        try:
            rows = pots.list_pots()
        except Exception:  # noqa: BLE001 - selection just read the same catalog.
            return None
        return next(
            (pot for pot in rows if pot.pot_id == pot_id and is_archived(pot)),
            None,
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
                    intent=request.intent,
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
            # A pot whose graph journal is capturing is refused before anything
            # is touched: the graph reset and the document purge each refuse on
            # their own, so checking once here keeps reset and archive from
            # stopping half-way (graph cleared, documents kept).
            refuse_while_journaling(
                getattr(self._services.backend, "journal", None),
                context.value,
                "pot reset",
            )
            result = self._services.backend.mutation.reset_pot(context.value)
            reset_ok = bool(result.get("ok", True))
            # Documents go only after the graph reset succeeded: a failed reset
            # must not leave live claims citing chunk ids that no longer exist.
            resources = self._resources_or_none()
            purged = (
                bool(resources.purge_pot(context.value))
                if reset_ok and resources is not None
                else None
            )
            return ResetContextResult(
                context_id=context.value,
                reset=reset_ok,
                resources_purged=purged,
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
                approved_by=request.approved_by,
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
                defer_verification=request.defer_verification,
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
                include_claims=request.include_claims,
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

    # --- document resources -------------------------------------------------

    async def resource_import(
        self, context: ContextIdentity, request: ResourceImportRequest
    ) -> Outcome[object]:
        # Contents only, never a path: see ``ResourceImportRequest``.
        return await self._call(
            lambda: self._resources().import_dir(
                pot_id=context.value,
                slug=_required_value(request.doc, "doc"),
                files=dict(request.files),
                source_ref=request.source_ref,
                source_kind=request.source_kind,
            )
        )

    async def resource_get(
        self, context: ContextIdentity, request: ResourceGetRequest
    ) -> Outcome[object]:
        def get() -> ResourceBatchResult:
            if not request.resource_ids:
                raise ValueError("resource_ids is required")
            result = self._resources().get(
                pot_id=context.value,
                resource_ids=tuple(request.resource_ids),
                with_neighbors=request.with_neighbors,
            )
            if isinstance(result, ResourceBatchResult):
                return result
            # Every id resolved: the facade keeps its historical tuple shape,
            # the typed boundary always answers one batch receipt.
            return ResourceBatchResult(
                chunks=tuple(result), outcomes=(), status="success"
            )

        return await self._call(get)

    async def resource_list(
        self, context: ContextIdentity, request: ResourceListRequest
    ) -> Outcome[object]:
        def list_sections() -> ResourceListResult:
            doc = _required_value(request.doc, "doc")
            return ResourceListResult(
                doc=doc,
                sections=tuple(
                    self._resources().list(
                        pot_id=context.value, slug=doc, section=request.section
                    )
                ),
            )

        return await self._call(list_sections)

    async def resource_rm(
        self, context: ContextIdentity, request: ResourceRmRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._resources().delete(
                pot_id=context.value, slug=_required_value(request.doc, "doc")
            )
        )

    async def resource_status(
        self, context: ContextIdentity, request: ResourceStatusRequest
    ) -> Outcome[object]:
        del request
        return await self._call(lambda: self._resources().status(pot_id=context.value))

    async def resource_index_status(
        self, context: ContextIdentity, request: ResourceIndexStatusRequest
    ) -> Outcome[object]:
        del request
        return await self._call(
            lambda: self._resources().index_status(pot_id=context.value)
        )

    async def resource_index_build(
        self, context: ContextIdentity, request: ResourceIndexBuildRequest
    ) -> Outcome[object]:
        def build() -> object:
            resources = self._resources()
            # Pending work is per pot, so ``doc`` narrows by re-deriving that
            # document's rows first; that is what makes it mean something on a
            # document whose index rows are missing entirely.
            if request.doc:
                resources.index_rebuild(pot_id=context.value, doc=request.doc)
            return resources.index_build(pot_id=context.value, wait=request.wait)

        return await self._call(build)

    async def resource_index_rebuild(
        self, context: ContextIdentity, request: ResourceIndexRebuildRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: ResourceIndexRebuildResult(
                reports=tuple(
                    self._resources().index_rebuild(
                        pot_id=context.value, doc=request.doc or None
                    )
                )
            )
        )

    async def commit_status(
        self, context: ContextIdentity, request: CommitStatusRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.commit_status(
                _required_value(request.plan_id, "plan_id"), pot_id=context.value
            )
        )

    async def verify_commit(
        self, context: ContextIdentity, request: VerifyCommitRequest
    ) -> Outcome[object]:
        return await self._call(
            lambda: self._services.graph_workbench.verify_commit(
                _required_value(request.plan_id, "plan_id"), pot_id=context.value
            )
        )

    async def journal_status(
        self, context: ContextIdentity, request: JournalStatusRequest
    ) -> Outcome[object]:
        del request
        return await self._journal(
            "journal_status",
            context,
            lambda commits: commits.journal_status_async(pot_id=context.value),
        )

    async def commits(
        self, context: ContextIdentity, request: CommitsRequest
    ) -> Outcome[object]:
        return await self._journal(
            "commits",
            context,
            lambda commits: commits.commits_async(
                pot_id=context.value,
                cursor=request.cursor,
                limit=request.limit,
                actor=request.actor,
                origin=request.origin,
                logical_key=request.logical_key,
            ),
        )

    async def commit_show(
        self, context: ContextIdentity, request: CommitShowRequest
    ) -> Outcome[object]:
        return await self._journal(
            "commit_show",
            context,
            lambda commits: commits.commit_show_async(
                _required_value(request.commit_id, "commit_id"),
                pot_id=context.value,
                offset=request.offset,
                limit=request.limit,
            ),
        )

    async def revert_preview(
        self, context: ContextIdentity, request: RevertPreviewRequest
    ) -> Outcome[object]:
        return await self._journal(
            "revert_preview",
            context,
            lambda commits: commits.revert_preview_async(
                _required_value(request.commit_id, "commit_id"),
                pot_id=context.value,
                expected_head=_required_value(request.expected_head, "expected_head"),
            ),
        )

    async def rollback_preview(
        self, context: ContextIdentity, request: RollbackPreviewRequest
    ) -> Outcome[object]:
        return await self._journal(
            "rollback_preview",
            context,
            lambda commits: commits.rollback_preview_async(
                _required_value(request.target_commit_id, "target_commit_id"),
                pot_id=context.value,
                expected_head=_required_value(request.expected_head, "expected_head"),
            ),
        )

    async def apply_preview(
        self, context: ContextIdentity, request: ApplyPreviewRequest
    ) -> Outcome[object]:
        return await self._journal(
            "apply_preview",
            context,
            lambda commits: commits.apply_preview_async(
                _required_value(request.preview_id, "preview_id"),
                pot_id=context.value,
            ),
        )

    async def disable_rollback(
        self, context: ContextIdentity, request: DisableRollbackRequest
    ) -> Outcome[object]:
        del request
        return await self._journal(
            "disable_rollback",
            context,
            lambda commits: commits.disable_rollback_async(pot_id=context.value),
        )

    async def rebuild_commits(
        self, context: ContextIdentity, request: RebuildCommitsRequest
    ) -> Outcome[object]:
        del request
        return await self._journal(
            "rebuild_commits",
            context,
            lambda commits: commits.rebuild_commits_async(pot_id=context.value),
        )

    async def _journal(
        self,
        operation: str,
        context: ContextIdentity,
        call: Callable[[Any], Awaitable[Any]],
    ) -> Outcome[object]:
        """Run one commit-service call inside this pot's commit grant.

        Every call that reaches here passed the resource manager, which
        authenticated the caller and authorized this operation for exactly
        this pot, so the grant is scoped to that pot and nothing wider.
        """

        try:
            with commit_grant(context.value):
                value = await call(self._services.graph_workbench)
        except CommitAccessDenied as exc:
            return Failure(
                DomainError(
                    code="commit_access_denied",
                    message=str(exc),
                    details={"operation": operation},
                )
            )
        except ValueError as exc:
            return Failure(
                DomainError(
                    code="validation_error",
                    message=str(exc),
                    details={"operation": operation},
                )
            )
        except Exception as exc:
            return await self._dependency_failure(operation, exc)
        return Success(GraphJournalResult.from_value(value))

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
            EngineOperation.RESOURCE_IMPORT: self.resource_import,
            EngineOperation.RESOURCE_GET: self.resource_get,
            EngineOperation.RESOURCE_LIST: self.resource_list,
            EngineOperation.RESOURCE_RM: self.resource_rm,
            EngineOperation.RESOURCE_STATUS: self.resource_status,
            EngineOperation.RESOURCE_INDEX_STATUS: self.resource_index_status,
            EngineOperation.RESOURCE_INDEX_BUILD: self.resource_index_build,
            EngineOperation.RESOURCE_INDEX_REBUILD: self.resource_index_rebuild,
            EngineOperation.COMMIT_STATUS: self.commit_status,
            EngineOperation.VERIFY_COMMIT: self.verify_commit,
            EngineOperation.JOURNAL_STATUS: self.journal_status,
            EngineOperation.COMMITS: self.commits,
            EngineOperation.COMMIT_SHOW: self.commit_show,
            EngineOperation.REVERT_PREVIEW: self.revert_preview,
            EngineOperation.ROLLBACK_PREVIEW: self.rollback_preview,
            EngineOperation.APPLY_PREVIEW: self.apply_preview,
            EngineOperation.DISABLE_ROLLBACK: self.disable_rollback,
            EngineOperation.REBUILD_COMMITS: self.rebuild_commits,
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

    def _resources_or_none(self) -> Any | None:
        return getattr(self._services, "resources", None)

    def _resources(self) -> Any:
        resources = self._resources_or_none()
        if resources is None:
            raise CapabilityNotImplemented(
                "resources",
                detail="this runtime does not compose a document resource store",
                recommended_next_action=(
                    "run against the local Potpie runtime, which composes one"
                ),
            )
        return resources

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
        except (ResourceStoreError, ResourceIndexError) as exc:
            # The store's own stable code, not ``validation_error``: an agent
            # retries a bad slug and an oversized chunk differently.
            return Failure(
                DomainError(
                    code=exc.code,
                    message=str(exc),
                    details={"detail": exc.detail} if exc.detail is not None else {},
                    recommended_next_action=exc.recommended_next_action,
                )
            )
        except JournalError as exc:
            if exc.code == JOURNAL_CAPTURE_ACTIVE:
                return Failure(
                    DomainError(
                        code=JOURNAL_CAPTURE_ACTIVE,
                        message=str(exc),
                        recommended_next_action=(
                            "inspect the pot's journal with 'potpie graph "
                            "journal-status'; no command retires journal "
                            "capture yet"
                        ),
                    )
                )
            return Failure(
                DomainError(
                    code="validation_error",
                    message=str(exc),
                    details={"detail": getattr(exc, "detail", None)},
                )
            )
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
                    # A graph store that refuses to serve possibly stale data
                    # names its own repair; keep it instead of the generic one.
                    recommended_next_action=(
                        getattr(exc, "recommended_next_action", None)
                        or "check backend/daemon readiness with 'potpie doctor'"
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
                    documents=operations,
                ),
            )
        )


def build_local_resource_manager(services: Any) -> ContextResourceManager:
    return ContextResourceManager(
        resolver=LocalContextSelectorResolver(services),
        authenticator=LocalCliAuthenticator(),
        authorizer=LocalCliAuthorizer(services),
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
