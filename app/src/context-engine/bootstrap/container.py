"""Compose adapters and ports (dependency injection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from adapters.outbound.connectors.github import (
    GitHubConnector,
    GitHubReadPort,
    PyGithubSourceControl,
)
from adapters.outbound.connectors.notion import NotionConnector
from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from adapters.outbound.reconciliation.context_graph_tools import (
    ContextGraphReconciliationTools,
)
from adapters.outbound.readers import (
    ChangeHistoryReader,
    DecisionsReader,
    GraphOverviewReader,
    OwnersReader,
    PrDiffReader,
    PrReviewContextReader,
    ProjectGraphReader,
    ReleaseNotesReader,
    SemanticSearchReader,
    TimelineReader,
)
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.intelligence.hybrid_graph import HybridGraphIntelligenceProvider
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from adapters.outbound.policy import DefaultPolicyAdapter
from adapters.outbound.postgres.agent_checkpoint_store import (
    SqlAlchemyAgentCheckpointStore,
)
from adapters.outbound.postgres.agent_execution_log import (
    PostgresAgentExecutionLog,
)
from adapters.outbound.postgres.batch_repository import SqlAlchemyBatchRepository
from adapters.outbound.postgres.ingestion_config import SqlAlchemyIngestionConfig
from adapters.outbound.postgres.delegating_event_query_service import (
    DelegatingEventQueryService,
)
from adapters.outbound.postgres.ingestion_event_store import (
    SqlAlchemyIngestionEventStore,
)
from adapters.outbound.postgres.ledger import SqlAlchemyIngestionLedger
from adapters.outbound.postgres.reconciliation_ledger import (
    SqlAlchemyReconciliationLedger,
)
from adapters.outbound.settings_env import EnvContextEngineSettings
from adapters.outbound.synthesis.null import NullAnswerSynthesizer
from adapters.outbound.synthesis.pydantic_ai_answer import (
    PydanticAIAnswerSynthesizer,
)
from application.services.context_reader_registry import ContextReaderRegistry
from application.services.context_resolution import ContextResolutionService
from application.services.source_connector_registry import SourceConnectorRegistry
from domain.ports.event_query_service import EventQueryService
from domain.ports.event_stream import (
    EventStreamPublisherPort,
    NoOpEventStreamPublisher,
)
from domain.ports.ingestion_config import IngestionConfigPort
from domain.ports.context_graph import ContextGraphPort
from adapters.outbound.graphiti.port import EpisodicGraphPort
from domain.ports.intelligence_provider import IntelligenceProvider
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
    NoOpContextGraphJobQueue,
)
from domain.ports.policy import PolicyPort
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.pot_source_listing import PotSourceListingPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.telemetry import TelemetryPort
from adapters.outbound.neo4j.port import StructuralReadPort
from domain.source_references import SourceReferenceRecord


@dataclass
class ContextEngineContainer:
    """Wired dependencies for use cases.

    Phase 2 collapsed the source-shaped slots (``source_for_repo``,
    ``source_resolver``, ``pot_source_listing``) into a single
    :class:`SourceConnectorRegistry`. Hosts register connectors before
    handing the container off; consumers route every source-typed call
    through the registry.
    """

    settings: ContextEngineSettingsPort
    episodic: EpisodicGraphPort
    structural: StructuralReadPort
    pots: PotResolutionPort
    connectors: SourceConnectorRegistry = field(default_factory=SourceConnectorRegistry)
    readers: ContextReaderRegistry = field(default_factory=ContextReaderRegistry)
    intelligence_provider: IntelligenceProvider | None = None
    resolution_service: ContextResolutionService | None = None
    reconciliation_agent: ReconciliationAgentPort | None = None
    jobs: ContextGraphJobQueuePort | None = None
    context_graph: ContextGraphPort | None = None
    pot_source_listing: PotSourceListingPort | None = None
    """Host-side per-pot source rows (last_sync_at, sync_mode, …). Distinct
    from ``connectors``, which advertises engine-side connector capabilities.
    Both are surfaced by ``context_status``."""
    telemetry: TelemetryPort | None = None
    """Cost + drift sink. ``None`` is treated as ``NoOpTelemetry``."""
    event_stream_publisher: EventStreamPublisherPort = field(
        default_factory=NoOpEventStreamPublisher
    )
    """Live activity / status publisher. NoOp by default; the API process
    swaps in a Redis adapter so the events screen can stream agent activity
    end-to-end."""

    def policy(self) -> PolicyPort:
        """Return the centralized authorization port for this container.

        Built from the container's current view of settings + adapters so
        capability checks (engine enabled, agent wired, episodic up) reflect
        the live configuration. Inbound adapters call this once per request
        and translate ``PolicyDecision.status_code`` to their transport.
        """
        return DefaultPolicyAdapter(
            settings=self.settings,
            pots=self.pots,
            reconciliation_agent_available=self.reconciliation_agent is not None,
            context_graph_available=self.context_graph is not None,
            episodic_available=getattr(self.episodic, "enabled", True),
        )

    def ledger(self, session: Session) -> SqlAlchemyIngestionLedger:
        return SqlAlchemyIngestionLedger(session)

    def reconciliation_ledger(self, session: Session) -> SqlAlchemyReconciliationLedger:
        return SqlAlchemyReconciliationLedger(session)

    def batch_repository(self, session: Session) -> SqlAlchemyBatchRepository:
        return SqlAlchemyBatchRepository(session)

    def agent_checkpoint_store(self, session: Session) -> SqlAlchemyAgentCheckpointStore:
        return SqlAlchemyAgentCheckpointStore(session)

    def agent_execution_log(self, session: Session) -> PostgresAgentExecutionLog:
        """Durable agent execution log: the live token/tool stream *and*
        the crash-resume substrate. Session-scoped like the checkpoint
        store; opens its own short sessions per write internally."""
        return PostgresAgentExecutionLog(session)

    def ingestion_config(self, session: Session) -> IngestionConfigPort:
        return SqlAlchemyIngestionConfig(session)

    def ingestion_event_store(self, session: Session) -> SqlAlchemyIngestionEventStore:
        return SqlAlchemyIngestionEventStore(session)

    def event_query_service(self, session: Session) -> EventQueryService:
        return DelegatingEventQueryService(
            self.ingestion_event_store(session), session=session
        )

    def ingestion_submission(self, session: Session) -> IngestionSubmissionService:
        from application.services.ingestion_submission_service import (
            DefaultIngestionSubmissionService,
        )

        return DefaultIngestionSubmissionService(
            settings=self.settings,
            pots=self.pots,
            reconciliation_agent=self.reconciliation_agent,
            reco_ledger=self.reconciliation_ledger(session),
            events=self.ingestion_event_store(session),
            batches=self.batch_repository(session),
            jobs=self.jobs or NoOpContextGraphJobQueue(),
            ingestion_config=self.ingestion_config(session),
        )


def _attach_reconciliation_context(
    agent: ReconciliationAgentPort | None,
    context_graph: ContextGraphPort | None,
) -> None:
    if agent is None or context_graph is None:
        return
    ctx_setter = getattr(agent, "set_context_tools", None)
    if ctx_setter is not None:
        ctx_setter(ContextGraphReconciliationTools(context_graph))
    graph_setter = getattr(agent, "set_context_graph", None)
    if graph_setter is not None:
        graph_setter(context_graph)


def _build_answer_synthesizer(*, telemetry: TelemetryPort | None = None):
    """Return an LLM synthesizer when CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL is set, else Null."""
    import os

    if os.getenv("CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL"):
        return PydanticAIAnswerSynthesizer(telemetry=telemetry)
    return NullAnswerSynthesizer()


def _default_reader_registry(
    *, episodic: EpisodicGraphPort, structural: StructuralReadPort
) -> ContextReaderRegistry:
    """Register every first-party reader.

    Adding a new evidence family means writing one reader module under
    ``adapters/outbound/readers/`` and adding one ``register()`` call
    here. No edits to ``application/`` or ``domain/`` are required.
    """
    registry = ContextReaderRegistry()
    registry.register(SemanticSearchReader(episodic=episodic, structural=structural))
    registry.register(ChangeHistoryReader(episodic=episodic, structural=structural))
    registry.register(TimelineReader(structural=structural))
    registry.register(OwnersReader(structural=structural))
    registry.register(DecisionsReader(episodic=episodic, structural=structural))
    registry.register(PrReviewContextReader(structural=structural))
    registry.register(PrDiffReader(structural=structural))
    registry.register(ProjectGraphReader(structural=structural))
    registry.register(GraphOverviewReader(episodic=episodic, structural=structural))
    # Phase 3 smoke test — proves the contract by adding a brand-new
    # family in a single file with no application/domain edits.
    registry.register(ReleaseNotesReader(structural=structural))
    return registry


def build_container(
    *,
    settings: ContextEngineSettingsPort | None = None,
    pots: PotResolutionPort,
    connectors: SourceConnectorRegistry | None = None,
    readers: ContextReaderRegistry | None = None,
    reconciliation_agent: ReconciliationAgentPort | None = None,
    jobs: ContextGraphJobQueuePort | None = None,
    telemetry: TelemetryPort | None = None,
    event_stream_publisher: EventStreamPublisherPort | None = None,
) -> ContextEngineContainer:
    s = settings or EnvContextEngineSettings()
    telemetry_sink = telemetry or _default_telemetry()
    stream_publisher = event_stream_publisher or _default_event_stream_publisher()
    episodic = GraphitiEpisodicAdapter(s)
    structural = Neo4jStructuralAdapter(s)
    intelligence_provider = HybridGraphIntelligenceProvider(
        episodic=episodic,
        structural=structural,
    )
    registry = connectors or SourceConnectorRegistry()
    reader_registry = readers or _default_reader_registry(
        episodic=episodic, structural=structural
    )
    # Resolution dispatches through the connector registry — the registry
    # is the post-Phase-2 replacement for ``CompositeSourceResolver``.
    resolution_service = ContextResolutionService(
        intelligence_provider,
        source_resolver=registry,
        telemetry=telemetry_sink,
    )
    context_graph = GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=reader_registry,
        resolution_service=resolution_service,
        answer_synthesizer=_build_answer_synthesizer(telemetry=telemetry_sink),
    )
    _attach_reconciliation_context(reconciliation_agent, context_graph)
    _attach_reconciliation_telemetry(reconciliation_agent, telemetry_sink)
    _attach_reconciliation_event_stream(reconciliation_agent, stream_publisher)
    return ContextEngineContainer(
        settings=s,
        episodic=episodic,
        structural=structural,
        context_graph=context_graph,
        pots=pots,
        connectors=registry,
        readers=reader_registry,
        intelligence_provider=intelligence_provider,
        resolution_service=resolution_service,
        reconciliation_agent=reconciliation_agent,
        jobs=jobs or NoOpContextGraphJobQueue(),
        telemetry=telemetry_sink,
        event_stream_publisher=stream_publisher,
    )


def _default_telemetry() -> TelemetryPort:
    """Default telemetry sink. Picks the Postgres adapter if a DB URL is set."""
    import os

    if not os.getenv("DATABASE_URL"):
        from domain.ports.telemetry import NoOpTelemetry

        return NoOpTelemetry()
    try:
        from adapters.outbound.postgres.telemetry import SqlAlchemyTelemetry

        return SqlAlchemyTelemetry()
    except Exception:
        from domain.ports.telemetry import NoOpTelemetry

        return NoOpTelemetry()


def _attach_reconciliation_telemetry(
    agent: ReconciliationAgentPort | None,
    telemetry: TelemetryPort,
) -> None:
    if agent is None:
        return
    setter = getattr(agent, "set_telemetry", None)
    if setter is not None:
        setter(telemetry)


def _attach_reconciliation_event_stream(
    agent: ReconciliationAgentPort | None,
    publisher: EventStreamPublisherPort,
) -> None:
    """Wire the live publisher into the agent if it accepts one.

    Falling back to NoOp lets old hosts (no Redis configured) continue to
    work — they just don't get live activity streaming.
    """
    if agent is None:
        return
    setter = getattr(agent, "set_event_stream_publisher", None)
    if setter is not None:
        setter(publisher)


def _default_event_stream_publisher() -> EventStreamPublisherPort:
    """Build the Redis publisher when REDIS_URL is set; NoOp otherwise.

    The Redis adapter constructor lazily connects, so failure to reach Redis
    surfaces only when an event actually fires — not at container build.
    Wrapped in try/except so a transient ImportError or misconfigured URL
    doesn't break the rest of the engine.
    """
    import os

    if not os.getenv("REDIS_URL"):
        return NoOpEventStreamPublisher()
    try:
        from adapters.outbound.event_stream.redis_publisher import (
            RedisEventStreamPublisher,
        )

        return RedisEventStreamPublisher()
    except Exception:  # noqa: BLE001 — fall back is intentional
        return NoOpEventStreamPublisher()


def _default_repo_resolver(
    pots: PotResolutionPort,
) -> Callable[[str, SourceReferenceRecord], str | None]:
    """Return a sync resolver that maps pot_id → primary repo_name."""

    def _resolve(pot_id: str, _ref: SourceReferenceRecord) -> str | None:
        resolved = pots.resolve_pot(pot_id)
        if resolved is None:
            return None
        repo = resolved.primary_repo()
        return repo.repo_name if repo else None

    return _resolve


def build_container_with_github_token(
    *,
    token: str,
    pots: PotResolutionPort,
    settings: ContextEngineSettingsPort | None = None,
    reconciliation_agent: ReconciliationAgentPort | None = None,
    jobs: ContextGraphJobQueuePort | None = None,
) -> ContextEngineContainer:
    """Build a container with the GitHub + Notion connectors pre-wired.

    Hosts that need Linear (or any other source) register their connector
    onto the returned ``container.connectors`` after construction.
    """
    try:
        from github import Auth, Github

        gh = Github(auth=Auth.Token(token))
    except ImportError:
        from github import Github

        gh = Github(token)

    def source_for_repo(_repo_name: str) -> GitHubReadPort:
        return PyGithubSourceControl(gh)

    import os

    registry = SourceConnectorRegistry()
    registry.register(
        GitHubConnector(
            source_for_repo=source_for_repo,
            repo_resolver=_default_repo_resolver(pots),
            webhook_secret=(os.getenv("GITHUB_WEBHOOK_SECRET") or "").strip() or None,
        )
    )
    # Notion is registered with no fetcher — capabilities advertise as
    # "available connector, no live read access" so the registry can still
    # surface it in ``context_status`` and prove the contract is wired.
    registry.register(NotionConnector())

    return build_container(
        settings=settings,
        pots=pots,
        connectors=registry,
        reconciliation_agent=reconciliation_agent,
        jobs=jobs,
    )
