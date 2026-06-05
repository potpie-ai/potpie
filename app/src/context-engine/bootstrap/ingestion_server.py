"""Composition root for the HTTP **ingestion/server** subsystem.

Distinct from ``bootstrap.host_wiring.build_host_shell`` (the in-process agent
spine behind the CLI + MCP). This root wires the async ingestion pipeline —
Neo4j writer/reader, the Postgres batch/ledger/execution-log stores, source
connectors, and the reconciliation agent — that backs the FastAPI surface in
``adapters/inbound/http``. It is intentionally kept separate while the pipeline
is migrated onto ``HostShell``; nothing on the CLI path imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from adapters.outbound.connectors.github import (
    GitHubConnector,
    GitHubReadPort,
    PyGithubSourceControl,
)
from adapters.outbound.connectors._bench_stubs import (
    AlertingStubConnector,
    DeployStubConnector,
    RepoDocsStubConnector,
    SlackStubConnector,
)
from adapters.outbound.connectors.notion import NotionConnector
from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.reconciliation.context_graph_tools import (
    ContextGraphReconciliationTools,
)
from adapters.outbound.graph import GraphWriterPort, Neo4jGraphWriter
from adapters.outbound.graph.backends.neo4j_backend import Neo4jGraphBackend
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
from application.services.source_connector_registry import SourceConnectorRegistry
from domain.ports.event_query_service import EventQueryService
from domain.ports.event_stream import (
    EventStreamPublisherPort,
    NoOpEventStreamPublisher,
)
from domain.ports.ingestion_config import IngestionConfigPort
from domain.ports.context_graph import ContextGraphPort
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
    NoOpContextGraphJobQueue,
)
from domain.ports.policy import PolicyPort
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.pot_source_listing import PotSourceListingPort
from domain.ports.observability import NoOpObservability, ObservabilityPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.telemetry import TelemetryPort
from domain.source_references import SourceReferenceRecord


@dataclass
class IngestionServerContainer:
    """Wired dependencies for use cases.

    Phase 2 collapsed the source-shaped slots (``source_for_repo``,
    ``source_resolver``, ``pot_source_listing``) into a single
    :class:`SourceConnectorRegistry`. Hosts register connectors before
    handing the container off; consumers route every source-typed call
    through the registry.
    """

    settings: ContextEngineSettingsPort
    graph_writer: GraphWriterPort
    pots: PotResolutionPort
    connectors: SourceConnectorRegistry = field(default_factory=SourceConnectorRegistry)
    reconciliation_agent: ReconciliationAgentPort | None = None
    jobs: ContextGraphJobQueuePort | None = None
    context_graph: ContextGraphPort | None = None
    pot_source_listing: PotSourceListingPort | None = None
    """Host-side per-pot source rows (last_sync_at, sync_mode, …). Distinct
    from ``connectors``, which advertises engine-side connector capabilities.
    Both are surfaced by ``context_status``."""
    telemetry: TelemetryPort | None = None
    """Cost + drift sink. ``None`` is treated as ``NoOpTelemetry``."""
    observability: ObservabilityPort = field(default_factory=NoOpObservability)
    """Tracing + metrics sink. NoOp by default (ships dark); the OTel
    adapter is wired only when an OTLP endpoint is configured. Distinct
    from ``telemetry`` (business cost/drift) — bridged, not merged."""
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
            episodic_available=getattr(self.graph_writer, "enabled", True),
        )

    def ledger(self, session: Session) -> SqlAlchemyIngestionLedger:
        return SqlAlchemyIngestionLedger(session)

    def reconciliation_ledger(self, session: Session) -> SqlAlchemyReconciliationLedger:
        return SqlAlchemyReconciliationLedger(session)

    def batch_repository(self, session: Session) -> SqlAlchemyBatchRepository:
        return SqlAlchemyBatchRepository(session)

    def agent_checkpoint_store(
        self, session: Session
    ) -> SqlAlchemyAgentCheckpointStore:
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


def build_ingestion_server(
    *,
    settings: ContextEngineSettingsPort | None = None,
    pots: PotResolutionPort,
    connectors: SourceConnectorRegistry | None = None,
    reconciliation_agent: ReconciliationAgentPort | None = None,
    jobs: ContextGraphJobQueuePort | None = None,
    telemetry: TelemetryPort | None = None,
    observability: ObservabilityPort | None = None,
    event_stream_publisher: EventStreamPublisherPort | None = None,
) -> IngestionServerContainer:
    s = settings or EnvContextEngineSettings()
    telemetry_sink = telemetry or _default_telemetry()
    observability_sink = observability or _default_observability()
    # Publish to the process-global accessor so middleware / the Celery
    # worker / infra wrappers see the same sink the container holds.
    from bootstrap.observability_runtime import set_observability

    set_observability(observability_sink)
    # Bridge (not merge) cost/drift into OTel metrics when observability is
    # live, so the same numbers reach Prometheus without touching the
    # record_cost / record_drift call sites.
    if not isinstance(observability_sink, NoOpObservability):
        from adapters.outbound.observability.telemetry_bridge import (
            ObservabilityTelemetryBridge,
        )

        telemetry_sink = ObservabilityTelemetryBridge(
            telemetry_sink, observability_sink
        )
    stream_publisher = event_stream_publisher or _default_event_stream_publisher()
    # FalkorDB (#819) ships as code-in-tree but is not yet wrapped behind the
    # GraphBackend port — follow-up. Until that adapter exists, only Neo4j can
    # be assembled into a ContextGraphService here.
    backend_kind = (s.graph_db_backend() or "neo4j").strip().lower()
    if backend_kind == "falkordb":
        raise NotImplementedError(
            "FalkorDB GraphBackend adapter is not wired yet; the FalkorDB "
            "reader/writer modules (#819) remain importable at "
            "adapters.outbound.graph.falkordb_* but need a GraphBackend port "
            "wrapper before they can plug into ContextGraphService. Use "
            "GRAPH_DB_BACKEND=neo4j until the adapter lands."
        )
    graph_writer = Neo4jGraphWriter(s)
    registry = connectors or SourceConnectorRegistry()
    # One graph substrate: the Neo4j GraphBackend (claim_query read trunk +
    # mutation write door). Share the one writer so the ingestion scan path
    # (container.graph_writer) and ContextGraphService don't open two pools.
    backend = Neo4jGraphBackend(s, writer=graph_writer)
    context_graph = ContextGraphService(backend=backend)
    # Fail fast if the read trunk's reader set has drifted from the advertised
    # ``READER_BACKED_INCLUDES`` (see domain.coherence).
    from domain.coherence import assert_runtime_coherence

    assert_runtime_coherence(reader_backed_includes=context_graph.backed_includes)
    _attach_reconciliation_context(reconciliation_agent, context_graph)
    _attach_reconciliation_telemetry(reconciliation_agent, telemetry_sink)
    _attach_reconciliation_observability(reconciliation_agent, observability_sink)
    _attach_reconciliation_event_stream(reconciliation_agent, stream_publisher)
    return IngestionServerContainer(
        settings=s,
        graph_writer=graph_writer,
        context_graph=context_graph,
        pots=pots,
        connectors=registry,
        reconciliation_agent=reconciliation_agent,
        jobs=jobs or NoOpContextGraphJobQueue(),
        telemetry=telemetry_sink,
        observability=observability_sink,
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


def _observability_enabled() -> bool:
    import os

    raw = os.getenv("CONTEXT_ENGINE_OBSERVABILITY", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _default_observability() -> ObservabilityPort:
    """Default observability sink. NoOp unless explicitly enabled.

    ``CONTEXT_ENGINE_OBSERVABILITY=console`` selects the dependency-free
    console adapter (local dev). Any other truthy value selects the OTel
    adapter, but only when an OTLP endpoint is actually configured — so the
    feature ships dark and never tries to export into the void.
    """
    import os

    if not _observability_enabled():
        return NoOpObservability()
    mode = os.getenv("CONTEXT_ENGINE_OBSERVABILITY", "").strip().lower()
    if mode == "console":
        try:
            from adapters.outbound.observability.console import ConsoleObservability

            return ConsoleObservability()
        except Exception:  # noqa: BLE001
            return NoOpObservability()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint:
        return NoOpObservability()
    try:
        from adapters.outbound.observability.otel import OtelObservability

        return OtelObservability()
    except Exception:  # noqa: BLE001 — missing extra / setup failure → dark
        return NoOpObservability()


def _attach_reconciliation_telemetry(
    agent: ReconciliationAgentPort | None,
    telemetry: TelemetryPort,
) -> None:
    if agent is None:
        return
    setter = getattr(agent, "set_telemetry", None)
    if setter is not None:
        setter(telemetry)


def _attach_reconciliation_observability(
    agent: ReconciliationAgentPort | None,
    observability: ObservabilityPort,
) -> None:
    """Wire the observability sink into the agent if it accepts one.

    Duck-typed like ``_attach_reconciliation_telemetry`` — old agents that
    don't expose ``set_observability`` simply don't get spans/metrics.
    """
    if agent is None:
        return
    setter = getattr(agent, "set_observability", None)
    if setter is not None:
        setter(observability)


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


def build_ingestion_server_with_github_token(
    *,
    token: str,
    pots: PotResolutionPort,
    settings: ContextEngineSettingsPort | None = None,
    reconciliation_agent: ReconciliationAgentPort | None = None,
    jobs: ContextGraphJobQueuePort | None = None,
) -> IngestionServerContainer:
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
            allow_unsigned=os.getenv("CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS", "")
            .strip()
            .lower()
            in ("1", "true", "yes"),
        )
    )
    # Notion is registered with no fetcher — capabilities advertise as
    # "available connector, no live read access" so the registry can still
    # surface it in ``context_status`` and prove the contract is wired.
    registry.register(NotionConnector())
    # Bench-time passive stubs for sources without production readers
    # yet (see adapters/outbound/connectors/_bench_stubs.py). They route
    # envelopes through reconciliation but advertise no fetch capability,
    # so production traffic that lacks a real reader will still fail
    # closed instead of silently grading against a stub.
    registry.register(SlackStubConnector())
    registry.register(RepoDocsStubConnector())
    registry.register(AlertingStubConnector())
    registry.register(DeployStubConnector())

    return build_ingestion_server(
        settings=settings,
        pots=pots,
        connectors=registry,
        reconciliation_agent=reconciliation_agent,
        jobs=jobs,
    )
