# Context Engine — Observability

End-to-end tracing, metrics, and structured logging for the context engine.
Backend-neutral (OTLP → Tempo / Prometheus / Loki); **ships dark** — the
NoOp sink is the default and there is zero overhead until an OTLP endpoint
is configured.

## Enabling it

```bash
CONTEXT_ENGINE_OBSERVABILITY=1                      # 1 | console | off (default off)
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=context-engine
CONTEXT_ENGINE_LOG_FORMAT=json                      # plain (default) | json
CONTEXT_ENGINE_LOG_LEVEL=INFO
```

`CONTEXT_ENGINE_OBSERVABILITY=console` selects a dependency-free adapter
that prints spans/metrics — useful locally. The real adapter needs the
`observability` extra (`pip install -e ".[observability]"`:
`opentelemetry-sdk`, `-exporter-otlp`, `-instrumentation-openai`). Collector
+ alert + dashboard configs are in [`deployment/observability/`](../../deployment/observability/).

## Architecture

A thin hexagonal port — `domain/ports/observability.py:ObservabilityPort`
(`span` / `baggage` / `current_traceparent` / `counter` / `histogram` /
`gauge`) — with three adapters: `NoOp` (default), `Console`, `Otel`. Nothing
outside `adapters/outbound/observability/otel.py` imports `opentelemetry`;
that is what keeps the backend swappable. Wired through `build_container`
exactly like the existing `TelemetryPort` (env-gated `_default_*` → NoOp,
duck-typed `set_*` attach). `bootstrap/observability_runtime.py` exposes the
same instance process-globally for the three composition-root concerns that
can't reach a container (ASGI middleware, the Celery worker, the infra
proxy).

### Provider-aware OTel adapter

The parent Potpie Celery worker already owns a Logfire `TracerProvider`
that instruments pydantic-ai. The adapter detects this: if a real SDK
provider exists it *attaches* a second OTLP processor to it (agent spans
ride the same provider and nest under our spans); standalone, it owns the
provider. This avoids the dual-provider trap that would silently break
correlation.

### Trace topology (the key decision)

The pipeline is fan-in/fan-out — `N events → 1 windowed batch → chunked
agent run → M mutations` — so "one trace per event" is wrong and a 5-minute
window kills live context propagation. The model:

1. **Ingress** is a short sync trace. Its `traceparent` is persisted into
   the (previously dead) `context_events.correlation_id` column.
2. **The batch run** is the primary async trace. On claim it starts a fresh
   trace with OTel **span-links** reconstructed from each event's persisted
   `correlation_id` — the correct primitive for delayed fan-in batches.
3. Each chunk's `agent.run_batch` span sets **baggage**
   (`ce.pot_id/batch_id/chunk/event_ids`) so pydantic-ai's own child spans
   inherit the ids and are traceable back to the agent run.
4. `graph.add_episode` records `graph.episode_uuid` — the durable
   event→graph-node link, in-trace.

## Span catalogue

| Span | Where | Key attributes |
|---|---|---|
| `HTTP {method} {route}` | ingress middleware | `http.*`, binds `trace_id` |
| `ingest.submit` | submission service | `pot_id`, `ingest.event_id`, `ingest.duplicate` |
| `batch.process` | worker entry | `batch_id`, `pot_id`, links→ingress |
| `agent.run_batch` | per chunk | `pot_id`, `batch_id`, `agent.chunk`, `agent.event_ids` |
| `graph.add_episode` | Graphiti write | `pot_id`, `event_id`, `graph.episode_uuid` |
| OpenAI SDK spans | auto (Graphiti) | nested under `graph.add_episode` |
| `context.resolve` | read path | `pot_id`, `resolve.intent/mode` |
| `reader.{family}` | per reader | `reader.family`, `reader.count` |
| `neo4j.{op}` | infra proxy | `db.system`, `db.op` |

## Metric catalogue

Counters (OTLP→Prometheus appends nothing; `.`→`_`):

- `ce.ingest.events_total{source}`, `ce.ingest.dedup_total{source}`
- `ce.batch.started_total`, `ce.batch.finished_total{result}`,
  `ce.batch.reaped_total` ← **the dead-letter signal, page on it**
- `ce.events.reconciled_total`, `ce.events.failed_total`
- `ce.agent.timeout_total`
- `ce.resolve.total{result}`, `ce.resolve.reader_fallback_total{reason}`
- `ce.llm.calls_total / tokens_total / input_tokens_total /
  output_tokens_total{kind,model}` (mirrored from `TelemetryPort`,
  **includes Graphiti's previously-invisible calls**)
- `ce.neo4j.errors_total{op}`

Histograms: `ce.batch.time_in_pending_ms` (windowed canary),
`ce.agent.tool_calls`, `ce.graph.add_episode_ms`, `ce.resolve.latency_ms`,
`ce.llm.latency_ms`, `ce.neo4j.query_ms{op}`.

Gauges: `ce.drift.{stale_refs,verification_failed_refs,source_access_gaps,
open_conflicts,missing_coverage}` (mirrored from drift snapshots),
`ce.dependency_up{dependency}`.

## Logging

`bootstrap/logging_setup.py` installs one root handler from every
entrypoint. A `CorrelationFilter` injects the active
`trace_id/event_id/pot_id/batch_id/run_id/...` onto **every** record, and
`JsonFormatter` serializes `extra=` payloads — so the formerly-dropped
operator-audit channel (`extra={"audit": ...}`) now renders, and all ~200
existing `getLogger` sites get structured, trace-correlated JSON **without
being rewritten**. (This is a deliberate, lower-risk realization of the
plan's "structlog migration": stdlib + a formatter is the swappable seam
for 200 stdlib call sites; forcing structlog would be churn for no added
capability.)

## Readiness

`GET /health` is unchanged (liveness). `GET /ready` probes Postgres +
Neo4j (hard) and Redis (optional), emits `ce.dependency_up`, and returns
503 when a hard dependency is down — standard k8s readiness semantics.

## Deviations from the plan (and why)

- **Logging:** stdlib JSON + correlation filter instead of a structlog
  rewrite of ~200 sites — identical outcome, no churn, formatter is the
  swappable seam.
- **`episode_uuid` backlink:** recorded as a `graph.add_episode` span
  attribute instead of a new `context_events` column. The host owns the
  schema (no migration framework here) and per-event attribution inside a
  multi-event chunk is ambiguous; the trace backend is where you pivot
  event→graph anyway, so this is faithful to the observability intent at
  far lower risk to the hot path.
- **Infra proxy scope:** wraps the Neo4j structural adapter only. The
  Graphiti episodic adapter is excluded (a route does
  `isinstance(container.episodic, GraphitiEpisodicAdapter)`, and its
  hottest call already has a dedicated span).
- **`record_drift`:** was already wired (`ContextResolutionService.
  _emit_drift_snapshot`); the original plan note was stale. Phase C added
  the OTel mirror, not the wiring.
