---
title: Observability
description: "Logs, traces, metrics, and the backend-neutral observability port."
---

## Overview

The domain and application layers emit spans and metrics through
`ObservabilityPort` in `domain/ports/observability.py`. Only the OTel adapter imports
`opentelemetry`, so core code does not depend on a telemetry backend. The default
`NoOpObservability` sends no spans or metrics. Callers can emit telemetry without
checking whether an operator enabled it. Observability failures cannot fail a
request.

Both composition roots wire the same port: the root-owned local runtime
(`potpie/runtime/composition.py`) and the standalone ingestion HTTP server
(`bootstrap/ingestion_server.py`). See [architecture.md](./architecture.md).

## Shape

```mermaid
flowchart LR
  obs_core["shared services<br/>Pot Mgmt · Graph Service<br/>Graph Workbench · Skill Manager"]
  obs_ingest["ingestion pipeline<br/>(HTTP server · batch worker)"]
  obs_port["ObservabilityPort<br/>(domain port)"]
  obs_noop["NoOpObservability<br/>(disabled by default)"]
  obs_console["ConsoleObservability"]
  obs_otel["OtelObservability → OTLP"]

  obs_core --> obs_port
  obs_ingest --> obs_port
  obs_port --> obs_noop
  obs_port --> obs_console
  obs_port --> obs_otel
```

Default behavior:

- Local OSS uses `NoOpObservability` by default. No spans, metrics, or remote
  telemetry leave the process unless explicitly configured.
- Logs always work (stdlib logging), but structured/JSON output and trace correlation
  are opt-in.
- Core code emits through the observability port, not vendor SDKs. The OTel adapter
  (`adapters/outbound/observability/otel.py`) is the *only* `opentelemetry` import.

## Configuration

```bash
# Sink selection (unset / 0 / false / off => NoOp, the default)
CONTEXT_ENGINE_OBSERVABILITY=console     # off | console | <any-truthy => OTLP>

# OTLP export only activates when an endpoint is present (else falls back to NoOp)
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=...   # alternative to the line above
OTEL_SERVICE_NAME=context-engine         # default: context-engine

# Logging (independent of the trace/metric sink)
CONTEXT_ENGINE_LOG_FORMAT=json           # plain (default) | json
CONTEXT_ENGINE_LOG_LEVEL=INFO            # default: INFO
```

`bootstrap/observability_wiring.py default_observability()` resolves the sink:
`console` builds `ConsoleObservability` with dependency-free trace output; any other
truthy value builds `OtelObservability` **only if** an OTLP endpoint is set, otherwise
it stays NoOp. A missing OTel extra or any setup failure also degrades silently to
NoOp. OTLP export therefore lives behind optional dependencies and explicit config.

## The port

`ObservabilityPort` (`domain/ports/observability.py`) exposes:

- `span(name, *, kind, attributes, links)` — context manager yielding a `Span`
  (`set_attribute(s)` / `add_event` / `record_exception` / `set_error`). `kind` is a
  small string set mapped to the OTel enum inside the adapter: `internal` (default),
  `server`, `client`, `producer`, `consumer`. `links` is a sequence of W3C
  `traceparent` strings — how the batch trace links back to each event's ingress trace.
- `current_traceparent()` — W3C traceparent of the active span (persisted into
  `context_events.correlation_id` at admission so the async batch can correlate across
  the windowed delay).
- `baggage(**items)` — attaches OTel baggage around the reconciliation agent run so the
  child spans pydantic-ai emits inherit pot/batch/run/event ids.
- `counter` / `histogram` / `gauge` — metric primitives.

### Correlation spine

`bootstrap/observability_context.py` carries a fixed, documented correlation key set
in a single `contextvars` var: **`trace_id, pot_id, event_id, batch_id, run_id, seq,
chunk, source`**. Anything outside this set is dropped so log/trace cardinality stays
bounded. Keys are bound at ingress and re-bound inside the batch worker, and injected
into every log record (see Logging) so any line or span ties back to its event.

## Trace map

The `Code boundary` column names the module each span instruments. These modules
also appear in the code map in [architecture.md](./architecture.md).

| Span | Kind | Meaning | Code boundary |
|---|---|---|---|
| `graph.<command>` | internal | One workbench command: `graph.status`, `graph.catalog`, `graph.describe`, `graph.search_entities`, `graph.read`, `graph.neighborhood`, **`graph.propose`**, **`graph.commit`**, `graph.bulk`, `graph.history`, `graph.inbox` (carries `operation`), `graph.quality` (carries `report`), plus the legacy `graph.mutate` / `graph.mutation_template` / `graph.nudge`. | `potpie/cli/commands/graph.py` (`_graph_command`) |
| `ingest.submit` | server | Inbound episode/event/record normalized and submitted. | `application/services/ingestion_submission_service.py` |
| `HTTP <method> <route>`, `http.ready` | server | Ingestion HTTP server request + readiness. | `adapters/inbound/http/_hardening.py`, `adapters/inbound/http/api/router.py` |
| `batch.process` | consumer | Windowed batch fan-in (N events → 1 run → M mutations); **links back** to each event's ingress traceparent. | `application/use_cases/context_graph_jobs.py` |
| `agent.run_batch` | internal | One reconciliation-agent chunk run. Emitted only when the agent planner is enabled (off by default). | `application/use_cases/process_batch.py` |
| `<prefix>.<op>` | client | Public methods of an instrumented infra adapter (e.g. `neo4j.<op>`), wrapped at the composition root. | `bootstrap/observability_proxy.py` (`instrument_adapter`) |

`graph.propose` and `graph.commit` are the canonical write-door spans (propose →
commit `--verify`); `graph.mutate` is the legacy wrapper that internally proposes then
commits. See [writing.md](./writing.md).

Notes:

- Reads attach `subgraph`, `view`, and `match_mode` attributes to `graph.read`
  instead of emitting per-reader spans. The read path (`ReadOrchestrator` → 10 readers →
  `EnvelopeBuilder`) is described in [querying.md](./querying.md).
- The per-event **`reconciliation_run`** is a durable ledger row
  (`context_reconciliation_runs`), *not* a span. It is linked to `batch.process` by
  `run_id` and surfaced through `graph status` / the explorer UI.
- Batch ingestion links event spans to the batch span to represent delayed fan-in
  without treating the batch as one long request. See
  [ingestion-nudge.md](./ingestion-nudge.md).

> **Roadmap (not yet wired):** external Event Ledger spans (`ledger.pull`,
> `ledger.query`, `ledger.cursor.update`). The managed/self-hosted ledger clients are
> TODO stubs, so these paths are non-functional against real providers today (see
> [ingestion-nudge.md](./ingestion-nudge.md)). Do not confuse the external Event Ledger
> with the live internal Postgres event store, which *is* instrumented above.

## Metrics

All metric names are `ce.*`. Graph-workbench and batch metrics are mirrored to both the
`ObservabilityPort` and a Sentry metrics runtime.

Workbench commands in `potpie/cli/commands/graph.py` emit one counter and
histogram per command root:

- `ce.graph.<root>_total` and `ce.graph.<root>_ms` for each root (`read`, `propose`,
  `commit`, `catalog`, `status`, `describe`, `search_entities`, `neighborhood`,
  `history`, `inbox`, `quality`, `bulk`, `mutate`, …). Bounded label set:
  `{result, error_code, pot_id, subgraph, view, risk, status, operation, report,
  backend_profile, match_mode}`.

CLI envelope:

- `ce.cli.invocations_total`, `ce.cli.duration_ms`.

Ingestion pipeline (the live internal event store + batch worker):

- `ce.ingest.events_total`, `ce.ingest.dedup_total` (admission);
- `ce.batch.started_total`, `ce.batch.finished_total`, `ce.batch.reaped_total`;
- `ce.batch.time_in_pending_ms` measures time waiting for the five-minute windowed
  flusher and helps detect a stalled flusher;
- `ce.events.reconciled_total`, `ce.events.failed_total`.

Reconciliation agent + cost telemetry (only when the planner is enabled):

- `ce.agent.tool_calls`, `ce.agent.timeout_total`;
- `ce.llm.calls_total`, `ce.llm.input_tokens_total`, `ce.llm.output_tokens_total`,
  `ce.llm.tokens_total`, `ce.llm.latency_ms`.

Instrumented infra adapters (`instrument_adapter`):

- `ce.<prefix>.query_ms{op}`, `ce.<prefix>.errors_total{op}` (e.g. `ce.neo4j.*`).

Setup + quality/drift:

- `ce.setup.runs_total`, `ce.setup.step_total`;
- gauges `ce.drift.missing_coverage`, `ce.drift.open_conflicts`,
  `ce.drift.source_access_gaps`, `ce.drift.stale_refs`,
  `ce.drift.verification_failed_refs`.

Readiness:

- gauge `ce.dependency_up{dependency}`.

> **Roadmap (not yet wired):** external Event Ledger metric families (`ce.ledger.*`,
> `ce.event_ledger.*`) are not emitted because the ledger clients are stubs.

## Logging

Logging is configured in one place (`bootstrap/logging_setup.py configure_logging()`,
idempotent, stdlib-only). A `CorrelationFilter` attaches the active correlation keys
(above) to every record, so any log line can be tied back to its trace / event / pot /
batch / run. Two formats:

- **`plain`** (default) — human format that still surfaces the trace:
  `… [trace=… event=… pot=…] message`.
- **`json`** — one JSON object per line: `ts/level/logger/msg` + the present
  correlation keys + any structured `extra=` fields (including the `audit` channel,
  which earlier versions discarded).

Every request, daemon action, or pipeline step should therefore carry: the active
`trace_id`/`request id`, `pot_id`, the `event/batch/run` ids when in the async pipeline,
and the service-boundary logger name (`daemon`, `pot_management`, `graph_service`,
`graph_backend`, `skill_manager`, `ingestion`, …). Backend name is included where safe.

Local logs are discoverable from:

```bash
potpie daemon logs
potpie doctor
```

> Managed-host routing is available. The external Event Ledger binding remains
> inactive, so do not interpret saved binding fields as evidence of live ledger
> traffic.

## Readiness

A running daemon can have unavailable graph storage or semantic search.
`potpie doctor` reports readiness from `backend.capabilities()` + `backend.mutation.readiness()`
(`BackendReadiness`) + `daemon.status()` + `ledger.status()`, and `potpie graph status`
returns `data_plane_status`. Keep each dependency status in diagnostics so
operators can identify which component is unavailable. See [cli-flow.md](./cli-flow.md)
for command flags.

Local readiness checks:

- daemon process and version;
- active pot;
- registered sources;
- Graph Service, plus the active `GraphBackend` name and its capability set (which of
  the six ports are real vs fail-closed — see [architecture.md](./architecture.md));
- semantic index and embedder (bundled local embedder by default);
- Skill Manager catalog and installed-vs-recommended skills.

Managed pot and host status commands use the configured managed host. Cloud skill
sync and external Event Ledger rebinding remain unimplemented. A saved ledger
binding does not establish provider health, cursor progress, or delivery lag.

## See also

- [architecture.md](./architecture.md) — composition roots, the Code Map, the
  `GraphBackend` port + capability coverage table.
- [ingestion-nudge.md](./ingestion-nudge.md) — the internal Postgres event store vs the
  external Event Ledger seam; windowed batching and the batch worker.
- [cli-flow.md](./cli-flow.md) — `doctor`, `daemon logs`, `graph status` command flags.
- [writing.md](./writing.md) — the propose → commit write door behind `graph.propose` /
  `graph.commit`.
