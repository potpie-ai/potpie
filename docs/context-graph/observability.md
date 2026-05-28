# Observability

Last reviewed: 2026-05-28.

Observability must work for both local OSS and managed cloud without making
local installs heavy.

Default behavior:

- Local OSS ships dark: local logs, no remote telemetry unless enabled.
- Managed cloud enables hosted tracing, metrics, logs, readiness, and cost
  telemetry through deployment config.
- The core engine emits through ports/adapters, not direct vendor SDK calls.

## Enabling

```bash
CONTEXT_ENGINE_OBSERVABILITY=1                      # 1 | console | off
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=context-engine
CONTEXT_ENGINE_LOG_FORMAT=json                      # plain | json
CONTEXT_ENGINE_LOG_LEVEL=INFO
```

`CONTEXT_ENGINE_OBSERVABILITY=console` is the local-friendly mode. OTLP export
belongs behind an optional extra and explicit config.

## Architecture

Use a hexagonal observability port:

- `domain/ports/observability.py`
- outbound adapters under `adapters/outbound/observability/`
- composition-root wiring in bootstrap/runtime code

The core should emit spans, metrics, and logs through this boundary. Local and
managed deployments decide where those signals go.

## Trace Shape

Important spans:

| Span | Meaning |
|---|---|
| `daemon.request` or `HTTP {method} {route}` | Local daemon or hosted API request. |
| `context.resolve` | Resolve/search request. |
| `reader.{include}` | One reader execution. |
| `context.record` | Structured record write. |
| `scanner.{name}` | Local scanner execution. |
| `graph.write` | Validated graph mutation apply. |
| `graph.query` | Claim query or graph neighborhood read. |
| `event.ledger.append` | Managed/event-ledger webhook capture. |
| `reconciliation.run` | Managed or optional local raw-event reconciliation. |

Batch ingestion traces should link source events to reconciliation runs instead
of pretending delayed fan-in is one synchronous request.

## Metrics

Minimum counters:

- `ce.resolve.total{result}`
- `ce.resolve.unsupported_include_total{include}`
- `ce.record.total{result,record_type}`
- `ce.scanner.total{result,scanner}`
- `ce.graph.write_total{result}`
- `ce.graph.query_total{result}`
- `ce.daemon.restart_total` for local
- `ce.event_ledger.events_total{source}` for managed/event-ledger
- `ce.reconciliation.total{result}` for raw-event reconciliation

Useful histograms:

- `ce.resolve.latency_ms`
- `ce.reader.latency_ms{include}`
- `ce.graph.write_ms`
- `ce.graph.query_ms`
- `ce.scanner.latency_ms{scanner}`
- `ce.reconciliation.latency_ms`

Readiness gauges:

- `ce.dependency_up{dependency}`
- `ce.daemon_up`
- `ce.graph_store_up`
- `ce.event_ledger_lag`

## Logging

Use structured logs when configured. Every request or daemon action should carry
the active pot id, request id, daemon profile (`local` or `cloud`), and graph
store adapter where safe.

Local logs should be easy to find from:

```bash
potpie daemon logs
potpie doctor
```

## Readiness

Local readiness should check:

- daemon process and version
- local auth/IPC
- active pot registry
- local graph store
- local state DB/migrations
- registered readers and scanners
- MCP config when installed

Managed readiness should check:

- API process
- auth/policy dependencies
- hosted graph store
- operational DB
- queue/worker dependencies
- event ledger
- configured source connectors

Liveness and readiness should stay separate. A running daemon/API can be live
while graph storage is not ready.
