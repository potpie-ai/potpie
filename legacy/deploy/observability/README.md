# Observability deployment — Loki + Promtail + Grafana

This is the production log-aggregation pipeline for the [observability
package](../../potpie/observability). Closes the audit's
"JSONL-to-stdout-is-inert-without-a-shipper" gap.

## What it does

1. The FastAPI and Celery containers emit one JSON object per line to stdout
   (the `json_stdout` sink in the observability package).
2. **Promtail** tails Docker container logs, parses the JSON, and ships each
   line to Loki with `level` as a label and the structured fields
   (`request_id`, `conversation_id`, `run_id`, `user_id`, `task_id`,
   `project_id`) as structured metadata.
3. **Loki** stores and indexes the logs.
4. **Grafana** is pre-provisioned with Loki as a data source; use it to query
   and dashboard.

## Quick start

```bash
# 1. Make sure your app stack is up first and shares a network named
#    'potpie_app-network' (or set COMPOSE_NETWORK in your env).
docker network ls | grep potpie_app-network

# 2. Tag your FastAPI / Celery containers so Promtail picks them up:
#    labels:
#      potpie.observability: "true"
#      potpie.service: "fastapi"     # or "celery-worker"
#      potpie.env: "production"      # or "staging"

# 3. Start the observability stack:
docker compose -f deploy/observability/docker-compose.observability.yml up -d

# 4. Open Grafana
open http://localhost:3001
# default login: admin / admin (override with GRAFANA_ADMIN_PASSWORD)
```

## Sample LogQL queries

```logql
# All errors across the stack in the last hour
{level="ERROR"} | json

# Trace one request end-to-end (request_id is a Loki structured-metadata key)
{service="fastapi"} | request_id="abc-123"

# Slow agent runs: filter Celery task logs by conversation_id and look at
# end events
{service="celery-worker"} | json | conversation_id="conv-9"

# Top error sources by logger module
sum by (logger) (count_over_time({level="ERROR"} | json [5m]))
```

## Why these specific labels

Loki gets pathologically slow when label cardinality explodes (millions of
unique values). The rule: **labels are low-cardinality dimensions you filter
on; structured metadata is high-cardinality fields you may search/extract**.

- **Labels** (small set of values): `service`, `env`, `level`, `container`
- **Structured metadata** (unbounded values): `request_id`, `conversation_id`,
  `run_id`, `user_id`, `task_id`, `project_id`, `logger`

This matches the observability package's design: high-cardinality
correlation IDs go in `obs_context` / `obs_fields`, are flattened into the
JSON output, and Promtail promotes them to structured metadata without
turning them into labels.

## What still needs to be done outside this commit

1. Tag the FastAPI / Celery containers in the actual deployment manifest
   (k8s, Helm, ECS task def, whatever this repo's deployment uses) with the
   labels listed above. The `dockerfile` in this repo doesn't set them by
   default; the deployment layer does.
2. Decide on retention. Default Loki keeps logs forever — set
   `limits_config.retention_period` in a Loki override if you want
   time-bounded storage.
3. Build dashboards. The provisioned data source unblocks queries; saved
   dashboards / alerts are a follow-up.

## Why not Datadog / Splunk / CloudWatch

The decision is recorded in `docs/deployment-architecture.md` — Loki +
Grafana matches the target stack alongside Prometheus and Langfuse, and is
self-hostable. If a managed backend is preferred later, swap the
`promtail.clients[].url` (or use the `cloudwatch` / `datadog` clients
upstream offers) and keep everything else.
