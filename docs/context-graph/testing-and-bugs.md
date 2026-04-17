# Context Engine Test Harness And Findings

This file tracks the practical test harness for context-engine and the bugs or gaps found while exercising it end to end.

## Harness

Primary script:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py --help
```

Mock-only E2E, no Potpie server or Neo4j required:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py mock-e2e --print-json
```

In-process HTTP E2E, no Potpie server, Neo4j, or API key required:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py http-e2e --print-json
```

Live API smoke test against the configured Potpie `/api/v2/context` server. This path intentionally requires a valid Potpie API key because it exercises the deployed API boundary:

```bash
cd app/src/context-engine
uv run potpie --json doctor
cd ../../..
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke --print-json
```

Live ingest with bundled sample episodes:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke --write --print-json
```

Live ingest plus `context_record`:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke --write --record --print-json
```

Reports are written to:

```text
app/src/context-engine/.tmp/context-engine-lab-report.json
```

Mock data lives at:

```text
app/src/context-engine/scripts/mock_context_data.json
```

## Coverage

The harness exercises:

- mock agent context wraps for `feature`, `debugging`, and `operations`
- in-process HTTP router calls without an API key through injected test auth
- in-process `context_status`
- in-process raw episode ingest with `sync=true`
- in-process semantic `context_search`
- in-process `context_resolve` for bundled recipes
- in-process reset
- recipe selection through `context_recipe_for_intent`
- response envelope validation, including `coverage`, `freshness`, `quality`, `source_refs`, and fallbacks
- context record type validation and deterministic source id generation
- live `GET /health` through the Potpie context API client
- live `context_status`
- live semantic `context_search`
- live `context_resolve` for bundled recipes
- optional live raw episode ingest
- optional live `context_record`

## Findings

### Fixed

| Area | Finding | Fix |
| --- | --- | --- |
| Mock operations coverage | `mock-e2e` initially returned partial coverage for the operations recipe because the mock provider did not return `runbooks`, `scripts`, `config`, or `local_workflows` project-map records. | Added representative mock `Runbook`, `Script`, `ConfigVariable`, and `LocalWorkflow` records so operations flow is testable without external services. |
| CLI readiness signal | `potpie doctor` previously reported `GET /health` success even when the stored API key was invalid, so authenticated `search`, `ingest`, and MCP calls still failed. | Added authenticated `GET /api/v2/context/pots` probe to `doctor` and surfaced `potpie_auth_ok` / `potpie_auth_message` in JSON and human output. |
| No-key API tests | Local tests for the context API were blocked by live Potpie API-key credentials even though they only needed to exercise our own router behavior. | Added `context_engine_lab.py http-e2e`, which mounts the context router with injected lab auth and in-memory graph adapters. This keeps production `/api/v2/context` auth intact while making local API tests deterministic. |

### Open Bugs / Gaps

The table below should be updated after each harness run.

| Status | Area | Finding | Impact | Next step |
| --- | --- | --- | --- | --- |
| Open | Local credentials | Live `api-smoke` reached `GET /health` on `http://127.0.0.1:8001`, but all authenticated `/api/v2/context` calls returned `401 Invalid API key`. `potpie --json doctor` now reports `potpie_auth_ok: false`. | Live ingest/query/resolve/record flows cannot be validated until credentials are refreshed. | Run `potpie login <valid-api-key> --url http://127.0.0.1:8001`, then rerun `api-smoke --write --record`. |
| Open | Live write coverage | Because authenticated API calls failed, live ingest and `context_record` were not executed. | End-to-end server-side ingestion/reconciliation remains unverified in this environment. | Rerun `api-smoke --write --record` after fixing the API key. |

## Latest Run

Date: 2026-04-17

Commands:

```bash
uv run python app/src/context-engine/scripts/context_engine_lab.py mock-e2e
uv run python app/src/context-engine/scripts/context_engine_lab.py http-e2e --print-json
uv run python app/src/context-engine/scripts/context_engine_lab.py api-smoke --print-json
uv run potpie --json doctor
```

Results:

- Mock E2E: passed. `feature`, `debugging`, and `operations` all returned complete coverage with `quality.status=watch`, as expected for unverified mock source refs.
- HTTP E2E: passed without a Potpie API key. `status`, three `ingest:*` calls, `search`, three `resolve:*` calls, record source-id normalization, and `reset` all succeeded through the context router.
- Live API smoke: failed after health. `health` returned `status_code=200`, but `status`, `search`, and all `resolve:*` calls returned `401 Invalid API key`.
- Doctor: now reports `potpie_health_ok=true` and `potpie_auth_ok=false`.
