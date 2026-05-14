# Agent Contract

The four-tool agent port. This is the single source of truth for any agent or skill that consumes the Context Engine.

The contract is enforced in code at:
- Tool intents, includes, recipes, record types: [`app/src/context-engine/domain/agent_context_port.py`](../../app/src/context-engine/domain/agent_context_port.py)
- Request/response models: [`app/src/context-engine/domain/intelligence_models.py`](../../app/src/context-engine/domain/intelligence_models.py)
- MCP tools: [`app/src/context-engine/adapters/inbound/mcp/server.py`](../../app/src/context-engine/adapters/inbound/mcp/server.py)
- HTTP surface: [`app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`](../../app/src/context-engine/adapters/inbound/http/api/v1/context/router.py)

If this doc disagrees with code, the code is right and this doc has rotted — fix it.

---

## The four tools

| Tool | Role | When to use |
|---|---|---|
| `context_resolve` | Primary | Start non-trivial work. Returns a bounded context wrap for a task. |
| `context_search` | Secondary | Narrow follow-up when an entity or phrase is already known. |
| `context_record` | Write | Save a durable decision, fix, preference, workflow, feature note, runbook note, or incident summary. |
| `context_status` | Status | Cheap pot/source health check. Returns the recommended recipe for an intent. |

Adding a new use case becomes a parameter on `context_resolve` (`intent`, `include`, `scope`, `mode`, `source_policy`), not a new tool.

---

## `context_resolve`

Pulls together evidence families needed for a task and returns a structured envelope with answer, evidence, source refs, freshness, coverage, fallbacks, and conflicts.

### Request

| Field | Type | Notes |
|---|---|---|
| `pot_id` | str | Required. Tenant boundary. |
| `query` | str | Required. Free-form task description. |
| `intent` | str | Optional. One of: `feature`, `debugging`, `review`, `operations`, `planning`, `docs`, `onboarding`, `refactor`, `test`, `security`, `unknown`. Drives default `include` set if omitted. |
| `scope` | object | Optional. Narrows the request — see below. |
| `artifact_ref` | `{kind, identifier}` | Optional. E.g. `{kind: "pr", identifier: "694"}`. |
| `include` | list[str] | Optional. Evidence families (see catalog below). Overrides the intent's default set. |
| `exclude` | list[str] | Optional. Suppresses families that would otherwise run. |
| `mode` | `fast` \| `balanced` \| `deep` \| `verify` | Default `fast`. Controls latency budget and whether activated families are mandatory. |
| `source_policy` | `references_only` \| `summary` \| `verify` \| `snippets` \| `full_if_needed` | Default `references_only`. How aggressively the resolver should fetch source bodies. |
| `budget` | `{max_items, max_tokens, timeout_ms, freshness}` | Default `{12, null, 4000, "prefer_fresh"}`. |
| `as_of` | datetime | Optional. Time-travel query. |
| `consumer_hint` | str | Optional. Telemetry only. |

### Scope fields

`repo_name`, `branch`, `file_path`, `function_name`, `symbol`, `pr_number`, `services[]`, `features[]`, `environment`, `ticket_ids[]`, `user`, `source_refs[]`.

### Include-family catalog

Drives which `ContextReader`s the planner activates. Defined in `agent_context_port.CONTEXT_INCLUDE_VALUES`:

`purpose`, `feature_map`, `service_map`, `repo_map`, `docs`, `tickets`, `decisions`, `recent_changes`, `owners`, `prior_fixes`, `diagnostic_signals`, `incidents`, `alerts`, `deployments`, `runbooks`, `local_workflows`, `scripts`, `config`, `preferences`, `agent_instructions`, `source_status`, `operations`, `discussions`, `artifact`, `semantic_search`, `causal_chain`.

### Default include sets per intent

Defined in `agent_context_port.DEFAULT_INTENT_INCLUDES`. Recipes wrap these with `mode` and `source_policy` defaults — see the recipe section.

### Response envelope

The full envelope is built by `bundle_to_agent_envelope()` from an `IntelligenceBundle`:

| Field | Shape | Notes |
|---|---|---|
| `ok` | bool | False on hard error; envelope still returned. |
| `answer.summary` | str | LLM-synthesized when available; deterministic count summary as fallback. |
| `answer.{artifacts, recent_changes, decisions, discussions, owners, project_map, debugging_memory}` | list | Per-family curated answer slices. |
| `facts.{changes, decisions, ownership, project_map, debugging_memory, causal_chain}` | list | Canonical graph results. |
| `evidence` | list | Semantic hits + discussions. Compact supporting facts. |
| `source_refs` | list[`SourceReferenceRecord`] | `{type, uri, fetchable, access, last_seen_at, last_verified_at, verification_state}`. |
| `source_resolution` | object | What the resolver actually fetched (source-policy-controlled). |
| `confidence` | float | Coverage-derived (`complete`=0.82, `partial`=0.55, `empty`=0.2). |
| `as_of` | iso str | Echoes request `as_of`. |
| `coverage` | `{status, available[], missing[], missing_reasons{}}` | `complete` \| `partial` \| `empty`. |
| `freshness` | `{status, last_graph_update, last_source_verification, stale_refs, needs_verification_refs}` | |
| `quality` | `{status, metrics, issues, recommended_maintenance, ...}` | From `GraphQualityReport`. |
| `verification_state` | `verified` \| `needs_verification` \| `verification_failed` \| `unverified` \| `unknown` | Derived from `source_refs`. |
| `fallbacks` | list[`SourceFallback`] | `{code, message, impact}`. Codes include `not_ingested`, `empty_result`, `source_unreachable`, `permission_denied`, `stale`, `not_supported`. |
| `open_conflicts` | list | Contested facts the agent should be aware of. |
| `recommended_next_actions` | list | E.g. `verify_entity`, `refresh_scope`, `resync_source_scope`, `expire_stale_facts`. |
| `errors` | list[`ResolutionError`] | Per-source errors with `recoverable` flag. |
| `meta` | `{provider, total_latency_ms, per_call_latency_ms{}, capabilities_used[], schema_version, cost{}}` | `meta.cost` (Phase 5) carries `resolve_ms`, `per_call_latency_ms`, and `synthesis: {model, input_tokens, output_tokens, total_tokens, latency_ms}` when the answer synthesizer ran. |
| `quality.drift` | `{status, signals{}, thresholds{}}` | Phase 5: explicit drift summary derived from `quality.metrics` so an agent can decide "verify before acting" from one field. `signals` exposes `stale_refs`, `needs_verification_refs`, `verification_failed_refs`, `source_access_gaps`, `missing_coverage`, `fallbacks`, `open_conflicts`. `status` mirrors `quality.status` (`good` \| `watch` \| `degraded` \| `unknown`). |
| `bundle` | object | Full raw `IntelligenceBundle` for callers that want it; everything else is derived from this. |

### Modes

- `fast` — default budget (~4 s), best-effort families, no mandatory enforcement.
- `balanced` — same budget; signals to the resolver to prefer richer families when cheap.
- `deep` — extends timeout to ≥ 8 s, doubles `max_items`, marks every activated family as mandatory so partial failures surface in `fallbacks`.
- `verify` — extends timeout to ≥ 6 s; if `source_policy=references_only`, escalates to `verify`.

### Source policies

Controls how aggressively the source resolver re-fetches:

- `references_only` — graph references; no source fetch (cheapest).
- `summary` — source headline + summary if cheap.
- `verify` — re-check the source ref still resolves; updates `verification_state`.
- `snippets` — small source snippets when materially helpful.
- `full_if_needed` — fetch full payloads when graph evidence is empty (most expensive).

---

## `context_search`

Narrow follow-up after `context_resolve`. Use when the entity or phrase is already known and you want a shape-controlled lookup, not a planner.

### Request

`pot_id`, `query`, `limit` (default 8), `node_labels` (CSV), `repo_name`, `source_description`, `include_invalidated` (bool), `as_of` (ISO).

### Response

`{ok, answer.summary, evidence[], source_refs[], coverage, freshness, fallbacks, recommended_next_actions}`. Same shape as `context_resolve` but only the families a narrow search produces.

---

## `context_record`

Generic write into project memory. Routes through the same reconciliation pipeline as ingested events.

### Request

| Field | Type | Notes |
|---|---|---|
| `pot_id` | str | Required. |
| `record_type` | enum | One of `decision`, `fix`, `bug_pattern`, `investigation`, `diagnostic_signal`, `preference`, `workflow`, `feature_note`, `service_note`, `runbook_note`, `integration_note`, `incident_summary`, `doc_reference`. Defined in `agent_context_port.CONTEXT_RECORD_TYPES`. |
| `summary` | str | Required. The headline fact. |
| `details` | str | Optional. Free-form body. |
| `source_refs` | list[str] | Optional. References back to source systems. |
| `confidence` | float | Default 0.7. |
| `visibility` | str | Default `project`. |
| `scope` | object | Optional. Same shape as `context_resolve` scope. |
| `idempotency_key` | str | Optional. Auto-derived from content if omitted. |
| `sync` | bool | Default false. When true, response is the applied result; when false, response is `queued`. |

### Response

When `sync=false`: `{ok, status: "queued", episode_uuid, event_id, job_id}`.
When `sync=true`: `{ok, status: "applied", episode_uuid, event_id, job_id}`.
On rejection: `{ok: false, status: "reconciliation_rejected", ...}`.

---

## `context_status`

Cheap pot/source health check. Returns the recommended recipe for an intent.

### Request

`pot_id`, optional `repo_name`, optional `source_refs` (CSV), optional `intent`.

### Response

Pot readiness, registered source connectors, last ingestion times, freshness gaps, the four-tool manifest from `context_port_manifest()`, the recommended recipe for the supplied intent (or `unknown`), and (Phase 3) the **registered reader manifest** under `readers`. Each reader entry exposes `family`, `description`, `intents`, `requires_scope`, `cost`, and `backend` so an agent can ask "what evidence families does this pot expose, and which scope fields do they need?" before issuing a `context_resolve` call.

---

## Recipes

Defined in `agent_context_port.CONTEXT_RESOLVE_RECIPES`. A recipe is a parameter preset for `context_resolve` — never a new tool.

| Intent | Mode | Source policy | When |
|---|---|---|---|
| `feature` | `fast` | `references_only` | Before feature work, behavior changes, or cross-repo implementation. |
| `debugging` | `fast` | `references_only` | Before investigating a bug, incident, failing workflow, alert, or flaky behavior. |
| `review` | `balanced` | `summary` | Before reviewing a PR or checking risky changes against project memory. |
| `operations` | `balanced` | `summary` | Before deployment, environment, runbook, alert, or production-impacting work. |
| `docs` | `fast` | `references_only` | When locating or validating project documentation and decision context. |
| `onboarding` | `fast` | `references_only` | When entering an unfamiliar pot, repo, service, or local workflow. |
| `planning` | `balanced` | `references_only` | Before roadmap, sprint, architecture, or cross-team coordination work. |
| `refactor` | `balanced` | `references_only` | Before restructuring code, migrating services, or cleaning up technical debt. |
| `test` | `fast` | `references_only` | Before writing, modifying, or reviewing tests and test infrastructure. |
| `security` | `balanced` | `verify` | Before security review, audit, vulnerability assessment, or hardening work. |
| `unknown` | `fast` | `references_only` | When the task does not match a more specific recipe. |

The default `include` set per intent is in `DEFAULT_INTENT_INCLUDES`.

---

## Usage rules (for agents and skills)

From `context_port_manifest()`:

1. Start non-trivial work with `context_status` or `context_resolve` for the active pot and task scope.
2. Use `intent` / `include` / `mode` / `source_policy` presets — do not ask for new context tools per use case.
3. Prefer `mode=fast` and `source_policy=references_only` first; escalate to `summary`, `verify`, `snippets`, or `deep` only when coverage or risk requires it.
4. Inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on graph memory.
5. If `quality.status` (or equivalently `quality.drift.status`) is `watch` or `degraded`, verify relevant facts against source truth before high-impact work.
6. Inspect `meta.cost` if you need to bound per-resolve LLM spend before issuing a follow-up.
7. Use `context_search` only for specific follow-up lookup when the needed entity or phrase is already known.
8. Use `context_record` when a durable decision, fix, workflow, preference, feature note, document reference, or incident summary should become reusable project memory.

---

## Current HTTP surface

The MCP tools above translate to HTTP routes. After Phase 4 there is exactly one ingestion path; agent / record / status / read all share the four-tool envelope shape.

| Purpose | Method + Path | Notes |
|---|---|---|
| Resolve / search | `POST /api/v1/context/query/context-graph` | One unified `ContextGraphQuery` endpoint with `goal` (`answer` \| `retrieve`) and `strategy` (`hybrid` \| `semantic` \| `auto`). |
| Record | `POST /api/v1/context/record` | Body matches `ContextRecordRequest`. Routes through `record_durable_context`. |
| Status | `POST /api/v1/context/status` | Body matches `ContextStatusRequest`. Routes through `report_status`. |
| Event submit | `POST /api/v1/context/events/reconcile` | Canonical inbound for connector and webhook events. Async by default. |
| Episode ingest | `POST /api/v1/context/ingest` | Raw episode write through `submit_raw_episode`; defaults to async, supports `sync=true` for inline apply. |
| Operator: hard reset | `POST /api/v1/context/...` (operator-scoped) | See router file for the operator surface. |

Routes prefixed `[operator]` in the router are not part of the agent contract. They exist for maintenance: hard reset, conflict resolve, edge reclassification.

### Authorization (Phase 5)

Every route funnels through one `PolicyPort.authorize` call before doing any work — see [`architecture.md` § Operability](./architecture.md#operability-and-observability). When a request is denied the response carries the policy `reason` taxonomy verbatim: `context_graph_disabled`, `reconciliation_disabled`, `agent_planner_disabled`, `reconciliation_agent_unavailable`, `unknown_pot`, `forbidden`, `maintenance_write_disabled`. The HTTP status code is dictated by the policy decision (`404` for `unknown_pot`, `403` for permission denials, `503` for capability gates, `400` for malformed resource/action). Clients can branch on the `reason` string instead of parsing free-text details.
