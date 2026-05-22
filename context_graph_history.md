# Context Graph Chat History

This file is the running notebook for this chat. We will update it as we inspect
or change the context graph system.

**Maintenance rule:** After every context-graph task (tests, wiring, PR work),
append or revise this file with what changed, verification commands, and what
is still open. Do not leave status only in chat.

## 2026-05-22

- Created this file at the start of the repo walkthrough.
- Initial discovery:
  - Runtime package: `app/src/context-engine/`
  - Monolith integration package: `app/modules/context_graph/`
  - Context graph docs: `docs/context-graph/`
  - Tests: `tests/unit/context_graph/`, `tests/integration-tests/context_graph/`,
    and `app/src/context-engine/tests/`
- Early shape:
  - `app/src/context-engine/` appears to be the standalone context engine with
    domain ports, application use cases, inbound adapters, outbound adapters,
    bootstrap/container wiring, benchmarks, and tests.
  - `app/modules/context_graph/` appears to be the Potpie monolith bridge:
    pots, pot sources, FastAPI routes, Celery queue bridge, sandbox
    provisioning, and context tools wiring.
  - Docs describe the target model as one ingestion path:
    source connector/event -> ingestion ledger -> batch/reconciliation ->
    validated plan -> Graphiti/Neo4j.

### Walkthrough: how context graph is wired

#### Package split

- `app/src/context-engine/` is the engine:
  - `domain/` defines canonical models, ontology, graph query models,
    reconciliation plans, ingestion event models, and source reference models.
  - `domain/ports/` defines boundaries such as `ContextGraphPort`,
    `ReconciliationAgentPort`, `IngestionSubmissionService`,
    `BatchRepositoryPort`, `ReconciliationLedgerPort`, `SourceConnectorPort`,
    `ContextReaderPort`, `PolicyPort`, and `TelemetryPort`.
  - `application/` orchestrates ports via services/use cases.
  - `adapters/inbound/` exposes HTTP, MCP, CLI, and worker entrypoints.
  - `adapters/outbound/` implements Graphiti/Neo4j/Postgres/connectors/readers/
    reconciliation/synthesis/policy/event-stream adapters.
  - `bootstrap/container.py` composes the engine.
- `app/modules/context_graph/` is the Potpie host bridge:
  - Defines pot/pot member/pot repo/pot source SQLAlchemy models.
  - Exposes pot tenancy APIs.
  - Builds a context-engine container with Potpie DB, auth, CodeProvider,
    GitHub/Linear connectors, Celery queue, sandbox tools, web tools, and
    user-scoped pot access.
  - Hosts the Celery tasks that process ingestion batches.

#### Container wiring

- `bootstrap/container.py::build_container(...)` creates:
  - `GraphitiEpisodicAdapter` for Graphiti/Neo4j writes and episodic memory.
  - `Neo4jStructuralAdapter` for direct structural reads.
  - `HybridGraphIntelligenceProvider` for bundled intelligence reads.
  - `ContextReaderRegistry` with readers:
    `semantic_search`, `change_history`, `timeline`, `owners`, `decisions`,
    `pr_review_context`, `pr_diff`, `project_graph`, `graph_overview`,
    `release_notes`.
  - `SourceConnectorRegistry`.
  - `ContextResolutionService`.
  - `GraphitiContextGraphAdapter`, the concrete `ContextGraphPort`.
  - Optional PydanticAI answer synthesizer/query agent depending on env.
  - Optional telemetry and event stream publisher.
- `app/modules/context_graph/wiring.py` adapts this to the monolith:
  - `build_container_for_session(db)` is worker/system scoped.
  - `build_container_for_user_session(db, user_id)` is request/user scoped and
    enforces pot access by returning `None` for inaccessible pots.
  - Registers GitHub and Linear connectors.
  - Adds sandbox, GitHub, Linear, and web tool surfaces to the reconciliation
    agent.
  - Sets `container.pot_source_listing` from `ContextGraphPotSource` rows.

#### Main ingestion path

Canonical path:

1. A caller submits an `IngestionSubmissionRequest`.
2. `DefaultIngestionSubmissionService.submit(...)` validates engine enabled,
   reconciliation agent present, pot exists, repo belongs to pot, and
   `source_id` is present.
3. It builds a `ContextEvent`.
4. `event_admission.admit_event(...)`:
   - appends/dedupes the event in `context_events` through
     `ReconciliationLedgerPort.append_event`;
   - marks it queued;
   - coalesces it into an open per-pot batch via
     `BatchRepositoryPort.upsert_open_batch_for_pot`;
   - either enqueues the batch immediately or leaves it pending when the pot
     is in `windowed` mode.
5. The queue backend calls a worker with `batch_id`.
6. `context_graph_process_batch` Celery task calls
   `context_graph_jobs.handle_process_batch(...)`.
7. `handle_process_batch` rebuilds a session-scoped container, claims the
   batch, then calls `process_batch(...)`.
8. `process_batch(...)`:
   - applies policy for `apply.write`;
   - loads batch events;
   - marks events processing;
   - opens per-event reconciliation run rows;
   - chunks large batches by `CONTEXT_ENGINE_MAX_CHUNK_EVENTS` (default 20);
   - calls `reconciliation_agent.run_batch(...)` for each chunk;
   - records work events, success/failure, and live stream status.
9. The Pydantic deep reconciliation agent uses read tools plus the
   `apply_graph_mutations` mutation tool.
10. `apply_graph_mutations` converts the LLM/tool plan into a
    `ReconciliationPlan`, stamps provenance from the source event, and calls
    `ContextGraphPort.apply_plan_async(...)`.
11. `GraphitiContextGraphAdapter.apply_plan_async(...)` calls
    `apply_reconciliation_plan_async(...)`.
12. `apply_reconciliation_plan_async(...)` validates the plan, writes episode
    drafts, then applies entity upserts, edge upserts, edge deletes, and
    invalidations through `EpisodicGraphPort`. In practice this is the
    Graphiti adapter using Graphiti's driver.

#### Ingestion triggers found

- HTTP `POST /api/v1/context/events/reconcile` in
  `adapters/inbound/http/api/v1/context/router.py`.
- HTTP `POST /api/v1/context/record`, backing durable context record writes.
- Agent tool `context_record` in
  `app/modules/intelligence/tools/context_tools/agent_context_tools.py`.
- Repo attach in `app/modules/context_graph/attach_repo_to_pot.py`:
  - upserts `context_graph_pot_repositories`;
  - mirrors into `context_graph_pot_sources`;
  - best-effort prewarms sandbox;
  - emits `repository.added` with `source_id=repo_added:{owner}/{repo}`.
- Source attach paths in `pot_sources_service.py` also emit attach/backfill
  events, e.g. Linear team attach.
- Raw episode HTTP `POST /api/v1/context/ingest` exists, but it is a lower-level
  Graphiti episode path rather than the normalized event/reconciliation path.

#### Queueing and batch maintenance

- Queue adapter: `app/modules/context_graph/celery_job_queue.py`.
- Worker tasks: `app/modules/context_graph/tasks.py`.
- Main task: `context_graph_process_batch(batch_id)` on queue
  `context-graph-etl`.
- Beat task: `context_graph_flush_windowed_batches` flushes pending windowed
  batches.
- Beat task: `context_graph_reap_stale_batches` marks stuck in-flight batches
  failed after the lease.

#### Read/query path

- HTTP `POST /api/v1/context/query/context-graph` calls
  `container.context_graph.query_async(body)`.
- `GraphitiContextGraphAdapter` routes:
  - `goal=ANSWER` through `resolve_context(...)` +
    `ContextResolutionService` + optional answer synthesizer.
  - `goal=INVESTIGATE` through optional query agent, otherwise answer fallback.
  - retrieve/timeline/aggregate/neighborhood style reads through
    `ContextReaderRegistry.execute(...)`.
- `ContextReaderRegistry` chooses readers from explicit `include`, query
  strategy, goal, and scope. It returns a single `ContextGraphResult` with
  per-reader leg metadata and fallbacks.
- Agent tools expose four user-facing tools:
  - `context_resolve`
  - `context_search`
  - `context_record`
  - `context_status`

#### Pot model

- Pot rows live in host tables:
  - `context_graph_pots`
  - `context_graph_pot_members`
  - `context_graph_pot_repositories`
  - `context_graph_pot_sources`
  - invitations/integrations tables around those.
- `SqlalchemyPotResolution` is broad/system scoped.
- `UserScopedContextGraphPotResolution` enforces tenant boundaries for user
  calls by hiding inaccessible pots.

### Test planning notes

User asked for a critical/exhaustive test plan for the context graph module,
assuming individual submodules mostly work. The plan should therefore focus on
cross-module contracts and failure modes:

- host bridge -> engine container wiring
- pot/repo/source attach -> normalized event submission
- event submission -> ledger/batch/queue semantics
- Celery job queue -> batch claim -> `process_batch`
- reconciliation agent mutation tool -> `ContextGraphPort.apply_plan_async`
- read APIs/tools -> `ContextGraphQuery`/reader registry/resolution envelopes
- policy/tenant boundaries across all user-facing paths
- idempotency/retry/windowing/stale-batch recovery
- no external services in normal unit/contract tests; use fakes for DB ports,
  Graphiti, Neo4j, Celery, source connectors, and agents.

### Independent code research (verifies + extends the walkthrough above)

Read directly from the source to ground the test plan in actual behaviour
(not just the file map). Highlights below are the load-bearing details that
will drive specific test choices.

#### Domain ports (real shape; 24 ports total under `domain/ports/`)

- `ContextGraphPort` (`domain/ports/context_graph.py`): unified read+write
  surface. Has BOTH `apply_plan` (sync) AND `apply_plan_async` — the agent
  tool uses the async variant to stay on the same event loop and avoid the
  documented "Future attached to a different loop" Neo4j cross-bind. Test
  plan should treat the sync path and the async path as distinct contracts.
- `BatchRepositoryPort` has more than the walkthrough listed: bulk
  `add_events_to_open_batch_for_pot`, `get_open_batch_id_for_pot`,
  `get_latest_batch_id_for_event`, `list_stale_in_flight_batches` (lease
  query), plus the per-state transition methods. Documents the
  `pending`→`claimed`→`running`→`done|failed` state machine and the
  in-flight semantics (events arriving while claimed/running open a NEW
  pending batch — fakes MUST replicate this).
- `ReconciliationLedgerPort` separates event lifecycle vs reconciliation
  run rows vs work events, plus bulk variants (`record_events_reconciled`,
  `record_events_failed`, `fail_inflight_events` — status-guarded for the
  reaper). All bulk ops are all-or-nothing. Fakes need both single + bulk.
- `ContextGraphJobQueuePort` is one method (`enqueue_batch`). Default
  `NoOpContextGraphJobQueue` ships in the port file for tests/CLI.
- `PotResolutionPort` has a `actor_scoped: bool` attribute that the policy
  adapter reads to decide whether per-actor pot authorization is enforced
  (see Policy below). This is the security seam.

#### Container composition (`bootstrap/container.py`)

- `ContextEngineContainer` is a dataclass with session-bound factories
  (`ledger(db)`, `reconciliation_ledger(db)`, `batch_repository(db)`,
  `ingestion_submission(db)`, …) — every adapter that needs a session is
  built **per call** from the live DB session, not stashed at construction
  time. Tests should NOT rely on adapter identity across calls.
- `build_container(...)` always installs a `GraphitiEpisodicAdapter` and
  `Neo4jStructuralAdapter` (Neo4j-bound) plus a default reader registry
  with 10 first-party readers. `_attach_reconciliation_context` wires the
  reconciliation agent to `ContextGraphReconciliationTools` + the context
  graph after the graph adapter is built — order matters.
- `_default_telemetry()` / `_default_event_stream_publisher()` fall back
  to NoOp when env vars are absent; both wrap the import in `try/except` so
  optional deps never break the engine.

#### Monolith bridge (`app/modules/context_graph/wiring.py`)

- Two container builders:
  - `build_container_for_session(db)` — system/worker scope.
    `SqlalchemyPotResolution` (NOT actor-scoped).
  - `build_container_for_user_session(db, user_id)` — request scope.
    `UserScopedContextGraphPotResolution(db, user_id)` with
    `actor_scoped = True`.
- Both share: `PotpieContextEngineSettings`, `_build_connector_registry`
  (GitHub + Linear, with `allow_unsigned=True` because the monolith verifies
  signatures at the HTTP ingress already), `try_pydantic_deep_reconciliation_agent`,
  and `_attach_agent_tools` (sandbox + github + linear + web).
- `_attach_agent_tools` is **best-effort per surface** — each tool family
  is wrapped in `try/except`, a failing import drops that surface only and
  the agent still runs with whatever loaded. Tests need to assert this:
  agent must remain usable when (e.g.) the sandbox module is unavailable.
- `_allowed_repos_for_pot` (inside `_attach_agent_tools`) gates the GitHub
  agent tools to the union of `owner/repo` rows attached to the pot —
  security review item C-5: a hijacked agent cannot pull a foreign repo
  through the shared org credential. **No test for this guard.**
- `SqlalchemyPotSourceListing` is attached to the container AFTER
  `build_container` returns. The container's `pot_source_listing` slot is
  `None` until then — order-sensitive.

#### Ingestion path (verified end-to-end)

`DefaultIngestionSubmissionService.submit` (app/src/context-engine/application/services/ingestion_submission_service.py):

1. Settings enabled check (raise `context_graph_disabled` if not).
2. Reconciliation agent must be wired (raise `no_reconciliation_agent`).
3. `pots.resolve_pot(request.pot_id)` — raise `unknown_pot_id` if None.
4. `resolve_write_repo(resolved, repo_name=explicit_repo)` chooses the
   write target. Explicit `repo_name` that's not in the pot raises
   `repo_not_in_pot`. No explicit + no primary repo falls back to a
   pot-level scope (provider from `event_scope_from_resolved_pot`).
5. `source_id` mandatory (raise if empty) — load-bearing for idempotency.
6. Build `ContextEvent` + `EventScope`, call `admit_event(...)`.
7. Duplicate? Return `EventReceipt(duplicate=True, status=ev.status)`
   refetched from the event store.
8. New? Return `EventReceipt(status="queued", job_id=batch_id)`. If
   `sync=True` or `wait=True`, block on `wait_for_terminal_ingestion_event`
   (default 300s timeout).

`admit_event(...)` (`application/services/event_admission.py`):

1. `reco_ledger.append_event(scope, event)` → `(event_id, inserted)`;
   dedupe is on scope + source_id.
2. Duplicate → return `EventAdmissionOutcome(inserted=False)` and bail.
3. `mark_event_queued(event_id)`.
4. `batches.upsert_open_batch_for_pot(pot_id, event_id)` → batch_id (the
   coalescing point — events arriving during a claimed batch open a new
   pending batch by contract).
5. Read mode from `ingestion_config.get(pot_id).mode` (default
   `immediate` if config port absent or read fails).
6. `windowed` → leave pending, return `enqueued=False`.
7. `immediate` → `jobs.enqueue_batch(batch_id)`, **swallow exceptions**
   (batch is durable; next event / flusher re-enqueues).

Note: `ingestion_event_store.get_event` vs `reco_ledger` are TWO different
ports. `submit` writes the event via the reco ledger but refetches the
duplicate event from the events store — fakes need both surfaces if the
test crosses the duplicate path.

#### Queue + Celery tasks (`app/modules/context_graph/`)

- `CeleryContextGraphJobQueue.enqueue_batch` does ONE thing:
  `context_graph_process_batch.delay(batch_id)`. Adapter for the port.
- `context_graph_process_batch` (Celery task) — bound, base
  `ContextGraphTask` (managed DB session), queue `context-graph-etl`,
  `autoretry_for=(Exception,)`, `max_retries=3` (env-overridable), backoff
  with jitter. Retries are for transient infra faults BEFORE
  `claim_batch_by_id` lands; agent failures convert to terminal `failed`
  inside `process_batch`, and a retry that lands after claim is a no-op
  (claim returns None).
- `context_graph_flush_windowed_batches` (beat) — every minute (per
  `celery_app.conf.beat_schedule`). Idempotent.
- `context_graph_reap_stale_batches` (beat) — lease default
  `CELERY_TASK_TIME_LIMIT + 900s` (15 min headroom). The lease MUST exceed
  the task time limit; this is a correctness invariant, not a tuning
  knob. **Worth a regression test.**

#### `process_batch` (`application/use_cases/process_batch.py`)

Bigger than the walkthrough suggests. Per-batch responsibilities:

- Policy gate: ONE `apply.write` check via `SYSTEM_ACTOR` per batch. Deny
  → mark batch failed, fail all pending events, publish failure status +
  end records, return early. Single policy boundary for every mutation
  inside the batch.
- Crash-resume substrate is `agent_execution_log` (durable append-only
  log), NOT the checkpoint store any more — `checkpoints` is kept as a
  param for ABI but is deleted in the body. Resume state seeds
  `already_done` event ids + the last seq so resumed runs don't redo work
  or collide on seq.
- Chunked execution: `CONTEXT_ENGINE_MAX_CHUNK_EVENTS` (default 20). One
  agent invocation per chunk, sequential. First chunk continues the prior
  message-history if resuming; later chunks share state via the graph.
- Two failure modes treated equivalently:
  - Agent **raises** (caught) → record failure trace on this chunk's
    runs, break the chunk loop.
  - Agent returns `outcome.ok=False` → same.
- Failure path: events from earlier successful chunks stay reconciled
  (`record_events_reconciled(aggregated_completed)`), failing-chunk events
  become `failed` (`record_events_failed`), batch becomes `failed`.
  Resume checkpoint is INTENTIONALLY NOT cleared so a retry resumes
  mid-batch with the durable completed set.
- Success path clears the resume checkpoint via `exec_log.clear`.
- Status/stream publishes are best-effort — every emit is wrapped so a
  Redis blip can never poison ingestion.
- One reconciliation run row per event is created up-front (`_start_runs_for_pending`)
  so the UI can render progress; on completion the agent's outcome trace
  is fanned to EVERY event's run row (duplicate by design — each event
  needs its own log for independent UI render).

#### Reconciliation agent (pydantic-deep, the live agent)

- Tool surface = `core_callables` (read x4 + `apply_graph_mutations` +
  `mark_events_processed` + `mark_event_processed` + `finish_batch`) plus
  `extra_callables` (github / linear / sandbox / web from `_attach_agent_tools`).
- `_enforce_playbook_tool_allowlist` filters extra tools to the union of
  the batch's playbook `tool_hints`. This is the **server-side prompt
  injection mitigation**: a hijacked agent cannot call a tool the
  event-kind was never authorized to use. **No test for this allowlist.**
- `apply_graph_mutations` has a **per-batch call cap**
  (`CONTEXT_ENGINE_MAX_APPLY_CALLS_PER_BATCH`, default in
  `_DEFAULT_MAX_APPLY_CALLS_PER_BATCH`). Exceeded → returns
  `{ok: False, error: "apply_call_cap_exceeded"}` and refuses further
  mutations. Runaway / prompt-injection guard. **No test.**
- Plan flow inside the tool: validate event_id is in batch → parse with
  `LlmReconciliationPlan.model_validate` → convert to domain
  `ReconciliationPlan` via `llm_plan_to_reconciliation_plan` (with
  `EventRef`) → build provenance from the event → call
  `apply_plan_async(plan, expected_pot_id=state.pot_id, provenance_context=prov)`.
  Each conversion stage can fail with a distinct error string.
- `CheckpointMiddleware` writes the durable resume state after every
  tool call via `_ExecutionLogCheckpointBridge`. Always on; a NoOp log
  makes it inert.
- Playbook-driven planner toggle: `playbooks_enable_planner(...)` flips
  the deep agent's todo/plan tools on for backfill seeds (single
  `*.added` event whose handling fans out into many enumerated
  artifacts). Normal live batches keep the planner off.

#### Read path / `ContextReaderRegistry`

- Family resolution is explicit `include` ∪ goal/strategy auto-pick:
  - `SEMANTIC` or `HYBRID` + non-empty `query` → adds `semantic_search`.
  - `NEIGHBORHOOD` (no explicit include) → adds `project_graph`.
  - `AGGREGATE` (no explicit include) → adds `graph_overview`.
  - `TIMELINE` + code-scoped (`file_path`/`function_name`/`pr_number`)
    → `change_history`; else → `timeline`.
- Unknown `include` token does NOT raise — recorded as fallback.
- Missing required scope (per `cap.requires_scope`) skips the reader
  and records `missing_scope` fallback.
- Reader exception → recorded as fallback, doesn't kill the request.
- Single reader → envelope `kind=family`. Multi → `kind="multi"`,
  `result` is `{family: result}`, meta carries leg metadata + fallbacks.
- `goal=ANSWER` and `goal=INVESTIGATE` are NOT readers — they're handled
  inside `GraphitiContextGraphAdapter` (resolve_context + synthesizer /
  optional query agent). The sync `query()` refuses these when a loop is
  already running — agent tools / FastAPI handlers must call
  `query_async()`. Worth a dedicated test on the dispatch contract.

#### Plan validation (`reconciliation_validation.py`)

- Hard checks: `event_ref.pot_id == expected_pot_id`,
  `MAX_EPISODES=32`, `MAX_GENERIC_ENTITY_UPSERTS=5000`,
  `MAX_GENERIC_EDGES=10000`, `MAX_INVALIDATIONS=2000`, no duplicate
  `entity_key`, ISO temporal property strings.
- Soft downgrade mode (`CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL=1` + non-strict):
  unknown labels dropped, no-canonical → Observation/ADR-Document
  fallback, lifecycle out of catalog → "unknown", unknown edge type →
  `RELATED_TO` (confidence 0.3), edge endpoint type mismatch → edge
  dropped, missing temporal edge props backfilled from episode
  reference_time. Every downgrade is recorded in
  `plan.ontology_downgrades` and (when room) attached as QualityIssue
  entities.
- Evidence warning: material plans (≥ 3 mutations) without evidence and
  without explicit `confidence` get a non-blocking warning appended.

#### Tenant boundary (HARD security contract — `DefaultPolicyAdapter`)

Three independent gates can allow a pot-scoped action:

1. Resolver `actor_scoped = True` (Potpie's `UserScopedContextGraphPotResolution`).
2. Actor surface ∈ {`system`, `webhook`} AND `auth_method` ∈
   {`system`, `webhook_signature`} (server-stamped only).
3. `CONTEXT_ENGINE_ALLOW_NO_AUTH=1` (dev-only).

Otherwise → 403. Crucially, `X-Potpie-Client` header can only assert
non-privileged surfaces (`cli`/`mcp`/`http`); `system`/`webhook` are
server-stamped via internal jobs / signature-verified webhook handlers.

#### Stale-batch reaper

- Per-stale-batch ordering: events first (via `fail_inflight_events`),
  THEN the batch (`mark_batch_failed`). A crash between the two writes
  leaves the batch in-flight and reapable again next tick — never
  ends in "batch failed but events still processing".
- `fail_inflight_events` is status-guarded so it doesn't clobber events
  a partially-completed batch already drove to `reconciled`.

#### `attach_repo_to_pot` (`app/modules/context_graph/attach_repo_to_pot.py`)

- Single idempotent verb wrapping (1) repo row upsert, (2) source mirror,
  (3) bootstrap `repository.added` event submit.
- Idempotency carve-out: a re-attach where the source mirror is missing
  (the repo was deleted via the Sources tab, orphaning the repo row) is
  treated as a fresh attach — re-emits the bootstrap event so ingestion
  re-queues. `already_attached=False` for that case.
- **SSRF guard**: `provider_host` must be in `_allowed_provider_hosts()`
  (security review M-2 — a pot owner could otherwise register an internal
  host and make the sandbox clone/fetch target it carrying the injected
  auth token). `github.com` is the only host by default; GHE hosts via
  `CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS`. **No test.**
- Bootstrap event emit is best-effort — submit failures (container build,
  context-engine import) are logged and swallowed. The DB-side attach
  stays successful.

### Existing test coverage (what NOT to re-cover)

Engine-side (`app/src/context-engine/tests/`): 80+ unit tests, 9
integration tests. Heavy coverage on:

- `test_event_admission.py` (3): insert + enqueue, dedupe, enqueue
  failure tolerance.
- `test_windowed_admission.py` (12): mode switching, flush
  ready/skipped/failure, force-flush, listing.
- `test_process_batch.py` (8): happy path, skip processed, empty/done
  batch noop, agent exception, run rows + work events fan, run failure
  trace, resume with prior messages.
- `test_process_batch_chunking.py` (9): chunk math, env override,
  per-chunk fresh history, mid-batch chunk failure preserves prior
  chunk's completions, per-chunk work events fan.
- `test_process_batch_streaming.py` (5): stream record emission.
- `test_policy_port.py` (18): every action gate, including the three
  tenant-boundary scenarios (wide resolver denied / dev escape / server-
  trusted actor / spoofed HTTP actor on wide resolver).
- `test_context_reader_registry.py` (15): register/lookup, explicit
  include, scope fallback, strategy auto-pick, goal auto-pick (incl.
  TIMELINE+code-scoped vs actor-scoped), multi-merge, exception →
  fallback, manifest, third-reader smoke.
- `test_reap_stale_batches.py` (4): no-op, ordering, lease passthrough,
  partial-failure resilience.
- `test_reconciliation_validation_edge_cases.py` (13): plan validation
  paths.
- `test_pydantic_deep_agent_batch.py` (5) + `test_pydantic_deep_agent_run_batch.py`
  (integration, 1): agent-level behavior.
- `test_hard_reset_pot.py` (4): pot reset.
- `test_batch_retry_endpoint.py` (integration, 8): bulk retry route.
- `test_wait_ingestion_event.py` (2): sync waiter.
- `test_queue_factory.py` (7): backend selection.
- Many ontology / extractor / extraction-edge / canonical / event
  playbook unit tests.

Monolith-side (`tests/unit/context_graph/`): mostly per-module unit
coverage — sources service, sandbox tools, attach, archive, pot member
roles, intelligence signals/policy, ingestion db status, extraction
edges, episode formatters. Less integration than the engine side.

### Test-plan-relevant gaps (the ones the new plan should cover)

These are the cross-module / contract-level holes I could not find tests
for. The walkthrough-stage test plan in the section above should be
narrowed to these, plus the items where existing coverage is per-unit but
the cross-module wiring is uncovered.

1. **HTTP `/events/reconcile` → admit → enqueue → claim → process_batch
   → apply_plan_async** end-to-end with all in-process fakes (no Celery,
   no Neo4j) — there's no test pinning the whole loop down on the
   reconcile entry. The closest are `test_event_admission` (admit only)
   and `test_process_batch` (post-claim only).
2. **`CeleryContextGraphJobQueue.enqueue_batch` adapter contract** —
   that the port impl actually calls `context_graph_process_batch.delay`
   with the batch_id. Trivial but uncovered.
3. **`build_container_for_session` vs `build_container_for_user_session`
   container shape** — `actor_scoped` flag on `pots`, `pot_source_listing`
   set, connectors registered, no `app.*` import cycles. The two builders
   are the seam between monolith and engine.
4. **Agent `apply_graph_mutations` tool routing** —
   `unknown_event_id`, `invalid_plan`, `plan_conversion_failed`,
   `apply_failed`, plus the `apply_call_cap_exceeded` cap. Direct test
   on the tool's input → port dispatch contract (mocked
   `ContextGraphPort.apply_plan_async`).
5. **`_enforce_playbook_tool_allowlist`** — ~~no existing test~~ **CGT-2 done:**
   `test_deep_agent_containment.py` (union, default playbook excluded).
6. **`_allowed_repos_for_pot`** — ~~no existing test~~ **CGT-2 done:**
   `test_github_agent_tools_repo_binding.py` + `test_wiring_github_repo_binding.py`.
7. **`attach_repo_to_pot` SSRF guard** — `provider_host` outside
   `CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS` raises; valid GHE host
   passes; deleted-then-re-added repo re-emits bootstrap event with
   `already_attached=False`. No existing test.
8. **Failure-path resume invariant in `process_batch`** — checkpoint
   is preserved on agent exception / `ok=False`, partially-completed
   chunks credited as reconciled, retry resumes mid-batch. Existing
   `test_process_batch` covers run failure but not the resume-after-fail
   invariant.
9. **Reader registry async dispatch contract** — `query()` (sync) must
   refuse ANSWER/INVESTIGATE inside a running loop; `query_async()` must
   honor them. Catches the documented Neo4j cross-loop crash.
10. **Reaper lease invariant regression** — lease must exceed
    `CELERY_TASK_TIME_LIMIT`. A small env-mocked test would prevent a
    config drift from creating a reaper-vs-live-worker race.
11. **`UserScopedContextGraphPotResolution` access checks** —
    member vs legacy owner row, archived pot returns None,
    `find_pots_for_repo` honors the per-user filter. Test plan
    should include the row-level access path, not just the policy gate.
12. **Per-pot ingestion config endpoints (`/pots/{id}/ingestion-config`,
    `/ingest/flush`)** — bridge from policy + ingestion_config port to
    route response. Integration shape tests already exist for the
    config port; missing is the route-layer wiring (mode validation
    via Literal, force-flush no-pending case, force-flush enqueue
    failure tolerance).
13. **`record_durable_context` flow** — converts a `ContextRecordPayload`
    into an `IngestionSubmissionRequest`; the engine-side flow is
    covered, but the route → `record_durable_context` → submission
    plumbing (`scope` packing, idempotency_key, occurred_at) is not.
14. **Stream / NDJSON shape on `GET /events/{event_id}/stream`** —
    `replay_and_tail` cursor-resume + idle-timeout, transient `end`
    record when no batch yet, error → terminal `end` record. Existing
    `test_process_batch_streaming` tests the publish side, not the
    replay/tail consumer.

These 14 are the natural backbone of the cross-module test plan. The
mechanical per-module units already exist for the rest.

**Status:** Independent research complete; ready to draft the actual
test plan against this gap list.

### Testing policy update

User clarified an important constraint:

- New context graph tests should live with the context graph module, not be
  scattered into the root `tests/` tree unless they are explicitly testing
  monolith-only host integration.
- The context graph suite should be independently runnable through CI. Treat
  `app/src/context-engine/tests/` as the primary home for engine tests, with a
  dedicated CI command/job that can run without unrelated Potpie tests.
- Do not add redundant tests for behavior already covered in root `tests/` or
  existing context-engine tests. New work should target the uncovered
  cross-module contracts listed above.
- For host bridge behavior that necessarily depends on `app/modules/context_graph/`
  (pot rows, repo attach, Celery adapter, user-scoped Potpie wiring), prefer a
  small dedicated module-level suite and keep it separately selectable in CI
  instead of mixing it into unrelated backend tests.
- Normal CI should use fakes/mocks for external systems: no live GitHub,
  Linear, Graphiti, Neo4j, Redis, Celery worker, or LLM dependency for the
  default context graph contract suite.

Implication for the plan:

1. Engine-owned tests go under `app/src/context-engine/tests/`.
2. Potpie host bridge tests should be isolated to the context graph module's
   existing test area and tagged/commanded so CI can run them as a context graph
   job.
3. Before adding any test, check existing root `tests/unit/context_graph/` and
   `app/src/context-engine/tests/` coverage and only add tests for a genuine
   uncovered contract.

### Two-day Linear-ready execution plan

Goal: add independent, non-redundant context graph contract coverage in two
days. Scope is not reduced; work is split so multiple engineers can execute in
parallel. Default tests must use fakes/mocks and avoid live GitHub, Linear,
Graphiti, Neo4j, Redis worker, Celery worker, or LLM calls.

#### Day 1 — Security, contracts, and CI skeleton

**Issue CGT-1 — Create independent context graph CI command/job**

- Owner: Infra/test harness.
- Scope:
  - Add or document a dedicated command for `app/src/context-engine/tests/`.
  - Add or document a dedicated command for Potpie host bridge context-graph
    tests only.
  - Ensure both commands are independently runnable without the whole backend
    suite.
  - Ensure default job excludes live/external-service tests.
- Acceptance:
  - CI can run context-engine tests independently.
  - CI can run host bridge context-graph tests independently.
  - Commands are documented in the PR description and/or repo test docs.

**Issue CGT-2 — Security regression tests: tool and repo access guards**

- Owner: Engine/reconciliation.
- Scope:
  - Add tests for `_enforce_playbook_tool_allowlist`: extra tools are filtered
    to the union of active playbook `tool_hints`; unauthorized extra tools are
    not callable.
  - Add tests for GitHub agent-tool repo binding: `_allowed_repos_for_pot`
    only permits repos attached to the pot and rejects foreign repos.
  - Keep tests fake-only; no GitHub API.
- Acceptance:
  - Prompt-injection tool expansion is pinned by tests.
  - Cross-pot/private-repo read escape is pinned by tests.

**Issue CGT-3 — Security regression tests: pot access and SSRF guard**

- Owner: Host bridge.
- Scope:
  - Add `attach_repo_to_pot` tests for `provider_host` allowlist:
    default allows `github.com`, rejects unknown/internal hosts, env allows
    configured GitHub Enterprise host.
  - Add row-level tests for `UserScopedContextGraphPotResolution`:
    member access, legacy owner access, archived pot hidden,
    `find_pots_for_repo` filters by user.
- Acceptance:
  - SSRF guard is covered.
  - Pot tenant-boundary behavior is covered at resolver level, not only policy
    level.

**Issue CGT-4 — Container and queue adapter contract tests**

- Owner: Host bridge / engine boundary.
- Scope:
  - Test `build_container_for_session(db)` vs
    `build_container_for_user_session(db, user_id)` container shape:
    actor-scoped resolver flag, connector registry contains GitHub/Linear,
    `pot_source_listing` is attached, jobs port is present.
  - Test `_attach_agent_tools` is best-effort per tool surface: one failing
    surface does not prevent other surfaces / agent from being usable.
  - Test `CeleryContextGraphJobQueue.enqueue_batch(batch_id)` calls
    `context_graph_process_batch.delay(batch_id)`.
- Acceptance:
  - Monolith-to-engine wiring is pinned without starting Celery or connecting
    external services.

**Issue CGT-5 — Reconciliation mutation tool contract tests**

- Owner: Engine/reconciliation.
- Scope:
  - Directly test `apply_graph_mutations` behavior with fake
    `ContextGraphPort.apply_plan_async`.
  - Cover `unknown_event_id`, `invalid_plan`, `plan_conversion_failed`,
    `apply_failed`, success mutation-count response, and
    `apply_call_cap_exceeded`.
  - Assert provenance/expected pot id are passed to apply.
- Acceptance:
  - Agent mutation tool input -> domain plan -> graph apply contract is pinned.
  - Runaway apply cap is covered.

#### Day 2 — Ingestion seam, read/API surface, streaming/operability

**Issue CGT-6 — End-to-end fake ingestion contract**

- Owner: Engine/application.
- Scope:
  - Build a fake-port test for:
    `IngestionSubmissionRequest` -> `DefaultIngestionSubmissionService.submit`
    -> `admit_event` -> batch enqueue -> `handle_process_batch`
    -> fake agent -> fake `apply_plan_async`.
  - No FastAPI, no Celery worker, no Neo4j. Use in-process fakes.
  - Cover duplicate event does not enqueue/process again.
  - Cover immediate vs windowed mode expectations where this flow touches
    admission.
- Acceptance:
  - The canonical event pipeline is pinned end-to-end at contract level.
  - The test proves rows/ports hand the correct shapes across boundaries.

**Issue CGT-7 — Failure and resume invariants**

- Owner: Engine/application.
- Scope:
  - Add focused `process_batch` tests for failure-path resume:
    checkpoint/execution-log state is preserved on agent exception and
    `ok=False`.
  - Partially-completed chunks are credited as reconciled; remaining events
    fail; retry resumes without redoing completed events.
  - Add regression for reaper lease invariant:
    stale-batch lease default is `CELERY_TASK_TIME_LIMIT + 900`.
- Acceptance:
  - Crash/retry behavior is pinned.
  - Reaper cannot race a live worker due to config drift.

**Issue CGT-8 — Read dispatch and async contract tests**

- Owner: Engine/read path.
- Scope:
  - Test `GraphitiContextGraphAdapter.query()` refuses ANSWER/INVESTIGATE
    while an event loop is running.
  - Test `query_async()` honors ANSWER/INVESTIGATE dispatch.
  - Add route/tool contract tests only where not already covered:
    `context_search` normalized envelope, unsupported include fallback,
    missing scope fallback if existing registry tests do not already cover the
    exact external envelope.
- Acceptance:
  - The documented event-loop safety contract is covered.
  - Agent-facing read envelope remains stable.

**Issue CGT-9 — Record and ingestion-config route plumbing**

- Owner: HTTP/API surface.
- Scope:
  - Test `POST /record` route plumbing into `record_durable_context`:
    scope packing, `idempotency_key`, `occurred_at`, actor/source channel,
    queued fallback shape.
  - Test ingestion config route wiring:
    mode validation, get/update response, force-flush no-pending case,
    enqueue failure tolerance if surfaced at route level.
  - Use fake container/session ports; no DB service unless absolutely needed.
- Acceptance:
  - User-facing write/config APIs are pinned at transport-contract level.

**Issue CGT-10 — Event stream / NDJSON consumer contract**

- Owner: HTTP/API surface.
- Scope:
  - Test event activity stream shape:
    replay/tail cursor behavior, transient `end` record when no batch exists,
    idle-timeout, and error -> terminal `end` record.
  - This complements existing publish-side tests and should not duplicate
    `test_process_batch_streaming`.
- Acceptance:
  - Frontend-facing stream protocol is pinned.

**Issue CGT-11 — Final dedupe pass and coverage audit**

- Owner: Any / reviewer.
- Scope:
  - Before merging, compare added tests against:
    `app/src/context-engine/tests/` and root `tests/unit/context_graph/`.
  - Remove or collapse redundant tests.
  - Ensure all new tests live in the agreed context graph locations.
  - Run the independent CI commands locally and record results.
- Acceptance:
  - No duplicate low-value coverage.
  - All new tests are independently runnable.
  - Residual gaps, if any, are explicitly documented.

#### Suggested two-day schedule

Day 1 morning:

- CGT-1, CGT-2, CGT-3 start in parallel.
- CGT-4 starts once CI command assumptions are clear.

Day 1 afternoon:

- Finish CGT-2 / CGT-3 security tests.
- Finish CGT-4.
- Start CGT-5.

Day 2 morning:

- Finish CGT-5.
- CGT-6, CGT-7, CGT-8 start in parallel.

Day 2 afternoon:

- CGT-9 and CGT-10.
- CGT-11 final dedupe, command runs, documentation, PR polish.

#### Linear grouping

- Epic: Context Graph Independent Test Suite
- Milestone: Two-day hardening pass
- Labels:
  - `context-graph`
  - `tests`
  - `ci`
  - `security`
  - `contract-tests`
- Priority:
  - P0: CGT-1, CGT-2, CGT-3, CGT-5, CGT-6
  - P1: CGT-4, CGT-7, CGT-8, CGT-9
  - P2: CGT-10, CGT-11

### PR #792 follow-up status

PR: `potpie-ai/potpie#792` / branch `test/context-graph-ci-wiring`.

Completed work:

- Initial PR review found `--context-graph-only` was not green. Fixed the
  runner by explicitly excluding known stale context-engine tests and
  correcting the engine test `conftest.py` comment.
- Verified after first fix:
  `uv run python scripts/run_tests.py --context-graph-only`
  -> `1262 passed, 3 deselected`.
- Group A completed: deleted three stale duplicate host-side tests because the
  engine copies already exist:
  - `tests/unit/context_graph/test_extraction_edges.py`
  - `tests/unit/context_graph/test_graph_quality.py`
  - `tests/unit/context_graph/test_ontology_classifier.py`
- Group B completed: moved engine-owned tests into the engine tree:
  - `tests/unit/context_graph/test_reconciliation_flags.py`
    -> `app/src/context-engine/tests/unit/test_reconciliation_flags.py`
  - `tests/unit/context_graph/test_sandbox_git_tools.py`
    -> `app/src/context-engine/tests/unit/test_sandbox_git_tools.py`
- Group B drift fixes completed:
  - Updated `ontology_soft_fail_enabled` default expectations to match current
    engine behavior.
  - Relaxed sandbox git command assertions so they tolerate the current git
    hardening command prefix/shape.
- Group C completed: fixed the event-loop-dependent
  `tests/unit/context_graph/test_archive_pot_cleanup.py` tests. The tests now
  use deterministic `asyncio.run(...)` instead of relying on
  `asyncio.get_event_loop().run_until_complete(...)`, which only passed when a
  previous test polluted the main thread with an event loop.

Verification (historical mid-PR):

- Focused:
  `uv run pytest app/src/context-engine/tests/unit/test_reconciliation_flags.py app/src/context-engine/tests/unit/test_sandbox_git_tools.py tests/unit/context_graph/test_archive_pot_cleanup.py -q`
  -> `68 passed`.
- Full context graph command (before post-review fixes):
  -> `1347 passed, 2 deselected`.

Pushed commits (branch `test/context-graph-ci-wiring`):

- `034670e0` test(context-graph): wire engine tests into CI; add --context-graph-only
- `554ca25a` test(context-graph): keep dedicated suite green
- `637b747d` test(context-graph): include context_ingest in agent manifest assertion
- `424fc15e` test(context-graph): move engine tests out of host suite
- `25a9382a` test(context-graph): fix stale tests and drop runner workarounds
- `cec32230` test(context-graph): complete CGT-2 security regression coverage

**Latest verification (2026-05-22, after post-review fixes + CGT-2):**

```bash
uv run python scripts/run_tests.py --context-graph-only
# -> 1361 passed, 0 deselected, exit 0
```

Post-review fixes on the same branch:

- Removed runner `--deselect` entries and host `--ignore` for wiring tests (no
  longer needed).
- Fixed `test_wiring_sandbox_tools.py` for `_attach_agent_tools`.
- Fixed `test_agent_installer.py` Claude bundle expectations (commands removed
  from template).
- PR still **OPEN** on GitHub (`potpie-ai/potpie#792`); mergeable.

Current note: unrelated untracked files remain in the worktree and were left
untouched.

## 2026-05-22 — CGT-1 landed (CI wiring)

Made `app/src/context-engine/tests/` discoverable through the root runner and
added a context-graph-only mode.

**Changes:**

- `app/src/context-engine/tests/conftest.py` — new. Auto-marks tests under
  `tests/unit/` as `unit` and `tests/integration/` as `integration` by path,
  so the root runner's `-m unit` / `-m integration` filters work without
  touching every engine test file (most lacked `pytestmark`).
- `scripts/run_tests.py` — added `CONTEXT_ENGINE_UNIT_TESTS_DIR` and
  `CONTEXT_ENGINE_INTEGRATION_TESTS_DIR`. Both included in `--unit-only`,
  `--integration-only`, and the full run. Added new `--context-graph-only`
  flag that runs engine + host-bridge tests in one pytest invocation.
- `Makefile` — added `make test-context-graph`.

**Commands now available:**

```bash
make test-context-graph                                  # engine + host bridge
uv run python scripts/run_tests.py --context-graph-only
uv run python scripts/run_tests.py --unit-only           # now includes engine unit
uv run python scripts/run_tests.py --integration-only    # now includes engine integration
```

**First wiring run (historical):** `--context-graph-only` initially collected
1556 tests with 61 stale host-side assertion failures. Those were cleared on
PR #792 (move/delete/fix), not by ignoring the whole host tree.

**Still excluded via `--ignore` in `scripts/run_tests.py` (CGT-11 follow-up):**

1. `benchmarks/test_benchmark_evaluator.py` — broken `benchmarks.evaluator` import.
2. `benchmarks/test_benchmark_dataset.py` — legacy fixture layout.
3. `test_edge_collapse_golden.py` — missing `tests/fixtures/edge_collapse_golden.json`.
4. `test_linear_issue_plan.py`, `test_linear_issue_resolver.py`,
   `test_linear_webhook_normalize.py` — missing `tests/data/linear/*.json`.

**Resolved since first wiring (no longer ignored / deselected):**

- `test_wiring_sandbox_tools.py` — fixed for `_attach_agent_tools`.
- Stale host duplicates (`test_extraction_edges`, `test_graph_quality`,
  `test_ontology_classifier`) — deleted.
- `test_reconciliation_flags`, `test_sandbox_git_tools` — moved to engine tree.
- Runner `--deselect` for agent installer / ontology — removed (tests green).

**CGT-1 gaps still open:**

- No GitHub Actions job yet that runs `make test-context-graph` (local/Makefile only).
- No separate engine-only vs host-only CLI flags (combined `--context-graph-only` only).
- `app/modules/context_graph/README.md` still says root `tests/unit/context_graph/`
  was removed — partially true for domain duplicates, but host bridge tests remain.

**Status:** CGT-1 done (local green baseline). CGT-2 done (see below). **Next: CGT-3.**

## 2026-05-22 — CGT-2 landed (security regression: tool + repo guards)

**Scope (Linear CGT-2):** Pin playbook `tool_hints` allowlist and GitHub repo
binding so a prompt-injected agent cannot expand tools or read foreign repos
through the shared org credential (review C-5).

**Tests added / strengthened:**

| Area | File | What it pins |
|------|------|----------------|
| Playbook allowlist | `app/src/context-engine/tests/unit/test_deep_agent_containment.py` | Drops undeclared tools for `pull_request/merged`; unions hints across batch events; default playbook hints are **not** an auth boundary |
| GitHub tools (adapter) | `app/src/context-engine/tests/unit/test_github_agent_tools_repo_binding.py` | (pre-existing) `unknown_repo`, case-insensitive match, fail-closed when unwired |
| GitHub tools (wiring) | `tests/unit/context_graph/test_wiring_github_repo_binding.py` | `_attach_agent_tools` passes `allowed_repos_for_pot` from pot repo rows; foreign repo blocked before `source_for_repo` |

**Verification:**

```bash
uv run pytest \
  app/src/context-engine/tests/unit/test_deep_agent_containment.py \
  app/src/context-engine/tests/unit/test_github_agent_tools_repo_binding.py \
  tests/unit/context_graph/test_wiring_github_repo_binding.py -q
# -> 15 passed

uv run python scripts/run_tests.py --context-graph-only
# -> 1361 passed, exit 0
```

**Commit:** `cec32230` on `test/context-graph-ci-wiring`.

**Status:** CGT-2 acceptance criteria met. **Next recommended: CGT-3** (SSRF on
`attach_repo_to_pot`, `UserScopedContextGraphPotResolution` row-level access).

## Current task backlog (CGT plan)

| Ticket | Priority | Status |
|--------|----------|--------|
| CGT-1 | P0 | Done (local); CI workflow + split commands still open |
| CGT-2 | P0 | **Done** |
| CGT-3 | P0 | Not started — SSRF + `UserScopedContextGraphPotResolution` |
| CGT-4 | P1 | Partial — `test_wiring_sandbox_tools` only |
| CGT-5 | P0 | Not started — `apply_graph_mutations` contract |
| CGT-6 | P0 | Not started — fake E2E ingestion (gap #1) |
| CGT-7 | P1 | Not started — `process_batch` resume + reaper lease |
| CGT-8 | P1 | Not started — `query` / `query_async` ANSWER/INVESTIGATE |
| CGT-9 | P1 | Not started — `/record` + ingestion-config routes |
| CGT-10 | P2 | Not started — event stream NDJSON consumer |
| CGT-11 | P2 | Partial — PR dedupe; 6 engine `--ignore`s remain |

Cross-module gaps **#1–#14** (research section) still map to CGT-3–CGT-10 above;
see table in «Test-plan-relevant gaps».

## 2026-05-22 — PR #792 review follow-up (repository bootstrap allowlist)

Reviewed the updated PR #792 branch (`test/context-graph-ci-wiring` at
`ff464a03`) against the context-graph history and recent CGT-2 changes.

**Finding fixed:** `github/repository/added` playbook text tells the agent to
hydrate backfilled PRs with PR details plus commits/review/issue comments where
useful, but the new hard `tool_hints` allowlist only kept the basic list/get
tools. That meant bootstrap backfill could silently lose
`github_get_pull_request_commits`,
`github_get_pull_request_review_comments`, and
`github_get_pull_request_issue_comments`.

**Changes made:**

- `app/src/context-engine/domain/event_playbooks.py` — added the three PR
  hydration tools to the repository bootstrap playbook allowlist.
- `app/src/context-engine/tests/unit/test_deep_agent_containment.py` — added a
  regression test proving `repository.added` keeps the expected GitHub
  hydration tools while still dropping unrelated Linear tools.

**Verification:**

```bash
uv run pytest app/src/context-engine/tests/unit/test_deep_agent_containment.py -q
# -> 8 passed

uv run python scripts/run_tests.py --context-graph-only
# -> 1362 passed, 8 warnings, exit 0
```

**Status:** ready to commit/push on top of PR #792.
