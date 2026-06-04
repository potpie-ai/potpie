# Context Engine — end-to-end integration test surface

What the live, end-to-end integration suite (`test_e2e_topology.py`) exercises
against a **live Neo4j** (`.env` → `NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD`),
post minimal-topology-ontology + Graphiti removal.

The suite is **deterministic** — no LLM calls. Every test creates a uniquely
keyed pot, writes through a real adapter, asserts via direct Cypher reads of the
canonical store, and resets the pot on teardown. If Neo4j is unreachable the
whole module skips.

## Environment & wiring
- **Settings:** `EnvContextEngineSettings` reads `NEO4J_*` + `CONTEXT_GRAPH_ENABLED`. `conftest.py` loads them from the repo `.env` (without overriding already-set vars) and skips the module if bolt isn't reachable.
- **Container:** `build_container(settings=…, pots=ExplicitPotResolution({pot: repo}))` wires `episodic` (`Neo4jCanonicalGraphAdapter` — the canonical writer/reader, no Graphiti) and `context_graph` (`ContextGraphService`) over the single `ReadOrchestrator` (the 4 P9 readers — `coding_preferences`/`infra_topology`/`timeline`/`prior_bugs` — over `Neo4jClaimQueryStore`). The legacy structural adapter + 6 structural readers were removed with Graphiti.
- **Capability:** `container.episodic.enabled` / `context_graph` reflect live Neo4j availability.

## Pot lifecycle (host-managed; engine resolves)
There is no engine-side pot *create* use case — pots are host-managed. The engine consumes a `PotResolutionPort`:
- `resolve_pot(pot_id) -> ResolvedPot | None`, `known_pot_ids()`, `find_pots_for_repo(RepoRef)`, `list_pot_repos(pot_id)`, `get_repo_in_pot(...)`.
- Tested via `ExplicitPotResolution` (a `pot_id -> repo` map) — resolve, list repos, reverse-lookup by repo, unknown-pot → `None`.

## Ingestion — deterministic paths (covered)
1. **Config/topology scanner path** — `ScanWorkingTreeUseCase(scanner_registry, canonical_writer=container.episodic)` over a synthetic working tree:
   - `KubernetesManifestScanner` → `Service DEPLOYED_TO Environment` (env stamped on edge).
   - `CodeownersScanner` → `OWNED_BY` (Service/Repository → Person/Team).
2. **Validated reconciliation-plan path** — `context_graph.apply_plan_async(ReconciliationPlan)` writes the structured topology (`DEFINED_IN`, `DEPENDS_ON`, `USES`, `HOSTED_ON`, `MEMBER_OF`). This is the apply side of the agent pipeline, tested without the agent.
3. **Writer invariants** — off-catalog predicates are rejected; `OWNED_BY` (singleton) supersedes a prior owner on a new deterministic claim.

## Ingestion — real Postgres + real LLM (`test_e2e_pipeline.py`)
- **Real Postgres** — `conftest.pg_test_db` creates a throwaway database *inside the configured Postgres instance* (`POSTGRES_SERVER`/`DATABASE_URL`), builds the schema via `Base.metadata.create_all`, and drops it on teardown. Event submission + batching round-trips through it (`ingestion_submission.submit` → `context_events`/batch rows → `batch_repository`).
- **Real LLM reconciliation** — the full `submit → claim_batch_by_id → process_batch(real PydanticDeepReconciliationAgent) → apply_plan → graph` pipeline, driven in-process (`NoOpContextGraphJobQueue`, no Celery). Skips if no LLM key.
- These run only when Neo4j **and** Postgres **and** an LLM key are available; otherwise they skip.

## Query — deterministic paths (covered)
The structural read stack was removed with Graphiti, so topology is asserted by
reading the canonical store directly (no reader indirection):
- `_count_entities(pot_id)` — direct Cypher count of `:Entity` nodes in the pot partition (replaces the old `structural.get_graph_overview(...)["totals"]`).
- `_label_counts(pot_id)` — direct Cypher per-label counts (services, environments, teams, …).
- Direct Cypher reads of the `:RELATES_TO {name}` edge for exact-shape assertions (env/strength/fact props, singleton liveness, and the bitemporal point-in-time owner selection via `_live_owner_as_of`).

## Query — real LLM (`test_e2e_pipeline.py`)
- `goal=ANSWER` — the real `PydanticAIAnswerSynthesizer` answers a query over deterministically-seeded topology (`query_async`). Skips without an LLM key.
- `goal=INVESTIGATE` (agentic loop) — available via the same wiring; not asserted here to keep runtime/cost bounded.
- The Neo4j-backed `ClaimQueryPort` (`Neo4jClaimQueryStore`) is now wired into the `ReadOrchestrator`, so the P9 readers (`infra_topology`/`coding_preferences`/`timeline`/`prior_bugs`) can read live; their routing/ranking is unit-tested by `test_read_orchestrator.py` + `test_p9_readers.py` against the in-memory store.

## Teardown
Each test resets its pot via `context_graph.reset_pot(pot_id)` (DETACH DELETE of the `group_id` partition), keeping the live DB clean and tests isolated.

## HTTP entrypoint (`test_e2e_http.py`, deterministic, live Neo4j)
The other suites call the use-case/adapter layer directly; this one drives the
**real FastAPI app** (`create_app()`) bound to the live container via
`app.dependency_overrides[get_container_or_503]`, so requests traverse the real
entrypoint — auth dependency, hardening middleware, request validation, the
policy tenant boundary, deps wiring, and `ContextGraphResult` → JSON.
- `TestHttpAuth` — the gate fails **closed**: 503 (no key, no escape hatch),
  401 (wrong key), 403 (valid key but the resolver isn't actor-scoped and
  `CONTEXT_ENGINE_ALLOW_NO_AUTH` is off — the policy tenant boundary), 200
  (dev no-auth mode).
- `TestHttpQuery` — `POST /query/context-graph` round-trips seeded topology
  (asserts the seeded node survives serialization); malformed body → 422.
- `TestHttpReset` — `POST /reset` clears the pot through the operator route.
- `TestHttpHardening` — security headers (`X-Content-Type-Options`) are present
  on a live response, proving the hardening middleware is installed.

## Batching state machine (`test_e2e_batching.py`, real Postgres, no LLM)
Covers the production reliability path the N=1 pipeline test skips. Uses the
driver-adaptive `pg_test_db` fixture (psycopg v3 when psycopg2 is absent).
- `TestCoalescing` — three events to one pot debounce into a single open batch.
- `TestEventIdempotency` — re-submitting the same `source_id` (webhook
  redelivery) dedups: `duplicate=True`, no second membership row.
- `TestWindowedFlush` — a windowed batch is enqueued only after the window
  elapses (injected clock); enqueue does not transition the batch.
- `TestReapStaleBatch` — a claim backdated past its lease is reaped → batch
  `failed`, its events surfaced for retry (crash/OOM recovery).
- `TestFreshBatchWhenInFlight` — a new event opens a fresh `pending` batch once
  the prior one is claimed.

## Test classes
`test_e2e_topology.py` (deterministic, live Neo4j):
- `TestEnvironmentAndContainer` — Neo4j reachable, settings, container wiring.
- `TestPotLifecycle` — resolve / list / reverse-lookup / unknown.
- `TestScannerIngestion` — k8s + CODEOWNERS working-tree → graph.
- `TestApplyPlanIngestion` — reconciliation plan → graph; off-catalog rejection; singleton supersession.
- `TestCanonicalReadback` — canonical `:Entity` count + per-label counts + `:RELATES_TO` topology edges, read via direct Cypher.
- `TestPotReset` — reset clears the partition.
- `TestBitemporalContract` — canonical `:RELATES_TO` bitemporal write/state:
  supersession stamps `invalid_at` at exactly the new claim's `valid_at`
  (+ `expired_at`/`superseded_by_object`); a point-in-time predicate
  (`valid_at <= T < invalid_at`) selects the right owner at any instant. **Note:**
  this is the *write/state* contract — no `as_of`-aware *reader* exists yet over
  the canonical `:RELATES_TO` edges, so the point-in-time selection is asserted
  via Cypher (`_live_owner_as_of`), documenting the predicate a future reader
  must implement.
- `TestPlanIdempotency` — re-applying the same plan (same `source_ref` MERGE
  keys) adds no duplicate entities or edges.

`test_e2e_pipeline.py` (real Postgres + real LLM):
- `TestPostgresPipeline` — event submission persists + batches (real Postgres, no LLM call).
- `TestLLMReconciliationPipeline` — submit → claim → `process_batch`(real agent) → graph; asserts the deploy event reconciles into the `Service-[:DEPLOYED_TO]->Environment` topology shape (labels + edge + `environment` prop), not just `entities >= 1`.
- `TestLLMQuery` — `goal=ANSWER` real synthesis over seeded topology.

## Removed
- The `/conflicts/list` and `/conflicts/resolve` operator endpoints (and their
  CLI commands + client methods) were deleted: predicate-family conflict
  detection/resolution was a Graphiti maintenance pass that no longer runs
  (`detect_family_conflicts` has no callers). Only deterministic singleton
  supersession remains, covered by `TestApplyPlanIngestion` /
  `TestBitemporalContract`.

## Fixtures (conftest.py)
- `live_env` / `settings` / `container` / `pot_id` / `repo_name` / `scanner_registry` — Neo4j-backed, deterministic.
- `pg_test_db` (session) — create/schema/drop a throwaway Postgres database in the configured instance.
- `db_container` — container with a constructed (not-invoked) agent for Postgres round-trips (no LLM key needed).
- `llm_env` — loads LLM keys + selects the model for all three LLM surfaces; bounds agent/query timeouts; skips without a key.
- `pipeline_container` — real reconciliation agent + real Postgres (`DATABASE_URL`) + real query agent/synthesizer; `NoOpContextGraphJobQueue` so batches run in-process.
