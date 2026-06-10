<!--
Audit method: 12-scope multi-agent fan-out over the canonical tree (potpie/context-engine),
one auditor per subsystem + 3 cross-cutting passes (drift / duplication / whole-tree dead-code),
then an adversarial skeptic refuted every "dead/deletable" claim before it reached this report.
59 agents, 90 raw findings → 42 confirmed-dead, 44 qualitative, 4 refuted (excluded here).
Date: 2026-06-09.

Hand-verified after synthesis:
- The "1555-line" figure belongs to the HTTP router adapters/inbound/http/api/v1/context/router.py
  (1555 lines), NOT ContextGraphService (160 lines) which is the facade the router binds — the
  two-facade duplication finding itself is confirmed.
- Empty husks confirmed: adapters/outbound/{github,neo4j,synthesis} = 0 .py; mcp/tools/__init__.py = 0 bytes.
- app/src/context-engine confirmed stale: 0 git-tracked, 0 real source files (sources gone to .pyc).
- "Broken at startup" CONFIRMED: legacy/app/modules/context_graph/hatchet_worker.py:16 imports
  adapters.inbound.hatchet.worker.run_hatchet_context_graph_worker, but the canonical
  adapters/inbound/hatchet/ holds only __init__.py — that import raises ImportError.
-->

# Context Engine — Tech-Debt & Drift Audit

## 1. Where we stand

The V1.5 surface is **substantially delivered and, in most places, honest**. The agent-facing read trunk is genuinely single: one `ReadOrchestrator` over the canonical Position-B `:RELATES_TO` claim store, one ranker (the geometric-mean veto is fixed), one write door through `backend.mutation`. CLI and MCP both fan into the same `AgentContextPort → DefaultGraphService`; the four `context_*` tools are intact; `context_record` flows through the same semantic lowerer as `graph mutate`; the LLM planner is correctly parked opt-in. The graph workbench (catalog/read/search-entities/mutate/nudge) is wired. This is a real, composed hexagon — the storage layer was successfully unified, and the bulk of the 546-file canonical tree is live and reachable through one of two composition roots.

The damage is concentrated, not diffuse. **Three structural fractures** account for most of the pain: (a) the service layer above the unified backend was *not* unified — `DefaultGraphService` (CLI/MCP) and `ContextGraphService` (the 1555-line HTTP router) are two facades doing the same read+write job in two DTO vocabularies; (b) there are **two non-converging record/ingest pipelines** (deterministic local vs. model-mediated managed HTTP) that re-derive claim shape independently; (c) the entire managed/Postgres/HTTP ingestion stack is a *wired-but-inert* parked subsystem — it type-checks and runs end-to-end on empty input but cannot pull a real event or author a meaningful claim, and the host product never exercises it. Around these sit a set of "honest seams" (ledger, cloud commands, managed-pot flags, stub backends) that are deliberately fail-closed, plus a thin tail of genuinely dead code.

The cruft tail is **small and well-bounded** — roughly 700–800 lines across ~10 modules plus several empty package husks — but it has accumulated across at least four visible eras (sandbox-agent era, server/synthesis era, multi-mode-read era, pre-V1.5 resolver era), and each left a ghost. The single biggest *process* problem is documentation drift: `architecture.md`, the playbooks, and the plan's own IR-01..IR-12 fix queue all describe a system the code no longer matches — in **both directions** (items marked Open are done; the documented default backend is wrong; the documented write contract `propose/commit/plan_id` never shipped). And the benchmark harness — the thing that's supposed to tell you whether any of this works — is **red**: its fixture corpus is gone and both engine-driving paths point at the deprecated server.

## 2. What doesn't make sense (structural)

Ranked by how much they hurt:

1. **Two graph-service facades over one backend** (`application/services/graph_service.py::DefaultGraphService` vs `adapters/outbound/graph/context_graph_service.py::ContextGraphService`). Both wrap the same `GraphBackend`, both build their own `ReadOrchestrator`, both route writes through `backend.mutation` — but in incompatible DTO vocabularies (`AgentEnvelope`/`ResolveRequest` vs `ContextGraphQuery`/`ContextGraphResult`). The `ContextGraphService` docstring even brags that storage was collapsed ("no second graph stack") — but the *service* on top was never collapsed. The `agent_context.py` claim that "every inbound adapter binds here and nowhere else" is now **false**: the HTTP router binds exclusively to `container.context_graph` and never touches `agent_context`. This is the single highest-leverage structural fix.

2. **Two non-converging record pipelines.** Local: `AgentContextService.record → DefaultGraphService.record → record_to_semantic → mutate (validate→lower→backend)`. Managed: `record_durable_context → IngestionSubmissionService.submit → debounced batch → process_batch → LLM reconciliation agent`. They share no lowering, no validation, no request/result types, and have *divergent semantics* (deterministic vs model-mediated). They converge only at `backend.mutation`. "Record context" has two meanings depending on which door you enter, and the managed door still **hard-fails with `no_reconciliation_agent`** (`ingestion_submission_service.py:120`) — directly violating the V1.5 non-negotiable that canonical writes need no Potpie LLM.

3. **Three composition roots, two of them duplicating policy.** `build_host_shell` (live agent spine), `build_ingestion_server` (deferred HTTP/Postgres), and `bootstrap/container.py::build_container` (an admitted legacy shim whose `_ResolutionServiceShim.set_source_resolver` writes a value nothing reads). The two live roots **independently** derive the default graph backend (`host_wiring` defaults `falkordb_lite` by Python version; `ingestion_server` defaults `neo4j` via `settings.graph_db_backend()`) and **copy-paste** the four bench-stub connector registrations. Two different answers to "which backend am I on?".

4. **The CLI breaks its own routing contract.** `host_cli.py`'s docstring asserts "every command routes CLI → HostShell → ports," yet `pots.py`'s `linear-team ingest` / `jira-project ingest` reach around HostShell into an outbound HTTP client (`PotpieContextApiClient`) and POST to a remote managed API. This remains a divergent provider-ingest path that bypasses the host.

5. **Inbound layer hand-rolls the envelope wire shape three times.** `AgentEnvelope` has no `to_dict()`, so `mcp/server.py::_envelope_dict`, `cli/commands/query.py::_envelope_payload`, and `cli/commands/graph.py::_read_payload` each copy-paste it; the nudge skills block is duplicated verbatim in `bootstrap.py` and `mcp/server.py`. Worse, a *richer* serializer (`application/services/envelope_builder.py::envelope_to_dict`) already exists but is used only by the outbound managed path — so the contract is drifting across four sites. Any wire-shape change is a 3–4-file lockstep edit.

6. **Two daemon architectures shipped side by side.** The real daemon (`host/daemon.py` + `daemon_main.py` + `daemon_rpc.py` + `daemon_client.py`) is wired and tested. A second, more elaborate plugin-registry/IPC-auth/health runtime (`host/daemon_runtime/*` + `domain/ports/daemon/*`, ~340 lines) is exercised **only by unit tests** — and the `DaemonRuntime` class its `__init__` docstring promises doesn't even exist.

7. **Observability is split-brain.** The entire OTLP/correlation/structured-logging bootstrap (`set_observability`, `instrument_adapter`, `correlation_scope`, `configure_logging`) is engaged *only* from `build_ingestion_server`. The live agent spine (`build_host_shell` / CLI / daemon) installs no sink, opens no correlation scope, and uses a separate `configure_cli_logging`. `logging_setup`'s docstring claims it's "called from every entrypoint (HTTP, MCP, CLI)" — false. The product users actually run is dark.

8. **Capability ports with live callers but no production-backend body.** `neo4j`/`falkordb` declare `inspection=False, snapshot=False`, but `graph.py` calls `host.backend.inspection.neighborhood/.path` and `snapshot.export/import_` unconditionally. On the *default* backend, those CLI commands fail closed with `CapabilityNotImplemented`. The "rebuildable projection never rebuilt" gap: ports exist, toy backends satisfy them, shipping backends don't.

9. **Lower-hurt smells worth naming:** five P9 readers copy-paste `_make_candidate_key`/`_corroboration`/`_payload_from_row`/anchor logic while `readers/_common.py` sits right there unused; `in_memory_reader` forks its own lexical scorer instead of `canonical_claim_query.embedding_score` (and the conformance ground-truth already ranks differently from production); two `StatusReport` types and two status paths for one "context_status" concept; the `RemoteSurface` `_NESTED`/`_REMOTE_ATTRS` whitelist in `daemon_client.py` must be hand-synced with `GraphBackend`'s capability ports or new attrs silently proxy as method-call closures.

## 3. Dead / unused / deletable

**Safe to delete now** (no production importer, claim confirmed, not refuted):

- `adapters/inbound/mcp/tools/` — empty package, zero importers, ships in the wheel as dead weight.
- `adapters/outbound/agent_tools/` (entire ~54KB cluster: `sandbox.py`, `_sandbox_git_tools.py`, `_sandbox_metrics.py`, `_path_safety.py`) — orphaned sandbox-agent-era island; the `add_extra_tools` extension point it targets has **no live caller anywhere**. Delete `tests/unit/test_path_safety.py` with it.
- `adapters/outbound/synthesis/`, `adapters/outbound/github/`, `adapters/outbound/neo4j/` — phantom dirs, no git-tracked files, only stale `.pyc`. Roles relocated (`connectors/github`, `graph/neo4j_*`); synthesis deleted in #851.
- `adapters/outbound/reconciliation/null_agent.py` (`NullReconciliationAgent`) — zero references, not even tests. The codebase standardized on a `None` sentinel for "no agent."
- `adapters/outbound/source_resolvers/github_pull_request.py` — 348-line byte-for-byte clone of `connectors/github/resolver.py`; the `__init__` re-exports the *connectors* copy. Its only import target (`domain.ports.source_control`) doesn't even exist in the canonical tree.
- `domain/ports/graph/claim_query.py` — re-export shim consumed only by the unused `__init__` aggregator; nothing imports the new namespace (the canonical `domain.ports.claim_query` won with 31 importers). Keep `domain/ports/graph/__init__.py` as a package marker, but its re-export body is dead.
- `host/daemon_runtime/*` + `domain/ports/daemon/*` (~340 lines) + the six `test_daemon_{registry,health,context,ipc_auth,ports,operations}.py` — the unwired parallel daemon design.
- `ContextGraphGoal.NEIGHBORHOOD` / `AGGREGATE` enum members — never constructed; `goal` is a non-dispatching passthrough label.
- The stale untracked **`/app/src/context-engine`** tree — gutted shadow copy (5174 `.py`, all a vendored `.venv`; sources already deleted to `.pyc`). Untracked, not gitignored, pollutes every repo-wide grep. `rm -rf` it; repoint the ~9 stale doc/comment path strings to `potpie/context-engine`.

**Probably dead — confirm intent first** (test-only or parked; deletable mechanically but may be a deliberate seam you want to keep):

- `belief.py`'s entire probabilistic confidence-fusion core (`derive_belief`, `score_object`, `Belief`, `ConfidenceLabel`, etc., ~525 lines) — referenced only by its own test; live scoring runs through `ranking.py`. Even the two "live" helpers `ranking.py` imports (`EvidenceStrength`, `decay_weight`) are non-functional re-exports. This is the "cosmetic confidence" abandoned design. Extract nothing; delete or wire deliberately.
- `application/services/temporal_search.py` (~96 lines) — `compute_temporal_flag`/`annotate_search_rows_temporally`, test-only; its sole output key (`superseded_label`) is read by a dead branch in `cli/ui/output.py:338`. Delete both.
- `adapters/outbound/reconciliation/timeline_plan.py` (`build_timeline_mutations`) — docstring claims "every plan builder calls" it; **nothing does**. Live timeline path is `semantic_mutation_lowering`. Either delete (clean up the dangling refs in `ontology.py:680` + 2 tests + 3 playbook `.md`) or hoist its `VERB_*` vocabulary into the live path.
- `domain/deterministic_extractors.py` — the parked deterministic-extraction strategy; only `test_extractors.py` imports it. A deliberate `TODO(decide)` seam, not accidental cruft.
- `connectors/github/review_threads.py::group_review_threads` — test-only; the live agent tool returns ungrouped review comments (a latent feature gap, not just dead code).
- `connectors/linear/*` end-to-end (`LinearConnector`, `build_linear_tools`, resolver, webhook, events; `LinearIssueFetcher` has no concrete impl) — fully authored, registered nowhere, duplicates a real client in `potpie/integrations`. Auth half is built; consumption half dangles.
- `NoOpReconciliationAgent`, `InMemoryEventStreamPublisher` — legitimate test doubles, but exported as peers of the real adapters in their package `__init__`s; reposition or annotate so they don't read as deployment options.

**Genuinely broken (not dead — fix or finish):** `adapters/inbound/hatchet/` is a 1-line husk; `legacy/app/modules/context_graph/hatchet_worker.py:16` imports `adapters.inbound.hatchet.worker.run_hatchet_context_graph_worker`, which was **deleted in #858** and now `ImportError`s at startup. Either restore `worker.py` (recoverable from `dbc92c23`) or delete the legacy entrypoint + README:32 reference.

## 4. Drift from the documented design

The plan and `architecture.md` describe a system the code doesn't match, in both directions:

- **The write contract in the doc never shipped.** `architecture.md` specifies `potpie graph propose|commit` with a server-created `plan_id` and a planned `domain/ports/graph_workbench.py`. None exist (`rg graph_workbench` → 0). The real surface is single-call `GraphService.mutate()`. An architect extending the write path from the doc would build against a phantom contract.
- **The plan's own IR-01..IR-12 fix queue had been stale both ways.** IR-03
  (inline-relation assembly) and IR-07 (vector search wired, `fact_embedding`
  written/queried, and V1.5 metadata hydrated on Neo4j/FalkorDB) are now marked
  fixed in the implementation plan; IR-08+ remain the active queue.
- **Default backend is documented wrong in three files.** `architecture.md` and `backends/__init__.py` call `embedded` the OSS default and never mention `falkordb_lite`; `backend.py` still tags `embedded`/`postgres`/`hosted` as `(TODO)`. The actual default (`host_wiring.py`, Python ≥3.12) is `falkordb_lite`. The subsystem contradicts itself about its own default.
- **Playbooks instruct the LLM to emit vocabulary the ontology doesn't have.** `event_playbooks.py` tells the reconciliation agent to seed `Module`, `Feature`, `Incident`, `DiagnosticSignal`, `DECIDES_FOR` — **none** exist in `ontology.py`. The validator rejects them and silently downgrades to generic `Document`/`RELATED_TO`. This is the exact "duplicate ontology in prose" anti-pattern the Rules section forbids, and `coherence.py`'s import-time checks don't cover playbook labels, so it's invisible at startup.
- **Backend-divergent claim metadata (fixed in working tree).** Neo4j/FalkorDB's
  shared `row_from_record` now hydrates first-class V1.5 `ClaimRow` fields
  (`truth`, `subgraph`, `environment`, `claim_key`, `valid_until`,
  `source_refs`, `evidence`, versions) and removes those contract keys from the
  legacy `properties` fallback bag.
- **Catalog over-advertises.** Views declare `ranking_inputs` like `truth_class`/`resolution_status` that `RankingService` never computes; `graph catalog --subgraph` is accepted then `del`'d and ignored. The contract claims behavior the engine doesn't implement.
- **Coherence guard runs on the wrong root.** `assert_runtime_coherence()` (reader/contract reconciliation) fires only from `build_ingestion_server`, not `build_host_shell` — so the canonical OSS spine can register a reader without updating the advertised contract and produce the exact silent-retrieval failure the check exists to prevent.
- **Daemon-default narrative inverted.** The daemon is the *runtime default* (`get_host()` builds `RemoteHostShell` unless `HOST_MODE=in_process`; `SetupPlan.host_mode='daemon'`), yet `shell.py`/`host_wiring.py` docstrings say "in-process by default" and the `Daemon` dataclass defaults `in_process=True`. A reader concludes the daemon is dev-only when it's the primary path.
- **Benchmark docs point at the deprecated server**: README references the stale `app/src/...` tree, gunicorn-on-8001 + Celery worker, `/events/reconcile` — none of which match the host-CLI engine. A reader cannot run a single scenario today.
- **Smaller honest-surface gaps:** `context_record` silently swallows `ContextRecordValidationError` (validation bypass, contradicting Step 8); `DefaultSkillManager.add()` returns a success-shaped no-op to a live `potpie skills add` command (should raise `CapabilityNotImplemented` like the engine's own convention).

## 5. Prioritized recommendations

1. **Collapse the two graph-service facades onto one DTO surface (L).** Port the HTTP router off `ContextGraphService`/`ContextGraphQuery` onto `DefaultGraphService`/`AgentEnvelope`, then delete `ContextGraphService` and the `ContextGraphPort`/`ContextGraphQuery` vocabulary. This is the highest-leverage fix; it also makes the `agent_context.py` "sole inbound binding" claim true again. Pairs naturally with adding `AgentEnvelope.to_dict()` (or reusing `envelope_to_dict`) to kill the 3–4-site wire-shape duplication.

2. **Sweep the confirmed-dead tail in one PR (S).** Delete the §3 "safe to delete now" list: empty husks (`mcp/tools`, `synthesis`, `github`, `neo4j` dirs), `agent_tools/` cluster, `null_agent`, the `source_resolvers` clone, the `graph/claim_query` shim, `daemon_runtime`/`ports.daemon` + their tests, the unused enum members, and `rm -rf app/src/context-engine`. Mechanical, low-risk, and it makes every future repo-wide grep trustworthy.

3. **Decide the managed-record principle and fix `no_reconciliation_agent` (M).** Either route managed `context_record` through the same deterministic semantic-mutation path as local (honoring the V1.5 non-negotiable, with a managed-path test that succeeds with no agent), or explicitly document the managed pipeline as model-required and stop claiming otherwise. This is the largest principle-vs-implementation gap.

4. **Reconcile the docs in a single pass (M).** Rewrite the `architecture.md` write-contract section (kill `propose/commit`/`plan_id`/`graph_workbench.py`), fix the default-backend tables to `falkordb_lite`, close/reopen the IR-01..IR-12 queue against reality, fix the daemon-default narrative, and either align playbook vocabulary to the ontology or add a `coherence.py` import-time check that playbook labels exist. Add the playbook-label check so this can't silently rot again.

5. **Restore the benchmark corpus and turn the smoke test green (M).** The `benchmarks/fixtures/raw_events/universe/acme` tree is missing (and root `.gitignore`'s blanket `*.json` makes it uncommittable). Restore it with a `fixtures/` allowlist exception, repoint `core/local_engine.py`/`core/query.py` off the deprecated server onto the host CLI / `DefaultGraphService`, and wire the V1.5-aligned `retrieval_eval.py` (R7) into `cli.py`. Without this you have no signal that any of the above changes work.

6. **Engage observability on the agent spine (S–M).** Call the existing `configure_logging` + `set_observability` from `build_host_shell` and open a `correlation_scope` per CLI/daemon invocation, and move `assert_runtime_coherence()` to fire on `build_host_shell` too. The machinery is built; it just isn't connected to the product people run.

7. **Consolidate the five P9 reader helpers into `readers/_common.py` (S).** `_make_candidate_key`, `_corroboration`, `_payload_from_row`, the scope→anchor logic — the shared home already exists and is unused for these. Low blast radius on the one subsystem that's actually on the live roadmap. While there, unify the lexical scorer so conformance ground-truth stops diverging from production.

Files of record for the above: `application/services/graph_service.py`, `adapters/outbound/graph/context_graph_service.py`, `application/services/ingestion_submission_service.py`, `bootstrap/{host_wiring,ingestion_server,container}.py`, `domain/event_playbooks.py`, `adapters/outbound/graph/canonical_claim_query.py`, `application/readers/_common.py`, `docs/context-graph/{architecture.md,graphv1-5-implementation-plan.md}`, `benchmarks/`.

## 6. Current CLI smoke-test findings and next work queue

Fresh-pot smoke test, 2026-06-10:

- Created `cli-usecase-smoke-20260610` / `pot_89bd2f33a905`.
- Backend: `falkordb_lite`, `match_mode=vector`.
- Wrote one semantic mutation batch covering:
  - project preferences,
  - infra/service topology,
  - recent timeline activity,
  - prior bug + fix memory.
- Dry-run accepted 8 operations with no warnings.
- Apply wrote 11 entity upserts and 11 claim edges.
- Restored active pot back to `potpie-monorepo` and removed the smoke pot's repo source to avoid current-repo inference ambiguity.

What this proved:

- The core V1.5 write path is usable from the installed CLI.
- Node summaries now surface through `graph read` / `graph search-entities`.
- The four target use cases are not blocked by storage anymore.
- The remaining problems are mostly read-shape, identity, ranking, and CLI ergonomics.

### 6.1 P0: Fix preference read shape

Finding: `preferences.active_preferences` returns `flat_claims`, despite the catalog advertising inline `POLICY_APPLIES_TO` relations. The query finds the right facts, but the agent gets a claim row instead of a preference entity with its applies-to scope.

Root cause:

- `application/readers/coding_preferences.py::_payload_from_row()` omits `predicate`.
- `application/services/graph_service.py::_assemble_inline_relation_items()` requires `predicate`, `subject_key`, and `object_key`.
- Because `predicate` is absent, inline assembly returns `inline_relation_assembly=no_relation_payloads`.

Next steps:

- Add `predicate`, `environment`, and `description` to the coding-preferences reader payload.
- Add a focused test that `graph read --view preferences.active_preferences` returns `read_shape=entity_relations`.
- Shape the preferred payload as:
  - preference entity: key, name, summary, description,
  - relation: `POLICY_APPLIES_TO`,
  - related scope entity: repo/service/path/code asset,
  - fact/prescription/source refs/truth/evidence strength.
- Add a CLI human output path that shows the prescription first, not the internal claim key.

Acceptance check:

```bash
potpie --json graph read \
  --view preferences.active_preferences \
  --scope service:web-app \
  --query "React TypeScript Tailwind frontend feature" \
  --limit 5
```

Expected: `read_shape=entity_relations`, entity key `preference:frontend-library-choices`, relation `POLICY_APPLIES_TO`, related entity `service:web-app`, and non-empty `summary`.

### 6.2 P0: Stop canonical label pollution

Finding: smoke-test entities accumulated wrong labels. Example: `service:web-app` came back as `["Entity", "Activity", "Service", "Environment", "APIContract"]`.

Root cause:

- Semantic mutations lower explicit `type` to one label in `application/services/semantic_mutation_lowering.py::_label_for()`.
- The graph writer only ever adds labels via `SET e:<Label>` and never removes or distinguishes inferred labels.
- The ontology classifier can infer labels from broad text patterns and property signatures (`API`, `environment`, `occurred_at`, etc.).
- Explicit type is not preserved as an authoritative primary type.

Next steps:

- Add an explicit `primary_label` or `canonical_type` property for semantic entity upserts.
- Treat top-level semantic `type` as authoritative unless the caller explicitly asks for inferred enrichment.
- Move speculative classifier output into `inferred_labels` metadata, not Neo4j/Falkor node labels, or gate it behind a confidence/source policy.
- Update `graph search-entities --type` to filter by authoritative primary label first, not accumulated labels.
- Add an audit command or test helper that reports entities whose key prefix disagrees with labels.

Acceptance check:

- `service:web-app` must have primary label `Service`.
- `graph search-entities "...frontend..." --type Service` may return it.
- `graph search-entities "...frontend..." --type Activity` must not return it unless it is explicitly an Activity.

### 6.3 P0: Timeline must use source event time, not graph write time

Finding: the timeline event accepted `occurred_at=2026-06-10T10:00:00+05:30`, but timeline output used the graph write time around `2026-06-10T03:33:25Z`.

Root cause:

- `application/services/semantic_mutation_lowering.py::_claim_properties()` sets `valid_at = op.valid_from or _now_iso()`.
- For `append_event`, `occurred_at` is only carried as an extra property.
- `application/readers/timeline_reader.py` filters/ranks by `row.valid_at`.
- CLI event formatting falls back to `valid_at` when no top-level `occurred_at` is present.

Next steps:

- For `append_event`, set claim `valid_at` to `op.valid_from or op.occurred_at or now`.
- In `TimelineReader._payload_from_row()`, surface top-level `occurred_at` from row extras when present.
- Ensure timeline window filtering uses event time for activity claims.
- Keep `observed_at` as ingestion/write observation time.

Acceptance check:

- Write an event with `occurred_at` two days ago.
- `graph read --view recent_changes.timeline --time-window 24h` must not return it.
- `graph read --view recent_changes.timeline --time-window 7d` must return it with the original source event time.

### 6.4 P0: Dedupe claim rows before inline assembly

Finding: bug reads can show the same `RESOLVED` or `REPRODUCES` relation repeated multiple times in one entity payload, even when raw graph inspection shows only one edge.

Root cause:

- Native vector reads can return repeated relationship rows.
- Readers do not consistently dedupe by `claim_key`.
- `_assemble_inline_relation_items()` appends every incoming `EvidenceItem` relation and does not dedupe by `claim_key` / predicate triple.

Next steps:

- Add a shared `dedupe_claim_rows(rows)` helper in `application/readers/_common.py`.
- Apply it in every P9 reader after `claim_query.find_claims()`.
- Add relation-level dedupe in `_assemble_inline_relation_items()` keyed by `claim.claim_key` first, then `(predicate, from_key, to_key, source_refs)`.
- Add a regression test with duplicate `ClaimRow`s and assert one relation in output.

Acceptance check:

- A bug query returning one `RESOLVED` claim emits one `RESOLVED` relation, not 2-3 copies.

### 6.5 P0: Bug/debug reads should always bundle fixes

Finding: querying the bug symptom returned the `BugPattern`; querying fix-specific terms returned the `Fix`. The system does not reliably bundle known fixes with a matched bug pattern.

Root cause:

- `application/readers/prior_bugs.py` performs one semantic claim search over `REPRODUCES`, `RESOLVED`, `ATTEMPTED_FIX_FAILED`, and `VERIFIED`.
- If the semantic query primarily matches the symptom claim, linked fix claims may not be in the candidate pool.
- Inline relation assembly can only assemble rows it already received.

Next steps:

- Convert `PriorBugsReader` into a two-phase reader:
  1. semantic search to find candidate `BugPattern` / `Fix` nodes,
  2. deterministic expansion around matched bug patterns to fetch `RESOLVED`, `ATTEMPTED_FIX_FAILED`, and `VERIFIED` claims.
- Emit a bug bundle shape: bug pattern, reproduces scopes, resolved fixes, failed attempts, verification evidence.
- Rank the bundle by best symptom match plus fix verification, not by individual claim only.

Acceptance check:

```bash
potpie --json graph read \
  --view bugs.prior_occurrences \
  --query "node summary missing empty graph explorer search" \
  --scope service:context-api \
  --limit 5
```

Expected: one bug-pattern item that includes the `fix:derive-node-summary-on-upsert` relation without requiring fix-specific query terms.

### 6.6 P1: Make environment-scoped infra reads useful by default

Finding: `--environment local` correctly filtered to `DEPLOYED_TO local`, but dropped unqualified global dependencies like `USES datastore:falkordb-lite` and incoming `DEPENDS_ON web-app -> context-api`.

Root cause:

- `InfraTopologyReader` enforces `qualified_only` when `environment` is present.
- The catalog mentions `include_unqualified_environment=true`, but the CLI has no direct flag for it.
- Agents debugging an environment usually need both environment-bound edges and global service dependencies.

Next steps:

- Add `--include-unqualified-environment` to `graph read`.
- For `infra_topology.service_neighborhood`, consider defaulting to include unqualified dependencies when a service anchor is present, while still tagging which edges are env-qualified.
- Split response metadata into `environment_qualified_edges` and `global_edges`.

Acceptance check:

```bash
potpie --json graph read \
  --view infra_topology.service_neighborhood \
  --scope service:context-api \
  --environment local \
  --include-unqualified-environment
```

Expected: `DEPLOYED_TO environment:local` plus global `USES` and incoming `DEPENDS_ON` edges.

### 6.7 P1: Make repo functionality / feature context a first-class use case

Finding: project functionality/topology is only partially represented. The ontology has `Feature`, `PROVIDES`, and `IMPLEMENTED_IN`, and `features.provided` exists in `graph_views.py`, but it is backed by the generic infra reader rather than a feature-specific reader and was not covered by the smoke test.

Next steps:

- Add a smoke mutation for:
  - `repo:<...> PROVIDES feature:<...>`,
  - `service:<...> PROVIDES feature:<...>`,
  - `feature:<...> IMPLEMENTED_IN repo/service`.
- Add a dedicated feature/capability reader or specialize the infra reader path for `features.provided`.
- Ensure the harness source-ingestion skills explicitly extract:
  - product capabilities,
  - major user workflows,
  - service responsibilities,
  - feature-to-service/repo implementation links.
- Add a CLI happy-path query: "what does this repo do?".

Acceptance check:

```bash
potpie --json graph read \
  --view features.provided \
  --scope anchor_entity_key:repo:local/potpie-usecase-smoke \
  --query "what does this repo do" \
  --limit 10
```

Expected: feature entities with summaries and implementation links.

### 6.8 P1: Add use-case-specific CLI mutation recipes

Finding: raw semantic mutation JSON works but is too verbose for routine agent/harness use. The smoke batch was useful but unwieldy.

Next steps:

- Add template helpers or subcommands:
  - `potpie graph add preference`,
  - `potpie graph add dependency`,
  - `potpie graph add event`,
  - `potpie graph add bug-fix`,
  - `potpie graph add feature`.
- Each command should print the semantic mutation JSON in `--dry-run` mode and then call the same `graph mutate` path.
- Keep raw `graph mutate` as the escape hatch.

Acceptance check:

- A one-command preference write can be queried back through `preferences.active_preferences`.
- A one-command bug/fix write can be queried back through `bugs.prior_occurrences`.

### 6.9 P1: Add CLI/daemon schema freshness checks

Finding: the first smoke-test dry-run failed with `GraphEntityRef.__init__() got an unexpected keyword argument 'summary'` until the daemon was restarted. The installed CLI was using the latest local code, but the detached daemon was still on an older in-memory schema.

Next steps:

- Include CLI and daemon code fingerprints in `potpie status` and `potpie doctor`:
  - module path,
  - package version,
  - git SHA or source-tree hash,
  - graph contract version,
  - ontology version,
  - supported semantic mutation schema version.
- Make `graph mutate` catch schema/version mismatch and return a clear "restart daemon" action.
- Consider auto-restarting the local daemon when the CLI detects that the daemon code fingerprint is stale.

Acceptance check:

- Editing a semantic DTO and running an old daemon should produce a deterministic diagnostic, not a Python constructor error.

### 6.10 P2: Add entity-summary semantic search, not just claim search

Finding: entity summaries now surface, but `graph search-entities` is still claim-backed. Entities with good summaries but few claims are not first-class vector search targets.

Next steps:

- Add node-level `summary_embedding` / `entity_embedding`.
- Add vector index support for entity summaries in FalkorDB and Neo4j.
- Merge entity-vector candidates with claim-derived candidates in `search_entities`.
- Keep claim facts as the primary evidence unit; use entity vectors for identity lookup and graph sense-making.

Acceptance check:

- A query matching only an entity summary, with no matching claim fact, can still return the entity.

## 7. Suggested immediate execution order

Treat this as the next implementation queue:

1. P0 preference read shape.
2. P0 claim dedupe in readers and inline assembly.
3. P0 timeline event-time semantics.
4. P0 authoritative primary labels / label pollution fix.
5. P0 bug bundle expansion to always include fixes.
6. P1 environment-scoped infra read ergonomics.
7. P1 feature/functionality smoke test and reader support.
8. P1 CLI mutation recipes for the four target memory use cases.
9. P1 CLI/daemon code-fingerprint freshness checks.
10. P2 entity-summary vector search.

The first five are correctness issues found by the fresh-pot smoke test. They should land before more ingestion breadth, otherwise the graph will accept useful facts but keep returning them in shapes that are hard for agents to consume reliably.
