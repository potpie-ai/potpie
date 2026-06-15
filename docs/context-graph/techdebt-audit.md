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

> **Status update (2026-06-11, `feat/graph-updates`):** the §3 dead-code sweep is **done** — every "safe to delete now" item and every "probably dead" item was resolved (deleted, except the two test doubles, which were annotated and un-exported), and the sweep went beyond the list: the **jira connector**, the **sync_history stack** (`adapters/outbound/sync_history/`, `domain/ports/sync_history.py`, `agent_tools/sync_history.py`), and the jira/linear playbooks + their CLI ingest commands and tests went with it (~7,600 lines removed). §2 items **4** (CLI provider-ingest bypass) and **6** (parallel daemon runtime) and the §3 hatchet item are resolved as a side effect. Items below carry individual ✅ marks; unmarked findings remain open.

> **Verification pass (2026-06-15, `feat/graph-updates` after merging `main` @ `f08d1370`):** a 12-way parallel re-audit checked every finding against the current tree. Net: the §3 dead-code sweep is **verified regression-free** — `python -m compileall adapters domain application host` passes, no deleted module has a live importer, and 1584 tests collect with 0 errors. Several structural findings **shrank or flipped** since 06-09 and the prose below now over-states them; each is corrected inline with a dated `*(2026-06-15 …)*` note and summarized here:
>
> - **§2.1 facade — much smaller than written.** `ContextGraphService` is no longer a 1555-line co-equal facade; it is a **~176-line legacy DTO translator** that delegates reads to `DefaultGraphService.resolve` and **disables writes** (`apply_plan` → `CapabilityNotImplemented`). It builds no `ReadOrchestrator` and routes no writes through `backend.mutation`. The read/write collapse is done, and the HTTP timeline/read path now uses `GraphService.resolve`; the remaining debt is the `ContextGraphService`/`ContextGraphPort` compatibility surface kept for hard reset and reconciliation-agent compatibility (and the `agent_context.py` docstring was fixed — it no longer falsely claims sole binding).
> - **§2.8 capability ports — half resolved.** `inspection` is now implemented on `falkordb`/`falkordb_lite`/`embedded`, so `graph inspect` no longer fails closed on the default backend. Still open: `snapshot` on falkordb, and both capabilities on neo4j.
> - **§4 ontology — `Feature` is now real** (`Feature`/`PROVIDES`/`IMPLEMENTED_IN` exist); only `Module`/`Incident`/`DiagnosticSignal`/`DECIDES_FOR` remain phantom playbook labels.
> - **§4 catalog `--subgraph`** is now plumbed through and filters (`graph_service.py:176-183`) — resolved; the `ranking_inputs` over-advertisement was also resolved in Tier 2. **§4 default backend** was documented wrong in **two** files (`architecture.md`, `backend.py`), not three — `backends/__init__.py` was corrected earlier, and the Tier 4 cleanup pass corrected the remaining two.
> - **§2.5 nudge-skills duplication resolved** (single `lru_cached` `load_bundle_skills`); the three inbound serializers + missing `AgentEnvelope.to_dict()` were also resolved in Tier 2. **§2.7** the CLI now has a *separate* Sentry crash sink (#865), so it is no longer fully dark; the daemon still is, and OTLP/correlation/structured logging stays ingestion-server-only.
> - **Still open after the Tier 4 cleanup pass:** §2.3 three composition roots + backend divergence, §5.5 benchmark corpus (now **worse** — the whole `benchmarks/fixtures/raw_events/` tree is gone; no scenario runs in any mode, even `smoke` is red), and the **entire §6 P0 queue** (reader/lowering files untouched since baseline `fd2876d7`). §2.2 is now partial: durable managed `context_record` uses the deterministic semantic-mutation path, but the broader non-record managed ingestion pipeline remains model-mediated. §4 write-contract/default-backend drift is resolved for `architecture.md` and `backend.py`.
> - **New since the merge:** a **Jira/Linear merge-reconciliation** cluster — code is consistent, connector-tool skill residue was pruned in Tier 2, and one orphaned module remains (new subsection after §3). The consolidated, prioritized cleanup backlog is the new **§8**.

> **Tier 4 cleanup pass (2026-06-15):** reviewed against this audit after the hardening work. `context_record` submissions on the managed HTTP path now route through `DefaultIngestionSubmissionService._submit_context_record()` → `GraphService.record()` and have a no-agent regression test; non-record ingestion still requires a reconciliation agent. The HTTP timeline endpoint no longer imports or constructs `ContextGraphQuery`, but `ContextGraphService` / `ContextGraphPort` remain for reset and reconciliation-agent compatibility, so the facade item is only partial. `architecture.md` and `domain/ports/graph/backend.py` now document the shipped `graph mutate` contract and `falkordb_lite` default. Benchmark restoration was intentionally not touched.

## 1. Where we stand

The V1.5 surface is **substantially delivered and, in most places, honest**. The agent-facing read trunk is genuinely single: one `ReadOrchestrator` over the canonical Position-B `:RELATES_TO` claim store, one ranker (the geometric-mean veto is fixed), one write door through `backend.mutation`. CLI and MCP both fan into the same `AgentContextPort → DefaultGraphService`; the four `context_*` tools are intact; `context_record` flows through the same semantic lowerer as `graph mutate`; the LLM planner is correctly parked opt-in. The graph workbench (catalog/read/search-entities/mutate/nudge) is wired. This is a real, composed hexagon — the storage layer was successfully unified, and the bulk of the 546-file canonical tree is live and reachable through one of two composition roots.

The damage is concentrated, not diffuse. **Three original structural fractures** account for most of the pain: (a) the service layer above the unified backend was *not* unified — `DefaultGraphService` (CLI/MCP) and `ContextGraphService` (the 1555-line HTTP router) are two facades doing the same read+write job in two DTO vocabularies; (b) record/ingest is still split for broader managed events (deterministic local vs. model-mediated managed HTTP), though durable managed `context_record` now shares the deterministic semantic-mutation path; (c) the managed/Postgres/HTTP ingestion stack remains a mostly parked subsystem for non-record events — it type-checks and runs end-to-end on empty input but cannot pull a real event without the reconciliation machinery, and the host product never exercises it. Around these sit a set of "honest seams" (ledger, cloud commands, managed-pot flags, stub backends) that are deliberately fail-closed, plus a thin tail of genuinely dead code.

The cruft tail is **small and well-bounded** — roughly 700–800 lines across ~10 modules plus several empty package husks — but it has accumulated across at least four visible eras (sandbox-agent era, server/synthesis era, multi-mode-read era, pre-V1.5 resolver era), and each left a ghost. The biggest remaining *process* problem is documentation drift outside the now-reconciled `architecture.md` graph-mutate/default-backend sections: the playbooks and the plan's own IR-01..IR-12 fix queue still describe a system the code no longer matches in both directions. And the benchmark harness — the thing that's supposed to tell you whether any of this works — is **red**: its fixture corpus is gone and both engine-driving paths point at the deprecated server.

## 2. What doesn't make sense (structural)

Ranked by how much they hurt:

1. **Two graph-service facades over one backend** (`application/services/graph_service.py::DefaultGraphService` vs `adapters/outbound/graph/context_graph_service.py::ContextGraphService`). Both wrap the same `GraphBackend`, both build their own `ReadOrchestrator`, both route writes through `backend.mutation` — but in incompatible DTO vocabularies (`AgentEnvelope`/`ResolveRequest` vs `ContextGraphQuery`/`ContextGraphResult`). The `ContextGraphService` docstring even brags that storage was collapsed ("no second graph stack") — but the *service* on top was never collapsed. The `agent_context.py` claim that "every inbound adapter binds here and nowhere else" is now **false**: the HTTP router binds exclusively to `container.context_graph` and never touches `agent_context`. This is the single highest-leverage structural fix. *(2026-06-15: largely collapsed already — `ContextGraphService` is now a ~176-line DTO translator delegating reads to `DefaultGraphService.resolve`, with writes disabled (`apply_plan`→`CapabilityNotImplemented`); it builds no `ReadOrchestrator` and routes no writes through `backend.mutation`, and the `agent_context.py` docstring was fixed. Tier 4 cleanup moved the HTTP timeline/read path off `ContextGraphQuery` and onto `GraphService.resolve`/`AgentEnvelope`. Remaining: delete or replace the `ContextGraphService`/`ContextGraphPort` compatibility surface still used for hard reset and reconciliation-agent compatibility.)*

2. **Two non-converging record/ingest pipelines.** Local: `AgentContextService.record → DefaultGraphService.record → record_to_semantic → mutate (validate→lower→backend)`. Managed record before cleanup: `record_durable_context → IngestionSubmissionService.submit → debounced batch → process_batch → LLM reconciliation agent`. They shared no lowering, no validation, no request/result types, and had *divergent semantics* (deterministic vs model-mediated). *(2026-06-15 Tier 4 cleanup: durable managed `context_record` now routes directly through `DefaultIngestionSubmissionService._submit_context_record()` → `GraphService.record()`, succeeds without a reconciliation agent, and local validation is symmetric because `record_to_semantic_request()` always calls `validate_record_payload()` for known schemas. Broader non-record managed ingestion remains model-mediated and still requires the reconciliation agent.)*

3. **Three composition roots, two of them duplicating policy.** `build_host_shell` (live agent spine), `build_ingestion_server` (deferred HTTP/Postgres), and `bootstrap/container.py::build_container` (an admitted legacy shim whose `_ResolutionServiceShim.set_source_resolver` writes a value nothing reads). The two live roots **independently** derive the default graph backend (`host_wiring` defaults `falkordb_lite` by Python version; `ingestion_server` defaults `neo4j` via `settings.graph_db_backend()`) and **copy-paste** the four bench-stub connector registrations. *(2026-06-15: now **five** stubs — `LinearStubConnector` was added by the merge — and the copy-paste is actually between `ingestion_server` and `standalone_container`, both ingestion-family roots; the agent spine registers none. Tier 2 factored `register_bench_stubs()`, but backend-default divergence remains.)* Two different answers to "which backend am I on?".

4. ✅ **The CLI breaks its own routing contract.** *(fixed in working tree 2026-06-11 — deleted, not rerouted: `pots.py`'s `linear-team ingest` / `jira-project ingest` commands and the `PotpieContextApiClient` bypass were removed outright (−299 lines) along with the jira/linear connectors; `pots.py` no longer reaches around HostShell.)* `host_cli.py`'s docstring asserted "every command routes CLI → HostShell → ports," yet `pots.py`'s ingest commands reached around HostShell into an outbound HTTP client and POSTed to a remote managed API.

5. ✅ **Inbound layer hand-rolls the envelope wire shape three times.** `AgentEnvelope` had no `to_dict()`, so `mcp/server.py::_envelope_dict`, `cli/commands/query.py::_envelope_payload`, and `cli/commands/graph.py::_read_payload` each copy-pasted it; the nudge skills block was duplicated verbatim in `bootstrap.py` and `mcp/server.py`. *(2026-06-15: nudge skills load from one `lru_cached` source, `AgentEnvelope.to_dict()` now owns the envelope wire shape, and the inbound serializers delegate to it.)*

6. ✅ **Two daemon architectures shipped side by side.** *(fixed in working tree 2026-06-11 — the parallel `host/daemon_runtime/*` + `domain/ports/daemon/*` runtime and its six unit tests were deleted; only the real daemon remains.)* The real daemon (`host/daemon.py` + `daemon_main.py` + `daemon_rpc.py` + `daemon_client.py`) is wired and tested. The second, more elaborate plugin-registry/IPC-auth/health runtime (~340 lines) was exercised **only by unit tests** — and the `DaemonRuntime` class its `__init__` docstring promised never existed.

7. **Observability is split-brain.** The entire OTLP/correlation/structured-logging bootstrap (`set_observability`, `instrument_adapter`, `correlation_scope`, `configure_logging`) is engaged *only* from `build_ingestion_server`. The live agent spine (`build_host_shell` / CLI / daemon) installs no sink, opens no correlation scope, and uses a separate `configure_cli_logging`. `logging_setup`'s docstring claims it's "called from every entrypoint (HTTP, MCP, CLI)" — false. The product users actually run is dark. *(2026-06-15: the CLI now engages a **separate** Sentry crash sink (`configure_cli_sentry`, #865), so it is no longer fully dark for crash reporting; the daemon entrypoint (`host/daemon_main.py`) still is. The OTLP/correlation/structured-logging bootstrap remains ingestion-server-only — `configure_logging` is now also reached by MCP, but never by the CLI or `build_host_shell`.)*

8. **Capability ports with live callers but no production-backend body.** `neo4j`/`falkordb` declare `inspection=False, snapshot=False`, but `graph.py` calls `host.backend.inspection.neighborhood/.path` and `snapshot.export/import_` unconditionally. On the *default* backend, those CLI commands fail closed with `CapabilityNotImplemented`. The "rebuildable projection never rebuilt" gap: ports exist, toy backends satisfy them, shipping backends don't. *(2026-06-15: half resolved — `inspection` is now implemented on `falkordb`/`falkordb_lite`/`embedded`, so `graph inspect` no longer fails closed on the default backend. Still failing closed: `snapshot` export/import on falkordb, and both capabilities on neo4j; `graph.py:537/557/569` still call them with no `capabilities()` pre-check.)*

9. **Lower-hurt smells worth naming:** two `StatusReport` types and two status paths for one "context_status" concept; the `RemoteSurface` `_NESTED`/`_REMOTE_ATTRS` whitelist in `daemon_client.py` must be hand-synced with `GraphBackend`'s capability ports or new attrs silently proxy as method-call closures. *(2026-06-15: the reader helper duplication and in-memory lexical scorer fork were resolved in Tier 2.)*

## 3. Dead / unused / deletable

**Safe to delete now** (no production importer, claim confirmed, not refuted) — ✅ **all deleted, 2026-06-11 working tree**:

- `adapters/inbound/mcp/tools/` — empty package, zero importers, ships in the wheel as dead weight.
- `adapters/outbound/agent_tools/` (entire ~54KB cluster: `sandbox.py`, `_sandbox_git_tools.py`, `_sandbox_metrics.py`, `_path_safety.py`) — orphaned sandbox-agent-era island; the `add_extra_tools` extension point it targets has **no live caller anywhere**. Delete `tests/unit/test_path_safety.py` with it.
- `adapters/outbound/synthesis/`, `adapters/outbound/github/`, `adapters/outbound/neo4j/` — phantom dirs, no git-tracked files, only stale `.pyc`. Roles relocated (`connectors/github`, `graph/neo4j_*`); synthesis deleted in #851.
- `adapters/outbound/reconciliation/null_agent.py` (`NullReconciliationAgent`) — zero references, not even tests. The codebase standardized on a `None` sentinel for "no agent."
- `adapters/outbound/source_resolvers/github_pull_request.py` — 348-line byte-for-byte clone of `connectors/github/resolver.py`; the `__init__` re-exports the *connectors* copy. Its only import target (`domain.ports.source_control`) doesn't even exist in the canonical tree.
- `domain/ports/graph/claim_query.py` — re-export shim consumed only by the unused `__init__` aggregator; nothing imports the new namespace (the canonical `domain.ports.claim_query` won with 31 importers). Keep `domain/ports/graph/__init__.py` as a package marker, but its re-export body is dead.
- `host/daemon_runtime/*` + `domain/ports/daemon/*` (~340 lines) + the six `test_daemon_{registry,health,context,ipc_auth,ports,operations}.py` — the unwired parallel daemon design.
- `ContextGraphGoal.NEIGHBORHOOD` / `AGGREGATE` enum members — never constructed; `goal` is a non-dispatching passthrough label.
- The stale untracked **`/app/src/context-engine`** tree — gutted shadow copy (5174 `.py`, all a vendored `.venv`; sources already deleted to `.pyc`). Untracked, not gitignored, pollutes every repo-wide grep. `rm -rf` it; repoint the ~9 stale doc/comment path strings to `potpie/context-engine`. *(Tree removed; ~8 stale `app/src/...` path strings still linger in `benchmarks/README.md`, `benchmarks/core/local_engine.py`, two `scripts/` docstrings, and two integration-test docstrings — the only residue of this list.)*

**Probably dead — confirm intent first** (test-only or parked; deletable mechanically but may be a deliberate seam you want to keep) — ✅ **all decided 2026-06-11: deleted, except the test doubles (kept, annotated, un-exported)**:

- `belief.py`'s entire probabilistic confidence-fusion core (`derive_belief`, `score_object`, `Belief`, `ConfidenceLabel`, etc., ~525 lines) — referenced only by its own test; live scoring runs through `ranking.py`. Even the two "live" helpers `ranking.py` imports (`EvidenceStrength`, `decay_weight`) are non-functional re-exports. This is the "cosmetic confidence" abandoned design. Extract nothing; delete or wire deliberately. *(Decided: deleted with `test_belief.py`; `ranking.py`'s dead re-imports removed.)*
- `application/services/temporal_search.py` (~96 lines) — `compute_temporal_flag`/`annotate_search_rows_temporally`, test-only; its sole output key (`superseded_label`) is read by a dead branch in `cli/ui/output.py:338`. Delete both. *(Decided: both deleted, with `test_temporal_search.py`.)*
- `adapters/outbound/reconciliation/timeline_plan.py` (`build_timeline_mutations`) — docstring claims "every plan builder calls" it; **nothing does**. Live timeline path is `semantic_mutation_lowering`. Either delete (clean up the dangling refs in `ontology.py:680` + 2 tests + 3 playbook `.md`) or hoist its `VERB_*` vocabulary into the live path. *(Decided: deleted; the `ontology.py` dangling ref cleaned up, the referencing tests and the jira/linear playbook `.md`s removed with the connector sweep.)*
- `domain/deterministic_extractors.py` — the parked deterministic-extraction strategy; only `test_extractors.py` imports it. A deliberate `TODO(decide)` seam, not accidental cruft. *(Decided: against the seam — deleted with `test_extractors.py`.)*
- `connectors/github/review_threads.py::group_review_threads` — test-only; the live agent tool returns ungrouped review comments (a latent feature gap, not just dead code). *(Decided: deleted; the grouping feature gap stands as an open note.)*
- `connectors/linear/*` end-to-end (`LinearConnector`, `build_linear_tools`, resolver, webhook, events; `LinearIssueFetcher` has no concrete impl) — fully authored, registered nowhere, duplicates a real client in `potpie/integrations`. Auth half is built; consumption half dangles. *(Decided: deleted end-to-end — and the **jira** connector went with it, plus both providers' playbooks, CLI ingest commands, and tests.)*
- `NoOpReconciliationAgent`, `InMemoryEventStreamPublisher` — legitimate test doubles, but exported as peers of the real adapters in their package `__init__`s; reposition or annotate so they don't read as deployment options. *(Decided: kept — docstrings now state "test double, not a deployment option," and `InMemoryEventStreamPublisher` is no longer re-exported from the package `__init__`.)*

**Genuinely broken (fixed in working tree — deleted, not restored):** `adapters/inbound/hatchet/` was a 1-line husk; `legacy/app/modules/context_graph/hatchet_worker.py:16` imported `adapters.inbound.hatchet.worker.run_hatchet_context_graph_worker`, which was **deleted in #858** and `ImportError`ed at startup. Since the server is deprecated to `legacy/`, the legacy entrypoint and the inbound husk package were deleted and the module README's worker row + dead `docs/hatchet-local.md` link removed. The outbound `event.push` adapter and `CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet` remain; consuming those events now requires an externally run Hatchet worker.

**Merge reconciliation — Jira/Linear residue (2026-06-15).** `main` re-added the Jira/Linear connectors, one-shot/diff-sync ingestion skills, CLI triggers, playbooks, and tests (#864/#857/#839/#836); the merge of `main` into `feat/graph-updates` (`f08d1370`) **re-deleted** them. The removal is **internally consistent in code** — `git grep` finds no surviving import of a deleted `connectors.{jira,linear}` module, no dangling Typer registration in `pots.py`, and `event_playbooks.py`/`_bench_stubs.py` are reconciled (the surviving `LinearStubConnector` is an intentional passive bench stub, not residue). 1584 tests still collect with 0 errors. What remains is **one orphaned module plus broader doc residue**:

- `domain/sync_cursor.py` (+ `tests/unit/test_sync_cursor.py`) — the diff-sync cursor parser is now **dead**; its only callers were the deleted jira/linear diff-sync commands, and its docstring still claims a live "incremental diff-sync paths" feature. Delete it, or strip the claim if reserving it for a future generic path.
- `application/use_cases/context_graph_jobs.py:8` cites a `Linear linear_team.added` backfill example, and `domain/playbooks/repo_one_shot_ingestion.md:14` says "Sibling to `linear_team_one_shot_ingestion`" — both reference deleted artifacts. Prune.
- Bundled reconciliation skills still recipe Linear tooling that has **no registered implementation**: `reconciliation/skills/backfill-enumerate-drain/SKILL.md:27` (`linear_get_*`), `graph-mutation-plan/SKILL.md:41-43` (`linear:*` key rows); `source-ingestion` `SKILL.md:91` and `change-timeline` `SKILL.md:11` still advertise Linear/Jira sources, and `pots.py:142` `source add` help still lists `linear` as a kind.
- ✅ **Strategic decision made in Tier 2:** harness-led generic ticket/issue ingestion remains intended (agent reads ticket text → writes mutations, no connector); deleted Jira/Linear connector-tool-name references were pruned from bundled skills/templates and CLI source help.

## 4. Drift from the documented design

Remaining docs drift is now narrower. `architecture.md` was corrected for the shipped write contract and default backend in the Tier 4 cleanup pass; the plan queue and some operational docs still describe a system the code doesn't match, in both directions:

- ✅ **The write contract in the doc never shipped.** `architecture.md` specified `potpie graph propose|commit` with a server-created `plan_id` and a planned `domain/ports/graph_workbench.py`. None exist (`rg graph_workbench` → 0). The real surface is single-call `GraphService.mutate()`. *(Resolved 2026-06-15: `architecture.md` now documents `potpie graph mutate` → `GraphService.mutate()` → `GraphBackend.mutation.apply`, with dry-run/apply semantics and no shipped `plan_id` contract.)*
- **The plan's own IR-01..IR-12 fix queue had been stale both ways.** IR-03
  (inline-relation assembly) and IR-07 (vector search wired, `fact_embedding`
  written/queried, and V1.5 metadata hydrated on Neo4j/FalkorDB) are now marked
  fixed in the implementation plan; IR-08+ remain the active queue.
- ✅ **Default backend was documented wrong in three files.** `architecture.md` and `backends/__init__.py` called `embedded` the OSS default and never mentioned `falkordb_lite`; `backend.py` still tagged `embedded`/`postgres`/`hosted` as `(TODO)`. The actual default (`host_wiring.py`, Python ≥3.12) is `falkordb_lite`. *(Resolved 2026-06-15: `backends/__init__.py` had already been corrected, and Tier 4 corrected `architecture.md` plus `domain/ports/graph/backend.py`.)*
- **Playbooks instruct the LLM to emit vocabulary the ontology doesn't have.** `event_playbooks.py` tells the reconciliation agent to seed `Module`, `Feature`, `Incident`, `DiagnosticSignal`, `DECIDES_FOR` — **none** exist in `ontology.py`. The validator rejects them and silently downgrades to generic `Document`/`RELATED_TO`. This is the exact "duplicate ontology in prose" anti-pattern the Rules section forbids, and `coherence.py`'s import-time checks don't cover playbook labels, so it's invisible at startup. *(2026-06-15: `Feature`/`PROVIDES`/`IMPLEMENTED_IN` were since added and **are real** (`ontology.py:383-395,657-672`); only `Module`/`Incident`/`DiagnosticSignal`/`DECIDES_FOR` remain phantom (`event_playbooks.py:250,252,288,290,320`). The validator still silently downgrades the phantoms to `Document`/`RELATED_TO`, and the startup blind-spot stands.)*
- **Backend-divergent claim metadata (fixed in working tree).** Neo4j/FalkorDB's
  shared `row_from_record` now hydrates first-class V1.5 `ClaimRow` fields
  (`truth`, `subgraph`, `environment`, `claim_key`, `valid_until`,
  `source_refs`, `evidence`, versions) and removes those contract keys from the
  legacy `properties` fallback bag.
- ✅ **Catalog over-advertises.** Views declared `ranking_inputs` like `truth_class`/`resolution_status` that `RankingService` never computes, and `semantic_match` aliases that are really `semantic_similarity`. *(2026-06-15: `graph catalog --subgraph` is plumbed through and filters views, and Tier 2 reconciled `graph_views.py` `ranking_inputs` with `RankingService._score_one`.)*
- **Coherence guard runs on the wrong root.** `assert_runtime_coherence()` (reader/contract reconciliation) fires only from `build_ingestion_server`, not `build_host_shell` — so the canonical OSS spine can register a reader without updating the advertised contract and produce the exact silent-retrieval failure the check exists to prevent.
- **Daemon-default narrative inverted.** The daemon is the *runtime default* (`get_host()` builds `RemoteHostShell` unless `HOST_MODE=in_process`; `SetupPlan.host_mode='daemon'`), yet `shell.py`/`host_wiring.py` docstrings say "in-process by default" and the `Daemon` dataclass defaults `in_process=True`. A reader concludes the daemon is dev-only when it's the primary path.
- **Benchmark docs point at the deprecated server**: README references the stale `app/src/...` tree, gunicorn-on-8001 + Celery worker, `/events/reconcile` — none of which match the host-CLI engine. A reader cannot run a single scenario today.
- **Smaller honest-surface gaps:** ✅ `context_record` validation asymmetry is fixed — validation errors surface via `mcp/server.py:221` `_error` and HTTP 400, managed HTTP validates in `record_durable_context.py`, and Tier 4 made `record_to_semantic_request()` always validate known schemas on the local MCP/CLI path. Remaining: `DefaultSkillManager.add()` returns a success-shaped no-op to a live `potpie skills add` command (should raise `CapabilityNotImplemented` like the engine's own convention).

## 5. Prioritized recommendations

1. **Collapse the two graph-service facades onto one DTO surface (L).** Port the HTTP router off `ContextGraphService`/`ContextGraphQuery` onto `DefaultGraphService`/`AgentEnvelope`, then delete `ContextGraphService` and the `ContextGraphPort`/`ContextGraphQuery` vocabulary. This is the highest-leverage fix; it also makes the `agent_context.py` "sole inbound binding" claim true again. Pairs naturally with adding `AgentEnvelope.to_dict()` (or reusing `envelope_to_dict`) to kill the 3–4-site wire-shape duplication.

2. ✅ **Sweep the confirmed-dead tail in one PR (S).** *(Done 2026-06-11 on `feat/graph-updates`, and went beyond the list: the entire §3 "probably dead" set was also resolved, plus the jira connector, the sync_history stack, and the broken hatchet entrypoint — ~7,600 lines removed.)* Delete the §3 "safe to delete now" list: empty husks (`mcp/tools`, `synthesis`, `github`, `neo4j` dirs), `agent_tools/` cluster, `null_agent`, the `source_resolvers` clone, the `graph/claim_query` shim, `daemon_runtime`/`ports.daemon` + their tests, the unused enum members, and `rm -rf app/src/context-engine`. Mechanical, low-risk, and it makes every future repo-wide grep trustworthy.

3. ✅ **Decide the managed-record principle and fix `no_reconciliation_agent` (M).** *(Done for durable `context_record`, 2026-06-15: managed records route through `GraphService.record`, local/managed structured validation is symmetric, and a no-agent managed-path test covers it. Non-record managed ingestion remains model-mediated.)* Either route managed `context_record` through the same deterministic semantic-mutation path as local (honoring the V1.5 non-negotiable, with a managed-path test that succeeds with no agent), or explicitly document the managed pipeline as model-required and stop claiming otherwise. This is the largest principle-vs-implementation gap.

4. **Reconcile the docs in a single pass (M).** *(Partially done 2026-06-15: `architecture.md` write-contract/default-backend drift and `backend.py` default-backend drift are fixed; the playbook vocabulary check landed in Tier 3.)* Remaining docs work: benchmark docs, the IR-01..IR-12 queue, daemon-default narrative, and a final operational-doc spot check.

5. **Restore the benchmark corpus and turn the smoke test green (M).** The `benchmarks/fixtures/raw_events/universe/acme` tree is missing (and root `.gitignore`'s blanket `*.json` makes it uncommittable). Restore it with a `fixtures/` allowlist exception, repoint `core/local_engine.py`/`core/query.py` off the deprecated server onto the host CLI / `DefaultGraphService`, and wire the V1.5-aligned `retrieval_eval.py` (R7) into `cli.py`. Without this you have no signal that any of the above changes work. *(2026-06-15: it is the **whole** `benchmarks/fixtures/raw_events/` tree that's gone, not just the universe seed; even the engine-free `smoke` gate is red. A server-free in-process driver already exists (`core/local_engine.py`, via `run --local`/`POTPIE_BENCH_INPROCESS=1`), so the repoint is partly done — make `smoke` then `run-light --local` green. `core/query.py` is still HTTP-only. The orphaned R7 harness is `retrieval_eval.py` (recall@k/MRR); `evaluators/retrieval.py` is already wired — don't conflate them.)*

6. **Engage observability on the agent spine (S–M).** Call the existing `configure_logging` + `set_observability` from `build_host_shell` and open a `correlation_scope` per CLI/daemon invocation, and move `assert_runtime_coherence()` to fire on `build_host_shell` too. The machinery is built; it just isn't connected to the product people run.

7. **Consolidate the five P9 reader helpers into `readers/_common.py` (S).** `_make_candidate_key`, `_corroboration`, `_payload_from_row`, the scope→anchor logic — the shared home already exists and is unused for these. Low blast radius on the one subsystem that's actually on the live roadmap. While there, unify the lexical scorer so conformance ground-truth stops diverging from production.

Files of record for the above: `application/services/graph_service.py`, `adapters/outbound/graph/context_graph_service.py`, `application/services/ingestion_submission_service.py`, `bootstrap/{host_wiring,ingestion_server,container}.py`, `domain/event_playbooks.py`, `adapters/outbound/graph/canonical_claim_query.py`, `application/readers/_common.py`, `docs/context-graph/{architecture.md,graphv1-5-implementation-plan.md}`, `benchmarks/`.

## 6. Current CLI smoke-test findings and next work queue

> **2026-06-15 verification:** the entire **P0 queue (6.1–6.5) is still open** — the reader/lowering files are untouched since the audit baseline commit `fd2876d7`. P1/P2 progress is annotated per-subsection below (6.6/6.7/6.8 partially landed; 6.9/6.10 open).

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

> _2026-06-15: **partially done** — `InfraTopologyReader` honors `include_unqualified_environment` (`infra_topology.py:56-71`) and the catalog hints it, but there is still no `--include-unqualified-environment` flag on `graph read` (only `--scope include_unqualified_environment:true`), no service-anchor default, and no `environment_qualified_edges`/`global_edges` split._

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

> _2026-06-15: **ingestion side addressed** — a `feature`/`repo-baseline` `graph mutation-template` kind and the `potpie-repo-baseline` skill landed. The **reader is still open**: `features.provided` is backed by the generic `InfraTopologyReader` (`graph_views.py:162-175`, `v1_include=infra_topology`); no dedicated `FeatureReader` exists._

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

> _2026-06-15: **partially done** — recipes shipped as `graph mutation-template --kind {repo-baseline,feature,preference,bug-fix,decision}` (`graph.py:275,437-463`) rather than `graph add …` subcommands. Still open: `dependency` and `event` kinds, and chaining the template into `graph mutate` (it currently only prints schema JSON)._

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

## 8. Cleanup next-steps (consolidated, 2026-06-15)

The verification pass turns the audit into a concrete cleanup backlog. §5 recommendations still hold; this section supersedes their status and folds in the merge residue and doc drift. Ordered by leverage ÷ risk, so a contributor can pick up Tier 1 with no design decisions.

**Tier 1 — mechanical, zero-risk, do first (one PR, makes every grep trustworthy):**

1. Find/replace the 8 stale `app/src/context-engine/...` path strings → `potpie/context-engine/...` (`benchmarks/README.md:73`, `benchmarks/core/local_engine.py:43`, `scripts/context_engine_lab.py:12`, `scripts/benchmark_context_engine.py:6-8`, `tests/integration/test_cli_auth_{atlassian,linear}_e2e.py:12`). Leave `domain/ingestion_kinds.py:11` (`app/src/integrations`) — that path is valid.
2. Delete `domain/sync_cursor.py` + `tests/unit/test_sync_cursor.py` (orphaned by the diff-sync removal; zero live callers) — or strip the misleading "incremental diff-sync paths" docstring if reserving it.
3. Prune the two dangling cross-refs: `context_graph_jobs.py:8` (`linear_team.added` example) and `repo_one_shot_ingestion.md:14` ("Sibling to `linear_team_one_shot_ingestion`").
4. Delete the dead `_ResolutionServiceShim.set_source_resolver` / `resolution_service` field from `bootstrap/container.py` (no readers anywhere in the tree).
5. Fix `bootstrap/logging_setup.py:8-9` docstring — it claims "called from every entrypoint (HTTP, MCP, CLI)" but the CLI never calls it.

**Tier 2 — small refactors (low blast radius):**

6. ✅ `AgentEnvelope.to_dict()` added and the three inbound serializers (`mcp/server.py`, `cli/commands/query.py`, `cli/commands/graph.py`) refactored onto it (2026-06-15).
7. ✅ Single `register_bench_stubs(registry)` helper factored and called from both `ingestion_server` and `standalone_container` (2026-06-15).
8. ✅ Shared claim key / corroboration / payload / anchor helpers hoisted into `readers/_common.py`; current readers use them, and `in_memory_reader` now uses `canonical_claim_query.embedding_score` for lexical ranking (2026-06-15).
9. ✅ `graph_views.py` `ranking_inputs` now match `ranking.py`: stale `truth_class` / `resolution_status` removed and `semantic_match` aliases renamed to `semantic_similarity` (2026-06-15).
10. ✅ Strategic decision: keep harness-led generic ticket/issue ingestion, but prune deleted Jira/Linear connector/tool references from bundled skills and CLI source help (2026-06-15).

**Tier 3 — engage the dark machinery (small–medium):**

11. ✅ Call `configure_logging` + `set_observability` and open a `correlation_scope` from `build_host_shell`; wire crash telemetry into `host/daemon_main.py` (done 2026-06-15: host shell now installs the shared observability sink, daemon startup configures logging/Sentry, and daemon health/RPC/attr handlers run under correlation + server spans).
12. ✅ Move `assert_runtime_coherence(...)` to fire from `build_host_shell` too, and add a 7th coherence check that asserts every playbook label/predicate exists in `ENTITY_TYPES`/`EDGE_TYPES` (done 2026-06-15: playbook vocabulary drift now fails runtime coherence; the stale `Module`/`Incident`/`DiagnosticSignal`/`DECIDES_FOR` prose was rewritten to canonical `CodeAsset`/`Observation`/`BugPattern`/`DECIDED`/`AFFECTS`/`IMPLEMENTED_IN` vocabulary).
13. ✅ Add a `capabilities()` pre-check (or graceful message) to `graph.py` inspect/export/import rather than relying on the deep `CapabilityNotImplemented`; implement `snapshot` for falkordb and inspection/snapshot for neo4j, or hide those commands until backed (done 2026-06-15 via graceful preflight: unsupported inspect/export/import now return structured `not_implemented` before touching the unimplemented port).

**Tier 4 — load-bearing structural / principle fixes (medium–large):**

14. **Partial:** Port the HTTP router off `ContextGraphQuery`/`ContextGraphResult` onto `AgentEnvelope`, then delete the `ContextGraphService` translator + `ContextGraphPort` DTO vocabulary — makes the host the sole inbound binding. (2026-06-15: the HTTP timeline/read path now uses `GraphService.resolve` + `AgentEnvelope`, and the public legacy `ContextGraphQuery` endpoint remains explicitly unsupported. Still open: `container.context_graph` is used for hard reset and reconciliation-agent compatibility, so the translator/port vocabulary cannot be deleted yet.)
15. ✅ Decide the managed-record principle and fix `no_reconciliation_agent` (`ingestion_submission_service.py:122-123`): managed `context_record` now short-circuits to deterministic `GraphService.record`; non-record submissions still require the reconciliation agent. Local validation is symmetric because `record_to_semantic_request()` no longer gates validation behind `has_structured_details()`. Covered by `tests/unit/test_ingestion_submission_service.py`.
16. ✅ Reconcile `architecture.md`: deleted the phantom `propose|commit`/`plan_id`/`graph_workbench.py` write contract and replaced it with the single-call `graph mutate` → `DefaultGraphService.mutate()` flow; fixed the default-backend docs in `architecture.md` and `domain/ports/graph/backend.py` to `falkordb_lite`.
17. **Open by choice for this pass:** Restore the `benchmarks/fixtures/raw_events/` corpus with a `.gitignore` allowlist (`!…/benchmarks/fixtures/**/*.json`), make `smoke` then `run-light --local` green (the server-free `core/local_engine.py` driver already exists), and add a `retrieval-eval` subcommand wiring the orphaned R7 `retrieval_eval.py`.

**Tier 5 — the §6 correctness queue (P0 first):** unchanged and entirely open — see §6/§7. Land 6.1–6.5 before more ingestion breadth.
