# Context Graph — Scaffolding & Handover Plan

The docs in `docs/context-graph/` define the **end state**. This file is the
work-breakdown to get `app/src/context-engine/` there, organized so **each
feature and flow is an independently-ownable unit** that can be handed to a
different person.

Last deep-dived: 2026-06-01 (full code-vs-docs reconciliation across all six
subsystems).

> **Wave 1 landed (2026-06-01).** Doc fixes **D1/D2/D4**, **HU1** (setup value
> objects + `preview`), **HU16** (state-store + migration seams), **HU5** (ledger
> `query()` + `use`/`disconnect` CLI), **HU10a** (postgres/chroma/hosted stub
> backends), and **HU8** (config wired to `LocalConfigService`; `login`/`logout`/
> `whoami`/top-level `use`; `pot list` filters; `ingest show/replay/retry/
> dead-letter`; `cloud skills sync`) are all in. 870 tests green; every
> `cli-flow.md` command now exists (real or structured not-implemented). Next:
> **Wave 2 — HU4** (durable consumer run store).

## How to read this

A flow is **handover-ready** when all five seam dimensions exist, even if the
body is a stub:

| Dim | Means |
|---|---|
| **Port** | a named interface in `domain/ports/…` |
| **Wiring** | composed in a root (`bootstrap/host_wiring.py` local, `bootstrap/ingestion_server.py` managed) — even to a stub |
| **Body** | a real implementation, OR a clean stub raising `domain.errors.CapabilityNotImplemented("<dotted.slot>")` (never a bare `NotImplementedError`) |
| **Inbound** | a CLI command / MCP tool / HTTP route, even if it returns the structured not-implemented contract |
| **CodeMap** | a row in `architecture.md` so the owner can find it |

Each item is tagged:

- **[SCAFFOLD]** — stand up the seam (port + stub + wiring + inbound slot + Code Map row). Mostly mechanical; do these first so everything is ownable.
- **[BODY]** — implement real logic behind an existing/scaffolded seam. This is the feature work an owner picks up.
- **[DECISION]** — a product/architecture choice must be made first.
- **[DOC]** — a spec-internal inconsistency to fix (the only doc-side edits; do **not** otherwise edit the spec to match the code).

> Format note: the docs are the target. Code grows toward them. The `[DOC]`
> items are genuine doc-internal inconsistencies (spec ahead of itself), not
> "lower the bar to match the code."

Old gap IDs (G1–G7) are carried in parentheses for continuity.

---

## Already matches the spec — do not touch

- **4-tool agent contract → `AgentEnvelope`.** `resolve`/`search`/`record` are real end-to-end. (`domain/ports/agent_context.py`, `application/services/agent_context.py`, `application/services/graph_service.py`.)
- **5 reader-backed includes** real and returning data: `coding_preferences`, `infra_topology`, `timeline`, `prior_bugs`, `raw_graph` (`application/readers/`, `application/services/read_orchestrator.py`). Planned includes correctly surface as `unsupported_includes(reason=not_implemented)`.
- **6 structured record types** with schemas + builders: `preference`, `policy`, `bug_pattern`, `fix`, `verification`, `decision` (`domain/context_records.py`, `domain/ontology.py`). Lowering `record_type → ontology predicate` works (`graph_service._lower_record`).
- **One read trunk, no server-side synthesis.** `ContextGraphService` is now a thin `ContextGraphPort` envelope over **one** `GraphBackend` — reads via `backend.claim_query`, writes via `backend.mutation.apply_async` (G1b done; `adapters/outbound/graph/context_graph_service.py`).
- **`HostShell` facade + `build_host_shell` + two composition roots** (`host/shell.py`, `bootstrap/host_wiring.py`, `bootstrap/ingestion_server.py`).
- **`in_memory` + `embedded` backends**: all six capability ports real, persistence (embedded) real, both pass the conformance suite (`adapters/outbound/graph/backends/`, `tests/conformance/test_graph_backend_conformance.py`).
- **4 working-tree scanners** real + `ingest scan` wired end-to-end through `IngestService → GraphMutationPort` (CODEOWNERS, dependency manifest, k8s, OpenAPI; `adapters/outbound/scanners/`).
- **CLI contract**: exit codes 0–4, JSON error shape (`code`/`message`/`detail`/`recommended_next_action`), and `CapabilityNotImplemented` rendered as the structured not-implemented contract (`adapters/inbound/cli/commands/_common.py`). MCP exposes exactly the 4 tools over `build_host_shell` (`adapters/inbound/mcp/server.py`).
- **Skill Manager**: catalog/install/update/remove/status real; `SkillNudge` surfaced in `context_status` (`application/services/skill_manager.py`, `adapters/outbound/skills/`).
- **Observability**: `ObservabilityPort` + console + OTLP adapters + runtime real (`domain/ports/observability.py`, `adapters/outbound/observability/`). Reconciles with `observability.md`; call-site span coverage is still being filled in (see HU18).
- **Per-(pot, source) ledger cursor** — verified correct: `LedgerCursor` carries `source_id`, store keys on `{pot_id}:{cursor.source_id}` (`domain/ports/ledger/cursor.py`, `adapters/outbound/ledger/cursor_store.py`). **(old G5 — RESOLVED, was a misread.)**
- **`ConfigService.set`** persists atomically at the service layer (`application/services/config_service.py:50`). *(CLI command not yet wired to it — see HU8.)*
- **Snapshot** real for `in_memory` + `embedded` (`domain/ports/graph/snapshot.py`).
- **Reconciliation `pydantic_deep_agent`** real and live in the managed ingestion path (`adapters/outbound/reconciliation/`, `bootstrap/ingestion_server.py`).

---

# Part A — Scaffolding to stand up now

These make each flow **ownable** even before its body exists. Small, mostly
mechanical, low-risk. Do these first.

## HU1. Setup/lifecycle value objects + `preview` shape — [SCAFFOLD] ✅ landed 2026-06-01
- **Priority** P1 · **Effort** S · **Depends on** none
- **Spec**: `SetupPlan` carries `host_mode` (`daemon | in_process`); `SetupOrchestrator.preview(plan) → SetupPreview(steps: tuple[PlannedSetupStep], ok_to_run)`; `LoginPlan` exists. (`architecture.md:209-256`.)
- **Current**: `domain/lifecycle.py` has `SetupPlan` (no `host_mode`), `StepResult`, `SetupReport` only — **missing** `PlannedSetupStep`, `SetupPreview`, `LoginPlan`. The port (`domain/ports/services/setup.py:23`) exposes `plan(plan) → list[StepResult]`, not `preview(plan) → SetupPreview`. Hard/soft is hardcoded per step in `setup_orchestrator.py:66-80`, not derived from `host_mode`.
- **Close**: add `host_mode` to `SetupPlan`; add `PlannedSetupStep`, `SetupPreview`, `LoginPlan` dataclasses; rename/extend the port to `preview() → SetupPreview` (keep `plan` as alias during migration); derive hard/soft from `host_mode` (daemon-install hard for `daemon`, skipped for `in_process`).
- **Acceptance**: `potpie setup --dry-run` returns a `SetupPreview` with per-step owner, hard/soft, and skip reason; `host_mode` flips daemon-install hardness.

## HU16. Relational state-store + migration seam — [SCAFFOLD] ✅ landed 2026-06-01
- **Priority** P2 · **Effort** S · **Depends on** none
- **Spec**: setup step 4 is `pot_service.init` + `state_store.provision` + `migrator.migrate` as independently-ownable methods (`architecture.md:196,269` Seam→owner map row 4).
- **Current**: `LocalPotManagementService.init` just `mkdir`s the home (`application/services/pot_management.py:29`). No migration port exists anywhere; `state_store.provision`/`migrator.migrate` are documented but have **no seam**.
- **Close**: add a `MigrationPort` (`migrate() → StepResult`) and a `StateStorePort.provision()` seam under `domain/ports/`; wire stubs into `host_wiring` and the orchestrator's `pot.init` step; flat-file store returns `SKIPPED`.
- **Acceptance**: the SQLite/Postgres owner has a `migrate()` slot to fill; flat-file profile reports `skipped` cleanly.

## HU5. `EventLedgerClientPort.query()` + ledger binding CLI — [SCAFFOLD] ✅ landed 2026-06-01
- **Priority** P2 · **Effort** S · **Depends on** none
- **Spec**: `ledger query` inspects history without touching consumer cursor (distinct from cursor-based `pull`); `ledger use managed|self-hosted`, `ledger disconnect`; the listing command is `ledger sources list`. (`cli-flow.md:240-264`.)
- **Current**: port has `fetch`/`sources`/`health` only — **no `query()`** (`domain/ports/ledger/client.py:72`). CLI has `status`/`sources`/`pull` only; **missing** `use`/`query`/`disconnect`; exposes `ledger sources` (doc says `ledger sources list`). (`adapters/inbound/cli/commands/ledger.py`.)
- **Close**: add `query(...)` to `EventLedgerClientPort` (+ stub on managed/self-hosted clients raising CNI, real on fixture); add `ledger use/query/disconnect` command slots; rename `ledger sources` → `ledger sources list`.
- **Acceptance**: all `ledger` subcommands in `cli-flow.md` exist (real or structured not-implemented); naming matches the doc.

## HU8. Complete the CLI command slots — [SCAFFOLD]  (old G6) ✅ landed 2026-06-01 (ingest history slots ride on HU4/HU6)
- **Priority** P2 · **Effort** M · **Depends on** HU1 (login), HU3 (routing), HU4 (ingest history)
- **Spec**: every command in `cli-flow.md` "Command Groups" (`cli-flow.md:133-318`).
- **Current** (gaps only):
  - **Missing slots**: `login`/`logout`/`whoami`; top-level `potpie use`; `pot list` filters `--local/--managed/--all` (accepted but ignored); `ingest show/replay/retry/dead-letter`; `ledger use/query/disconnect` (HU5); `cloud skills sync`.
  - **Advisory-but-advertised-real**: CLI `config get` reads `os.getenv` and `config set` returns `persisted: False` — **not wired to `LocalConfigService`** (`adapters/inbound/cli/commands/bootstrap.py`), even though the service persists. `ingest status`/`runs` are placeholders; `backend use` advisory; `cloud push/pull/login/status` raise CNI.
  - **Signature drift**: `source add <kind> <location>` vs doc `source add repo <path>`.
- **Close**: add the missing command slots (most ride on HU3/HU4); **wire `config get/set` to `LocalConfigService`**; align `source add` + `ledger sources list`.
- **Acceptance**: every `cli-flow.md` command is real or maps to an open HU; `config get/set` round-trips through the persisting service.

## HU10a. Backend profile registry stubs — [SCAFFOLD] ✅ landed 2026-06-01
- **Priority** P3 · **Effort** S · **Depends on** none
- **Spec**: default profiles include `postgres/pgvector`, `chroma`, and a hosted profile behind the same ports (`architecture.md:546-553`).
- **Current**: `build_backend` resolves only `in_memory`, `embedded`, `neo4j`; `KNOWN_PROFILES` omits the rest; an unknown profile raises a bare `ValueError` (`adapters/outbound/graph/backends/__init__.py`).
- **Close**: register `postgres`, `chroma`, `hosted` as stub backends whose capability ports raise `CapabilityNotImplemented("graph.<profile>.<cap>.<method>")` and whose `provision()` returns `not_implemented` — so `backend use <profile>` is a real (if unbuilt) seam, not a crash.
- **Acceptance**: `potpie backend list` shows all doc profiles; selecting an unbuilt one returns the structured not-implemented contract.

---

# Part B — Bodies to hand over

Each is an independently-ownable feature behind a seam that exists (or is
scaffolded in Part A).

## HU4. Event Ledger durable consumer ingestion ledger — [SCAFFOLD+BODY]  (old G4)
- **Priority** P1 (largest missing port) · **Effort** L · **Depends on** none
- **Spec**: `ledger pull --apply` durably **enqueues** events, **advances the cursor after enqueue** (not after apply), then processes from a run store through the six states `pending/processing/applied/failed_retryable/failed_terminal/timed_out` with retry, lease timeout, and dead-letter. Backed by `LedgerEventRunStorePort` (`run_store.py`). (`architecture.md:368-426`, state table `407-414`.)
- **Current**: **no `run_store` port or adapter** — `domain/ports/ledger/` has `client/cursor/reconciler` only. `LedgerFacade.pull` (`host/shell.py:67-79`) is synchronous: fetch → reconcile → advance cursor immediately. No per-event state, retry, lease, or dead-letter.
- **Close**: [SCAFFOLD] add `domain/ports/ledger/run_store.py` (`LedgerEventRunStorePort`) + a local adapter; [BODY] rework `LedgerFacade.pull --apply` into enqueue → advance-cursor → process-with-state-machine; add retry/timeout/dead-letter.
- **Acceptance**: pulled events land in the run store with state; cursor advances after durable enqueue; a conformance test exercises the six-state machine + replay.

## HU7. Local async ingest worker — [BODY]
- **Priority** P2 · **Effort** M · **Depends on** HU4
- **Spec**: pulled events are processed from consumer state through ingestion into the Graph Service (`architecture.md:384-405`).
- **Current**: the async pipeline (`process_batch`, `flush_windowed_batches`, `reap_stale_batches`) is real but lives only in the managed/deferred root (`application/use_cases/`, `bootstrap/ingestion_server.py`); the local spine has no worker. After HU4 enqueues events, nothing drains them locally.
- **Close**: wire a local worker that leases from the run store, runs the `EventReconcilerPort`, and marks applied/failed. The `EventReconcilerPort` body is the **parked LLM-vs-deterministic decision** (`adapters/outbound/ledger/reconciler.py` is a trivial `RELATES_TO` placeholder; candidate bodies: `adapters/outbound/reconciliation/pydantic_deep_agent.py`, `domain/deterministic_extractors.py`). See HU15-adjacent decision.
- **Acceptance**: `ledger pull --apply` followed by worker drain yields applied claims; failures land in retry/dead-letter.

## HU6. Ingest run-history CLI — [SCAFFOLD+BODY]  (part old G6)
- **Priority** P2 · **Effort** M · **Depends on** HU4
- **Spec**: `ingest status/runs/show/replay/retry/dead-letter` (`cli-flow.md:218-227`).
- **Current**: `scan` real; `status`/`runs` are stubs ("not persisted"); `show/replay/retry/dead-letter` missing (`adapters/inbound/cli/commands/ingest.py`).
- **Close**: back these by the HU4 run store; add the command slots.
- **Acceptance**: every `ingest` subcommand reflects real run state.

## HU2. Detached local daemon — [SCAFFOLD+BODY]  (old G2)
- **Priority** P2 · **Effort** L · **Depends on** HU1 (`host_mode`)
- **Spec**: `potpie setup` installs/starts a real background daemon owning process lifecycle, IPC, health, log files, service-manager registration; "daemon first, then dependencies." (`architecture.md:106-136,171-179`.)
- **Current**: `host/daemon.py` is an in-process stand-in — `status`/`health` report `in_process`; `install/start/stop/restart` raise `CapabilityNotImplemented`; `ensure()` returns `SKIPPED` in-process. `Daemon` is a **concrete dataclass, not a Protocol port**.
- **Close**: [SCAFFOLD] extract a `Daemon` Protocol so in-process and detached variants are swappable; [BODY] implement detached process + socket/HTTP IPC + log files + service-unit install/start behind the `daemon.ensure` **hard** dep for `host_mode=daemon` (keep in-process for dev/tests).
- **Acceptance**: `potpie daemon start/stop/restart/status/logs` drive a real process; `setup` registers + starts it; in-process still works for tests.

## HU3. Managed login + managed pot routing — [SCAFFOLD+BODY]  (old G3)
- **Priority** P2 · **Effort** L · **Depends on** HU1 (`LoginPlan`)
- **Spec**: `potpie login` authenticates to `cloud.backend_url`, stores the session, makes managed pots visible in the same pot surface; commands route by selected-pot origin; `cloud push/pull` + managed skill sync explicit. (`architecture.md:274-293`, `cli-flow.md:58-93`.)
- **Current**: no `login/logout/whoami`; no managed routing; `cloud push/pull/login/status` raise `CapabilityNotImplemented` (`adapters/inbound/cli/commands/cloud.py`). Everything resolves to the local in-process host. The managed HTTP surface + parent-app adapter exist (`adapters/inbound/http/`, `app/modules/context_graph/`) but the CLI never routes to them.
- **Close**: implement login/session storage, managed pot listing, origin-aware routing in `commands/_common.py` pot resolution, and explicit `cloud push/pull`/skill-sync.
- **Acceptance**: after `potpie login`, a managed pot can be selected and `resolve/search/record/status` route to the managed backend.

## HU9. Finish managed-stack unification — [BODY]  (old G1)
- **Priority** P1 · **Effort** L · **Depends on** HU2/HU3 landing first
- **Spec**: the same **Graph Service** runs in the local daemon and the managed backend API; only host + storage adapters change. Pot Management is **not** shared — the backend keeps its own pots behind the thin `PotResolutionPort` (scope decision 2026-05-29). (`architecture.md:46-49,577-592`; see also `[DOC] D2`.)
- **Current**: data plane is unified (**G1b done** — `ContextGraphService` is one `GraphBackend` facade). What remains: the managed async ingestion pipeline still has its own composition root (`bootstrap/ingestion_server.py`: connectors + reconciliation + Postgres) not yet migrated onto `HostShell`.
- **Close**: migrate the async pipeline onto `HostShell` over time; connectors/reconciliation/ledger hang off the record/ingestion seam, not a parallel stack.
- **Acceptance**: managed and local differ only in backend profile + storage adapters; no second data-plane implementation.

## HU10b. Neo4j projections + real embedded vectors — [BODY]  (Impl Order step 3)
- **Priority** P2 · **Effort** L · **Depends on** none
- **Spec**: embedded backend has real vector search; neo4j is a full production profile (`architecture.md:546-553`, Implementation Order step 3 `:738`).
- **Current**: embedded/in_memory semantic search is a **Jaccard token-overlap placeholder**; `fact_embedding` is declared but never populated (`adapters/outbound/graph/in_memory_reader.py:120-134`). Neo4j `semantic`/`inspection`/`snapshot` are CNI stubs and `provision` is a CNI stub (`adapters/outbound/graph/backends/neo4j_backend.py`); neo4j is **not** in the conformance `FULL_PROFILES`.
- **Close**: real embedder + local vector index for embedded (SQLite + vectors); implement neo4j semantic/inspection/snapshot/provision; parametrize conformance for a live neo4j.
- **Acceptance**: embedded semantic search uses real vectors; neo4j passes the full conformance grid.

## HU11. Snapshot (neo4j) + cloud push/pull — [BODY]  (Impl Order step 8)
- **Priority** P3 · **Effort** M · **Depends on** HU10b (neo4j snapshot)
- **Spec**: portable pot export/import + explicit `cloud push/pull` of snapshots (`architecture.md:104,540`, `cli-flow.md:229-235`).
- **Current**: snapshot real for in_memory/embedded; neo4j snapshot is a CNI stub; `cloud push/pull` raise CNI.
- **Close**: implement neo4j `GraphSnapshotPort` (mirror in-memory) and wire `cloud push/pull` to snapshot transfer.
- **Acceptance**: `graph export/import` works on neo4j; `cloud push/pull` move a pot snapshot.

## HU12. `context_status` owner-sectioned structure — [BODY]
- **Priority** P3 · **Effort** M · **Depends on** HU4 (ledger section)
- **Spec**: six owner-scoped sections — `host`, `pot`, `graph_service`, `backend`, `ledger`, `skills` (`architecture.md:498-510`).
- **Current**: `StatusReport` is flat — `data_plane` + `pot_summary` + `skills` (`application/services/agent_context.py:48-69`). **Missing** a real `host` section (only `daemon_up: bool`, hardcoded), a separate `backend` section, and the `ledger` section entirely; `graph_service` omits record types / split supported-vs-unsupported includes.
- **Close**: reshape `StatusReport` into the six sections, each populated by its owner.
- **Acceptance**: `potpie status --json` returns all six sections; each is debuggable independently.

## HU13. Planned reader includes (decisions / docs / owners) — [BODY]
- **Priority** P3 · **Effort** M · **Depends on** none
- **Spec**: `decisions`, `docs`, `owners` move from `unsupported_includes` to backed includes (`architecture.md:463-468`).
- **Current**: correctly surfaced as `unsupported(not_implemented)`. `decision` records already lower to `DECIDED` (`domain/ontology.py:1381`) but there is no `DecisionsReader`; `doc_reference` emits `RELATES_TO` (no dedicated predicate/reader); `owners` is structural-only (OWNED_BY traversal) with no reader.
- **Close**: build `DecisionsReader`, `DocsReader`, `OwnersReader`; register in `read_orchestrator` + `READER_BACKED_INCLUDES`.
- **Acceptance**: each include returns ranked evidence; coherence checks pass.

## HU14. Make the retrieval dials act — [BODY]
- **Priority** P3 · **Effort** M · **Depends on** none
- **Spec**: `mode` (fast/balanced/verify/deep) is a retrieval-depth dial; `source_policy` (references_only/summary/verify/snippets) is an evidence-policy dial (`architecture.md:452-453`).
- **Current**: both are accepted and threaded into `metadata` but **not acted on** (`application/services/graph_service.py:75-86`).
- **Close**: `mode` → reader traversal depth / `max_items` / freshness weighting; `source_policy` → payload filtering/redaction in readers or envelope builder.
- **Acceptance**: changing `mode`/`source_policy` measurably changes the envelope.

## HU17. Skill target template-file install — [BODY]
- **Priority** P3 · **Effort** S · **Depends on** none
- **Spec**: `skills install` writes harness files into the agent harness (`architecture.md:555-575`).
- **Current**: `ClaudeAgentTarget` records install state in a JSON file but **does not copy** the bundle template files (`adapters/outbound/skills/claude_target.py`, TODO stage-N).
- **Close**: copy `templates/{claude_bundle,agent_bundle}` into the target on install/update.
- **Acceptance**: `skills install` materializes the skill files; `status` reflects on-disk drift.

## HU18. Fill observability call-site spans/metrics — [BODY]
- **Priority** P3 · **Effort** M · **Depends on** none
- **Spec**: the trace map + metric set in `observability.md`.
- **Current**: port + adapters real; some call sites still un-instrumented vs the doc's trace map; readiness metric set is a minimum subset.
- **Close**: instrument the remaining documented spans/metrics at call sites.
- **Acceptance**: the trace map in `observability.md` is fully emitted.

---

# Part C — Decisions

## HU15. The 9 free-form record types — [DECISION]
- **Priority** P3 · **Depends on** product input
- `investigation`, `diagnostic_signal`, `workflow`, `feature_note`, `service_note`, `runbook_note`, `integration_note`, `incident_summary` (and `doc_reference`) currently lower to a generic `RELATES_TO` claim with no dedicated retrieval path (`domain/ontology.py:1395-1466`, `domain/context_records.py` `FreeFormRecord`).
- **Decide**: give each a dedicated predicate + reader (semantic retrieval), or accept them as advice-only graph facts and document that. Drives whether HU13-style readers are needed for them.

---

# Part D — Doc reconciliation (spec-internal fixes only)

## D1. Interfaces table names a port that doesn't exist — [DOC] ✅ landed 2026-06-01
- `architecture.md:646,654,681` list `domain/ports/ledger/run_store.py` / `LedgerEventRunStorePort` (and a `run_store` adapter) in the **stable Interfaces** table, but they don't exist yet (HU4). The `EventReconcilerPort` that *does* exist is absent from the Code Map.
- **Close**: annotate the `run_store` rows `_(planned — HU4)_` and add `EventReconcilerPort` (`domain/ports/ledger/reconciler.py`) to the Code Map.

## D2. "Same Pot Management in managed" is stale — [DOC] ✅ landed 2026-06-01
- `architecture.md:47` says the *same Pot Management module* runs in managed; the 2026-05-29 scope decision (HU9) unifies the **Graph Service only** — managed keeps its own pots behind `PotResolutionPort`.
- **Close**: reword `architecture.md:46-49` to "same **Graph Service** module."

## D3. Setup skeleton: `preview`/`SetupPreview` vs `plan`/`list` — [DOC + tracked by HU1]
- `architecture.md:209-256` shows `preview(plan) → SetupPreview` + `PlannedSetupStep`/`LoginPlan`/`host_mode`; the code has `plan(plan) → list[StepResult]` and none of those dataclasses. Docs are the target — HU1 builds them. No doc edit needed beyond confirming HU1 lands; if HU1 keeps `plan()` as an alias, note it.

## D4. Remove the in-tree CLI README that contradicts the spec — [DOC]  (old G7) ✅ landed 2026-06-01
- `adapters/inbound/cli/README.md:68-69` still documents the removed `goal=answer` / `answer.summary` / `potpie overview` surface (the spec dropped server-side synthesis: `architecture.md:632-636`).
- **Close**: rewrite that README to the host-routed commands, or delete it and point to `cli-flow.md`.

---

## Suggested order

**Wave 1 — make everything ownable (Part A, mechanical): ✅ DONE 2026-06-01**
1. ~~**D4** + **D1/D2** — doc fixes (trivial; unblocks accurate Code Map).~~ ✅
2. ~~**HU1** — setup value objects + `preview` (unblocks HU2/HU3).~~ ✅
3. ~~**HU5**, **HU10a**, **HU16** — ledger `query`/CLI seam, backend profile stubs, migration seam.~~ ✅
4. ~~**HU8** — wire `config get/set`, add remaining CLI slots that don't depend on bodies.~~ ✅

**Wave 2 — durable ingestion (the biggest missing port):**
5. **HU4** — run store + state machine.
6. **HU7**, **HU6** — local worker + ingest history CLI.

**Wave 3 — host & managed:**
7. **HU2** — detached daemon.
8. **HU3** — managed login + routing; remaining **HU8** slots fall out.
9. **HU9** — finish managed-stack unification.

**Wave 4 — depth & polish:**
10. **HU10b**, **HU11** — real vectors / neo4j / snapshots / cloud sync.
11. **HU12**, **HU13**, **HU14**, **HU17**, **HU18** — status sections, planned readers, dials, skill files, spans.
12. **HU15** — free-form record-type decision (any time).
