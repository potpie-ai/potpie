# Context Engine — trackable issue candidates (pre-PR backlog)

> Generated 2026-06-17 from a fan-out mine of `.prsplit/*.md` (the 11 `cg-NN` PR-split files)
> plus codebase verification against `feat/graph-updates`. **68 raw candidates → 38 deduped.**
> Nothing here is filed yet except the two already-open chores: **#893** (graphiti scrub +
> dead resolution-service teardown) and **#894** (pyo3 0.28→0.29 bump).

Each item notes the related `cg-NN` PR, risk, whether it **blocks** that PR's merge, and the
verified source (file:line where the agents confirmed it on the live tree).

## How to use this list

- **Group A (10)** — blocking gates; file now and tie each to its `cg-NN` PR. These are the
  carry-forward decisions + flagged risks that must not silently ship.
- **Group B (7)** — roadmap epics; fail-closed stubs that are *advertised but unbuilt*. File as
  a few epics, not scattered notes.
- **Group C (5)** — genuine standalone tech-debt worth their own issues.
- **Group C′ (16)** — real but better folded into the PR review-focus bullets than filed as issue noise.

Recommendation: file **A + B + C ≈ 22 issues**; keep C′ as PR-description checklist items.

## A. Blocking gates — file before the named PR merges

### 1. chore(daemon): resolve daemon_runtime ship gate — carve dead assembler (shell.py/__main__.py/config.py) and re-home or drop managed-services tier before cg-08 merges

- **PR:** `cg-08`  ·  **Category:** deferred-decision  ·  **Risk:** 🔴 high  ·  **BLOCKS**
- **Why:** cg-08 carries an explicit DECISION (2026-06-17): do NOT ship PR-20's daemon_runtime as-is. Verified on feat/graph-updates: both daemons coexist — host/daemon_main.py (kept detached daemon) plus the full host/daemon_runtime/ tier. The DaemonRuntime assembler (shell.py), `python -m host.daemon_runtime run` entrypoint (__main__.py) and config.py have ZERO production consumers (only __main__/tests reference DaemonRuntime), so they are genuinely dead. But context.ShellContext, registry.Registry, health.HealthRegistrar and ipc_auth ARE still imported by adapters/inbound/daemon_http/transport.py, the three adapters/outbound/managed_services backends, and application/services/managed_service_manager.py. So this is a CARVE not a blanket delete: kill shell.py+__main__.py+config.py, then re-home or drop context/registry/health/ipc_auth with their managed-services + daemon_http + `service` CLI consumers. Also resolve the two-daemon overlap (FastAPI host vs async UDS/TCP OperationRegistry) so only one runnable entrypoint and one lifecycle/pidfile/discovery owner remain.
- **Source:** potpie/context-engine/host/daemon_runtime/ (shell.py/__main__.py/config.py dead; context.py/registry.py/health.py/ipc_auth.py used by daemon_http/transport.py + managed_services backends + managed_service_manager.py) + .prsplit/feat__cg-08-ce-daemon.md Carry-forward + docs/context-graph/pr-split-plan.md:392-411,741

### 2. test(daemon): add real daemon tests and confirm deleted unit-test coverage is subsumed before cg-08 merges

- **PR:** `cg-08`  ·  **Category:** test-gap  ·  **Risk:** 🟠 med  ·  **BLOCKS**
- **Why:** cg-08's carry-forward from PR-19 is 'add real daemon tests before merge.' The branch deletes six unit tests (test_daemon_context/health/ipc_auth/operations/ports/registry) and replaces them with a new integration suite plus new unit tests. The plan asks reviewers to confirm the deleted scenarios are genuinely subsumed, not lost, so the restored detached daemon path has genuine coverage rather than a gap hidden by test churn.
- **Source:** .prsplit/feat__cg-08-ce-daemon.md — 'Review focus' Test churn + 'Carry-forward decisions' PR-19; .prsplit/CONSOLIDATION-PLAN.md:118

### 3. chore(ui): drop committed frontend/dist build artifacts and add dist/ to .gitignore before cg-09 merges

- **PR:** `cg-09`  ·  **Category:** chore-cleanup  ·  **Risk:** 🟢 low  ·  **BLOCKS**
- **Why:** cg-09 commits four git-tracked build artifacts under adapters/inbound/http/ui/frontend/dist/ (index-45K60Bie.js = 352KB minified, index-B0oSWky4.css, logo_dark SVG, index.html) while the Makefile already builds the SPA on cli-install (ui-build -> npm run build), and static.py already falls back to a placeholder with build instructions when the bundle is absent. The frontend .gitignore ignores node_modules/.vite/*.local but NOT dist/, so artifacts keep getting committed. Carry-forward says drop them and build-on-install; when unignoring dist/, re-verify the static.py mount and the cli-status dist/index.html check still pass.
- **Source:** .prsplit/feat__cg-09-ce-ui-repo.md — Carry-forward + Review focus 'Committed dist/ build artifacts'; verified tracked via git ls-files; static.py:33-54; Makefile ui-build

### 4. security(cg-09): verify removing blanket *.json gitignore does not start tracking local/secret JSON tree-wide

- **PR:** `cg-09`  ·  **Category:** security  ·  **Risk:** 🟠 med  ·  **BLOCKS**
- **Why:** cg-09's .gitignore change removes the blanket *.json ignore and adds .tmp/. Removing *.json is broad and the prior !-allowlist entries become moot once the blanket ignore is gone, so previously-ignored local or secret JSON across the tree could start being tracked. Needs a tree-wide confirmation (git status / git check-ignore audit) before merge to avoid committing credentials/config.
- **Source:** .prsplit/feat__cg-09-ce-ui-repo.md — Review focus '.gitignore change'

### 5. fix(ce): reconcile embedder vector dimensionality — HashingEmbedder emits 256 but index creators default to 1536 magic value when embedder unwired

- **PR:** `cg-05`  ·  **Category:** bug-risk  ·  **Risk:** 🔴 high  ·  **BLOCKS**
- **Why:** Verified: local_embedder.py:38 sets _DEFAULT_DIMENSIONS=256 (the bundled OSS default), but all three index creators default to 1536 via getattr(self._embedder, 'dimensions', 1536) — neo4j_writer.py:80, falkordb_writer.py:233/238, cypher.py:759 ensure_canonical_indexes(embedding_dim=1536). Consistent when an embedder is wired (reads .dimensions=256). Hazard: writers accept embedder=None (neo4j_writer.py:42, embedded_backend.py:44) — if ensure_indexes runs with no embedder it creates a 1536-dim index while embedding writes use the real 256-dim embedder, so the vector index mismatches and semantic reads silently degrade to lexical fallback. Confirm index creation and embedding writes share one source-of-truth dimension on both Neo4j and FalkorDB, and remove the 1536 magic-default (derive from the bundled embedder or fail closed). Explicit 'confirm/fix; deferred' carry-forward.
- **Source:** local_embedder.py:38 + adapters/outbound/graph/{neo4j_writer.py:80,falkordb_writer.py:233-238,cypher.py:759-806} + .prsplit/feat__cg-05-ce-backends.md Carry-forward + docs/context-graph/pr-split-plan.md:226,748

### 6. fix(ce): verify record->semantic write path cannot auto-apply supersede/merge mutations via the approval shortcut

- **PR:** `cg-06`  ·  **Category:** bug-risk  ·  **Risk:** 🔴 high  ·  **BLOCKS**
- **Why:** cg-05/cg-06 rewire the V1 context_record write to flow through the same semantic-mutation path as mutate (record_to_semantic_request -> validate -> lower -> apply), removing the private direct-lowering path; record_to_semantic.py marks the request approved so medium-risk decisions auto-apply. Two load-bearing safety checks must be confirmed: (1) the approval shortcut cannot let supersede/merge operations slip through to auto-apply, bypassing the validator's atomic-batch review_required policy (a slip lets high-impact mutations write without review); and (2) the conversion preserves the old record_type->predicate mapping (e.g. preference -> POLICY_APPLIES_TO) and free-form fallback so recorded preferences still surface in coding_preferences, with a recorded-then-resolved round trip working on the in_memory backend.
- **Source:** .prsplit/feat__cg-05-ce-backends.md Review focus 'Semantic mutation pipeline correctness' (record_to_semantic.py) + .prsplit/feat__cg-06-ce-services.md Review focus (graph_service.py highest-risk)

### 7. chore(cross-root): verify cg-04/cg-09 installer code and remnant tests match cg-12 agent-surface assets (deleted skills, claude_plugin tree, remnant test placement)

- **PR:** `cross-cutting`  ·  **Category:** follow-up  ·  **Risk:** 🔴 high  ·  **BLOCKS**
- **Why:** The agent-surface templates+docs live on the independent docs/cg-12-agent-surface root (3 skills deleted: potpie-agent-context/potpie-cli-troubleshooting/potpie-pot-scope; new claude_plugin/ tree added), while the installer code (cg-04 skill manager) and the PR-28*/PR-32* remnant tests (test_agent_templates_v15, test_claude_plugin_manifest, test_repo_baseline_skill, test_nudge_e2e, test_cli_ergonomics, test_source_cli_contract) ride on the CE stack (cg-09). Confirm: (1) no installer manifest or hard-coded skill-name list still references the three deleted skills and the new claude_plugin/ tree is enumerated/picked up; (2) the remnant tests are only green if cg-12's assets exist at the cg-09 stack state — if not, move them to cg-12. This cross-root coupling can break tests/installs at merge depending on order.
- **Source:** .prsplit/docs__cg-12-agent-surface.md Review focus bullet 1 + .prsplit/feat__cg-09-ce-ui-repo.md Review focus 'Remnant tests' / 'Why this grouping'

### 8. test(cli): pin policy gate ordering so pot 403/404 resolves before reconciliation/feature-flag gates

- **PR:** `cg-01`  ·  **Category:** security  ·  **Risk:** 🔴 high  ·  **BLOCKS**
- **Why:** cg-01's policy ordering fix moves the pot_id-resolution block (tenant-boundary 403 / unknown-pot 404) ahead of require_reco/feature-flag gates in adapters/outbound/policy/default.py (~147-190) so flag configuration is never leaked for a pot the caller cannot access. The plan only says test_policy_port.py 'should pin the new ordering' (conditional, not confirmed) and asks to confirm no path now double-resolves the pot and that deny reasons/status codes are unchanged. This is a tenant-boundary information-disclosure surface; the ordering invariant should be an explicitly-pinned test, not review-time scrutiny.
- **Source:** .prsplit/fix__cg-01-login-posthog.md — Review focus 'Policy ordering fix' (policy/default.py:147-190)

### 9. chore(ce): confirm belief.py evidence-strength decay (predicate-family TTL + source-authority) is re-homed, not lost, before cg-11 deletion

- **PR:** `cg-11`  ·  **Category:** deferred-decision  ·  **Risk:** 🟠 med  ·  **BLOCKS**
- **Why:** belief.py is ALREADY deleted on feat/graph-updates (verified: removed in 2f380d79, still on main) with zero live importers, so the deletion is done — but the carry-forward re-homing concern is real and unresolved. domain/ranking.py only re-homed recency exponential decay (_recency_score, half-life). belief.py's distinct mechanisms — predicate-family TTL decay_weight (linear decay to 0 at observed_at+2*TTL via family_ttl_hours), the EvidenceStrength IntEnum, and per-source source_authority(source_system) weighting — were NOT carried over (the live is_source_authority in graph_contract.py is an unrelated string validator). Decide whether losing family-aware TTL decay and source-authority weighting is an acceptable retrieval regression or re-home it; needs bench/eval confirmation since ranking moved from weighted geometric to weighted arithmetic mean.
- **Source:** domain/belief.py (deleted in 2f380d79; exists on main) + domain/ranking.py:191-215 + docs/context-graph/pr-split-plan.md:740,749

### 10. test(ce): verify no stale ClaimQueryPort imports survive cg-11 re-home and the full suite import-resolves

- **PR:** `cg-11`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  **BLOCKS**
- **Why:** cg-11 re-homes ClaimQueryPort/ClaimRow: domain/ports/graph/__init__.py drops them and points callers at canonical domain.ports.claim_query while domain/ports/graph/claim_query.py is deleted. The Review focus names dangling-import risk the dominant concern and the single most likely place a stale `from domain.ports.graph import ClaimQueryPort` survives. Confirm import resolution via a green test suite so the re-home is proven, not assumed.
- **Source:** .prsplit/removal__cg-11-ce-removals.md:127 (Review focus dangling-import); domain/ports/graph/__init__.py, deleted domain/ports/graph/claim_query.py

## B. Roadmap epics — fail-closed stubs (advertised but unbuilt)

### 24. feat(graph): build out registered-but-stubbed graph backends (postgres/pgvector, chroma) and finish Neo4j/FalkorDB capabilities (invalidate, inspection, snapshot)

- **PR:** `cg-05`  ·  **Category:** tech-debt  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** Storage seams are advertised but unbuilt. The postgres (pgvector, with shipped [postgres] extra: sqlalchemy>=2.0, psycopg[binary]>=3.2) and chroma profiles are selectable via 'backend list' but resolve to a fail-closed StubGraphBackend whose six capability ports + provision() raise CapabilityNotImplemented (adapters/outbound/graph/backends/__init__.py:44,110-111; stub_backend.py:30-75). Neo4j — the 'shape-first production target' — has three of six ports unbuilt: _Neo4jMutation.invalidate raises (neo4j_backend.py:126-132), inspection + snapshot wired to Unimplemented* (readiness False at :148-149), plus reset_pot/apply TODO(stage-N) markers. FalkorDB/FalkorDBLite mutation.invalidate likewise raises for claim-key invalidation (falkordb_backend.py:109-117). Claim-key invalidation and portable snapshot/export are load-bearing for production correctness and migrations; track per-profile so the catalog stays honest.
- **Source:** adapters/outbound/graph/backends/{__init__.py:43-44,110-111, stub_backend.py:30-75, neo4j_backend.py:9-14,126-151, falkordb_backend.py:109-117} + pyproject.toml:30-33

### 25. feat(cloud): implement managed/cloud profile end-to-end (login, push/pull, ledger, pot routing, skill sync, hosted backend) as one epic

- **PR:** `cross-cutting`  ·  **Category:** follow-up  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** The entire managed/cloud track is a fail-closed skeleton: cloud.py login/status/push/pull and skills sync all raise CapabilityNotImplemented (adapters/inbound/cli/commands/cloud.py:23-59); the managed ledger client returns empty pages / unavailable health (adapters/outbound/ledger/managed_client.py:50-64); managed pot listing/routing are not implemented (pots.py:78, bootstrap.py:319); managed skill catalog sync is TODO; the 'hosted' graph profile resolves to a fail-closed StubGraphBackend; and LocalAuthService.init_local is unbuilt (auth_service.py:25). This is a coherent roadmap-sized feature, not scattered notes — track as one epic so 'cloud is on the roadmap' messaging maps to real work.
- **Source:** cloud.py:1-59; managed_client.py:8-64; pots.py:78; bootstrap.py:319; auth_service.py:25

### 26. feat(connectors): build real source connectors (Slack/Linear/deploy/alerting/repo_docs) + self-hosted ledger pull

- **PR:** `cross-cutting`  ·  **Category:** tech-debt  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** There are no live source readers: _bench_stubs.py registers passive Slack/RepoDocs/Alerting/Deploy/Linear connectors with fetch_capable=False that exist only so source-system lookup resolves for replayed fixtures (adapters/outbound/connectors/_bench_stubs.py:1-138; comment at :120-124 notes the real LinearConnector 'was removed without ever being registered'). The self-hosted ledger client is a dummy: fetch() returns an empty LedgerPage with a TODO for HTTP cursor pull and query() raises not-implemented (adapters/outbound/ledger/self_hosted_client.py:38-65). Outside the bench fixture path, ingestion from these providers does not work — a significant missing-reader gap worth tracking per-provider under one epic. (Verify the bench corpus still resolves the linear/issue source kind via the passive LinearStubConnector after the real connector's deletion in cg-11.)
- **Source:** adapters/outbound/connectors/_bench_stubs.py:1-138; adapters/outbound/ledger/self_hosted_client.py:38-65; .prsplit/removal__cg-11-ce-removals.md:130

### 27. chore(persistence): replace flat-file JSON control plane (pots/sources/cursors) with local state DB (SQLite + migrations)

- **PR:** `cg-06`  ·  **Category:** tech-debt  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** Two control-plane stores persist to ad-hoc JSON and both carry TODO(stage-N) to move into the real local state DB: LocalPotStore writes pots/sources/active-pot to <home>/pots.json (adapters/outbound/pots/local_pot_store.py:8-11) and LocalLedgerCursorStore keeps per-(pot,source) cursors in a JSON file (adapters/outbound/ledger/cursor_store.py:6). cli-flow.md specifies a SQLite + migrations control plane. Flat-file persistence has no concurrency/locking or migration story and is fragile as pot/source/cursor state grows; consolidate both onto the state DB as one task.
- **Source:** adapters/outbound/pots/local_pot_store.py:8-11; adapters/outbound/ledger/cursor_store.py:6

### 28. feat(host): implement LocalInstaller PATH install + OS service-unit registration + uninstall

- **PR:** `cg-08`  ·  **Category:** tech-debt  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** All three install bodies are fail-closed stubs: install_cli (PATH), register_service (systemd/launchd), and uninstall each raise CapabilityNotImplemented (adapters/outbound/install/local_installer.py:28-46), and is_installed unconditionally returns True because the POC CLI is already importable. Detached/daemonized host operation depends on real OS service-unit registration, so this blocks a production install/run-as-service story for non-pip users; track as environment-specific (pip/pipx/homebrew/deb) work.
- **Source:** adapters/outbound/install/local_installer.py:23-46

### 29. feat(skills): implement skill catalog add (register local-path/URL skill)

- **PR:** `cg-06`  ·  **Category:** tech-debt  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** DefaultSkillManager.add returns a hardcoded 'catalog add not implemented' SkillOperationResult instead of registering a skill, with a TODO(stage-N) to register a local-path/URL skill into the catalog (application/services/skill_manager.py:222-228). Catalog extension is a user-facing surface ('potpie skills add') that currently silently no-ops; track rather than leave as an inline TODO.
- **Source:** application/services/skill_manager.py:222-228

### 30. tech-debt(graph): replace V1.5 placeholders — subgraph version counter, resolver capability matrix, deterministic-only edge contradiction

- **PR:** `cg-06`  ·  **Category:** deferred-decision  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** Several V1.5/V2 placeholders are explicitly carried forward and should be tracked as deferred design decisions: graph_service._subgraph_versions returns a single global claim-count counter as a 'V1.5 stub' for V2 optimistic concurrency (application/services/graph_service.py:526-530); the resolver capability matrix advertises richer modes as placeholders until source resolvers are wired in Phase 4 (domain/context_status.py:83-85, domain/ports/source_resolver.py:13); and singleton_predicates.py:9-13 notes the LLM-driven dedupe_edges.resolve_edge complement path is deferred, leaving only the deterministic edge-contradiction path. Group as one 'V1.5 deferred decisions' issue to keep the V2 runway visible.
- **Source:** application/services/graph_service.py:526-530; domain/context_status.py:83-85; domain/singleton_predicates.py:9-13; domain/ports/source_resolver.py:13

## C. Standalone tech-debt / hardening

### 12. build: ship Claude Code plugin JSON + nudge hook in the built wheel (currently source-only, broken for pip installs)

- **PR:** `cross-cutting`  ·  **Category:** tech-debt  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** Verified: agent_installer.install_agent_bundle for --agent claude-plugin copies plugin.json, marketplace.json, hooks.json and potpie_nudge.py from templates/claude_plugin/, but the wheel build only globs templates/**/*.md, templates/**/*.yaml and ui/assets/**/*.json (pyproject.toml:97-110; force-include lists only logo assets). The plugin's 3 JSON files and potpie_nudge.py do NOT ship, so claude-plugin works from a source checkout but is broken for pip/wheel installs. The in-code comment blames a global '*.json' gitignore that no longer exists (files are tracked), so the only real blocker is the missing wheel include glob. Deliberately deferred dev/source-only; add a templates/claude_plugin/**/*.json + potpie_nudge.py include.
- **Source:** potpie/context-engine/pyproject.toml:97-110 (wheel include) + adapters/outbound/skills/agent_installer.py:309-322

### 14. security(daemon): bound the daemon_rpc.py arbitrary class-import deserialization trust boundary across the CLI<->daemon seam

- **PR:** `cg-08`  ·  **Category:** security  ·  **Risk:** 🔴 high  ·  follow-up
- **Why:** host/daemon_rpc.py reconstructs dataclasses/enums by importing module:qualname directly from the wire request payload — an arbitrary-class-import trust boundary keyed off input crossing the CLI<->daemon seam. The Review focus asks to scrutinize this boundary and round-trip fidelity for every DTO. Even on loopback/UDS, an unconstrained import-by-name deserializer is a hardening concern: bound the importable namespace to an allowlist and verify round-trip fidelity for all DTOs.
- **Source:** .prsplit/feat__cg-08-ce-daemon.md — Review focus 'RPC serialization correctness' (host/daemon_rpc.py)

### 22. docs(ce): update automation/skills/docs for removed `ledger pull --apply`, removed `ingest`, and changed `source add` JSON keys

- **PR:** `cg-11`  ·  **Category:** docs  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** cg-11 changes user-visible CLI behavior: the ingest command is removed, ledger pull drops --apply and its applied/claims_written output keys, and source add becomes registration-only with a new api_unreachable failure path and reworked positional args. The Review focus asks to update automation/skills/docs that parse the old JSON keys — consumers parsing removed keys will silently break.
- **Source:** .prsplit/removal__cg-11-ce-removals.md:129 (Review focus CLI behavior change); ledger.py, pots.py, deleted ingest.py

### 23. chore(domain): remove ReconciliationPlan->MutationBatch back-compat shim and decide fate of parked agentic-reconciliation tier

- **PR:** `cg-03`  ·  **Category:** tech-debt  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** Two cg-03 cleanup items: (1) PR-2's rename keeps `ReconciliationPlan = MutationBatch` plus the EvidenceRef/ReconciliationRequest re-export, documented in domain/reconciliation.py:34-38,69-72 as 'a thin back-compat shim for one iteration' — needs tracked removal once all importers move to MutationBatch or it lives forever; (2) cg-03 ships a dormant 'parked' agentic-reconciliation tier (domain/llm_reconciliation.py, domain/ports/reconciliation_tools.py, domain/reconciliation_flags.py) declared non-canonical in V1.5, gated behind opt-in CONTEXT_ENGINE_AGENT_PLANNER_ENABLED (default off); reviewers must 'confirm they are inert' and a wire-it-or-remove-it decision should be tracked rather than riding unwired indefinitely.
- **Source:** .prsplit/feat__cg-03-ce-foundation.md Review focus (Rename ripple PR-2); domain/reconciliation.py:34-38,69-72 + domain/llm_reconciliation.py:1-16, domain/reconciliation_flags.py:24-36

### 31. refactor(hexagonal): outbound agent_installer reaches into adapters.inbound.cli for bundle templates

- **PR:** `cross-cutting`  ·  **Category:** tech-debt  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** agent_installer.py now lives in adapters/outbound/skills/ but its _iter_bundle_files resolves bundle templates via resources.files('adapters.inbound.cli').joinpath('templates', ...) (agent_installer.py:52) — an outbound adapter importing resources from the inbound layer, violating hexagonal layering (arrows should be inbound -> domain <- outbound). The CLI inbound reorg left templates in inbound 'to avoid packaging churn' and flagged this as the minor remaining smell after the CredentialStore port lift. Fix by relocating bundle templates to a layer-neutral/outbound-owned package or exposing them through a port.
- **Source:** adapters/outbound/skills/agent_installer.py:52

## C′. Reviewer-checklist items — fold into PR review-focus, not standalone issues

### 11. chore(ce): confirm legacy app and integrations packages no longer reference removed Linear sync/episode modules after cg-11

- **PR:** `cg-11`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** cg-11 deletions reach across packages into legacy/app/modules/context_graph/hatchet_worker.py and three potpie/integrations/.../linear/* adapters (episodes.py, ingest_linear_issue.py, linear_sync.py). The Review focus asks to confirm nothing in the legacy app or integrations package still references the removed Linear sync/episode modules. Cross-package dangling references would not be caught by the CE test suite alone, warranting a tracked check before merge.
- **Source:** .prsplit/removal__cg-11-ce-removals.md:132 (Review focus cross-package); deleted legacy/app/.../hatchet_worker.py + potpie/integrations/.../linear/{episodes,ingest_linear_issue,linear_sync}.py

### 13. fix(claude-plugin): keep potpie_nudge.py NUDGE_EVENTS/PostToolUse classification in sync with domain.nudge.NudgeEvent and verify the fail-safe (exit 0, no output)

- **PR:** `cg-12`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** claude_plugin/hooks/potpie_nudge.py (the only executable on cg-12) defines a NUDGE_EVENTS set and a PostToolUse(Bash) -> test_failed/test_passed classification whose own comment asserts it must match domain.nudge.NudgeEvent — but that enum lives on cg-03 with no compile-time link across roots, so the event vocabulary can silently drift and break nudge forwarding. Separately the Review focus requires the hook to exit 0 with no output on any error, missing potpie binary, or unparseable stdin so a hook failure can never block a Claude Code session; this fail-safe contract is asserted but not verified by an automated test. Track a cross-branch enum-sync check plus a fail-safe test.
- **Source:** .prsplit/docs__cg-12-agent-surface.md Review focus bullet 2 + potpie_nudge.py

### 15. fix(cli): ensure genuine GitHub auth failures are not misclassified as login cancel

- **PR:** `cg-01`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** _is_github_login_cancel is used across multiple except arms in auth/github_commands.py (device-verification wait, top-level except (Abort, KeyboardInterrupt, EOFError), and a broad except Exception). The plan asks to confirm a genuine GitHub auth failure is still routed to capture_github_auth_event and not mis-classified as a cancel — which would silently exit 130 and drop the failure telemetry/error. This swallow-real-failure-as-cancel path is a verifiable behavioral risk warranting a tracked test.
- **Source:** .prsplit/fix__cg-01-login-posthog.md — Review focus 'Cancel detection consistency'

### 16. fix(cli): verify auto-open countdown degrades gracefully in non-TTY/CI (non-selectable stdin, \r\033[K redraw)

- **PR:** `cg-01`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** The new _wait_for_enter_or_auto_open in auth/github_commands.py branches on win32 (msvcrt) vs POSIX (select.select) with an except (OSError, TypeError, ValueError) fallback that sleeps 1s and treats as no-input. Two unverified concerns: the POSIX path degrading gracefully when stdin is not selectable, and whether the inline \r\033[K prompt redraw is acceptable in non-TTY/CI logs (control-code spam). Onboarding under CI/non-TTY warrants a tracked verification covering the non-interactive path.
- **Source:** .prsplit/fix__cg-01-login-posthog.md — Review focus 'Cross-platform cancel/countdown'

### 17. test(ce): backend conformance suite must exercise all four backends equivalently and ClaimQueryFilter forwarding must be exhaustive

- **PR:** `cg-05`  ·  **Category:** test-gap  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** cg-05's embedding-aware claim-query fan-out (canonical_claim_query.py, claim_query_semantic.py, claim_query_analytics.py + reader/writer edits) touches in-memory, embedded, Neo4j, and FalkorDB. Two risks: confirm test_graph_backend_conformance.py actually exercises all four backends equivalently, and that ClaimQueryFilter field forwarding in claim_query_semantic.py is exhaustive — it manually re-copies each filter field, so a dropped field silently degrades query filtering on one backend only.
- **Source:** .prsplit/feat__cg-05-ce-backends.md — Review focus 'Claim-query fan-out'

### 18. test(ce): guard the NudgeService/embedder zero-LLM and HashingEmbedder cross-process determinism invariants

- **PR:** `cg-06`  ·  **Category:** test-gap  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** Two related embedder/nudge invariants are asserted only by docstring/manual review: (1) NudgeService (application/services/nudge_service.py) must make NO model calls — only the local embedder behind a graph read plus the in-session agent — yet no automated guard fails if an API client is constructed on the nudge path; (2) the embedded backend is JSON-persisted and runs process-per-call, so HashingEmbedder (local_embedder.py) must use a stable cross-process digest, not Python's salted built-in hash(), or persisted embeddings won't match across invocations and vector lookups silently fail. Add regression guards for both.
- **Source:** .prsplit/feat__cg-06-ce-services.md Review focus (NudgeService zero-LLM) + .prsplit/feat__cg-05-ce-backends.md Review focus 'Determinism of the hashing embedder'

### 19. fix(ce): LocalInjectionLedger lock-free read-modify-write can duplicate nudge injections across hook processes

- **PR:** `cg-06`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** LocalInjectionLedger (adapters/outbound/session/injection_ledger.py) does read-modify-write per call across separate short-lived hook processes with no lock; the accepted worst case is one duplicate injection (never corruption). This explicitly-accepted concurrency tradeoff should be tracked so the duplicate-injection behavior is guarded by a test or revisited if it becomes a real annoyance. Also verify the JSON path resolves under $CONTEXT_ENGINE_HOME/~/.potpie correctly.
- **Source:** .prsplit/feat__cg-06-ce-services.md — Review focus (LocalInjectionLedger concurrency)

### 20. chore(deps): verify redis>=7.1,<8 upper bound and python_version markers, and confirm potpie-daemon script entry resolves

- **PR:** `cg-05`  ·  **Category:** tech-debt  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** cg-05 promotes falkordblite>=0.10.0 and redis>=7.1,<8 from extras into default deps, so every CLI install pulls FalkorDBLite + redis (the <8 bound exists because redis 8 maintenance notifications break FalkorDBLite's hostless embedded connection). Confirm the redis<8 bound and that the Python 3.12+ support floor is reflected in package metadata. Also confirm the potpie-daemon = 'host.daemon_main:main' script entry resolves — verified the module exists on this tip, but in an intermediate cg-05-only install (before cg-08) it may dangle, so gate/verify it doesn't expose a broken console script.
- **Source:** pyproject.toml:23-26 (redis/falkordblite), :89 (potpie-daemon) + .prsplit/feat__cg-05-ce-backends.md Review focus 'Dependency surface'

### 21. chore(ce): apply_plan rename + writer-port seam has no dangling old-name imports and all writers satisfy the port

- **PR:** `cg-05`  ·  **Category:** chore-cleanup  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** cg-05 renames the writer-port seam via apply_plan.py (apply_plan rename), writer_port.py, and falkordb/neo4j writer edits. Watch for dangling imports of the old name and confirm all writer implementations satisfy the new port — a missed import or unimplemented port method surfaces only at runtime on a specific backend. Also covers the parallel ContextGraphPort demotion to a legacy compatibility shim (apply_plan retyped to MutationBatch/MutationResult, intentionally allowed not-implemented): confirm managed callers tolerate the not-implemented path.
- **Source:** .prsplit/feat__cg-05-ce-backends.md Review focus 'apply_plan rename' + .prsplit/removal__cg-11-ce-removals.md:131 (ContextGraphPort shim)

### 32. chore(cg-07): clean up _common.py Abort-helper bleed from the PR-25 hunk

- **PR:** `cg-07`  ·  **Category:** chore-cleanup  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** The 'Known cosmetic caveat' admits relaxed purity: PR-25's self-contained CLI Abort helper in _common.py also appears in the CE stack starting at cg-07 because no branch carries a 'PR-22 hunk without PR-25' state, so cg-07's _common.py diff shows the Abort helper alongside its own json_error_formatter. Documented idempotent/harmless at merge, but it's an impure diff worth tracking so the cosmetic bleed is cleaned up rather than silently normalized into the stack.
- **Source:** .prsplit/CONSOLIDATION-PLAN.md:94-98 (Known cosmetic caveat)

### 33. chore(cli): confirm usage-analytics double-capture in query.py is intentional, not a duplicate signal

- **PR:** `cg-01`  ·  **Category:** follow-up  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** query.py now calls capture_usage_command_succeeded in addition to capture_activation_succeeded / context-activation paths. In code, _capture_context_activation (query.py:158) calls both capture_activation_succeeded and capture_usage_command_succeeded back-to-back, with another capture_usage_command_succeeded at query.py:101 — a real analytics-integrity question that should be resolved/documented rather than left implicit.
- **Source:** .prsplit/fix__cg-01-login-posthog.md — Review focus 'Usage telemetry double-capture' (query.py:101,164)

### 34. chore(claude-plugin): handle hooks.json python3-on-PATH assumption for the nudge hook invocation

- **PR:** `cg-12`  ·  **Category:** bug-risk  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** claude_plugin/hooks/hooks.json wires SessionStart/PreToolUse/PostToolUse/Stop to the adapter via a python3 invocation that assumes python3 is on PATH. On environments without python3 (or only python), the hook fails; combined with the fail-safe requirement this should be confirmed and possibly hardened. Tracked as an environment-portability follow-up.
- **Source:** .prsplit/docs__cg-12-agent-surface.md Review focus bullet 3; claude_plugin/hooks/hooks.json

### 35. fix(cli): harden repo_location.resolve_repo_location remote normalization (SSH vs HTTPS, broad except, ~/relative)

- **PR:** `cg-09`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** repo_location.resolve_repo_location shells out to git remote get-url origin with a 2s timeout and a broad except, then normalizes .git/git@ refs and a relative/~ expansion branch. This resolved ref becomes the durable match key for repo sources, so inconsistent normalization between SSH and HTTPS remotes (or swallowed errors) would silently misclassify or fail to match a repo's pot. The SSH/HTTPS and ~/relative branches need explicit verification.
- **Source:** .prsplit/feat__cg-09-ce-ui-repo.md — Review focus 'repo_location.resolve_repo_location'

### 36. security(ui): verify /ui router is read-only/loopback-only and prefix-display policy matches V1.5 identity ontology

- **PR:** `cg-09`  ·  **Category:** security  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** router.py (cg-09) is intended read-only and loopback-only and delegates to HostShell read surfaces. Confirm no route mutates the graph, that pot resolution (?pot= vs active pot) is correct, and that the _PREFIX_LABEL key-prefix display policy stays consistent with the V1.5 identity ontology. As a new browser-exposed surface over project memory, these read-only/loopback guarantees and identity-prefix consistency should be tracked and verified rather than assumed.
- **Source:** .prsplit/feat__cg-09-ce-ui-repo.md — Review focus '/ui router safety' (router.py, test_ui_router.py)

### 37. bug-risk(daemon): verify commands/daemon.py _restart() AttributeError-swallow fallback degrades cleanly and DaemonStartError emits EXIT_UNAVAILABLE envelope

- **PR:** `cg-08`  ·  **Category:** bug-risk  ·  **Risk:** 🟠 med  ·  follow-up
- **Why:** commands/daemon.py _detached_daemon() re-instantiates Daemon(in_process=False) and _restart() swallows AttributeError to fall back to stop+start. Verify this degrades cleanly and that DaemonStartError paths emit the contract error envelope (EXIT_UNAVAILABLE). Swallowing AttributeError to drive control flow is a fragile fallback worth tracked verification/hardening. Also confirm the hunk-split shared files _common.py (from ce-19) and host_cli.py (from ce-20) — taken from different source branches — are mutually consistent (host_cli.py adds `service` to the import tuple/add_typer; _common.py recovery-tool helpers).
- **Source:** .prsplit/feat__cg-08-ce-daemon.md — Review focus 'CLI seam fallback' + 'Shared-file cumulative state'

### 38. chore(daemon): verify setup --daemon and in-process default behavior preserved through heavy bootstrap/host-wiring rework

- **PR:** `cg-08`  ·  **Category:** follow-up  ·  **Risk:** 🟢 low  ·  follow-up
- **Why:** host_wiring.py (181 lines), host/daemon.py (203 lines), and bootstrap.py (80 lines) carry the bulk of cg-08's behavioral change, plus a new observability_wiring.py hook. Confirm `setup --daemon` / in-process default behavior is preserved across this rework; given the size of the change a tracked verification of the preserved setup/default paths is warranted.
- **Source:** .prsplit/feat__cg-08-ce-daemon.md — Review focus 'Heavy bootstrap rework'
