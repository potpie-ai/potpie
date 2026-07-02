# Agent Surface Ingestion And Query Review

Date: 2026-06-17

This note captures a review of Potpie's agent-facing ingestion, querying, UI,
quality, and skill surfaces. It is now structured as a planning seed: evidence
first, then a deduplicated backlog and implementation workstreams.

## Executive Summary

Potpie has a strong conceptual model for agent-led project memory: live graph
catalog/describe, semantic mutation proposals, identity resolution, evidence,
truth classes, confidence, history, inbox, and skills that push agents toward
source-backed ingestion.

The weak point is enforcement. Many important behaviors are written in skills
instead of encoded in command contracts, commit gates, status health, or UI
defaults. A careful agent can do good work; a rushed or confused agent can still
write shallow memory, miss freshness work, or trust misleading health output.

Approximate assessment after Workstreams 1 through 5:

- Agent guidance quality: 8.5/10.
- Ingestion enforcement: 6/10.
- Automatic freshness / keep-up-to-date behavior: 3/10.
- Query/read ergonomics: 8/10.
- Query token efficiency: 7/10.

Highest-leverage planning themes:

1. Preserve text-first and bounded routine reads.
2. Keep pot selection and empty-graph visibility in the verified path.
3. Make status/quality/freshness reliable enough to gate ingestion.
4. Add hard post-ingestion verification instead of relying on skill text.
5. Fix stale docs and skill command examples with validation.
6. Improve use-case coverage now that exact lookup and drilldown work.

## Evidence Snapshot

### Current Graph State

The current repo-local default for `github.com/potpie-ai/potpie` resolves to
pot `pot_73b01a108e23` (`potpie-repo-usecase-test-20260615`). This is now the
smooth agent path for this repo: agent-facing commands resolve to the populated
pot even when the global active pot is still `test` / `pot_1e1cd80ab7de`.

`graph status` for that pot reported:

- Backend: `falkordb_lite`, ready.
- Sources: 2.
- Claims: 152.
- Entities: 95.
- Predicates: 17.
- Summary quality: `ok`.
- Empty-pot warnings: none while the repo default is set.

Earlier Workstream 1 and 2 measurements used the populated review graph
`pot_03d60f3b579d`, which had 82 claims. Deeper quality reports for that graph
showed a different picture:

- `projection-drift`: `degraded`, 49 findings.
- `conflicting-claims`: `watch`, 10 findings.
- `duplicate-candidates`: `ok`, 0 findings.
- `low-confidence`: `ok`, 0 findings.
- `orphan-entities`: `ok`, 0 findings.
- `stale-facts`: `ok`, 0 findings.

The graph contains useful memory around features, preferences, and recent
GitHub/Linear activity, but readbacks are often sparse and the quality signal is
not trustworthy enough to use `graph status` as graph health. Planning direction:
`graph status` should show a clean, compact health summary by default, with
detailed reports remaining in `graph quality`.

Pot-scope drift was observed before Workstream 3: the active local repo could
resolve to `test` / `pot_1e1cd80ab7de`, which has 0 claims, while populated
repo-linked pots existed. Workstream 3 now makes that state visible and
actionable. Clearing the repo default reproduces an empty-pot warning that
suggests:

```bash
potpie pot default set --repo current pot_73b01a108e23
```

### Token Cost Hotspots

The graph is not mostly noise. The populated pot returns useful feature,
preference, infra, and GitHub timeline context. The default agent-facing JSON
path is expensive because it wraps useful facts in repeated metadata.

Rough token estimates below use `chars / 4` and were measured against
`pot_03d60f3b579d`:

| Query | Approx tokens | Notes |
| --- | ---: | --- |
| `graph catalog --task ... --json` | 3.4k | Full contract, all views, entity types, predicates, ranking detail. |
| `graph describe infra_topology --view service_neighborhood --examples --json` | 4.0k | Full subgraph schema plus mutation/source-authority rules. |
| `graph read --subgraph features --view feature_context --limit 20 --json` | 12.3k | 13 items, 40 relation copies. |
| `graph read --subgraph recent_changes --view timeline --limit 10 --json` | 17.6k | 17 relation items plus a 10-event projection. |
| `graph read --subgraph infra_topology --view service_neighborhood --limit 20 --json` | 10.2k | 17 items, 32 relation copies. |
| `graph search-entities "context graph" --limit 10 --json` | 6.8k | Each entity carries supporting claims. |
| `graph quality conflicting-claims --limit 10 --json` | 4.3k | Verbose report, and current findings are partly semantic noise. |

Text output is dramatically cheaper for routine agent reads:

| Query | Approx tokens | Notes |
| --- | ---: | --- |
| `graph read --subgraph recent_changes --view timeline --limit 3 --format table` | 285 | Human-readable event list. |
| `graph read --subgraph recent_changes --view timeline --limit 3 --format jsonl` | 502 | One compact event row per line. |
| `graph read --subgraph features --view feature_context --limit 5` | 140 | Useful feature summaries without relation metadata. |
| `graph search-entities "context graph" --limit 3` | 53 | Enough for identity selection in many workflows. |

Direction: do not make JSON the default for agent read paths. Use text output
for routine context gathering, and reserve JSON for workflows that need exact
machine-checked structure, such as mutation proposals, commits, quality
automation, and tests.

### Query Surface Evidence

The current surface has the right basic primitives:

- `graph search-entities`: semantic entity lookup with type, predicate,
  subgraph, scope, truth, time, environment, and external-id filters.
- `graph read`: named use-case views with query, scope, repo, time, environment,
  depth, direction, limit, sort, dedupe, and format controls.
- `graph neighborhood`: node expansion by entity, predicate, depth, direction,
  and limit.
- `admin.inspection_slice`: broad raw graph view for the explorer.

What worked well:

- `search-entities "potpie"` found services, repo, and features.
- `search-entities "context graph"` found the repo, context-graph feature,
  context-engine service, preferences, and related activity.
- `graph neighborhood --entity service:potpie-legacy --predicate USES` returned
  a compact topology slice with Postgres, Neo4j, Redis, Qdrant, and Hatchet.
- Text `graph read` output is useful for quick feature, infra, preference, and
  timeline orientation.
- `--type`, `--predicate`, timeline date filters, time windows, query, and
  service scope can produce useful reductions.

What did not work well:

- `admin.inspection_slice` is claim-centered and not useful for broad graph
  orientation.
- Some node labels are polluted by projection drift, weakening type-based
  exploration.
- Existing commands still do not provide enough compact pot-level orientation by
  subgraph, entity type, predicate, source, truth, environment, or time window.
- Before Workstream 3, active repo scope could resolve to empty
  `pot_1e1cd80ab7de` unless the caller pinned a populated review pot. That is
  now covered by repo-local defaults, `pot linked`, and empty-pot warnings.

Post-Workstream 2 implementation review:

- `graph catalog --task ... --profile read` now exposes compact ranking metadata
  in table and JSON output: `rank`, `score`, `matched_terms`, `reason`, and
  `next_read`.
- `graph read` and `graph search-entities` now accept `--source-ref`, and exact
  source refs are threaded through `GraphReadRequest`, `GraphEntitySearchRequest`,
  `ClaimQueryFilter`, in-memory, FalkorDB, Neo4j, and semantic query paths.
- `graph search-entities --external-id github:potpie-ai/potpie#issue/881` now
  finds the corresponding timeline activity because claim source refs are checked.
- `preferences_for_scope` now accepts direct `service:<service>` scope.
- `infra_topology.service_neighborhood` now accepts
  `include_unqualified_environment`; strict environment reads can return no facts,
  while explicit unqualified inclusion returns unqualified topology.
- `graph neighborhood --detail summary` is relation-first and includes predicate,
  from/to keys, fact, source refs, truth, environment, score, claim key, and
  source system. `--detail full` keeps the raw node/edge projection available.
- Installed Codex and Claude skills were remove/reinstalled from the generated
  templates and now teach `--source-ref` and neighborhood summary drilldown.
- Focused Workstream 2 verification passed:
  `165 passed in 1.21s` for graph surface, CLI contract, backend reader, template,
  reader, entity summary, and quality unit tests.
- Remaining command-surface gap: `--source-family github` did not find the GitHub
  issue even though `--source-ref github:potpie-ai/potpie#issue/881` worked.
  Source-family filtering should derive provider family from source refs as well
  as first-class `source_system` values.
- Remaining ranking gap: catalog can rank `decisions.preferences_for_scope` first
  when the task explicitly says "coding preferences", but generic build tasks with
  "error handling", "logging", and "tests" can still rank preferences too low.
- The current quality summary is too shallow for agent trust. Backend analytics
  reports `status=ok` while workbench reports such as `projection-drift` and
  `conflicting-claims` show degraded/watch states.
- Projection drift checks literal endpoint-label pairs instead of the ontology's
  endpoint semantics (`@Scope`, `@Activity`, `CodeAsset`, wildcard), causing
  valid `Preference -> Service` and `Activity -> Repository/Service/Feature`
  claims to look invalid.
- Predicate-family conflict detection treats multi-valued `USES` dependencies as
  mutually exclusive. Conflict reporting should be limited to `singleton=True`
  predicates or families explicitly marked exclusive.

Post-Workstream 3 implementation review:

- Repo-local pot defaults are persisted and participate in pot resolution before
  current-repo linked-pot fallback and active-pot fallback.
- `source add repo` sets a repo-local default by default; `--no-default` keeps
  registration-only behavior available.
- `pot linked --repo current` and `pot default show/set/clear --repo current`
  expose multi-pot links, counts, active state, and default state.
- `graph status`, `graph read`, `graph search-entities`, timeline reads, and
  `source list` surface resolved pot identity and counts.
- Empty-pot output warns when another pot linked to the current repo has claims,
  and includes an exact `pot default set` command.
- `potpie ui --pot <pot>` opens the explorer with `?pot=<pot>`, and the UI pot
  selector returns per-pot claim/entity/predicate counts.
- The current repo is now defaulted to `pot_73b01a108e23`; `graph status`
  resolves there with 152 claims and no warning.

### Use-Case Coverage

| Use case | Current score | What works | Main gap |
| --- | ---: | --- | --- |
| Project preferences | 6/10 | Direct service scope works; exact preference search returns relevant context graph policies; catalog ranks preferences first when the task explicitly asks for coding preferences. | Coverage is narrow; no broad error handling, tests, file structure, framework/library, or logging preferences. Generic build-task ranking still needs preference synonyms. |
| Infra setup and architecture | 5/10 | `service:potpie-legacy` has datastore/dependency facts; `USES` node exploration works; strict vs include-unqualified environment reads are now explicit. | Sparse environment/deploy/adapter/service dependency topology; `potpie-context-engine` infra read returns only repo/feature; owners/docs are empty. |
| Timeline | 8/10 | Recent PRs/issues are queryable by window, date, service, topic, exact `--source-ref`, and `--external-id`; text output is useful. | Topic query still returns broad events after the top hit; source-family filtering does not yet infer `github` from source refs. |
| Bug/debug/troubleshooting | 1/10 | The `prior_occurrences` view exists. | Current graph has no `BugPattern` or `Fix` entities; setup/graph-read issues only appear as timeline activity, not reusable debug memory. |

### Command Drift

Examples found during review:

- Historical status drift: CLI docs described `potpie status` as context status
  while live behavior defaulted to integration auth unless `--host` or
  non-default options were used. Keep this covered by live CLI tests.
- The installed `potpie-cli` skill advertises `potpie --json search "query" -n
  15`, but top-level `search` does not support `-n`.
- Nudge instructions still mention `graph mutate`, while the graph workbench now
  prefers `graph propose` / `graph commit`.

These stale examples are planning blockers because agents will follow
right-sounding instructions and receive the wrong output or a CLI error.

## Deduplicated Backlog

The original review had the same issues repeated across evidence, findings,
pickup lists, suggested order, and chunk plans. The backlog below is the single
source of truth for planning.

| Priority | Theme | Consolidates | Planning outcome |
| --- | --- | --- | --- |
| P0 | Text-first and bounded agent reads | JSON cost, relation duplication, timeline dual shapes, expensive search, verbose catalog/describe | Agents get cheap routine context and can request full JSON only when needed. |
| P0 | Pot-scope visibility | UI pot confusion, active empty pot, missing pot/count output | Completed in Workstream 3: users and agents can tell immediately which pot is selected, set a repo default, and get actionable empty-pot warnings. |
| P1 | Existing-command graph exploration and drilldown | Raw inspection gaps, sparse neighborhood output, missing faceted orientation in current commands | Core drilldown flow completed in Workstream 2; repo/pot orientation completed in Workstream 3. |
| P1 | Filter normalization and source-ref lookup | Inconsistent scope aliases, weak environment filtering, missing source-ref filter | Core exact source-ref/service/environment flow completed in Workstream 2; source-family provider inference remains a follow-up. |
| P1 | Quality health cleanup | Misleading status, projection-drift noise, multi-binding conflict noise | Health signals become trustworthy enough for read confidence and ingestion gates. |
| P1 | Hard ingestion verification | Skill-only ingestion standards, `context_record` usage in skills | Broad graph writes get readback and quality checks through commands, not optional prose. |
| P1 | Docs and skill command validation | Stale CLI docs, stale skill examples, old `graph mutate` references | Workstream 2 graph/source-ingestion skill examples were refreshed; remaining CLI/status/search/mutation examples still need validation. |
| P2 | Freshness and coverage | TTLs not enforced, sparse preferences/infra/debug memory | Stale facts are detected and the graph becomes useful for the target workflows. |

## Workstreams

### 1. Text-First And Bounded Agent Reads

Goal: stop spending large JSON envelopes for routine context gathering.

Status: done. The bounded read/search/catalog surface is implemented, installed,
and verified. Deeper full-JSON relation de-duplication and formal payload-size
regression tests are intentionally skipped for this workstream.

Completed on 2026-06-17:

- Added `graph read` detail controls through the service DTO and CLI:
  `--detail compact|full` and `--relations summary|full`.
- Made compact read JSON the default by summarizing relation metadata instead
  of emitting full relation payloads from `GraphReadResult.to_dict()`.
- Kept explicit full-detail behavior available for UI/debug workflows with
  `--detail full --relations full`.
- Made timeline JSON `events` and `jsonl` output omit bulky `items` unless full
  detail is requested.
- Added `graph search-entities --supporting-claims N`; default search JSON no
  longer includes supporting claims.
- Updated the graph explorer API to opt into full read detail and bounded
  supporting claims so the UI keeps its richer behavior.
- Added compact catalog discovery through `graph catalog --profile read` and
  `graph catalog --profile read --format table`.
- Updated bundled/generated agent templates and skills so routine
  `graph read` / `graph search-entities` examples use text output by default,
  timeline reads prefer `--format table`, and JSON is reserved for machine
  contracts, writes, history verification, or full evidence/debug payloads.
- Added focused tests for compact read output, full-detail opt-in, timeline
  event payloads without `items`, bounded entity-search supporting claims,
  compact catalog discovery, and text-first agent templates.
- Reinstalled the editable CLI, restarted the daemon, refreshed installed
  Claude/Codex skills, and verified the installed command and skill copies use
  the new text-first/compact behavior.
- Measured payload-size improvements on `pot_03d60f3b579d`: catalog compact JSON
  13,537 -> 3,632 chars and table 452 chars; timeline limit-1 full JSON 64,373
  -> compact events 2,452 chars and table 1,145 chars; search limit-10 with
  claims 27,108 -> compact 6,378 chars and text 221 chars; feature read limit-5
  full JSON 14,387 -> compact 4,279 chars and text 570 chars.

Validated with:

- `uv run pytest tests/unit/test_graph_surface_lite_contract.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`
- `uv run ruff check adapters/inbound/cli/commands/graph.py adapters/inbound/http/ui/router.py application/services/graph_service.py domain/ports/services/graph_service.py tests/unit/test_graph_surface_lite_contract.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`
- `uv run ruff format --check adapters/inbound/cli/commands/graph.py adapters/inbound/http/ui/router.py application/services/graph_service.py domain/ports/services/graph_service.py tests/unit/test_graph_surface_lite_contract.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`
- `potpie doctor`
- `potpie graph catalog --profile read --format table`
- `potpie --json graph read --pot pot_03d60f3b579d --subgraph recent_changes --view timeline --limit 1 --format events`
- `potpie --json graph search-entities --pot pot_03d60f3b579d "context graph" --limit 10`

Deferred / skipped:

- Deduplicate relation claims in full JSON by emitting each claim once and
  referencing claim ids from entity relations. Full detail remains available
  with `--detail full --relations full`.
- Add formal token-size regression tests for the key agent paths. Manual
  before/after measurements were captured instead.

Scope:

- Update installed and generated agent skills so read-only examples omit
  `--json` by default.
- Prefer text `graph read` for features, infra, preferences, decisions, and
  debugging reads.
- Prefer `graph read --subgraph recent_changes --view timeline --format table`
  or non-JSON `--format jsonl` for timeline reads.
- Make compact responses the default across read/search JSON surfaces; use
  explicit flags such as `--detail full` or `--relations full` to request the
  current evidence-rich payloads.
- For JSON `--format events` and `--format jsonl`, stop returning bulky
  relation-backed `items` unless full detail is requested.
- Add `--detail compact|full`, `--relations summary|full`, or equivalent
  controls for `graph read`.
- Deduplicate relation claims in JSON by emitting each claim once and
  referencing claim ids from entity relations.
- Add `--detail compact|full` or `--supporting-claims N` to
  `graph search-entities`.
- Add a compact read-only contract discovery mode, such as
  `graph catalog --format table` or `--profile read`.

Acceptance:

- Skills clearly say: text is for agent context; JSON is for machine contracts.
- `graph read --subgraph recent_changes --view timeline --limit 1 --format
  events --json` returns one event and does not include the full relation graph
  by default.
- Feature and infra JSON reads return compact entity summaries by default
  without full relation payloads.
- `search-entities "context graph" --limit 10 --json` no longer includes full
  supporting claims by default.
- Explicit full-detail modes preserve current evidence-rich behavior for
  UI/debug workflows.
- Tests assert bounded output shape for the key agent paths.

### 2. Existing-Command Exploration And Filter Normalization

Goal: let agents explore broadly, then drill into the right node/subgraph
without pulling full evidence for the whole graph.

Status: implemented and verified on 2026-06-17. Accepted for the core
existing-command drilldown flow: direct service scope, exact source-ref
read/search, external-id fallback to source refs, infra strict-vs-unqualified
environment reads, compact catalog ranking metadata, and relation-first
neighborhood summaries all work against the populated review pot. Remaining
improvements are tracked below and in later workstreams.

Verified agent flow:

```bash
potpie graph status
potpie graph catalog --task "<task>" --profile read
potpie graph read --subgraph decisions --view preferences_for_scope --scope repo:<repo>,path:<path> --limit 8
potpie graph search-entities "<service or feature>" --type Service --limit 5
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:<service> --depth 1 --limit 8
potpie graph neighborhood --entity service:<service> --predicate USES --detail summary --limit 20
potpie graph read --subgraph recent_changes --view timeline --source-ref <github/pr-or-issue> --format table
```

Completed on 2026-06-17:

- Kept broad exploration inside existing commands instead of adding a separate
  map command: `graph status` for compact pot health/orientation, `graph catalog`
  for available views, `search-entities` for faceted identity lookup, `graph read`
  for use-case context, and `graph neighborhood` for node expansion.
- Added `source_ref_in` to the claim-query contract and implemented it in
  in-memory, FalkorDB, Neo4j, and semantic query paths.
- Added `--source-ref` to `graph read` and `graph search-entities`, then threaded
  source refs through service DTOs and view readers.
- Made `--external-id` match claim source refs for activity / ticket / PR style
  handles.
- Normalized view contracts so `decisions.preferences_for_scope` accepts direct
  `service:<service>` scope and `infra_topology.service_neighborhood` accepts
  `include_unqualified_environment`.
- Kept aliases consistent for repo, path, file path, service, feature,
  `anchor_entity_key`, environment, and `source_ref`.
- Preserved task-ranked catalog ordering and exposed compact ranking metadata in
  text and JSON: rank, score, matched terms, reason, and `next_read`.
- Added `graph neighborhood --detail summary|full`; summary is relation-first and
  includes from/to keys, predicate, fact, source refs, truth, environment, score,
  claim key, and source system.
- Added explicit full-detail flags on existing commands for selected nodes/claims.
- Updated generated and installed Codex/Claude skills after command contracts
  were tested, so agents learn `--source-ref` and neighborhood summary drilldown.
- Reinstalled the editable CLI, restarted the daemon, and remove/reinstalled
  Codex and Claude managed Potpie skills.

Validated with:

- `potpie graph read --pot pot_03d60f3b579d --subgraph recent_changes --view timeline --source-ref github:potpie-ai/potpie#issue/881 --format table --limit 5`
- `potpie graph search-entities --pot pot_03d60f3b579d "issue 881" --source-ref github:potpie-ai/potpie#issue/881 --limit 8`
- `potpie graph search-entities --pot pot_03d60f3b579d "issue 881" --external-id github:potpie-ai/potpie#issue/881 --limit 8`
- `potpie graph read --pot pot_03d60f3b579d --subgraph decisions --view preferences_for_scope --scope service:potpie-context-engine --query "error handling logging tests file structure preferred libraries frameworks" --limit 8`
- `potpie graph read --pot pot_03d60f3b579d --subgraph infra_topology --view service_neighborhood --scope service:potpie-legacy,include_unqualified_environment:true --environment prod --depth 2 --direction both --limit 12`
- `potpie graph neighborhood --pot pot_03d60f3b579d --entity service:potpie-legacy --predicate USES --detail summary --limit 8`
- `uv run pytest tests/unit/test_graph_surface_lite_contract.py tests/unit/test_graph_cli_contract.py tests/unit/test_neo4j_claim_query.py tests/unit/test_falkordb_reader.py tests/unit/test_agent_templates_v15.py tests/unit/test_p9_readers.py tests/unit/test_graph_entity_summary.py tests/unit/test_graph_workbench_quality.py`

Acceptance result:

- Accepted: service preference scope, exact source-ref read/search, external-id
  lookup via source refs, strict/unqualified environment behavior, catalog
  ranking metadata, summary/full neighborhood detail, and installed skill examples.
- Carried to Workstream 3: compact pot/repo orientation and empty-pot warnings.
- Carried as Workstream 2 follow-up / Workstream 6 validation: source-family
  provider inference should treat `github:` source refs as GitHub family even
  when the claim's stored `source_system` is `cli`.
- Carried as Workstream 2 follow-up / Workstream 7 coverage: catalog task ranking
  needs preference synonyms so generic build tasks mentioning error handling,
  logging, tests, frameworks, and file layout naturally surface
  `decisions.preferences_for_scope`.
- Carried to Workstream 7: graph data coverage remains sparse for broad coding
  standards, context-engine infra/deploy topology, and reusable debug memory.

### 3. Pot-Scope Visibility

Goal: prevent empty-pot confusion from looking like graph failure.

Status: implemented, reinstalled, and verified end to end on 2026-06-17. Pot
resolution now supports a repo-local default that is set automatically by
`source add repo` and setup repo registration paths. `graph status`,
`graph read`, timeline reads, `search-entities`, and `source list` surface the
resolved pot and graph counts, and empty-pot output warns when the current repo
is linked to a populated alternative. `pot linked` and
`pot default set/show/clear` expose multi-pot repo links and let users pick the
default. `potpie ui --pot` opens the explorer against a chosen pot, and the UI
selector shows per-pot claim counts.

Completed on 2026-06-17:

- Added local `repo_defaults` persistence to the local pot store and pot
  management service.
- Made `source add repo` set the repo default by default, with `--no-default`
  for edge cases.
- Added `pot default set/show/clear` and `pot linked --repo current`.
- Updated pot resolution so explicit `--pot` wins, then repo default, then
  linked current-repo pots, then active pot.
- Added pot id/name, source count, claim count, and entity count to key
  agent-facing text surfaces.
- Added actionable empty-pot warnings with an exact `pot default set` command.
- Added `potpie ui --pot` and UI selector counts.
- Built the UI bundle and verified focused CLI/UI behavior.
- Reinstalled Potpie from editable local packages with Python 3.13, restarted
  the daemon, and verified installed CLI commands against the live daemon.
- Removed/reinstalled Codex and Claude global Potpie skills and confirmed the
  installed `potpie-cli` skill teaches repo defaults, `pot linked`, and
  `source add repo --no-default`.
- Set this repo's default to `pot_73b01a108e23` so agent commands resolve to
  the populated pot instead of the active empty `test` pot.

End-to-end verification:

- `potpie daemon status` reports `daemon: detached (up=True)`.
- `potpie --json skills status --agent codex --scope global` and the Claude
  equivalent report all eight Potpie skills installed with no missing/outdated
  entries.
- `potpie --json pot linked --repo current` lists multiple pots linked to
  `github.com/potpie-ai/potpie`, including populated candidates with counts.
- `potpie pot default set --repo current pot_73b01a108e23` makes
  `graph status` resolve to `pot_73b01a108e23` with 152 claims and no warnings.
- `potpie pot default clear --repo current` makes `graph status` fall back to
  active `pot_1e1cd80ab7de` and emit the actionable populated-pot warning.
- `potpie --json source list` resolves through the repo default and returns the
  populated pot's repo and Linear sources.
- `potpie --json graph search-entities potpie --limit 5` resolves through the
  repo default and returns populated graph entities.
- `potpie --json ui --pot pot_73b01a108e23 --no-open` returns a URL containing
  `/ui?pot=pot_73b01a108e23`.
- `/ui/api/pots` returns pots with `counts` payloads for the UI selector.
- In an isolated `CONTEXT_ENGINE_HOME`, `source add repo .` normalized
  `remote.origin.url` to `github.com/potpie-ai/workstream3-smoke`, set the
  repo default automatically, and a later `source add repo ... --no-default`
  did not overwrite that default.

Acceptance result:

- Accepted: the current repo shows whether it is using `test` or the populated
  repo default, including pot id/name and graph counts.
- Accepted: empty graph output includes an actionable pot-selection hint when a
  populated linked pot exists.
- Accepted: the UI can be launched directly against a chosen pot and surfaces
  pot counts in the selector.

Scope:

- Include resolved pot id/name, source count, and claim count in text output for
  `source list`, `graph status`, `graph read`, and related read commands.
- Warn when the current pot has zero claims and another known pot has a source
  for the current repo with non-zero claims.
- Add or improve `potpie ui --pot`.
- Show pot id and claim/entity counts in the UI selector.

Acceptance:

- A user can see in one line whether they are querying `test` or the populated
  repo default `pot_73b01a108e23`.
- Empty graph output includes an actionable pot-selection hint when applicable.
- The UI can be launched directly against a chosen pot.

### 4. Quality Health Cleanup

Goal: make health and quality reports trustworthy enough to guide agents and
gate ingestion.

Status: done in the current tree. Existing implementation and tests now make the
quality summary aggregate deep reports, distinguish command success from health,
use ontology endpoint semantics for projection drift, and avoid treating
multi-binding predicates such as `USES` as singleton conflicts. This is the
baseline that Workstream 5's verified commit gate builds on.

Scope:

- Make `graph status` include a clean, compact health summary by default.
- Keep detailed quality investigation in `graph quality`.
- Distinguish command success from memory health in JSON, such as `ok` vs
  `health_status`.
- Include projection drift, conflicts, stale facts, duplicates, orphans, and
  low-confidence counts in the visible status surface.
- Fix projection drift endpoint matching to use ontology semantics for
  `@Scope`, `@Activity`, `CodeAsset`, and wildcard endpoints.
- Fix conflict detection so multi-binding predicates like `USES` are not
  treated as mutually exclusive unless a predicate or family is explicitly
  singleton.
- Make `quality summary` aggregate bounded counts from deep quality reports
  instead of delegating solely to backend analytics.
- Use `EdgeTypeSpec.allows()` or equivalent ontology helper logic for projection
  drift endpoint checks.
- Split predicate-family metadata into at least two concepts: grouping for
  recall/maintenance, and exclusivity for conflict/supersession.

Acceptance:

- Valid `Preference -> Service` and `Activity -> Repository/Service` claims stop
  appearing as projection drift solely due to macro endpoint handling.
- `USES` multi-bindings for `service:potpie-legacy` are no longer reported as
  conflicts unless there is a real singleton violation.
- Status distinguishes command success from graph health.
- Deep quality counts are visible in the default status/health path.

### 5. Ingestion Verification And Record Boundaries

Goal: prevent shallow or quality-breaking graph writes.

Status: implemented and verified on 2026-06-18. The chosen shape is
`graph commit --verify` rather than a separate ingestion command, so verification
stays attached to the exact persisted plan that was committed.

Plan:

- Keep `graph propose` as the validation / persistence step.
- Add a verified commit gate on `graph commit --verify`.
- During verified commit, snapshot quality before apply, apply the stored lowered
  batch, read back committed claim keys, then snapshot quality again.
- Fail the verification gate when committed claim keys do not read back or when
  selected quality counts/status regress after commit.
- Warn, but do not confuse with write failure, when existing graph health is not
  clean and did not regress.
- Keep `context_record` public only as an MCP compatibility fallback.
- Update broad ingestion skills and examples to use verified commit as the hard
  gate; leave manual read/quality commands as follow-up diagnostics when the
  gate warns or fails.

Completed on 2026-06-18:

- Added `GraphIngestionVerificationResult` and embedded it in
  `GraphMutationCommitResult.to_dict()`.
- Added `verify` to `GraphWorkbenchService.commit(...)`.
- Implemented claim-key readback through `ClaimQueryFilter.claim_key_in`.
- Reused the Workstream 4 quality summary path for post-commit checks and
  compared pre/post quality counts and health status.
- Added `graph commit --verify` CLI support and compact human output:
  verification status, readback count, quality status, missing claim keys, and
  quality regressions.
- Made the CLI return a non-zero exit after a successful commit when the
  verification gate itself fails, while preserving the commit result and
  verification payload in JSON.
- Updated generated Codex/Claude agent templates and Potpie skills so graph
  write examples use `graph commit --verify`.
- Reframed source-ingestion verification: `graph commit --verify` is the gate;
  affected reads and `graph quality ...` commands are diagnostics when the gate
  warns or fails.
- Kept `context_record` references only under MCP compatibility framing.
- Added tests for verified commit readback, quality regression detection, CLI
  flag threading, and template enforcement that commit examples include
  `--verify`.

Validated with:

- `uv run pytest tests/unit/test_graph_workbench_plans.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`
- `uv run ruff check domain/graph_plans.py application/services/graph_workbench.py adapters/inbound/cli/commands/graph.py tests/unit/test_graph_workbench_plans.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`
- `uv run ruff format --check domain/graph_plans.py application/services/graph_workbench.py adapters/inbound/cli/commands/graph.py tests/unit/test_graph_workbench_plans.py tests/unit/test_graph_cli_contract.py tests/unit/test_agent_templates_v15.py`

Acceptance result:

- Accepted: a verified broad commit can fail the gate on readback misses or
  immediate quality regressions.
- Accepted: verified commit reports readback, quality status, quality deltas,
  regressions, and suggested next action in JSON, with compact text output.
- Accepted: broad ingestion skills route source-backed writes through
  `graph propose` / `graph commit --verify`; `context_record` remains only as an
  MCP compatibility fallback.

### 6. Docs And Skill Command Validation

Goal: keep agent-facing instructions synchronized with live commands.

Status: partially improved by Workstream 2. Generated and installed Codex/Claude
graph/source-ingestion skill examples now include `--source-ref` and
`graph neighborhood --detail summary`, but broader stale CLI examples remain.

Scope:

- Generate or validate command examples against live Typer help.
- Add tests that run documented snippets in help or dry-run mode.
- Add a validation case for `--source-family github` so GitHub PR/issue refs
  match through provider-family inference, not only through exact `--source-ref`.
- Replace `graph mutate` references in nudge instructions with
  `graph propose` / `graph commit`.
- Keep `potpie status` docs and live default command behavior aligned.
- Fix the installed `potpie-cli` search example that uses unsupported `-n`.

Acceptance:

- Stale command examples fail tests before they reach installed skills or docs.
- The status/search/mutation examples in docs and skills match live CLI
  behavior.
- Workstream 2 examples cover exact lookup handles: source refs, external ids,
  service scope, path scope, environment, and neighborhood drilldown.
- Tests pin compact catalog ranking output so scores/reasons do not disappear
  from the agent-facing read profile.

### 7. Freshness And Use-Case Coverage

Goal: improve the graph after the surface is cheaper and quality gates are less
noisy.

Status: still open after Workstream 2 verification. The query surface can now
reach exact handles and service/environment scopes, but the populated review graph
is still sparse for broad coding standards, context-engine infra/deploy topology,
owners/docs, and reusable debug memory.

Scope:

- Apply ontology freshness TTLs in stale-fact reporting.
- Surface refresh recommendations in status/health output.
- Re-run or broaden repo baseline ingestion for service/deploy/API/data
  topology, owners, docs/runbooks, and active decisions.
- Ingest coding standards for error handling, tests, file structure, logging,
  libraries/frameworks, security, and API style.
- Add catalog ranking synonyms so build-code tasks mentioning error handling,
  logging, tests, framework/library choice, file structure, API style, or naming
  surface `decisions.preferences_for_scope` even when the user does not say
  "coding preferences" explicitly.
- Model timeline events as correlation leads and link them to bug patterns,
  fixes, decisions, and infra facts only when source evidence supports it.
- Ingest verified setup issues, recurring failures, failed attempts, fixes, and
  verification status as `BugPattern`/`Fix` facts.

Acceptance:

- Stale facts can be detected without explicit stale markers.
- Baseline reads for this repo become less sparse after quality gates are fixed.
- Each target use case has non-empty reads for this repo.
- Coverage reports can show which use-case entity families are still missing.
- Timeline-only incidents that become reusable troubleshooting knowledge are
  represented in debug memory with evidence.

## Preserve

Keep these existing strengths while planning the cleanup:

- The graph workbench primitives are the right foundation: catalog, describe,
  read, search-entities, propose, commit, history, inbox, and quality.
- Semantic mutation validation catches unknown predicates, bad endpoint types,
  bad truth classes, invalid confidence, malformed timestamps, and missing
  evidence for high-authority claims.
- The source-ingestion and repo-baseline skills are thorough and correctly push
  agents toward source-backed, harness-led ingestion.
- The tests pin many important content contracts: four MCP tools, no stale
  include names, canonical V2 read syntax, harness-led boundaries, and required
  ingestion workflow sections.
- The current graph already contains useful features, preferences, and recent
  GitHub activity.

## Planning Decisions

- `graph status` should show a clean, compact health summary by default.
- Broad exploration should be supported through existing commands instead of a
  separate `graph map` or `graph summary` command.
- Exact source refs are first-class graph handles, not query text. Agents should
  be able to jump directly from a PR, issue, ticket, doc, or deploy ref to the
  corresponding timeline/debug/doc facts.
- Catalog is the agent's capability overview. The compact read profile should
  explain which view to try next and why, without forcing full ontology JSON.
- Neighborhood is the drilldown surface. Its default output should be a bounded
  fact list with source refs and truth metadata, not only structural counts.
- `context_record` can remain public, but it should not be used in the reviewed
  use cases and should be removed from related skills and examples.
- Compact responses should be the default. Explicit flags should be required to
  request full data.
