# Extending the Context Engine

This is the contributor guide for adding to the engine without touching the application layer. The `SourceConnector` registry (Phase 2) and the `ContextReader` registry (Phase 3) are both live.

---

## Adding a new source

A source is anything that contributes events, entities, or evidence to a pot: GitHub, Linear, Notion, Slack, Sentry, PagerDuty, a documentation site, a CI pipeline.

Adding a source means writing **one** module under `app/src/context-engine/adapters/outbound/connectors/<your_source>/` that implements [`SourceConnectorPort`](../../app/src/context-engine/domain/ports/source_connector.py) and registering it in `bootstrap/container.py`. The application layer never imports your connector.

The contract:

```python
class SourceConnectorPort(Protocol):
    def kind(self) -> str: ...                                                # "notion"
    def capabilities(self) -> Sequence[SourceCapability]: ...                 # which (provider, source_kind) pairs you serve
    def list_artifacts(self, scope: ConnectorScope) -> Iterable[SourceRef]: ...
    def normalize_webhook(self, payload: bytes, headers: Mapping[str, str]) -> ContextEvent | None: ...
    async def fetch(self, *, pot_id, refs, source_policy, budget, auth) -> SourceResolutionResult: ...
    def propose_plan(self, event: ContextEvent, context_graph) -> ReconciliationPlan | None: ...
```

`propose_plan` is **optional** — deterministic connectors return a plan; passive connectors return `None` and let the reconciliation agent plan from the raw event. Every other verb is similarly optional: declare what you do via the `fetch_capable / list_capable / webhook_capable / plan_capable / sync_capable` flags on each `SourceCapability` and the registry will route around the verbs you don't implement.

### Worked example: the Notion connector (Phase 2 smoke test)

The whole connector lives in [`adapters/outbound/connectors/notion/connector.py`](../../app/src/context-engine/adapters/outbound/connectors/notion/connector.py) — a single file, ~250 lines, with no edits required outside the connector dir except the one-line registration.

1. **Define the connector class.** Implement the six methods. For Notion the connector advertises `(provider="notion", source_kind="page")` with policies `summary` and `snippets`, sets `webhook_capable=False` (Notion's webhook payloads aren't stable yet), and emits a small `ReconciliationPlan` with one `Document` entity per page.

2. **Inject your read surface.** Notion takes a `NotionPageFetcher` Protocol so tests can pass a fake. The host wires up the live HTTP client.

3. **Register in `bootstrap/container.py`:**

   ```python
   registry = SourceConnectorRegistry()
   registry.register(NotionConnector(fetcher=my_notion_fetcher))
   container = build_container(pots=pots, connectors=registry, ...)
   ```

4. **Verify** with `tests/unit/test_source_connector_registry.py::test_third_source_smoke_test_notion_connector_loads` — that test fails the moment the contract drifts and the connector can no longer be added without touching `application/` or `domain/`.

### What does not change

- The agent contract: connectors do not appear in `agent-contract.md`. Their capabilities surface through `context_status.connectors`.
- The write path: every plan a connector proposes is still validated against the ontology and applied through Graphiti.
- The graph: connectors do not write directly. They produce events and plans; the registry routes; the reconciliation pipeline applies.

### What gets registered, not invented

- Identifying conventions (e.g. `notion:page:<page_id>`, `github:pr:<repo>:<number>`) are connector-private. Pick something stable and stick with it; the registry doesn't care about the format.
- Authentication and credential lookup live entirely in the connector. The engine threads a `ResolverAuthContext` through but does not inspect it.

---

## Adding a new evidence family

An evidence family is one read-side leg of the context graph: `decisions`, `change_history`, `owners`, `project_graph`, `graph_overview`, etc. Adding a family means writing **one** module under `app/src/context-engine/adapters/outbound/readers/<your_family>.py` that implements [`ContextReaderPort`](../../app/src/context-engine/domain/ports/context_reader.py) and adding one `register()` call in `bootstrap/container.py`. The application layer never imports your reader.

The contract:

```python
class ContextReaderPort(Protocol):
    def family(self) -> str: ...                          # "release_notes"
    def capability(self) -> ReaderCapability: ...         # intents, requires_scope, cost, backend
    def read(self, request: ContextGraphQuery) -> ReaderResult: ...
```

`ReaderCapability` is the routing contract:

- `family` — stable key the registry routes on.
- `intents` — `frozenset[str]` of agent intents that should auto-include the reader (e.g. `{"review", "operations"}`).
- `requires_scope` — `frozenset[str]` of scope fields the reader needs. Recognised values: `file_path`, `function_name`, `pr_number`, `repo_name`, `branch`, `user`, `query`. Missing scope becomes a `missing_scope` fallback in the response — the reader is not invoked.
- `cost` — `ReaderCost("cheap" | "medium" | "expensive", estimated_ms)`. Used by the router (and Phase 5's policy/budget enforcement).
- `backend` — informational: `"structural"`, `"graphiti"`, or `"hybrid"`.
- `compat=True` — flags legacy readers (the registry stamps `meta.compat=True` on the response).

`ReaderResult` carries `result`, optional `count`, optional `error`, optional `fallback_reason` (e.g. `"semantic_fallback"`), and optional `compat`.

### Worked example: the release-notes reader (Phase 3 smoke test)

The whole reader lives in [`adapters/outbound/readers/release_notes.py`](../../app/src/context-engine/adapters/outbound/readers/release_notes.py) — a single file, ~80 lines, with no edits required outside the reader file except the one-line registration.

1. **Define the reader class.** Implement the three methods. `release_notes` advertises `intents={"operations","review","planning"}`, no `requires_scope`, `cost="cheap"`, and reuses the existing `get_change_history` helper to pull recent merged PRs, then filters them by title/label heuristics.

2. **Register in `bootstrap/container.py`** (inside `_default_reader_registry`):

   ```python
   registry.register(ReleaseNotesReader(structural=structural))
   ```

3. **Verify** with `tests/unit/test_context_reader_registry.py::test_third_reader_smoke_test_release_notes_loads` — that test fails the moment the contract drifts and a new reader can no longer be added without touching `application/` or `domain/`.

### What does not change

- `context_resolve` parameters are unchanged. New family = new include value, not a new tool.
- `agent-contract.md`'s include catalog stays the source of truth for the agent-facing vocabulary; reader families are the engine-internal surface that the application layer routes through.
- The answer path: `goal=ANSWER` does not run readers — it composes `resolve_context` + the answer synthesizer. Readers are for evidence-family retrieval, not synthesis.

### What gets registered, not invented

- Naming conventions: family keys are lowercase, snake_case, stable. The registry is case-insensitive on lookup but stores the lowercase form.
- Backend choice: a reader may compose any combination of `EpisodicGraphPort` (vector / episodic) and `StructuralReadPort` (canonical reads). Use the existing helpers under `adapters/outbound/graphiti/query_helpers.py`; introduce new helpers there before reaching past them.

---

## Adding a new agent record type

Record types are the union for `context_record.record_type` — defined in `agent_context_port.CONTEXT_RECORD_TYPES`. To add one:

1. Add the value to `CONTEXT_RECORD_TYPES`.
2. Update [`agent-contract.md`](./agent-contract.md) record-type table.
3. Make sure the reconciliation pipeline knows how to turn the new type into a canonical mutation. (For `decision`, `fix`, `preference`, `workflow`, `feature_note`, etc., this is already wired; for a genuinely new shape, you may need to extend the agent's plan validator.)

Do not invent new record types lightly. The full list lives in code; the criterion for a new one is "this is a new kind of durable project memory that doesn't fit any existing slot."

---

## Adding a new intent

Intents drive the default include set per task. Defined in `agent_context_port.CONTEXT_INTENTS` and `DEFAULT_INTENT_INCLUDES`. To add one:

1. Add the intent string to `CONTEXT_INTENTS`.
2. Add a default include set in `DEFAULT_INTENT_INCLUDES`.
3. Add a recipe entry in `CONTEXT_RESOLVE_RECIPES` with `mode`, `source_policy`, and a `when` rule.
4. Update the recipes table in [`agent-contract.md`](./agent-contract.md).

The intent surface is intentionally narrow. Most "I need a new intent" instincts are actually "I need a new recipe preset over an existing intent." Resist adding intents that overlap existing ones.

---

## What this guide will *not* cover

- Adding a new graph store. There is one. See `vision.md`.
- Adding a new public agent tool. There are four. See `agent-contract.md`.
- Adding a "compatibility path" for the old version of any of the above. Replace, do not parallel.
