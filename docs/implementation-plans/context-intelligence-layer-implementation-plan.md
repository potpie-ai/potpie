# Context Intelligence Layer — Implementation Plan

Branch: `feat/context-engine`

Spec reference: [context-graph-intelligence-layer-spec.md](./context-graph-intelligence-layer-spec.md)

---

## 1. Objective

Implement a provider-agnostic context intelligence layer inside `context-engine` that:

- follows the existing `domain` / `application` / `adapters` / `bootstrap` structure,
- uses the current graph stack as the first provider implementation,
- can be consumed by Potpie agents and external clients,
- reduces repeated context tool chaining for history-heavy questions.

---

## 2. Guiding Constraints

### 2.1 Architectural constraints

- The runtime must depend on `IntelligenceProvider`, not directly on Neo4j or Graphiti.
- The current graph-backed implementation must live behind an outbound adapter.
- The main entry point must be a use case in `application/use_cases/`.
- Provider methods must be async. Sync backends use `asyncio.to_thread()` inside the adapter.
- Authentication is the caller's responsibility, not the use case's.

### 2.2 Product constraints

- Potpie should be the first consumer, not the only consumer.
- The first release should improve historical-context retrieval without forcing a major agent rewrite.
- Existing Potpie context tools (`get_change_history_tool`, `get_decisions_tool`, etc.) remain available during transition.
- The design must support HTTP and MCP exposure using the same application contract.

### 2.3 Scope constraints

The intelligence layer resolves historical/contextual evidence only. It does **not** own:

- code-node lookups (fetching source code from node IDs),
- file-structure enumeration,
- live code search / embedding queries.

These remain the caller's responsibility.

### 2.4 Rollout constraints

- Start with one provider implementation: current hybrid graph stack.
- Keep existing context query endpoints working and unchanged.
- Ship behind a feature flag for agent-runtime adoption.

---

## 3. Target File Layout

```text
context-engine/src/
├── domain/
│   ├── intelligence_models.py       # NEW
│   ├── intelligence_signals.py      # NEW
│   ├── intelligence_policy.py       # NEW
│   └── ports/
│       └── intelligence_provider.py # NEW
├── application/
│   ├── services/
│   │   └── context_resolution.py    # NEW
│   └── use_cases/
│       └── resolve_context.py       # NEW
├── adapters/
│   ├── outbound/
│   │   └── intelligence/
│   │       ├── __init__.py          # NEW
│   │       ├── hybrid_graph.py      # NEW
│   │       └── mock.py              # NEW
│   └── inbound/
│       ├── http/api/v1/context/router.py  # EXTEND
│       └── mcp/server.py                  # EXTEND
└── bootstrap/
    └── container.py                       # EXTEND
```

Existing files are not removed or renamed. All changes are additive or extend existing files.

---

## 4. Phase 0 — Domain Contracts

**Goal:** Define the stable interface and models before writing runtime logic.

### 4.1 Add `domain/ports/intelligence_provider.py`

Define a provider-neutral async protocol:

- `search_context(project_id, query, limit)` — semantic/embedding search
- `get_artifact_context(project_id, artifact)` — artifact lookup (PR, issue, etc.)
- `get_change_history(project_id, scope, limit)` — deterministic history lookup
- `get_decision_context(project_id, scope, limit)` — deterministic decision lookup
- `get_related_discussions(project_id, scope, limit)` — deterministic discussion lookup
- `get_ownership(project_id, scope, limit)` — deterministic ownership lookup
- `get_capabilities()` — sync, returns `CapabilitySet`

Important:

- no graph-specific terms,
- no storage-vendor assumptions,
- all data methods are `async`,
- return normalized domain models from `intelligence_models.py`.

### 4.2 Add `domain/intelligence_models.py`

Define all request, response, and evidence models as specified in the spec (Section 6.1):

Request models:

- `ContextResolutionRequest` (project_id, query, consumer_hint, artifact_ref, scope, timeout_ms)
- `ArtifactRef` (kind, identifier)
- `ContextScope` (file_path, function_name, symbol)

Response models:

- `IntelligenceBundle` (request, artifacts, changes, decisions, discussions, ownership, coverage, errors, meta)
- `CoverageReport` (status, available, missing, missing_reasons)
- `ResolutionError` (source, error, recoverable)
- `ResolutionMeta` (provider, total_latency_ms, per_call_latency_ms, capabilities_used, schema_version)
- `CapabilitySet` (boolean per evidence family)

Evidence records:

- `ArtifactContext`
- `ChangeRecord`
- `DecisionRecord`
- `DiscussionRecord`
- `OwnershipRecord`

### 4.3 Add `domain/intelligence_signals.py`

Implement pure functions for:

- PR number extraction (regex: `#\d+`, `PR \d+`, `pull request \d+`)
- file path detection (regex: paths with `/` and common extensions)
- symbol detection (CamelCase / snake_case tokens)
- history / rationale cue detection (keywords: `why`, `when`, `who`, `changed`, `history`, `decision`, `discussion`, `review`, `rationale`, `removed`, `added`, `introduced`, `refactor`)
- ownership cue detection (keywords: `who`, `owner`, `owns`, `maintainer`, `worked on`)

Output: a `SignalSet` dataclass with boolean flags and extracted entities (PR number, file paths, symbols).

### 4.4 Add `domain/intelligence_policy.py`

Implement:

- signal-to-evidence-plan mapping,
- per-consumer profile defaults,
- mandatory evidence calculation per signal combination,
- timeout budget selection,
- capability-aware planning: skip evidence families the provider doesn't support.

Policy output: an `EvidencePlan` dataclass listing which provider methods to call, which are mandatory, and the timeout budget.

**Deliverable:** the domain layer compiles and has no dependency on Graphiti, Neo4j, HTTP, MCP, or Potpie agents.

---

## 5. Phase 1 — Resolution Runtime

**Goal:** Create the application-layer orchestration service.

### 5.1 Add `application/services/context_resolution.py`

Class: `ContextResolutionService`

Constructor receives:

- `provider: IntelligenceProvider`
- optional config overrides

`resolve(request: ContextResolutionRequest) -> IntelligenceBundle`:

1. Extract signals from `request.query` (and `request.artifact_ref`, `request.scope` if provided).
2. Build evidence plan using policy + provider capabilities.
3. Launch all planned provider calls concurrently via `asyncio.gather()`.
4. Apply a shared timeout (`request.timeout_ms`).
5. For each call: record result, latency, or error.
6. Assemble `IntelligenceBundle` with normalized evidence.
7. Compute `CoverageReport` from plan vs results.
8. Return bundle.

Error handling:

- Each provider call is wrapped in a try/except.
- Timeouts are handled via `asyncio.wait_for()` per task or via a global deadline.
- Errors produce `ResolutionError` entries; they do not crash the resolution.
- If all calls fail, return an empty bundle with `coverage.status = "empty"` and all errors listed.

### 5.2 Add `application/use_cases/resolve_context.py`

Thin wrapper:

```python
def resolve_context(
    service: ContextResolutionService,
    project_id: str,
    query: str,
    *,
    consumer_hint: str | None = None,
    artifact_ref: ArtifactRef | None = None,
    scope: ContextScope | None = None,
    timeout_ms: int = 4000,
) -> IntelligenceBundle:
    request = ContextResolutionRequest(
        project_id=project_id,
        query=query,
        consumer_hint=consumer_hint,
        artifact_ref=artifact_ref,
        scope=scope,
        timeout_ms=timeout_ms,
    )
    return await service.resolve(request)
```

This matches the existing use-case style in `query_context.py`.

### 5.3 Add tests for the application layer

Use `mock.py` provider for deterministic tests.

Test cases:

- Signal extraction for history-heavy queries
- Signal extraction for code-navigation queries
- Policy produces correct evidence plan per signal set
- Capability-aware degradation (semantic search disabled)
- Partial coverage when one call times out
- Empty bundle when provider is completely down
- Coverage `missing_reasons` populated correctly
- Timeout enforcement does not exceed budget

**Deliverable:** one stable application entry point resolves context without depending on Potpie code.

---

## 6. Phase 2 — First Provider Implementation

**Goal:** implement the first `IntelligenceProvider` using the current graph stack.

### 6.1 Add `adapters/outbound/intelligence/hybrid_graph.py`

Class: `HybridGraphIntelligenceProvider(IntelligenceProvider)`

Constructor receives:

- `episodic: EpisodicGraphPort`
- `structural: StructuralGraphPort`

Method mapping:

| Provider method | Implementation |
|----------------|---------------|
| `search_context` | `episodic.search()` via `to_thread()` → normalize to `list[dict]` |
| `get_artifact_context` | `structural.get_pr_review_context()` via `to_thread()` → normalize to `ArtifactContext` |
| `get_change_history` | `structural.get_change_history()` via `to_thread()` → normalize to `list[ChangeRecord]` |
| `get_decision_context` | `structural.get_decisions()` via `to_thread()` → normalize to `list[DecisionRecord]` |
| `get_related_discussions` | `structural.get_pr_review_context()` via `to_thread()` → extract threads → normalize to `list[DiscussionRecord]` |
| `get_ownership` | `structural.get_file_owners()` via `to_thread()` → normalize to `list[OwnershipRecord]` |
| `get_capabilities` | Return `CapabilitySet` based on `episodic.enabled` and structural availability |

**Episodic-disabled handling:**

When `episodic.enabled` is `False`:
- `search_context` returns `[]`
- `CapabilitySet.semantic_search` is `False`
- All structural methods work normally

**Normalization:**

Every method converts raw `dict` output from the lower-level ports into the typed domain models. No raw Cypher labels, graph-specific keys, or Graphiti objects should leak through.

### 6.2 Add `adapters/outbound/intelligence/mock.py`

Class: `MockIntelligenceProvider(IntelligenceProvider)`

Returns canned data from a fixture dict.

Supports:

- configurable capabilities,
- per-project fixture data,
- injectable failures for error-path testing.

### 6.3 Tests for the provider adapter

Test cases:

- Each provider method normalizes output correctly
- Episodic-disabled path reports correct capabilities
- `to_thread()` wrapping does not block the event loop
- Raw graph-specific keys do not appear in output

**Deliverable:** the current context-graph stack satisfies the new provider contract with proper normalization.

---

## 7. Phase 3 — Container Wiring

**Goal:** wire the new layer into `context-engine` bootstrap.

### 7.1 Update `bootstrap/container.py`

Add to `ContextEngineContainer`:

```python
intelligence_provider: IntelligenceProvider
resolution_service: ContextResolutionService
```

In `build_container()`:

```python
intelligence_provider = HybridGraphIntelligenceProvider(
    episodic=episodic_adapter,
    structural=structural_adapter,
)
resolution_service = ContextResolutionService(
    provider=intelligence_provider,
)
```

This reuses the existing `episodic` and `structural` instances. No duplication.

### 7.2 Keep existing members intact

No removal of:

- `episodic: EpisodicGraphPort`
- `structural: StructuralGraphPort`
- existing query use cases

They remain lower-level building blocks used by:
- the new `HybridGraphIntelligenceProvider`
- existing direct query routes and tools (during transition)

**Deliverable:** the new layer is constructible from the standard container.

---

## 8. Phase 4 — Inbound Exposure

**Goal:** make the new layer consumable through existing context-engine surfaces.

### 8.1 HTTP

Update `adapters/inbound/http/api/v1/context/router.py`:

- Add `ResolveContextRequest` Pydantic model matching `ContextResolutionRequest` fields
- Add `POST /query/resolve-context` endpoint
- Auth: same `require_api_key` + `_require_project_access` pattern as existing routes
- Response: serialize `IntelligenceBundle` as JSON

Do not include rendered prompt text in the HTTP response.

### 8.2 MCP

Update `adapters/inbound/mcp/server.py`:

- Add `context_resolve` tool
- Auth: same `assert_mcp_project_allowed` pattern
- Response: bundle as dict

### 8.3 Response contract

HTTP and MCP both return:

```json
{
  "bundle": { ... },
  "coverage": { ... },
  "errors": [ ... ],
  "meta": { ... }
}
```

Include `meta.schema_version` for forward compatibility.

**Deliverable:** external callers can consume the same intelligence contract through HTTP and MCP.

---

## 9. Phase 5 — Potpie Runtime Adoption

**Goal:** use the new layer from Potpie agents without tightly coupling Potpie to the graph stack.

### 9.1 Initial adoption path

Potpie calls `resolve_context` before agent reasoning.

Where to call it:

- **Option A:** In `AutoRouterAgent.run()` before dispatching to the selected agent. Makes intelligence a platform behavior.
- **Option B:** In `QnAAgent._enriched_context()`, replacing its current direct calls. Simpler migration but remains per-agent.

Recommended: **Option A** for long-term, but start with **Option B** for validation (less risky, easier to compare before/after).

### 9.2 Potpie-side prompt renderer

Create a Potpie-specific renderer (e.g. `app/modules/context_graph/bundle_renderer.py`) that:

- takes an `IntelligenceBundle`,
- produces a prompt-safe text block,
- includes coverage status and missing-evidence instructions for the LLM,
- tells the model to skip redundant tool calls when coverage is complete.

This is Potpie's code, not context-engine domain code.

### 9.3 Transition of existing tools

Phase 1: existing tools remain. `resolve_context` prefetches evidence. Agent has both.

Phase 2: update agent prompts:
- if the prefetched intelligence block is present and coverage is complete, do not re-call the same tools.
- if coverage is partial, allow one targeted drill-down.

Phase 3: monitor tool-call telemetry. If tools are rarely called after prefetch, consider removing them from default agent toolsets.

### 9.4 Feature flag

Gate adoption behind `CONTEXT_INTELLIGENCE_ENABLED=true` (default `false`).

When disabled, agents use the old `_enriched_context` path unchanged.

**Deliverable:** fewer repeated context tool calls for history-heavy questions.

---

## 10. Phase 6 — Evaluation and Observability

**Goal:** prove the layer improves behavior instead of just adding abstraction.

### 10.1 Golden-query evaluation

Create a small evaluation set (~10 queries) covering:

- PR rationale questions (e.g. "Why was retry policy added?")
- review discussion questions (e.g. "What was discussed in PR #694?")
- file history questions (e.g. "Who changed config.py recently?")
- ownership questions (e.g. "Who maintains the payment module?")
- code-navigation questions (e.g. "What does the webhook handler do?" — should NOT force deep graph fetch)

Validate per query:

- signals extracted correctly
- evidence plan matches expected families
- bundle content is non-empty where expected
- coverage status is correct
- rendered summary (if Potpie renderer) is well-formed

### 10.2 Operational metrics

Track:

- `resolve_context` total latency
- per-provider-call latency
- coverage status distribution (complete / partial / empty)
- error rate per provider method
- repeated context-tool calls after prefetch (should decrease)

### 10.3 Failure behavior validation

Test:

- provider timeout degrades to partial coverage (not crash)
- single failing call does not fail the whole resolution
- completely down provider returns empty bundle with errors
- caller (Potpie agent / HTTP) handles partial coverage gracefully

**Deliverable:** measurable confidence that the new layer is helping.

---

## 11. Implementation Order

1. `domain/intelligence_models.py`
2. `domain/ports/intelligence_provider.py`
3. `domain/intelligence_signals.py`
4. `domain/intelligence_policy.py`
5. `adapters/outbound/intelligence/mock.py`
6. `application/services/context_resolution.py`
7. `application/use_cases/resolve_context.py`
8. tests for domain + application (using mock provider)
9. `adapters/outbound/intelligence/hybrid_graph.py`
10. tests for hybrid_graph adapter
11. `bootstrap/container.py` extension
12. HTTP route extension
13. MCP tool extension
14. Potpie-side renderer
15. Potpie runtime integration (Option B then Option A)

Models first. Mock second. Tests third. Real provider fourth. Exposure fifth. Potpie last.

---

## 12. What Not To Do

- Do not put policy logic into `structural.py` or `episodic.py`.
- Do not make Potpie agent prompts the primary contract.
- Do not expose raw graph nodes as the public interface.
- Do not skip the mock provider; it is important for keeping the runtime independently testable.
- Do not handle authentication inside the use case; that's the caller's job.
- Do not put prompt rendering into the domain layer.
- Do not remove existing Potpie context tools before validating the new path.
- Do not assume the episodic layer is always available — handle disabled gracefully.
- Do not make the resolution service synchronous — it must be async to enable parallel provider calls.

---

## 13. Risks

### Risk 1: abstraction without behavior improvement

Mitigation: ship golden-query evaluation, compare tool-call counts before and after adoption.

### Risk 2: leaking graph-specific assumptions into domain models

Mitigation: keep domain models evidence-shaped, not node-shaped. Review interfaces before implementation starts.

### Risk 3: Potpie stays coupled to old direct query tools

Mitigation: adopt `resolve_context` in one agent first, update prompts to trust prefetched context, then phase out duplicated enrichment.

### Risk 4: provider contract too broad too early

Mitigation: start with evidence families already supported by the current graph stack. Add workflow-specific capabilities later behind `CapabilitySet`.

### Risk 5: sync-to-async wrapping adds overhead or deadlocks

Mitigation: test `to_thread()` wrapping under concurrent load. Ensure no shared mutable state in the Neo4j/Graphiti adapters that could cause issues under parallel execution.

### Risk 6: prompt rendering drifts from bundle schema

Mitigation: Potpie-side renderer should be tested against the same golden queries as the bundle evaluation. If the bundle adds a field, the renderer should handle it or ignore it gracefully.

### Risk 7: external consumers depend on unstable bundle shape

Mitigation: include `schema_version` in `ResolutionMeta`. Document which fields are stable vs experimental. Use the HTTP route's `/v1/` prefix as a versioning boundary.

---

## 14. Final Deliverable

At the end of this plan, `context-engine` should provide:

- a provider-neutral context intelligence contract,
- a graph-backed first implementation with proper sync-to-async handling,
- one shared resolution use case with parallel execution and fail-open behavior,
- HTTP and MCP access with versioned response schema,
- a clean path for Potpie and external agents to consume historical context without direct dependence on the current graph stack,
- golden-query evaluation proving the layer improves answer quality and reduces tool chaining.
