# Context Intelligence Layer — Product Spec

Branch: `feat/context-engine`

Related docs:
- [GitHub Intelligence Layer — Architecture](./github-intelligence-layer-architecture.md)
- [GitHub Intelligence Layer — Implementation Plan](./github-intelligence-implementation-plan.md)
- [Context Intelligence Layer — Implementation Plan](./context-intelligence-layer-implementation-plan.md)

---

## 1. Problem

Today, Potpie has useful context data, but agent consumption is still too tool-driven and too Potpie-specific.

Current behavior:

- QnA enriches `ChatContext.additional_context` with a few context-engine calls.
- Other agents mostly do not consume that historical context.
- The model still often needs multiple tool calls to assemble one conclusion.
- The runtime is coupled to the current context-graph stack rather than a stable intelligence contract.

This creates four product problems:

1. historical context is not a platform capability,
2. multiple agents behave differently for the same evidence need,
3. external agents cannot consume the layer cleanly,
4. the runtime is too aware of storage shape instead of evidence shape.

---

## 2. Goal

Build a provider-agnostic context intelligence layer that:

1. resolves historical/contextual evidence before reasoning,
2. reduces repeated tool chaining for graph-heavy questions,
3. remains modular enough to expose outside Potpie,
4. maps cleanly onto the existing `context-engine` architecture,
5. allows the current graph stack to be the first provider implementation, not the permanent architecture.

---

## 3. Core Principles

### 3.1 Provider-agnostic runtime

The runtime should depend on an intelligence interface, not on Neo4j, Graphiti, Cypher, or specific node labels.

### 3.2 Evidence-first contract

Agents should consume normalized evidence such as:

- artifact context,
- change history,
- decisions,
- discussions,
- ownership,
- workflow context,

not raw graph nodes.

### 3.3 Context-engine native structure

The design should follow the existing `context-engine` layering:

- `domain`
- `application`
- `adapters`
- `bootstrap`

### 3.4 Potpie is one consumer

The intelligence layer should be usable by:

- Potpie agents,
- external agents,
- HTTP clients,
- MCP clients,
- future SDK consumers.

### 3.5 The current graph stack is an adapter

Neo4j + Graphiti + Postgres-backed ingestion is the first implementation of the intelligence provider, not the architecture itself.

### 3.6 Scope boundary

The intelligence layer resolves **historical / contextual evidence** only.

It does **not** own:

- code-node lookups (fetching source code from node IDs),
- file-structure enumeration,
- live code search / embedding queries.

Those remain in the caller's domain (Potpie `ToolService`, code-graph tools, etc.).

The caller is responsible for combining code-level context with the intelligence bundle if both are needed.

---

## 4. Architectural Decision

Introduce a **Context Intelligence Layer** inside `context-engine` with three logical levels:

1. **Provider-neutral contracts and policy**
2. **Resolution/orchestration runtime**
3. **Backend-specific provider adapters**

This separates:

- how evidence is requested,
- how evidence is resolved,
- where evidence is stored.

---

## 5. High-Level Runtime Flow

```text
Caller (Potpie agent / HTTP / MCP / external client)
   ↓
resolve_context use case
   ↓
ContextResolutionService
   ├─ extract query signals
   ├─ build evidence plan
   ├─ call provider(s) in parallel (async over sync via to_thread where needed)
   ├─ assemble IntelligenceBundle
   ├─ compute coverage
   └─ produce output (structured bundle + optional rendered text)
   ↓
caller receives:
   - structured bundle (always)
   - rendered summary (if requested)
   - coverage metadata (always)
   - errors (if any)
```

This is intentionally not graph-shaped. A graph-backed provider can power it, but the runtime does not depend on graph semantics.

---

## 6. Context-Engine Mapping

The implementation should follow the current `context-engine/src` structure.

### 6.1 `domain/`

Provider-neutral contracts, models, and pure decision logic live here.

Recommended files:

```text
context-engine/src/domain/
├── intelligence_models.py
├── intelligence_signals.py
├── intelligence_policy.py
└── ports/
    └── intelligence_provider.py
```

#### `domain/ports/intelligence_provider.py`

Defines the stable contract for any backing intelligence system.

The provider exposes two kinds of operations:

**Semantic search** (query-driven, may use embeddings, variable latency):

- `search_context(...)`

**Deterministic lookups** (scope-driven, exact, fast):

- `get_artifact_context(...)`
- `get_change_history(...)`
- `get_decision_context(...)`
- `get_related_discussions(...)`

**Introspection:**

- `get_capabilities()`

This port should not mention:

- Neo4j
- Graphiti
- Cypher
- `Entity:PullRequest`
- graph node labels

Important: the port methods should be **async**. When the underlying implementation is synchronous (as with the current Neo4j driver), the adapter wraps calls with `asyncio.to_thread()`.

#### `domain/intelligence_models.py`

Defines normalized request/response models.

**Request models:**

```python
@dataclass
class ContextResolutionRequest:
    project_id: str
    query: str
    consumer_hint: str | None = None    # e.g. "codebase_qna_agent", "debugging_agent", "external"
    artifact_ref: ArtifactRef | None = None  # e.g. PR number, issue number
    scope: ContextScope | None = None   # e.g. file path, function name
    timeout_ms: int = 4000

@dataclass
class ArtifactRef:
    kind: str       # "pr", "issue", "commit", "task"
    identifier: str # "694", "LINEAR-123", etc.

@dataclass
class ContextScope:
    file_path: str | None = None
    function_name: str | None = None
    symbol: str | None = None
```

**Response models:**

```python
@dataclass
class IntelligenceBundle:
    request: ContextResolutionRequest
    artifacts: list[ArtifactContext]
    changes: list[ChangeRecord]
    decisions: list[DecisionRecord]
    discussions: list[DiscussionRecord]
    ownership: list[OwnershipRecord]
    coverage: CoverageReport
    errors: list[ResolutionError]
    meta: ResolutionMeta

@dataclass
class CoverageReport:
    status: str                  # "complete", "partial", "empty"
    available: list[str]         # evidence families that returned data
    missing: list[str]           # evidence families that were expected but empty
    missing_reasons: dict[str, str]  # why each is missing: "not_fetched", "empty_result", "timeout", "error", "not_supported"

@dataclass
class ResolutionError:
    source: str      # e.g. "search_context", "get_change_history"
    error: str       # human-readable message
    recoverable: bool

@dataclass
class ResolutionMeta:
    provider: str
    total_latency_ms: int
    per_call_latency_ms: dict[str, int]
    capabilities_used: list[str]

@dataclass
class CapabilitySet:
    semantic_search: bool = False
    artifact_context: bool = False
    change_history: bool = False
    decision_context: bool = False
    discussion_context: bool = False
    ownership_context: bool = False
    workflow_context: bool = False
```

**Evidence records:**

```python
@dataclass
class ArtifactContext:
    kind: str                    # "pr", "issue"
    identifier: str
    title: str | None = None
    summary: str | None = None
    author: str | None = None
    created_at: str | None = None
    url: str | None = None
    extra: dict[str, Any] | None = None

@dataclass
class ChangeRecord:
    file_path: str | None = None
    function_name: str | None = None
    artifact_ref: str | None = None  # e.g. "PR #694"
    summary: str | None = None
    date: str | None = None

@dataclass
class DecisionRecord:
    decision: str
    rationale: str | None = None
    source_ref: str | None = None    # e.g. "PR #694 thread 1"
    file_path: str | None = None

@dataclass
class DiscussionRecord:
    source_ref: str | None = None
    file_path: str | None = None
    line: int | None = None
    participants: list[str] | None = None
    summary: str | None = None
    full_text: str | None = None

@dataclass
class OwnershipRecord:
    file_path: str
    owner: str
    confidence_signal: str | None = None  # e.g. "3 PRs in last 90 days"
```

#### `domain/intelligence_signals.py`

Pure query-signal extraction:

- PR number detection
- file path detection
- symbol detection
- history / rationale cues
- ownership cues

#### `domain/intelligence_policy.py`

Pure retrieval planning logic:

- what evidence families are relevant,
- what is mandatory,
- per-consumer defaults,
- timeout budget,
- capability-aware planning (skip what the provider doesn't support).

---

### 6.2 `application/`

Orchestration belongs here.

Recommended files:

```text
context-engine/src/application/
├── services/
│   └── context_resolution.py
└── use_cases/
    └── resolve_context.py
```

#### `application/services/context_resolution.py`

Main runtime service.

Responsibilities:

- run signal extraction,
- apply policy,
- consult provider capabilities,
- call provider methods in parallel,
- assemble the final bundle,
- compute coverage,
- collect errors.

Important implementation detail: the provider's underlying calls may be synchronous (e.g. Neo4j driver). The resolution service should use `asyncio.gather()` with provider methods that internally handle `to_thread()` wrapping.

#### `application/use_cases/resolve_context.py`

Thin application entry point.

This does **not** handle authentication. Authentication is the caller's responsibility:

- HTTP inbound adapter checks API key and project access before calling.
- MCP inbound adapter checks allowed projects before calling.
- Potpie agent runtime checks user/project access before calling.

This matches the existing pattern where `query_context.py` use cases don't do auth; the router/tool layer does.

---

### 6.3 `adapters/outbound/`

Backend-specific provider implementations live here.

Recommended files:

```text
context-engine/src/adapters/outbound/
└── intelligence/
    ├── __init__.py
    ├── hybrid_graph.py
    └── mock.py
```

#### `adapters/outbound/intelligence/hybrid_graph.py`

First concrete provider implementation.

It internally composes:

- `EpisodicGraphPort` (for semantic search)
- `StructuralGraphPort` (for deterministic lookups)

**Sync wrapping:** both ports are currently synchronous. This adapter wraps each call in `asyncio.to_thread()` so the resolution service can call them in parallel.

**Episodic-disabled degradation:** Graphiti can be disabled (`enabled=false`). When disabled:

- `search_context` returns empty results,
- `CapabilitySet.semantic_search` is reported as `false`,
- all structural-only methods continue to work normally.

The policy layer should consult capabilities and skip planning semantic search when it's unavailable.

#### `adapters/outbound/intelligence/mock.py`

Simple deterministic provider for:

- tests,
- local development,
- provider conformance checks.

Returns canned data for known project/query combinations.

---

### 6.4 `adapters/inbound/`

Expose the context intelligence layer through existing transports.

#### HTTP

Extend:

- `context-engine/src/adapters/inbound/http/api/v1/context/router.py`

Add a route such as:

- `POST /query/resolve-context`

Authentication: same pattern as existing routes (API key + project access check before calling use case).

Response: structured `IntelligenceBundle` serialized as JSON. No rendered prompt text in HTTP responses — that's a consumer concern.

#### MCP

Extend:

- `context-engine/src/adapters/inbound/mcp/server.py`

Add a tool such as:

- `context_resolve`

Response: structured bundle as JSON dict.

Both should call the same `resolve_context` use case.

---

### 6.5 `bootstrap/`

Wire the provider and resolver in the container.

Files:

- `context-engine/src/bootstrap/container.py`

Add to `ContextEngineContainer`:

- `intelligence_provider: IntelligenceProvider`
- `resolution_service: ContextResolutionService`

The `HybridGraphIntelligenceProvider` is constructed from the existing `episodic` and `structural` members. No duplication.

---

## 7. Rendering Strategy

Rendering is **not** in the domain layer. The domain produces structured `IntelligenceBundle` only.

Rendering is a **consumer concern**:

| Consumer | Rendering |
|----------|-----------|
| Potpie agent runtime | Render bundle → prompt-safe text block, inject into `additional_context` |
| HTTP API | Return bundle as JSON — no rendering |
| MCP tool | Return bundle as JSON dict — no rendering |
| Future SDK | Consumer decides how to present |

A shared rendering utility can live in `application/services/` or in Potpie's own agent integration code. But the domain contract is: return structured data, not formatted text.

This avoids coupling the domain to any specific LLM prompt format.

---

## 8. Intelligence Provider Contract

```python
class IntelligenceProvider(Protocol):
    async def search_context(
        self, project_id: str, query: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        ...

    async def get_artifact_context(
        self, project_id: str, artifact: ArtifactRef
    ) -> ArtifactContext | None:
        ...

    async def get_change_history(
        self, project_id: str, scope: ContextScope, limit: int = 10
    ) -> list[ChangeRecord]:
        ...

    async def get_decision_context(
        self, project_id: str, scope: ContextScope, limit: int = 20
    ) -> list[DecisionRecord]:
        ...

    async def get_related_discussions(
        self, project_id: str, scope: ContextScope, limit: int = 10
    ) -> list[DiscussionRecord]:
        ...

    async def get_ownership(
        self, project_id: str, scope: ContextScope, limit: int = 5
    ) -> list[OwnershipRecord]:
        ...

    def get_capabilities(self) -> CapabilitySet:
        ...
```

Important:

- All data methods are `async`.
- Adapters with sync backends use `asyncio.to_thread()` internally.
- `get_capabilities()` is sync because it's a static property of the provider, not an I/O call.

---

## 9. Capability Model

Different providers will support different evidence families.

Example for current hybrid graph:

```json
{
  "semantic_search": true,
  "artifact_context": true,
  "change_history": true,
  "decision_context": true,
  "discussion_context": true,
  "ownership_context": true,
  "workflow_context": false
}
```

Example when Graphiti is disabled:

```json
{
  "semantic_search": false,
  "artifact_context": true,
  "change_history": true,
  "decision_context": true,
  "discussion_context": true,
  "ownership_context": true,
  "workflow_context": false
}
```

The policy consults capabilities before planning. If a capability is `false`, the evidence plan skips it and coverage reports it as `"not_supported"` rather than `"empty_result"`.

---

## 10. Error Model

Errors during resolution should not crash the request.

Each provider call that fails is recorded as a `ResolutionError` in the bundle:

```json
{
  "source": "get_change_history",
  "error": "Neo4j connection timed out after 3000ms",
  "recoverable": true
}
```

Coverage reflects the impact:

```json
{
  "status": "partial",
  "available": ["decisions", "discussions"],
  "missing": ["change_history"],
  "missing_reasons": {
    "change_history": "timeout"
  }
}
```

The caller decides how to handle:

- Potpie agents: mention missing evidence in the prompt, proceed with what's available.
- HTTP callers: inspect `coverage` and `errors` fields, decide whether to retry.

---

## 11. What Happens to Existing Potpie Context Tools

Today, Potpie has individual tools that each create their own `Neo4jStructuralAdapter`:

- `get_change_history_tool.py`
- `get_decisions_tool.py`
- `get_pr_review_context_tool.py`
- `get_pr_diff_tool.py`

These will **not** be removed immediately.

Transition path:

1. **Phase 1:** `resolve_context` prefetches evidence into the bundle. Existing tools remain as drill-down fallbacks.
2. **Phase 2:** Agent prompts are updated to prefer bundle evidence and only call tools when coverage is partial.
3. **Phase 3:** If tool-call telemetry shows the tools are rarely needed after prefetch, consider removing them from default agent toolsets.

This avoids a breaking change and allows gradual validation.

---

## 12. Potpie Consumption Model

Potpie should become one consumer of the context intelligence layer, not the owner of its only runtime path.

Recommended path:

```text
Potpie agent runtime
   ↓
call resolve_context(project_id, query, consumer_hint=agent_id, ...)
   ↓
receive IntelligenceBundle
   ↓
render bundle to prompt text (Potpie-side renderer)
   ↓
attach rendered text to ChatContext.additional_context
   ↓
optionally attach structured bundle to ChatContext.context_bundle
   ↓
agent answers with fewer context tool calls
```

Important: the Potpie renderer is Potpie's code, not context-engine's domain code. The renderer knows about prompt formatting, coverage instructions for the LLM, and how to tell the model to skip redundant tool calls.

---

## 13. Exposure Outside Potpie

To make the layer usable outside Potpie, expose it through the same application contract.

Recommended order:

### 13.1 Internal Python usage

First consumer: Potpie runtime calls the use case directly.

### 13.2 HTTP API

Expose: `POST /query/resolve-context`

Response is always JSON, never prompt text.

### 13.3 MCP

Expose: `context_resolve` tool.

### 13.4 Versioning

The bundle schema should include a `schema_version` field in `ResolutionMeta`. Start at `"1"`. Increment when breaking changes are made. HTTP route should be under `/v1/` (already the case).

---

## 14. What Stays Specific to the Current Graph Stack

The following remain implementation details of the first provider:

- Graphiti semantic search
- Neo4j structural traversal
- `PullRequest`, `Decision`, `Issue`, `Feature`, `Developer` labels
- bridge edges such as `MODIFIED_IN` and `TOUCHED_BY`
- raw event payload usage for recovery/debugging
- sync-to-async wrapping via `to_thread()`

These are valuable, but they belong behind `hybrid_graph.py`, not in the runtime contract.

---

## 15. What This Design Solves

### 15.1 Fewer repeated tool calls

For historical questions, the runtime can resolve evidence once and hand the agent a ready bundle instead of forcing multiple lookups just to reach a conclusion.

### 15.2 Cross-agent consistency

QnA, debug, blast-radius, and external agents can all consume the same intelligence contract.

### 15.3 Backend flexibility

Future changes in storage do not force a rewrite of prompts, agent runtime, HTTP API, MCP tools, or external clients.

### 15.4 Cleaner architecture

The design fits naturally into the existing `context-engine` structure instead of creating a parallel subsystem.

---

## 16. Non-Goals

This spec does not require:

- replacing the current graph stack,
- removing existing direct query endpoints like `get_pr_diff`,
- removing existing Potpie context tools immediately,
- building a universal workflow schema on day one,
- extracting `context-engine` into a separate repository immediately,
- handling code-node lookups or file-structure enumeration (those are out of scope).

This is an internal architecture improvement with a clean externalization path.

---

## 17. Open Questions

1. Should Potpie call `resolve_context` directly (in-process) or route through HTTP even internally? Direct is simpler and faster; HTTP is more decoupled but adds latency and deployment complexity.
2. Should the bundle be cached per conversation turn, or re-resolved every message? Caching avoids redundant fetches for follow-ups but risks stale data.
3. Should the Potpie-side prompt renderer live in `app/modules/intelligence/` or in `app/modules/context_graph/`?
4. Should `resolve_context` support merging results from multiple providers, or is single-provider sufficient for v1?

---

## 18. Recommendation

Proceed with:

- a provider-neutral intelligence contract in `domain`,
- fully specified request/response models,
- a resolution service in `application`,
- a graph-backed first implementation in `adapters/outbound/intelligence`,
- HTTP and MCP exposure through existing `adapters/inbound`,
- rendering as a consumer concern, not a domain concern,
- Potpie as the first consumer, not the defining boundary,
- gradual deprecation of direct context tools after validation.
