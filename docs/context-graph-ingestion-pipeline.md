# Project Scope Document: Context Graph Ingestion Pipeline (Revised)

*Updated to reflect feasibility assessment, latency considerations, and spec-agent context-bloat mitigations.*

---

## 1. Overview

Potpie will implement a **Context Graph Ingestion Pipeline** that enriches the existing code knowledge graph (Neo4j) with VCS, issue-tracker, documentation, and (optionally) observability data. The goal is to make spec generation and validation **context-aware** (PR history, ticket AC, policies, incidents) without bloating the spec agent or adding unacceptable latency.

---

## 2. Phases (unchanged intent, implementation notes added)

### Phase 1: The Immediate Context Layer (VCS & Code Review)

**Focus:** Capture the immediate "Why" and "Who" behind existing code.

**Data sources:** GitHub APIs.

**Data to extract:** Pull request descriptions, merge commit messages, linked GitHub Issues, critical code-review comments.

**Implementation notes:**

- Extend or add a **separate ETL job** (e.g. Celery) that uses existing `GithubService` / `GitHubProvider` to pull PR history (e.g. last 12–24 months).
- Map commit hash + file + patch line ranges to existing AST nodes via `(file_path, start_line, end_line)`; create `(PullRequest)`, `(Commit)` nodes and edges (e.g. `(NODE)-[:MODIFIED_IN]->(Commit)`).
- **Deliverable:** A **context service** or **tool** that, given target files/functions (or node_ids), returns a **curated, summarized** list of relevant PRs (e.g. "Function X was heavily modified in PR #405 due to race conditions; consider thread safety"). This is **injected only when in scope** (see §5).

---

### Phase 2: The Intent & Constraint Layer (Issue Trackers)

**Focus:** Map business requirements and acceptance criteria to the codebase.

**Data sources:** Linear, Jira (OAuth / API).

**Data to extract:** Ticket titles, descriptions, acceptance criteria, status, tags (e.g. P0, Security).

**Implementation notes:**

- **Deterministic linking:** Parse branch names (e.g. `feat/PROJ-123`) and PR titles to extract ticket IDs; create hard edges `(PullRequest)-[:REFERENCES]->(Ticket)`.
- **Semantic linking:** Embed ticket descriptions; use vector similarity (e.g. threshold > 0.85) to create `(Ticket)-[:SEMANTICALLY_MATCHES]->(Class)` or directory/file aggregates. Prefer deterministic linking; use semantic as fallback or for Epic → area mapping.
- **Deliverable:** Spec agent can **query** for the Epic/ticket in scope and pull related AC as **structured constraints**, again **curated and scoped** (see §5).

---

### Phase 3: The Governance & Rules Layer (Documentation)

**Focus:** Embed architectural standards and compliance rules as graph constraints.

**Data sources:** Local repo (`/docs/adr/`), Notion, Confluence.

**Data to extract:** ADRs, PRDs, security policies → rules, constraints, context.

**Graph:** New nodes `(Policy)`, `(ArchitecturalDecision)`, `(DesignDoc)`; edges `[:GOVERNS]`, `[:CONSTRAINS]` to directories/files.

**Implementation notes:**

- Lightweight LLM (e.g. GPT-4o-mini / Claude Haiku) with strict JSON-schema extraction; create nodes and link to paths (e.g. policy about billing → `/src/billing/`).
- **Deliverable:** **Spec-validation** step: before finalizing a spec, query `(Policy)-[:GOVERNS]->(target files)` and check spec against **summarized** policy bullets (not full doc dumps).

---

### Phase 4: The Operational Reality Layer (Observability & Chat) — Advanced

**Focus:** Inject runtime health and tacit knowledge into the graph.

**Data sources:** Sentry, Datadog, Slack.

**Data to extract:** Unresolved exception clusters, latency flags, summarized Slack threads (e.g. emoji-triggered summarization → `(SlackDecision)`).

**Deliverable:** Spec agent can **optionally** consider "this file has active production incidents" and suggest fallbacks or caution. **Scoped and capped** (see §5).

---

## 3. Technical Architecture: 4-Step Context ETL Pipeline

- **Extract:** Connectors (API polling / webhooks) for GitHub, Linear, Jira, Notion, Confluence; incremental updates via `last_synced`; Celery + retries/backoff for rate limits.
- **Transform:** LLM summarization/extraction (small, cheap models) to produce JSON summaries (intent, constraints, warnings). Avoid dumping raw 50-comment PR threads into the graph.
- **Embed:** Reuse or align with existing embedding pipeline (e.g. `InferenceService.generate_embedding()`) for semantic edges; store vectors in Neo4j (existing or new vector index).
- **Load:** Batch insert into Neo4j; new node/edge types; idempotent where possible.

**Note:** ETL runs **asynchronously** (e.g. Celery). It does **not** run during spec generation and does not add to spec-generation latency.

---

## 4. Risks & Mitigations (existing + additions)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graph bloat / noise | Latency, LLM noise | **Temporal decay:** `last_updated` on edges; weight recent PRs/tickets; prune or ignore >2 years in traversal. |
| LLM processing costs | High API bills | Use small/fast models for ETL summarization; reserve frontier models for spec generation. |
| API rate limits | Pipeline failure on bulk ingest | Retry queues (Celery/Redis); ingest incrementally (batching). |
| False semantic links | Wrong ticket–code mapping | Prefer **deterministic** linking (IDs, branch names); semantic only as fallback; threshold (e.g. 0.85). |
| **Spec-generation latency** | Slower specs | **Parallel** context queries; **precompute/cache** context when user opens task; **optional** feature; **narrow scope** (files in scope only). |
| **Spec-agent knowledge bloat** | Worse quality, distraction | **Curate context:** scope (files + time + relevance), **summarize** (no raw dumps), **cap** (e.g. 3–5 PRs, N policy bullets), **optional/tiered**; prefer **tool-based retrieval** over dumping into prompt. |

---

## 5. Spec-Agent Context: Avoid Bloat (New Section)

To avoid overwhelming the spec agent with too much knowledge:

1. **Strict scope**  
   Only attach context for **files/nodes in scope** for the current spec (e.g. from research agent's `file_impact` or user selection). Apply **time filters** (e.g. PRs/incidents last N months; open/recent tickets).

2. **Summarize, do not paste raw**  
   Pre-process PR threads, tickets, policies through a small model or rules into **short bullets** (e.g. 1–2 lines per PR/ticket/policy). Inject only these summaries into the spec flow.

3. **Structured, bounded block**  
   Add a single section, e.g. "Relevant context (use only if it applies to this spec):" with a **capped** list (e.g. max 3–5 PRs, 5–10 policy bullets). Keep it distinct from "Requirements" and "Research findings."

4. **Optional / tiered**  
   **Default:** No PR/ticket/policy/incident context (current behavior). **Opt-in "deep context":** When enabled, add the curated, scoped, summarized block. This keeps the default spec agent lean.

5. **Retrieval over injection**  
   Where possible, expose context via **tools** (e.g. `get_relevant_prs_for_files`, `get_governing_policies`) so the agent **pulls** context only when needed, instead of receiving a large dump in the system prompt.

6. **Runtime performance**  
   At spec time: run context queries **in parallel**; prefer **cached/precomputed** context (e.g. when user opens the task) so the spec path does a small read instead of multiple round-trips.

---

## 6. Backend Feasibility (Summary)

- **Potpie backend** already has: Neo4j (NODE with FILE/CLASS/FUNCTION/INTERFACE, `file_path`, `start_line`, `end_line`, vector index), repo parsing (RepoMap → CodeGraphService), GitHub (list_pull_requests, get_pull_request, commits, files), Jira/Linear/Confluence clients (tool-level), embeddings (InferenceService), Celery+Redis.
- **To build:** ETL jobs for each phase; new node/edge types in Neo4j; commit→node and ticket→code mapping logic; **context service** (or tools) that return **curated, summarized** context; hooks in spec flow (or in potpie-workflows) to consume this context under the rules in §5.

---

## 7. Recommended Target Architecture Plan

This section captures the recommended implementation strategy to balance capability, cost, and runtime speed.

### 7.1 Architectural Direction

- Use a **hybrid architecture** instead of graph-only or file-only:
  - **Graph plane (Neo4j):** Relationship intelligence and constrained traversal.
  - **Retrieval plane (Context Packs):** Fast, denormalized, pre-curated context retrieval for spec generation.
- Keep GraphQL/REST as an **API layer**, not as a performance mechanism by itself.
- Run heavy extraction/summarization/embedding asynchronously via Celery; keep spec-time calls lightweight.

### 7.2 Two-Plane Data Model

#### A) Graph Plane (Neo4j)

Purpose: Source-of-truth for deterministic and semantic links.

- Example deterministic edges:
  - `(NODE)-[:MODIFIED_IN]->(Commit)-[:PART_OF]->(PullRequest)`
  - `(PullRequest)-[:REFERENCES]->(Ticket)`
  - `(Policy)-[:GOVERNS]->(NODE or Directory)`
  - `(ProductionIncident)-[:AFFECTS]->(NODE or File)`
- Example semantic edges:
  - `(Ticket)-[:SEMANTICALLY_MATCHES {score, model, created_at}]->(NODE)`
- Query-time policy:
  - Prefer deterministic edges first.
  - Use semantic edges as fallback with confidence threshold.

#### B) Retrieval Plane (Context Packs)

Purpose: Spec-time fast retrieval with bounded prompt payloads.

- Materialize per-scope context artifacts (Markdown/JSON), keyed by:
  - `project_id`
  - `scope_hash` (derived from file set / target nodes)
  - `version_ref` (branch or commit)
- Include:
  - curated warnings
  - acceptance-criteria constraints
  - governing policy bullets
  - active incident cautions
  - source provenance references
- Store metadata for ranking and freshness:
  - `updated_at`, `expires_at`, `source_count`, `quality_score`, `token_estimate`

### 7.3 Spec-Time Retrieval Flow (Fast Path)

1. Resolve scope (`project_id`, target files or node_ids, optional `version_ref`).
2. Compute `scope_hash`.
3. Fetch best valid Context Pack.
4. If cache miss/stale:
   - return minimal deterministic context immediately;
   - asynchronously rebuild Context Pack.
5. Inject only bounded top-k context into spec prompt (see guardrails).

### 7.4 Guardrails and Limits (Mandatory)

- **Top-k caps:** e.g., max 3 PRs, 3 Tickets, 5 Policy constraints, 2 Incidents.
- **Temporal filter:** default 12 months, hard cap 24 months for active retrieval.
- **Traversal depth cap:** runtime graph traversal depth <= 2 (except offline ETL jobs).
- **Prompt budget cap:** enforce token ceiling for context block.
- **Confidence gating:** semantic links below threshold are excluded.
- **Source provenance:** every injected item must include source pointer (PR/Ticket/Doc/Incident id).

### 7.5 API Contract for Spec Context

Expose a single bounded retrieval endpoint/tool for spec generation:

- `get_spec_context(project_id, scope, version_ref=None, mode="lean|deep")`

Response shape (example):

- `warnings[]`
- `constraints[]`
- `policies[]`
- `incidents[]`
- `sources[]`
- `token_budget_used`

`mode` behavior:

- `lean` (default): deterministic + highest-signal items only.
- `deep`: includes additional semantic context under strict caps.

### 7.6 Rollout Plan

#### Stage A (Immediate)

- Implement GitHub PR/commit ingestion with deterministic mapping to file/function nodes.
- Build Context Pack cache with lean-mode retrieval.
- Integrate `get_spec_context` in spec flow.

#### Stage B

- Add Jira/Linear deterministic linking and AC extraction into constraints.
- Add policy ingestion + pre-validation constraints.

#### Stage C

- Add semantic matching for ticket/doc to code with confidence scoring and audits.

#### Stage D (Advanced)

- Add Sentry/Datadog/Slack-derived operational context (incidents and decisions).

### 7.7 Success Metrics

- Spec generation latency p95 remains within target SLO.
- Context block token usage stays under configured budget.
- Context precision improves (fewer irrelevant warnings).
- Acceptance of generated constraints/warnings increases over baseline.

### 7.8 Anti-Patterns to Avoid

- Do not dump raw PR threads, full tickets, or full policy docs into prompts.
- Do not rely on unbounded graph traversals at request time.
- Do not treat GraphQL adoption as a direct performance optimization.
- Do not replace graph semantics with markdown-only storage without provenance links.

---

*End of revised scope document.*
