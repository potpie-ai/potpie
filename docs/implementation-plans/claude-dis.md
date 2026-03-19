# Designing a context graph layer for AI coding agents

**AI coding agents fail not because they lack intelligence, but because they lack memory.** Current agents treat every session as a blank slate — they understand code syntax but not *why* it was built, *who decided*, or *what broke last time*. This report designs a modular "project memory" system: a context graph layer that captures the scattered, temporal, cross-platform knowledge that experienced engineers carry in their heads. The architecture extends Potpie.ai's existing Neo4j + Python knowledge graph into a full engineering context graph with ingestion from GitHub, Slack, Linear, CI/CD, and more — served to any coding agent via MCP tools and hybrid retrieval APIs. **The core thesis: code is not text to be searched; it is a living system to be understood through its history, decisions, and relationships.**

---

## 1. The landscape of engineering context tools reveals a critical gap

No existing product unifies the three layers coding agents need: **code structure**, **operational context**, and **organizational knowledge**. The market fragments into four categories, each solving a piece of the puzzle while missing the whole.

### Developer intelligence platforms solve cataloging, not comprehension

**Backstage** (Spotify, CNCF) and **Cortex** provide service catalogs with YAML-based metadata, but their rigid entity models and manual maintenance create staleness. Backstage's own VP admits external adoption often stalls below 10%. **OpsLevel** automates catalog discovery but offers no semantic understanding of code. These platforms answer "what services exist?" but not "why was this service built this way?"

**Sourcegraph Cody** comes closest to code comprehension through its **SCIP (Source Code Intelligence Protocol)** — a code graph capturing symbols, definitions, references, and call relationships across 100,000+ repositories. Its hybrid retrieval (BM25 + semantic + graph) represents state-of-the-art code context. But Cody is code-only: it cannot integrate tickets, design decisions, Slack discussions, or deployment context. Sourcegraph's recent pivot to "Amp" for agentic workflows acknowledges this limitation.

**Swimm** excels at auto-generating hallucination-free documentation via static analysis + LLM grounding — a pattern architecturally relevant to context graph construction — but its internal model isn't exposed as a queryable API.

### Agent memory platforms are general-purpose, not engineering-aware

**Mem0** (41K+ GitHub stars, $24M raised, selected by AWS as exclusive memory provider) implements a sophisticated hybrid datastore combining graph, vector, and key-value stores with memory lifecycle management (extraction, consolidation, decay). It achieves **26% higher accuracy** than OpenAI's memory and **91% lower p95 latency**. But it stores conversational preferences, not code relationships.

**Zep/Graphiti** offers the most architecturally relevant foundation. Graphiti is an open-source temporal knowledge graph engine on Neo4j with **bi-temporal modeling** (event time + ingestion time), prescribed and learned ontologies, and provenance tracking to source episodes. It achieves **94.8% on the DMR benchmark** and sub-200ms retrieval. However, it requires significant custom ontology work for engineering domains — there are no built-in entity types for functions, services, or architectural decisions.

**Cognee** (backed by OpenAI and FAIR founders, $7.5M seed) provides the most flexible pipeline architecture: `cognify` → `memify` → `search`, with 14 retrieval modes including graph completion. Its self-improving memory via the `memify` pipeline (pruning stale nodes, strengthening frequent connections, reweighting edges) is a pattern directly applicable to engineering context. **MemGPT/Letta** introduces the elegant OS-inspired metaphor of agents managing their own memory via paging between "RAM" (context window) and "disk" (archival store), but its underlying storage is unstructured text blocks.

### Potpie.ai is closest but needs three extensions

Potpie already converts entire codebases into Neo4j knowledge graphs capturing files, functions, classes, imports, call relationships, and AI-generated docstrings. Its multi-agent supervisor/subagent architecture with custom history processing (maintaining ~40-45% context utilization) produced the **#1 score on SWE-Bench Lite at 63%**. Tools like `AskKnowledgeGraphQueries`, `GetProbableNodeID`, and `GetNodeNeighbours` bridge the gap between natural language problem descriptions and actual code locations. One enterprise customer with a **40-million-line codebase cut root-cause analysis from ~1 week to ~30 minutes**.

But Potpie's graph is primarily structural (AST-level). The three critical extensions needed are: **(1) temporal awareness** — tracking how code and decisions evolve over time, **(2) organizational context** — decisions, discussions, and tribal knowledge from beyond the codebase, and **(3) cross-platform entity resolution** — linking the same feature across a GitHub PR, Linear ticket, and Slack thread.

### Products that signal where the market is heading

**Archgate** makes ADRs executable and enforceable in CI, queryable by AI agents via MCP server — "AI agents read your ADRs before writing code." **Dexicon** captures knowledge from AI coding sessions (Claude Code, Cursor, Codex), extracting atomic knowledge items and building a searchable knowledge graph. LinkedIn's **CAPT framework** (Contextual Agent Playbooks and Tools) encodes organization-wide instructions via MCP, producing a **20% increase in AI coding adoption** and **~70% reduction in issue triage time**. These emerging tools validate the thesis that coding agents need structured project memory.

---

## 2. What coding agents actually need — and where they fail

### The context gap taxonomy

Research across SWE-bench failure analyses, production agent deployments, and academic papers reveals a consistent hierarchy of context needs. Code structure (AST, imports, call graphs) is now well-served by multiple tools. But five critical context types remain almost entirely unserved:

- **Decision provenance** — *why* something was built a certain way, what alternatives were considered, who decided. No coding agent natively consumes ADRs. As one practitioner noted: "Agents don't have tenure; every session starts from scratch."
- **Discussion context** — Slack threads, PR review comments, ticket discussions where critical reasoning lives. A typical enterprise codebase contains "performance decisions from production issues, architectural patterns from infrastructure migrations, and business logic distributed across databases, middleware, and legacy methods" — none of it in the code itself.
- **Feature↔code mapping** — the relationship between business features and code modules. Developers think in features; code is organized by packages. This mapping is universally tribal.
- **Temporal evolution** — not just *what* changed (git log provides this) but the full narrative of *how understanding evolved*. SWE-bench Pro found that "commit messages are often unstructured, incomplete, or entirely missing."
- **Change impact analysis** — what breaks if you modify file X. Most agents discover dependencies reactively during execution rather than consulting a pre-computed impact graph.

### SWE-bench failures reveal context as the bottleneck

**SWE-bench Pro** (Scale AI) drops agent performance to **≤23.3%** on enterprise codebases, versus >70% on the original Verified set. Larger models "fail on semantic or algorithmic correctness in large, multi-file edits" while smaller models "fail on tool use and syntax errors." A benchmark mutation study found SWE-bench Verified "systematically overestimates agent capabilities by up to ~20%" because real developer communication is far less structured than formal GitHub issue descriptions.

**CodeRabbit's study of 470 GitHub PRs** found AI-generated code creates **1.7x as many bugs** as humans, **1.5-2x more security vulnerabilities**, and **~8x more excessive I/O operations**. These aren't model capability failures — they're context failures. Agents that don't understand deployment topology write code that works locally but fails at scale.

The most telling evidence comes from Surge AI's trajectory analysis: when Gemini encountered a truncated file read, it "spiraled into 693 lines of hallucinated code — filling in the blanks with its best guesses." GPT-5, by contrast, "explicitly went back to re-check rather than guessing." The difference is whether agents have sufficient context to know when they're operating in the dark.

### How current agents handle context — and where each breaks

| Agent | Context Mechanism | Key Strength | Critical Gap |
|-------|------------------|-------------|-------------|
| **Cursor** | Tree-sitter AST → embeddings → Turbopuffer vector DB, Merkle tree incremental reindexing | Two-stage retrieval (vector → AI reranking), @-mention system | No *why* context — git history shows diffs, not rationale |
| **Claude Code** | Agentic search (grep, find, read), CLAUDE.md hierarchy | No pre-built index needed; subagents get isolated context | Each session from scratch; relies on developer-maintained CLAUDE.md |
| **Aider** | Tree-sitter AST → NetworkX graph → PageRank ranking | Smart token budgeting via personalized PageRank | Purely structural — weaker models "get confused by the repo map" |
| **OpenHands** | Event-stream + LLM summarizing condenser + microagents | ~2x cost reduction via context condensation | No persistent cross-session memory |
| **SWE-agent** | Custom ACI with windowed file viewing, ReAct loop | Designed to prevent context flooding | Only filesystem interface — no access to discussions, tickets, decisions |
| **Cody** | SCIP code graph + BM25 + semantic search + embeddings | Multi-repo awareness (10+ repos simultaneously) | Code-structural only; no decision or discussion integration |

The "Code Digital Twin" paper (arXiv:2503.07967) captures the fundamental barrier: **"The required context is scattered across artifacts and entangled across time, beyond the capacity of LLMs to reliably capture, prioritize, and fuse."** Even as context windows grow to 1M+ tokens, the "Lost in the Middle" problem means models attend strongly to information at the beginning and end but lose information in the middle — with **over 30% performance degradation** at high context utilization.

---

## 3. A graph schema that models software as a living system

The schema must represent not just code structure (Potpie's current strength) but the full sociotechnical system: who built it, why, what was discussed, what changed, and what depends on what. The design draws on W3C PROV-O, Kruchten's architectural decision ontology, and Graphiti's temporal model.

### Twenty node types across five domains

**Code entities** (extending Potpie's existing schema):
```
(:File {path, language, loc, last_modified, hash, embedding})
(:Function {name, signature, docstring, complexity, start_line, end_line, embedding})
(:Class {name, bases, docstring, embedding})
(:Module {name, path, package})
(:Service {name, type, description, base_url})
(:API {endpoint, method, request_schema, response_schema})
```

**Version control entities** (new):
```
(:Commit {sha, message, author, timestamp, files_changed, insertions, deletions})
(:PullRequest {number, title, description, state, created_at, merged_at, branch})
(:CodeReview {id, reviewer, verdict, comments_count, submitted_at})
```

**Issue and feature entities** (new):
```
(:Issue {id, title, description, status, priority, type, source, created_at, resolved_at})
(:Feature {id, name, description, status, target_release})
(:Epic {id, title, description, business_value})
```

**Decision and discussion entities** (new):
```
(:Decision {id, title, status, context, rationale, consequences, decided_at})
(:Alternative {id, description, pros, cons, rejected_reason})
(:Discussion {id, source, title, timestamp, summary, embedding})
(:Comment {id, body, author, timestamp, thread_id, embedding})
```

**People and teams** (new):
```
(:Person {id, name, email, github_handle, role, team})
(:Team {name, description, domain})
```

### Thirty relationship types encode the connective tissue

The relationships fall into six categories that agents query constantly:

**Attribution chains**: `AUTHORED`, `CHANGED`, `REVIEWED`, `APPROVED`, `DECIDED_BY`, `MEMBER_OF` — answering "who owns this?" and "who should I ask?"

**Code-to-discussion linkage**: `DISCUSSED_IN`, `HAS_COMMENT` — bridging the gap between code and the conversations that shaped it.

**Implementation traceability**: `IMPLEMENTS`, `PART_OF`, `FIXES` — mapping code to features, tickets, and decisions. This enables agents to answer "what feature does this function serve?" and "which PR introduced this bug?"

**Bug causation chains**: `CAUSED_BY`, `FIXES` between commits and issues — enabling automated root-cause analysis by traversing from a failing test to the commit that broke it, to the PR that introduced it, to the ticket that requested the change.

**Decision evolution**: `SUPERSEDES`, `CONSIDERED`, `ENABLES`, `CONSTRAINS`, `CONFLICTS_WITH` — modeled after Kruchten's architectural decision ontology. A Decision node links to Alternatives that were considered, enabling agents to understand not just what was decided but what was rejected and why.

**Change impact**: `AFFECTS`, `IMPACTS`, `DEPENDS_ON`, `CALLS_API` — the pre-computed impact graph that agents currently lack. When an agent asks "what breaks if I modify this file?", the graph can traverse `DEPENDS_ON` edges with `blast_radius_score` weights.

Every relationship carries temporal and confidence metadata:
```
{
  valid_from: datetime,    // when the relationship became true
  valid_to: datetime,      // null = currently valid
  confidence: float,       // 0.0-1.0 for AI-extracted relationships
  source: string,          // "git_log" | "llm_extraction" | "manual"
}
```

### Bi-temporal modeling captures how truth evolves

Static graphs fail for software systems because code changes continuously, decisions get superseded, and information arrives out of order (rebases, retroactive discovery from meeting notes). The schema implements **bi-temporal modeling** with two time dimensions:

| Dimension | Definition | Example Query |
|-----------|-----------|--------------|
| **Valid time** | When a fact was true in the real world | "Was this function part of auth-service in January?" |
| **Transaction time** | When a fact was recorded in the database | "When did we learn about this dependency?" |

The implementation follows Graphiti/Zep's proven pattern: every entity has `created_at` and `expired_at` fields; old facts are **invalidated, not deleted**, when superseded. Every derived fact traces back to source **episodes** (raw data as ingested), providing full provenance. Change events are first-class nodes:

```
(:ChangeEvent {type: "function_modified", timestamp, commit_sha, 
               diff_summary: "Added rate limiting", before_hash, after_hash})
  -[:CHANGED]->(:Function)
```

For efficient current-state queries (the 95% case), a `CURRENT_VERSION` shortcut relationship avoids traversing temporal chains.

### Seven-level hierarchy enables multi-granularity retrieval

```
Level 0: Statement     →  Level 1: Function  →  Level 2: Class
Level 3: File          →  Level 4: Module     →  Level 5: Service
Level 6: Feature/Epic
```

Connected via `PART_OF` edges (child→parent), with pre-computed aggregate properties at each level (`file_count`, `function_count`, `complexity_sum`). Variable-length path queries enable cross-level traversal: "find all functions in auth-service" traverses 1-4 hops. **Shortcut relationships** (`BELONGS_TO_SERVICE`) materialize frequently traversed paths for performance.

### Decision provenance follows W3C PROV-O

The decision provenance chain maps directly onto PROV-O's three core classes: **Entity** (code artifact, ADR), **Activity** (change event, review process), and **Agent** (person, team, CI system). A complete provenance chain for "why does this function exist?" traverses:

```
(:Function {name: "process_order_event"})
  -[:IMPLEMENTS]->(:Decision {title: "Use event sourcing for orders"})
  -[:DISCUSSED_IN]->(:Discussion {source: "slack", channel: "#architecture"})
  -[:DECIDED_BY]->(:Person {name: "Alice"})
  -[:MOTIVATED_BY]->(:Feature {name: "Order audit trail"})
  -[:PART_OF]->(:Epic {name: "Compliance 2026"})
```

Each Decision node links to Alternative nodes with `pros`, `cons`, and `rejected_reason` — preserving the full decision space, not just the outcome.

---

## 4. Ingestion architecture: receive fast, process asynchronously, link everything

### The queue-first pattern

The architecture follows Netflix's Real-Time Distributed Graph pattern: each data source gets its own message queue topic, with thin webhook receivers that validate and immediately enqueue, returning 2xx within seconds. Processing workers consume from queues, transform events into graph operations, perform entity extraction and resolution, and write to Neo4j.

```
┌───────────────────────────────────────────────────────────┐
│  GitHub  ·  Linear  ·  Slack  ·  Notion  ·  CI/CD        │
└────┬──────────┬────────┬────────┬──────────┬──────────────┘
     │ webhooks │        │        │          │ polling
     ▼          ▼        ▼        ▼          ▼
┌───────────────────────────────────────────────────────────┐
│  FastAPI Receivers → Redis Streams (per-source topics)    │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Processing Workers:                                       │
│  Transformer → Entity Extractor → Entity Resolver          │
│  Delta Processor → MERGE ops → Edge Inheritance            │
└─────────────────────────┬─────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Neo4j: bi-temporal nodes + vector indexes + BM25         │
└───────────────────────────────────────────────────────────┘
```

**Key infrastructure choices**: Redis Streams for initial deployment (lighter weight, Potpie already uses Redis), graduating to Kafka when throughput demands it. FastAPI webhook receivers with Pydantic models for validation. neo4j-migrations (FlywayDB-style versioned Cypher scripts) for schema versioning.

### Three-tier entity resolution bridges the naming gap

The same feature appears as Linear ticket `FE-1234`, GitHub branch `fe-1234-auth-flow`, PR title `[FE-1234] Implement OAuth2 flow`, and Slack message "the auth flow for the new login." Linking these requires three tiers:

**Tier 1 — Convention-based (deterministic, high precision)**: Regex extraction of ticket IDs from PR titles, branch names, commit messages. GitHub linked issues, Jira development panel links. This alone resolves **60-70%** of cross-platform references.

**Tier 2 — Structural/temporal heuristics**: Temporal co-occurrence (PR merged within minutes of deployment → `DEPLOYED_VIA`), author overlap (same person creates ticket + PR + Slack message within time window), file-path overlap (PR modifying `auth-service/` links to tickets tagged `auth-service`).

**Tier 3 — Semantic/ML-based (probabilistic, broad coverage)**: Embedding similarity between ticket descriptions, PR descriptions, and Slack messages. LLM-based entity extraction from unstructured text. Graph-based link prediction using shared neighbors.

The critical lesson from Anyshift (building a temporal infrastructure KG at scale): **"The code that creates and maintains relationships is more complex than all the extraction pipelines combined."** Entity resolution is the hard part, not source ingestion.

### Delta processing, not snapshots

Full snapshot ingestion doesn't survive contact with production. The architecture implements delta processing: on each event, extract the entity identifier, check if it exists in the graph, compute the diff, and apply only changes. **Edge inheritance** is critical — when a node is updated, existing relationships must carry over to the new version unless explicitly removed.

Schema evolution uses an additive-only strategy: new labels, properties, and relationship types are added without removing old ones. A dual-write period handles renames. Raw source data is preserved as JSON properties alongside structured fields, enabling schema-on-read for flexibility.

---

## 5. Retrieval: three-layer hybrid search with token-budget awareness

### The retrieval pipeline

Agents need three types of queries, served by a three-layer pipeline:

**Layer 1 — Candidate generation (optimize for recall)**: Four parallel signals — BM25/keyword search for exact identifiers, vector/semantic search for conceptual matches, graph pattern matching for structural queries, and metadata filtering for temporal/authorship constraints. The Sourcegraph Cody team's key finding: **"context retrieval sources should be complementary"** — keyword and semantic search retrieve distinct relevant items.

**Layer 2 — Graph enrichment (expand context)**: From retrieved entry-point nodes, traverse 1-2 hops to include related decisions, discussions, and dependencies. Attach parent/sibling context (file → module → service). Include temporal context (recent changes, trends). This is where the graph outperforms pure vector search — it provides the connective tissue between fragments.

**Layer 3 — Ranking and reranking (optimize for precision)**: Reciprocal Rank Fusion across retrieval signals, recency weighting, structural relevance (closer in graph = higher score), token budget optimization, and diversity enforcement. Neo4j's `HybridCypherRetriever` implements this natively.

### Context hydration serves pre-packaged knowledge

Before an agent starts a task, the system assembles a **context packet** — a structured bundle of everything relevant:

```typescript
interface ContextPacket {
  task: TaskDescription;
  target_files: FileContext[];
  related_decisions: Decision[];      // ADRs affecting these files
  recent_changes: Change[];           // PRs/commits in last N days
  ownership: OwnershipInfo;           // Who owns/reviews this area
  dependencies: DependencyInfo;       // What breaks if you change this
  relevant_discussions: Discussion[]; // Matched by semantic similarity
  token_budget: { total, allocated_per_category, remaining };
}
```

Context loads progressively in four tiers: **Tier 0** (always loaded) — project conventions, file structure, tool schemas; **Tier 1** (task-specific) — target file content, immediate dependencies, recent decisions; **Tier 2** (on-demand) — extended discussions, historical PRs, cross-repo dependencies; **Tier 3** (search-triggered) — broad semantic results, expert identification.

The token budget manager allocates dynamically: 40% for core files, 20% for decisions, 15% for recent changes, 10% for discussions, 10% for dependencies, 5% for summary. Items below relevance thresholds are compressed to summaries or excluded entirely.

### MCP tools give agents clean access patterns

The MCP server exposes six core tools with `token_budget` parameters on every call:

```typescript
// Primary entry point — comprehensive task context
get_context_for_task(task_description, file_paths[], depth, token_budget)

// Structural queries
get_decision_history(entity, entity_type, include_discussions)
get_change_impact(file_path, change_type)  // "what breaks if I modify this?"
get_experts(entity, limit)                  // "who should I ask about this?"
get_dependencies(entity, direction, depth)  // upstream/downstream graph

// Semantic search across all context types  
search_context(query, scope, filters{since, author, service}, token_budget)
```

GraphQL serves as the intermediate API layer between MCP tools and Neo4j, providing self-documenting schemas that LLMs can reason about autonomously, precise data fetching (agents request exactly what they need), and relationship traversal in single queries. Neo4j's native `@neo4j/graphql` maps resolvers directly to Cypher traversals.

**Key API design insight from OpenAI's function calling guide**: Apply the "intern test" — can a new team member correctly use the function given only its description? Tools should be single-purpose, use enums for constrained parameters, and offload complexity from the model.

---

## 6. The six failure modes that will kill this system if ignored

### Context flooding degrades agent performance more than missing context

The "Lost in the Middle" problem (Liu et al., 2023) shows **>30% performance degradation** when relevant information moves from document edges to the middle. But the damage compounds: Chroma's research found that models "grow increasingly unreliable as input length grows," and Du et al. proved this isn't a retrieval issue — **even replacing irrelevant tokens with blank spaces still degrades performance**. Factory.ai treats context as "a scarce, high-value resource." The system must aggressively filter, not just retrieve.

**Mitigation**: Token budgets on every API call. Progressive context loading. Relevance thresholds with hard cutoffs. Context packet summaries at the top (exploiting primacy bias). Asymmetric context loading (more pre-change context, less post-change, following Qodo's pattern).

### Staleness compounds silently across the graph

Every commit, merge, or refactor can invalidate cached graph data. Claude Code v2.1.62 had a P1 bug where stale KV cache entries produced outdated context the model couldn't distinguish from current context. Model version drift adds another layer: GPT-4 outperforms GPT-3.5 by **20-30% on entity extraction**, so model upgrades produce different extractions that conflict with existing graph data.

**Mitigation**: Event-driven invalidation (invalidate affected nodes on each webhook event). Version-based cache keys (model version + prompt hash + KB version). Short TTLs for high-volatility entities (active files, in-progress PRs), longer for stable entities (archived ADRs, released features). Periodic full reconciliation against source APIs.

### Entity disambiguation is harder in code than in natural language

Code entities have unique challenges: the same identifier can refer to different functions across modules; concepts like "the auth bug" require temporal and conversational context to resolve; and naming conventions drift across teams and tools. Research shows entity disambiguation models overfit to frequent entities and struggle with **rare, project-specific terms** — exactly what engineering knowledge graphs contain.

**Mitigation**: Canonical identifiers with source-of-truth priorities. Confidence scores on all AI-extracted links (with human review for low-confidence matches). Multiple identifier aliases per entity. Context-aware disambiguation using graph neighborhood (if two nodes share many neighbors, they're likely the same entity).

### Graph quality degrades as entropy accumulates

**91% of ML models degrade over time**, and this compounds across every entity and relationship. Orphaned nodes appear as code is deleted but references remain. Stale relationships persist after refactoring. LLM extraction errors accumulate — recall is the limiting factor, as models favor high-precision extraction while missing valid connections. Anyshift learned this when their cleanup job accidentally deleted permanent "hub" nodes.

**Mitigation**: Mark structural anchor nodes as `permanent = true`. Implement Cognee's `memify` pattern — periodic post-processing that prunes stale nodes, strengthens frequent connections, and reweights edges. Hybrid LLM + human validation for critical relationships. Automated graph health metrics (orphan rate, relationship freshness, embedding staleness).

### Schema rigidity blocks evolution; schema flexibility blocks querying

Rigid schemas (predefined node labels and relationship types) enable reliable querying but require anticipating all entity types upfront. Dynamic schemas accommodate heterogeneous data but risk semantic drift. Neo4j's schema-optional nature is both blessing and curse.

**Mitigation**: Fixed schema for structural code entities (functions, classes, modules, dependencies) — these change rarely. Dynamic schema for contextual knowledge (decisions, discussions, incidents) — these evolve constantly. Additive-only schema evolution. Schema consolidation via LLM (Neo4j's documented pattern) to standardize redundant labels periodically.

### Adoption determines success more than architecture

**67% of platform engineering initiatives fail**, primarily due to adoption rather than technical issues. Platforms fail when "built in a vacuum, based on assumptions about what developers need rather than active, empathetic listening." The context graph must demonstrate immediate, concrete value or engineers will ignore it.

**Mitigation**: Start with the highest-value, lowest-friction use case: automated bug context hydration (the system assembles a context packet whenever a bug is assigned, requiring zero manual effort). Build from there to decision capture and tribal knowledge. Integrate into existing workflows (VS Code extension, Slack, PR comments) rather than requiring new tools.

---

## 7. Academic foundations and research that informs this design

The design draws on four streams of academic research, each contributing specific architectural patterns.

### Software engineering knowledge graphs

The systematic review by Wang et al. (2023, *Information and Software Technology*) covers 55 papers analyzing KG applications in SE. The key finding: **code details, architecture, and security reports** are the primary knowledge stored, with ontology modeling, NER, and knowledge fusion as construction methods. **GraphGen4Code** (Abdelaziz et al., 2020) demonstrated scalability — building code KGs from 1.3M Python files with 2B+ triples. **REPOGRAPH** (ICLR 2025) showed code graph databases improve both agent and procedural frameworks by **~32.8%** on SWE-bench, with the insight that "file-level indexing can only identify semantically similar but not genuinely related code snippets." **CODEXGRAPH** (NAACL 2025) uses Neo4j code graph databases as LLM agent interfaces supporting multi-hop reasoning.

### Decision provenance and design rationale

Kruchten's "Ontology of Architectural Design Decisions" (2004) defines first-class decision entities with attributes (rationale, state, cost, risk) and relationships (constrains, forbids, enables, subsumes, conflicts_with). Falessi et al. (2013, *ACM TOSEM*) proved empirically that design rationale documentation **significantly improves effectiveness of decision-making** in controlled experiments. Historical frameworks — IBIS (Kunz & Rittel, 1970), DRL, QOC, SEURAT — provide schema models for representing design decisions. De Boer's "Architecture Knowledge Graphs" (SATrends 2024) argues for linked data principles to break architectural knowledge out of isolated repositories.

### Temporal knowledge graphs and GraphRAG

Microsoft's **GraphRAG** paper (Edge et al., 2024, arXiv:2404.16130) demonstrated that LLM-built entity knowledge graphs with hierarchical community summaries via Leiden clustering substantially improve RAG for global sensemaking questions. The two-stage process (entity/relationship extraction → community summarization) is directly applicable. Zep's **Graphiti** paper formalized temporal context graphs with bi-temporal modeling, episode-based provenance, and hybrid retrieval. The bitemporal property graph model by Rost et al. (2021, arXiv:2111.13499) provides the formal foundation for temporal queries.

### Repository-level code understanding

The "Code Digital Twin" paper (arXiv:2503.07967) is the most directly relevant academic work, proposing a persistent knowledge infrastructure coupling "physical" artifacts with a "conceptual" layer encoding design rationales and bidirectional traceable links. The "Codified Context" paper (arXiv:2602.20478) demonstrated that documentation-as-infrastructure works: in 757 agent invocations, **57% used project-specific specialists** defined in context infrastructure, and **over 80%** of human prompts were ≤100 words — confirming that pre-loaded context reduces in-prompt explanation needs. A comprehensive survey (arXiv:2508.00083) identifies that "how to enable agents to efficiently understand and utilize non-public, highly contextualized information is a critical challenge."

---

## Conclusion: from code graph to context graph in four phases

The architecture proposed here extends Potpie's existing Neo4j + Python stack through four implementation phases, each delivering standalone value:

**Phase 1 (weeks 1-3)**: Extend the existing code graph with Commit, PullRequest, and Person nodes from Git history. Add `AUTHORED`, `CHANGED`, `AFFECTS` relationships. Implement convention-based entity linking (ticket IDs in PR titles). This alone enables "who last modified this function?" and "what PRs touched this module?" — queries agents cannot answer today.

**Phase 2 (weeks 4-6)**: Add issue tracker integration (Linear/Jira) with `IMPLEMENTS` and `FIXES` relationships. Add Slack ingestion for discussion context. Implement bi-temporal model on all nodes. This enables decision provenance: "why was this function written this way?" traces through the PR, to the ticket, to the Slack discussion.

**Phase 3 (weeks 7-9)**: Semantic entity resolution (embedding-based fuzzy matching for cases without explicit cross-references). LLM-based entity extraction from unstructured text. Build hybrid retrieval (vector + BM25 + graph traversal). This enables "what context is related to authentication failures?" — a query that spans code, tickets, discussions, and decisions.

**Phase 4 (weeks 10-12)**: MCP server with the six core tools. Context hydration with progressive loading. Token budget management. GraphQL API for agent consumption. This makes the context graph a first-class tool that any coding agent (Claude Code, Cursor, OpenHands, Potpie's own agents) can call before and during tasks.

The key architectural insight is that **the context graph is not a database feature — it is an application-layer service** with its own ingestion, resolution, retrieval, and serving logic. The backend store (Neo4j, Postgres+pgvector, or future alternatives) is an implementation detail. The value lives in the schema design (what entities and relationships to model), the entity resolution (linking the same concept across platforms), the temporal model (tracking how truth evolves), and the retrieval API (serving exactly the right context within token budgets). This is the layer that transforms coding agents from powerful text processors into systems that understand software the way experienced engineers do — through history, relationships, and the accumulated decisions that shaped every line of code.