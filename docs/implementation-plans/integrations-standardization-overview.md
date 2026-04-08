# Integrations & Sources — CTO Brief (Technical Overview)

*Short, shareable technical brief. Full deep-dive: [integrations-standardization-implementation-plan.md](./integrations-standardization-implementation-plan.md).*

---

## What we’re building

Potpie should understand **all the tools a team actually uses** — not only GitHub. Each project can attach **sources** (repos, issue trackers, channels, etc.). Data from those sources flows into a unified **context graph** so the product can answer questions using code history, issues, decisions, and links between them.

**Today:** GitHub is the main path.  
**Direction:** Same patterns for Linear, Jira, Slack, and others — without rebuilding the whole pipeline for each tool.

---

## Open source vs commercial

- **Open repo:** GitHub integration, shared “shell” (how connections and sources work), APIs, and a generic Sources experience.
- **Closed / commercial packages:** Concrete implementations for tools like Linear (OAuth, APIs, ingestion). No secrets or proprietary logic in the public tree.

The public codebase defines the **platform** (interfaces, schemas, orchestration, APIs, UI shell). Commercial code supplies provider implementations (Linear/Jira/Slack/etc.) via a plugin mechanism.

---

## Core model (connection vs source)

### Connection (account-level)

A **connection** is a user/workspace credential to a provider (OAuth tokens + granted scopes). One connection can back multiple project sources.

### Source (project-level)

A **source** attaches a provider scope to a specific project (e.g., one repo, one Linear team, one Slack channel) and has sync state/health.

---

### Minimal data model (high-level)

- **Connection**: `provider`, `user_id`, `auth_data` (encrypted), `scope_data`, `status`, `webhook_secret`
- **ProjectSource**: `project_id`, `provider`, `source_kind`, `scope_json`, `sync_enabled`, `sync_mode`, `webhook_status`, `last_sync_at`, `last_error`, `health_score`

**Important:** Detaching a source does **not** revoke the connection. Disconnecting a connection cascades to disable its sources.

---

## Provider plugin contract (how integrations “plug in”)

Providers are registered via a **registry** (no `if provider == "linear"` conditionals in shared code). Each provider supplies:

- **Adapter**: implements the relevant port (`SourceControlPort` for code hosts, `IssueTrackerPort` for Linear/Jira, etc.)
- **Ingestion**: raw artifact → narrative “episode” → graph ingestion + ledger write
- **Backfill**: cursor-based catch-up (safety net)
- **Webhook handler**: event → targeted incremental ingest
- **OAuth/refresh**: token exchange, refresh, revocation hooks

Commercial packages register providers at startup; the API process and Celery workers must load the same registry for correctness.

---

## How data gets into the graph (sync + ingestion)

Potpie uses a **knowledge graph** with an episodic/narrative layer (Graphiti). For each meaningful change (merged PR, issue updated, comment added), we generate a rich text **episode** and ingest it into the episodic graph. This extraction step yields typed entities/edges; we also write deterministic structural links where possible.

### Two sync modes (both required)

1. **Webhook-driven incremental sync (primary)**: provider event → enqueue a task to ingest *that one artifact*.
2. **Periodic backfill (recovery/onboarding)**: cursor-based scan to catch missed events and bootstrap new sources. Ledger dedup prevents double-ingest.

### Episode contract (provider-agnostic)

Each ingested artifact becomes an episode with:

- `name` (stable identifier)
- `episode_body` (rich narrative markdown)
- `source_description` (human label)
- `source_id` (dedup key)
- `reference_time` (time anchor)

Graphiti extracts entities/edges from `episode_body` using a shared schema; providers don’t need bespoke graph code if the narrative is good.

### Cross-provider correlation (deterministic)

When artifacts reference each other (Linear issue mentions a GitHub PR URL, PR body mentions a Linear ID), we write explicit edges using **namespaced entity keys** (e.g., `github:pr:owner/repo:1234`, `linear:issue:ENG-5678`). This avoids relying on LLM extraction for correlation.

### Source-aware context retrieval

When multiple sources exist, retrieval should be able to filter/weight by source type (code-host vs issue-tracker) and apply freshness decay when a source hasn’t synced recently.

---

## Security & compliance (non-negotiables)

- **Encrypted credentials**: OAuth tokens are stored encrypted at rest (application-level envelope encryption), not plaintext JSONB.
- **Refresh + revocation**: tokens refresh proactively; disconnect revokes tokens and disables linked sources.
- **Least privilege**: scopes are minimized; write-back capabilities (future) require explicit write scopes.
- **Auditability**: per-source sync events and failures are logged with project/source/provider identifiers.

---

## Product surface (what ships)

- A **Sources** area that is entirely API-driven:
  - provider catalog (available/disabled)
  - connections (what accounts are connected)
  - project sources (what’s attached per project)
  - per-source health: last sync time, error state, manual re-sync
- New providers appear automatically when the backend registry exposes them (no UI rewrites).

---

## Execution plan (high level, dependency-ordered)

Work is sequenced by dependency:

- Define **provider registry + ports**
- Implement **credential vault** and connection lifecycle
- Add **project_sources** and refactor runtime to support multiple sources per project
- Build **Sources APIs + webhook router**
- Refactor **Sources UI** to be data-driven
- Add the first commercial provider (Linear) to validate the architecture end-to-end
- Harden: per-provider queues, backpressure, health scoring, observability

---

## Key risks (and what addresses them)

- **Split GitHub auth state today**: solve via a deliberate aggregation layer until GitHub is migrated into the unified connection model.
- **Multi-source noise**: source-aware retrieval + freshness decay.
- **Webhook gaps**: periodic backfill + ledger dedup.
- **Provider API churn**: adapter versioning + provider health tracking + staged rollouts.
- **Operational load**: per-provider queues, rate limits, and backpressure.

---

## One-line summary

**Unify account connections and project sources, run a webhook-first + backfill-safe sync pipeline, ingest provider episodes into a correlated context graph, and add new tools as plugins (GitHub OSS; others commercial).**
