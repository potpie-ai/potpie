---
name: "context-engine-agent-context"
description: "Use when an agent needs project context from the context engine through MCP. Provides context_resolve recipes for feature work, debugging, review, operations, docs, and onboarding without adding separate context tools."
---

# Context-Engine Agent Context

Use this skill when the task is to gather, verify, or record project context through the context-engine MCP tools.

## Tool Surface

Use only the minimal context port:

- `context_resolve`: primary task context wrap.
- `context_status`: cheap pot readiness, freshness gaps, and recommended recipe.
- `context_search`: narrow follow-up lookup after `context_resolve`.
- `context_record`: durable learnings such as decisions, fixes, preferences, workflows, feature notes, doc references, and incident summaries.

Do not request or invent separate tools for feature context, debugging context, operations context, source lookup, or docs context. Express those through `context_resolve` parameters.

## Operating Loop

1. Resolve or ask for the active `pot_id`.
2. Build a narrow `scope` from known repo, branch, files, services, features, environment, PR, ticket, user, or source refs.
3. Call `context_status` when setup/readiness is uncertain or the task is broad.
4. Call `context_resolve` with the matching recipe below.
5. Inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on the answer.
6. Escalate only when needed:
   - `source_policy="summary"` for compact source-backed summaries.
   - `source_policy="verify"` and `mode="verify"` before production-impacting or disputed facts.
   - `source_policy="snippets"` or `mode="deep"` only for bounded source detail within budget.
7. Use `context_record` when the work discovers reusable project memory.

## Recipes

Feature work:

```json
{
  "intent": "feature",
  "include": ["purpose", "feature_map", "service_map", "docs", "tickets", "decisions", "recent_changes", "owners", "preferences", "source_status"],
  "mode": "fast",
  "source_policy": "references_only"
}
```

Debugging:

```json
{
  "intent": "debugging",
  "include": ["prior_fixes", "diagnostic_signals", "incidents", "alerts", "recent_changes", "config", "deployments", "owners", "source_status"],
  "mode": "fast",
  "source_policy": "references_only"
}
```

Review:

```json
{
  "intent": "review",
  "include": ["artifact", "discussions", "owners", "recent_changes", "decisions", "preferences", "source_status"],
  "mode": "balanced",
  "source_policy": "summary"
}
```

Operations:

```json
{
  "intent": "operations",
  "include": ["deployments", "runbooks", "alerts", "incidents", "scripts", "config", "owners", "source_status"],
  "mode": "balanced",
  "source_policy": "summary"
}
```

Docs:

```json
{
  "intent": "docs",
  "include": ["docs", "decisions", "source_status"],
  "mode": "fast",
  "source_policy": "references_only"
}
```

Onboarding:

```json
{
  "intent": "onboarding",
  "include": ["purpose", "repo_map", "service_map", "docs", "local_workflows", "agent_instructions", "source_status"],
  "mode": "fast",
  "source_policy": "references_only"
}
```

## Recording Durable Memory

Use `context_record` for:

- `decision`: architectural or product decisions.
- `fix`: concrete fix that resolved a bug or incident.
- `bug_pattern`: recurring symptom/root-cause pattern.
- `investigation`: useful diagnostic path or ruled-out hypotheses.
- `diagnostic_signal`: log line, metric, alert, or error signature.
- `preference`: team/user/project preference.
- `workflow`: local, release, deployment, or operational workflow.
- `feature_note`, `service_note`, `runbook_note`, `integration_note`, `doc_reference`, `incident_summary`.

Keep records compact and source-reference-first. Include source refs when available instead of copying full documents, logs, threads, or diffs.

## Quality And Drift

Treat graph memory as orientation unless quality says it is fresh enough for the task.

- If `quality.status` is `good`, graph context is acceptable for low-risk orientation.
- If `quality.status` is `watch`, verify stale or unverified facts before high-impact action.
- If `quality.status` is `degraded`, prefer source truth and consider recording an alignment correction.
- Follow `quality.recommended_maintenance` when the response suggests jobs such as `verify_entity`, `refresh_scope`, `resync_source_scope`, `repair_code_bridges`, or `expire_stale_facts`.
