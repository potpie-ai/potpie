<!-- potpie-start -->
# Context Engine

This project uses the Potpie context engine for project memory. Before non-trivial work use it to orient yourself. After work use it to record durable learnings.

## Quick Start

```bash
potpie doctor          # check setup
potpie search "topic"  # search project memory
potpie ingest "..."    # record an episode
potpie pot pots        # list pots
```

## The Four MCP Tools

When Potpie MCP is configured use only the minimal port:

- **`context_resolve`** — primary context wrap for any task. Pass `intent`, `scope`, `include`, `mode`, `source_policy`, and `budget`.
- **`context_status`** — cheap pot readiness, freshness gaps, and recommended recipe. Run before broad or ambiguous tasks.
- **`context_search`** — narrow follow-up lookup after `context_resolve` when a specific entity or phrase is known.
- **`context_record`** — save durable project learnings: decisions, fixes, preferences, workflows, bug patterns, feature notes, incident summaries.

Do not add separate tools for each context type. Express all use cases as `context_resolve` parameter recipes.

## Recipes

**Feature work:**
```json
{"intent":"feature","include":["purpose","feature_map","service_map","docs","tickets","decisions","recent_changes","owners","preferences","source_status"],"mode":"fast","source_policy":"references_only"}
```

**Debugging:**
```json
{"intent":"debugging","include":["prior_fixes","diagnostic_signals","incidents","alerts","recent_changes","config","deployments","owners","source_status"],"mode":"fast","source_policy":"references_only"}
```

**Review:**
```json
{"intent":"review","include":["artifact","discussions","owners","recent_changes","decisions","preferences","source_status"],"mode":"balanced","source_policy":"summary"}
```

**Operations:**
```json
{"intent":"operations","include":["deployments","runbooks","alerts","incidents","scripts","config","owners","source_status"],"mode":"balanced","source_policy":"summary"}
```

**Onboarding:**
```json
{"intent":"onboarding","include":["purpose","repo_map","service_map","docs","local_workflows","agent_instructions","source_status"],"mode":"fast","source_policy":"references_only"}
```

## Working Rules

- Run `context_status` or a quick `context_resolve` before broad or ambiguous tasks.
- Start with `mode=fast` and `source_policy=references_only`. Escalate only when coverage, freshness, or task risk requires it.
- Always inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and `source_refs` before relying on graph memory.
- If `quality.status` is `watch` or `degraded`, verify relevant facts against source truth before high-impact work.
- Use `context_record` after discovering reusable project memory — fixes, decisions, preferences, workflows, incident summaries.
- Keep records compact and source-reference-first.

## Slash Commands

Use `/potpie-feature`, `/potpie-debug`, `/potpie-review`, `/potpie-ops`, `/potpie-status`, and `/potpie-record` for guided context workflows.
<!-- potpie-end -->
