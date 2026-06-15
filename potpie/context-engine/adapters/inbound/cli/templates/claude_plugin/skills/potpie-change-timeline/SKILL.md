---
name: "potpie-change-timeline"
version: "3"
recommended: true
description: "Use when the agent needs recent or historical change context: merged PRs, tickets, docs, incidents, deployments, regressions, and source-history ingestion."
---

# Potpie Change Timeline

Use this skill when debugging a regression, explaining what changed recently,
reviewing a risky change, or ingesting source history from GitHub, tickets,
docs, or links.

## Read First

For "what changed recently in this project?", do not pass a repo. A pot is the
project boundary and can contain multiple repositories; the CLI infers the pot
from the current repo source when possible, then reads the whole project timeline
across registered repo sources:

```bash
potpie --json timeline recent --limit 20
potpie --json timeline recent --time-window 7d --limit 20
```

Only narrow the timeline when the user asks for a service-specific slice:

```bash
potpie --json timeline recent --service payments-api --query "timeout retry deployment" --limit 20
```

`graph read` is still available for lower-level reads. Timeline reads default to
event-shaped output, deduplicated by source ref and sorted by occurrence time:

```bash
potpie --json graph read --view recent_changes.timeline --limit 20
potpie --json graph read --view recent_changes.timeline --format raw --limit 20
```

MCP equivalent:

```json
{"intent":"debugging","include":["timeline","prior_bugs","infra_topology"],"mode":"fast","source_policy":"references_only"}
```

Use time windows when the user gives one, and include exact dates in your own
reasoning. Timeline context is recorded/source-backed project history; it does
not include uncommitted local worktree changes unless those have been recorded.
Timeline context is for correlation, not proof: verify source refs before
blaming a change.

## Ingest Source History

Timeline ingestion is harness-led. The harness enumerates source records, reads
the authored content, decides what happened, and writes `append_event` or related
semantic mutations. Potpie should not scan the working tree and deterministically
update the graph.

Change-history ingestion is timeline scope only. Do not use PR/issue history to
infer a repo's baseline architecture — that is the separate `potpie-repo-baseline`
procedure.

When the harness has already hydrated a repo link, PR, issue, document, or web
page, write directly with `graph mutate`.

## Record An Event

Use `append_event` for "something happened" records:

```json
{
  "graph_contract_version": "v1.5",
  "pot_id": "local/default",
  "idempotency_key": "mutation:timeline:github-pr-412",
  "created_by": {"surface": "cli", "harness": "codex"},
  "operations": [
    {
      "op": "append_event",
      "subgraph": "recent_changes",
      "truth": "timeline_event",
      "subject": {"key": "activity:github:pr:acme/payments:412", "type": "Activity"},
      "verb": "github_pr_merged",
      "occurred_at": "2026-06-08T16:20:00Z",
      "actor": {"key": "person:alice", "type": "Person"},
      "targets": [{"key": "repo:acme-payments", "type": "Repository"}],
      "description": "PR 412 merged bounded retry and timeout logging for payments outbound calls; relevant to refund timeout, settlement retry, and HTTP client flakiness investigations.",
      "evidence": [{"source_ref": "github:pr:412", "authority": "external_system"}]
    }
  ]
}
```

For PRs, also add `Fix`, `BugPattern`, `Decision`, or infra links only when the
source explicitly supports them. For issues and tickets, do not emit a `Fix` just
because the issue is closed or done; fixes come from merged code changes or an
explicit shipped-resolution source.
