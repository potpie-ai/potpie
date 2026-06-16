---
name: potpie-source-ingestion
description: "Use when the user explicitly asks to ingest a repository, PR, issue, ticket, runbook, incident report, document, or web link into Potpie. The harness reads source material, classifies durable facts, and writes semantic graph workbench mutations; no deterministic scanning or connector queue ingestion."
---

# Potpie Source Ingestion

Use this skill for explicit ingestion requests. The agent is the harness: gather
source data, read it, decide what is durable, resolve identity, and write semantic
graph mutations with evidence. Do not use pot-level connector ingestion commands.

## Ingestion Loop

1. Define scope: source kind, pot/project, source URL/path, time window, and
   desired memory shape: baseline, history, docs, infra, debug memory, or all.
2. Parallelize independent discovery when available: local `rg`/`rg --files`,
   GitHub CLI/app/MCP, Linear/Jira/Confluence/Slack/MCP, linked docs, web, and
   long-running background processors. Use sub-agents only for read-only
   discovery slices; the main agent owns source selection, identity resolution,
   mutation proposals, and commits.
3. Hydrate hosted context through the agent's integration tools/connectors and
   whatever other agent tools are available: repo metadata, README/docs, PRs,
   issues, releases, linked tickets/docs, CI/deploy records, runbooks, and
   product websites linked from the source. Potpie integrations are one evidence
   path, not the exclusive source path.
4. Read existing graph context and resolve identity:

```bash
potpie --json graph catalog --task "ingest source"
potpie --json graph search-entities "<repo service feature>" --limit 10
```

5. Classify durable facts:
   - Preferences: `Preference` and `POLICY_APPLIES_TO`.
   - Infra: services, environments, dependencies, datastores, adapters, deploys.
   - Features: `Feature` entities, repo/service `PROVIDES`, and feature
     `IMPLEMENTED_IN`.
   - Timeline: source-time activity events.
   - Debugging: bug patterns, fixes, verifications, failed attempts.
   - Decisions/docs: decisions, document references, runbook notes.
6. Write with the workbench:

```bash
potpie graph mutation-template --kind <kind>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
```

For many source-backed operations, write JSON/NDJSON semantic mutations and use
chunked apply:

```bash
potpie --json graph bulk apply --file mutations.ndjson --chunk-size 100 --manifest graph-bulk-manifest.json --verify
```

Use dry-run mode before committing large writes and `--start-chunk <n>` to resume
after fixing a failed chunk. The bulk command applies agent-authored mutations;
it does not gather evidence or decide what belongs in memory.

Use `graph inbox add` when evidence may matter but the canonical graph update is
uncertain.

## Repository Baseline

For repository ingestion, run baseline before change history. Use
`potpie-repo-baseline`, but drive it with a baseline-quality discovery pass:

- Product/context: README, docs, ADRs, package metadata, linked websites, public
  docs, and repo-mentioned product pages.
- Repo map: manifests, top-level apps/packages, major modules, route/API
  entrypoints, models, clients, adapters, and framework config.
- Runtime/deploy: Dockerfiles, compose, Kubernetes, Terraform, CI workflows,
  deploy scripts, environment templates, and feature flags.
- Integrations/history: GitHub repo metadata, topics, releases, recent
  PRs/issues, linked tickets, and external docs.

Record source-backed purpose, application type, features, service/module map,
API contracts, datastores, integrations, environments, deploy shape, ownership,
and explicit project preferences. Then use `potpie-change-timeline` for recent
or historical activity. Do not infer baseline architecture from PR titles or
issue status.

## Source Rules

- Tickets and issues can record timeline events, bug patterns, decisions, and
  docs. They do not prove a fix unless tied to a merged PR, commit, deployment,
  or explicit shipped-resolution source.
- Documents can record preferences, decisions, runbook notes, service notes, and
  infra facts only when they explicitly say them.
- Logs and transcripts can record diagnostic signals, investigations, fixes, and
  verifications. Keep raw logs out of descriptions except for short distinctive
  error text.

Every write needs source refs, truth class, and a retrieval-grade description.
Ingestion is harness-led; do not use scanner-driven graph updates or connector
queues to decide what belongs in memory.
