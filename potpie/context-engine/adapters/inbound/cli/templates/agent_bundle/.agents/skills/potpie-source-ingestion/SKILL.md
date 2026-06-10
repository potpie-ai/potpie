---
name: "potpie-source-ingestion"
version: "5"
recommended: true
description: "Use when ingesting a repo link, document, PR, issue, ticket, runbook, or web link into Potpie. The harness reads and interprets the source, then writes semantic graph mutations; no local code scanning or deterministic graph updates."
---

# Potpie Source Ingestion

Use this skill when the user asks to ingest a repository, link, document,
ticket, PR, issue, runbook, incident report, or external source into Potpie.

## Principle

The harness is the intelligence. It decides what data it has, reads the relevant
source, extracts durable facts, resolves graph identity, and writes semantic
mutations with evidence. Potpie validates and stores the graph update. Do not run
a working-tree scan, derive facts from filenames alone, or apply deterministic
graph updates outside the harness.

## Ingestion Loop

1. Identify the source kind: repo history, PR, issue, ticket, doc, runbook,
   incident, deployment note, or arbitrary link.
2. Gather the source data needed for the ingestion goal. For repository ingest,
   do both the baseline repo-understanding pass and the change-history pass
   below; for narrower sources, read only the directly relevant material.
3. Read existing graph context and resolve identity:

```bash
potpie --json graph search-entities "<service or repo name>" --type Service --limit 10
potpie --json timeline recent --limit 10
```

4. Classify each durable fact into the right memory family:

- Preferences: `Preference` + `POLICY_APPLIES_TO`.
- Infra architecture: `Service`, `Environment`, `DataStore`, `Dependency`, and
  topology predicates such as `DEPENDS_ON`, `DEPLOYED_TO`, `USES`, `EXPOSES`.
- Repository purpose, features, and functionality: `Repository`, `Service`,
  `Feature`, `CodeAsset`, `APIContract`, and topology predicates such as
  `DEFINED_IN`, `USES`, `EXPOSES`, `PROVIDES` (repo/service → feature), and
  `IMPLEMENTED_IN` (feature → repo/service/code). Give every `Feature` a
  compact `summary` and a retrieval-grade `description`.
- Timeline: `Activity` via `append_event`.
- Bugs/debug: `BugPattern`, `Fix`, `Verification`.
- Decisions/docs: `Decision`, `Document`, `DECIDED`, `AFFECTS`, or doc records.

5. Write a batch with `potpie --json graph mutate --file mutation.json --dry-run`.
   Apply only after validation is clean or the remaining warnings are intentional.

## Source-Specific Rules

Repository link:
Run two harness-led passes. The baseline pass has a dedicated procedure in
the `potpie-repo-baseline` skill — follow it when present.

1. Baseline repo-understanding pass:
   - Register the repo source in the current pot.
   - Read authored docs and metadata first: README, docs, ADRs, deployment docs,
     env templates, package/app manifests, CI/deploy workflows, route/API specs,
     framework config, and visible entrypoints.
   - Read code only when it is the source of truth for a durable fact, such as
     routes, exported feature modules, service clients, adapters, deployment
     targets, or API contracts. The harness should inspect and summarize the
     relevant files; it must not run a deterministic scanner that directly
     mutates the graph.
   - Record repository purpose, application type, primary user-facing
     functionality, feature areas, runtime/deploy shape, environments,
     service/API dependencies, datastores, and important integrations when
     supported by the read source.
   - Use `source_observation` or `authoritative_fact` for explicit repository
     evidence, `agent_claim` for reasoned lower-authority interpretation, and
     include evidence refs such as `repo:<repo>#README`, `repo:<repo>#package`,
     `repo:<repo>#route:<path>`, or GitHub URLs.

2. Change-history pass:
   - Read recent merged PRs and standalone issues.
   - Record timeline activities, clear fixes, explicit decisions, and obvious
     bug patterns.

Do not walk the file tree to invent modules, features, services, or dependencies.
Do read selected source files when the harness needs them to understand and
record actual topology, features, and functionality.

Document or web link:
Record `doc_reference`, `decision`, `preference`, `runbook_note`, `service_note`,
or infra claims only when the document explicitly says them. Preserve the URL or
document id in `source_refs` or mutation evidence.

Linear/Jira:
Tickets and issues can create timeline activities, bug patterns, docs, and
decisions. Do not emit `Fix` from an issue/ticket alone; fixes require a merged PR,
commit, deployment, or explicit shipped-resolution source.

Debug transcript/log:
Record bug patterns, diagnostic signals, investigations, fixes, and verifications.
Keep raw logs out of descriptions except for the shortest distinctive error text.

## Anti-Patterns

- Do not call a local code scan command.
- Do not update the graph from deterministic file/config scanning.
- Do not create a service because a directory has a service-like name.
- Do not stop at PR/issue timeline events for a repository ingest; capture the
  repo's durable purpose, topology, feature areas, and user-facing
  functionality when source evidence is available.
- Do not create dependencies because a package file mentions them unless the task
  is explicitly about dependency inventory and the harness has read that source.
- Do not hard-delete or silently overwrite claims; retract or end validity.
- Do not omit retrieval-grade descriptions.
