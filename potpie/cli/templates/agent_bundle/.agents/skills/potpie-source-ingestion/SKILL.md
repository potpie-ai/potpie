---
name: potpie-source-ingestion
description: "Use when the user explicitly asks to ingest, refresh, or deeply understand a repository, PR, issue, ticket, runbook, incident report, document, or web link into Potpie. The harness performs todo-driven discovery, uses local/GitHub/integration tools and read-only subagents when available, builds evidence-backed semantic mutations, and writes through graph propose/verified commit."
---

# Potpie Source Ingestion

Use this skill for explicit ingestion requests. The harness is the intelligence:
it gathers source data, reads it, decides what is durable, resolves identity, and
writes semantic graph mutations with evidence. Potpie validates and stores; it
does not decide what source material means.

## Non-Negotiables

- Use a todo/checklist for every repository or multi-source ingestion. Do not
  jump directly from a README to graph writes.
- Local inspection is required for repo understanding. Scanner-driven graph
  updates are forbidden. Inspect files with `rg`, `rg --files`, `git`, and
  structured tooling; do not run legacy or deterministic ingestion/scanner
  commands that walk the tree and write graph facts.
- Use subagents only for read-only discovery slices. The main agent owns source
  selection, identity resolution, mutation proposals, commits, and final
  synthesis.
- Do not write until each required discovery lane is complete, explicitly
  unavailable, or intentionally scoped out by the user.
- Every write needs source refs, source authority, truth class, confidence,
  compact summary, and retrieval-grade description.

## Phase 0: Scope And Preflight

1. Define source kind, pot/project, repo/path/URL, time window, and target memory
   shape: baseline, history, docs, infra, debug memory, preferences, or all.
2. Verify Potpie scope and graph availability:

```bash
potpie --json pot info
potpie --json source list
potpie --json graph status
potpie --json graph catalog --task "harness-led source ingestion"
```

3. If the repo is not registered, register metadata only. Use explicit `--pot`
   when pot scope is ambiguous:

```bash
potpie source add repo . --pot <pot-id-or-name>
```

4. Describe the views you expect to write/read before authoring mutations:

```bash
potpie --json graph describe features --view feature_context --examples
potpie --json graph describe infra_topology --view service_neighborhood --examples
potpie --json graph describe recent_changes --view timeline --examples
potpie --json graph describe decisions --view preferences_for_scope --examples
potpie --json graph describe debugging --view prior_occurrences --examples
```

5. If the CLI is unavailable or broken, continue discovery, build the proposed
   evidence matrix, and stop before committing graph writes. Use MCP
   `context_record` only as a compatibility fallback when explicitly available.

## Phase 1: Todo Plan

Create and maintain todos with at least these lanes for repository ingestion:

- Scope, pot, source registration, and graph contract preflight.
- Product/docs: README, docs, ADRs, runbooks, public docs, linked websites.
- Local repo map: manifests, packages/apps, entrypoints, route/API surfaces,
  tests, framework config, major modules, generated/API specs.
- Runtime/deploy: Dockerfiles, compose, Kubernetes, Terraform, CI workflows,
  deploy scripts, environment templates, feature flags.
- API/data/integrations: service clients, adapters, datastores, models,
  queues, auth providers, external APIs.
- GitHub/history: repo metadata, topics, releases/tags, recent merged PRs, open
  issues, linked tickets/docs, CI/deploy records.
- Preferences/workflows: explicit coding style, test commands, local dev setup,
  release/deploy/runbook workflows.
- Synthesis: evidence matrix, candidate graph facts, identity resolution,
  proposal, verified commit, and gate-driven follow-up checks.

Update todos as lanes finish. Preserve uncertain findings for the inbox instead
of forcing them into canonical graph claims.

## Phase 2: Parallel Discovery

Parallelize independent read-only slices when tools allow it. Recommended
subagent prompts:

- Docs/product: "Read README, docs, ADRs, runbooks, package metadata, and linked
  product pages. Return product purpose, features, explicit decisions,
  preferences, workflows, and source refs. Do not write mutations."
- Local architecture: "Inspect manifests, top-level apps/packages, entrypoints,
  route/API surfaces, tests, and framework config. Return service/module map,
  likely features, explicit source files, uncertainty, and no mutations."
- Runtime/deploy: "Inspect Docker/compose/Kubernetes/Terraform/CI/deploy/env
  templates. Return environments, deploy shape, config variables, workflows,
  datastores, and source refs. Do not write mutations."
- API/data/integrations: "Inspect route specs, client/adapters, models,
  datastores, queues, auth, and external integrations. Return candidate
  APIContract/DataStore/Adapter/Dependency facts with evidence. No mutations."
- GitHub history: "Use GitHub tools/CLI for repo metadata, releases, recent
  merged PRs, open issues, linked tickets/docs, and CI signal. Return timeline,
  fixes, decisions, bug patterns, and source refs. No mutations."
- Preferences: "Find explicit preferences in docs, config, tests, contribution
  guides, PR templates, and comments. Return only explicit reusable policies
  with evidence. No mutations."

The main agent should continue non-overlapping discovery while subagents run.

## Phase 3: Local Repo Inspection Targets

Use structured, bounded local inspection. Examples:

```bash
rg --files -g 'README*' -g 'docs/**' -g '*ADR*' -g 'package.json' -g 'pyproject.toml' -g 'Cargo.toml' -g 'go.mod'
rg --files -g 'Dockerfile*' -g 'docker-compose*' -g '.github/workflows/**' -g '*.tf' -g 'k8s/**' -g '.env*'
rg -n "FastAPI|APIRouter|express\\(|router\\.|Django|Flask|NextResponse|route\\(" .
rg -n "postgres|mysql|redis|mongo|s3|kafka|rabbit|queue|oauth|stripe|slack|github|linear|jira" .
rg -n "pytest|vitest|jest|playwright|make test|npm test|uv run|cargo test" README* docs .github pyproject.toml package.json Makefile
```

Do not infer durable facts from filenames alone. Use files to locate sources of
truth, then read the relevant snippets or authored docs.

## Phase 4: Hosted/GitHub Hydration

Use the agent's integration tools/connectors, including GitHub app/MCP/CLI
tools when available, not Potpie queue ingestion commands. For GitHub-backed
repo ingestion, gather the items below. Do not use Potpie CLI queue ingestion.

- Repository metadata: full name, default branch, description, topics,
  visibility, homepage, license, archived/fork status.
- Documentation: README, contributing guide, CODEOWNERS, PR/issue templates,
  docs linked from the repo.
- Releases/tags when they describe shipped behavior or deploy cadence.
- Recent merged PRs: title, body, author, merged date, branch, labels, linked
  issues, changed filenames; fetch patches only when author text is insufficient.
- Open/high-signal issues: title, body, labels, state, author, comments when
  available; issues can record requests/bugs but do not prove fixes.
- CI/workflows: workflow files and failed/passing runs only when relevant to
  durable workflows, release process, or recurring failures.
- Linked systems: Linear/Jira/Confluence/Slack/docs mentioned in repo, PRs, or
  issues when corresponding tools are available.

Respect API limits and user scope. Prefer recent/high-signal items over
exhaustive pagination unless the user explicitly asks for full history.

## Phase 5: Evidence Matrix

Before writing, build a compact matrix:

| Candidate | Graph family | Source refs | Authority | Truth class | Confidence | Action |
|---|---|---|---|---|---|---|
| Feature/service/dependency/etc. | features/infra/etc. | file, PR, doc, issue | authoritative_code, repository_metadata, external_system, user_statement, agent_observation | authoritative_fact, source_observation, agent_claim, preference, timeline_event | 0.0-1.0 | commit / inbox / skip |

Guidelines:

- Use `authoritative_fact` for explicit docs/config/code ownership facts.
- Use `source_observation` for observed source material that may not be policy.
- Use `agent_claim` for lower-authority synthesis from multiple weak signals.
- Use `preference` only for explicit reusable project preferences.
- Use `timeline_event` for source-time activity from PRs, tickets, releases, or
  deployments.
- Put uncertain but potentially useful findings into `graph inbox add`.

## Phase 6: Identity Resolution

Resolve before linking. Use specific filters when known:

```bash
potpie graph search-entities "<repo service feature dependency>" --limit 10
potpie graph search-entities "<service>" --type Service --environment prod --limit 10
potpie graph search-entities "<github-or-ticket-id>" --source-ref <github-or-ticket-ref> --limit 10
```

Reuse canonical keys. If duplicate candidates appear, stop and use inbox or a
review-required correction flow; do not create near-duplicate entities.

## Phase 7: Write

Author semantic mutation JSON from the live `graph catalog` and `graph describe`
examples. `graph mutation-template` is only a skeleton helper, not the source of
truth.

```bash
potpie --json graph propose --file mutation.json
```

Inspect proposal status, diff, warnings, rejected operations, conflicts, and
review flags:

- `invalid` or rejected operations: fix the mutation or skip the weak fact.
- `conflict` or duplicate risk: resolve identity or use inbox.
- `review_required`: ask for approval or commit only with the required
  `--approved-by` value when policy allows.
- `validated` / low-risk: commit with `--verify`.

```bash
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

For large batches of agent-authored mutations, use `graph bulk apply` with
dry-run, chunking, manifest, and verify. Bulk apply must only apply facts the
harness already selected.

## Phase 8: Verify And Quality Gate

`graph commit --verify` reads back committed claims and checks quality. When it
warns or fails, drill down with affected reads and quality reports:

```bash
potpie graph read --subgraph features --view feature_context --scope anchor_entity_key:<repo-key> --limit 50
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:<service> --depth 2 --direction both --limit 50
potpie graph read --subgraph recent_changes --view timeline --scope repo:<repo> --limit 50 --format table
potpie --json graph quality duplicate-candidates --limit 20
potpie --json graph quality low-confidence --limit 20
potpie --json graph quality conflicting-claims --limit 20
potpie --json graph quality orphan-entities --limit 20
```

If the verified commit misses expected facts, fix the mutation or record an inbox item.
Report what was ingested, what was skipped, and what remains uncertain.

## Repository Baseline

For repository ingestion, run baseline before change history. Use
`potpie-repo-baseline` in deep mode when the user asks to ingest or understand a
repo deeply. Record source-backed purpose, application type, features,
service/module map, API contracts, datastores, integrations, environments,
deploy shape, ownership, and explicit project preferences. Then use
`potpie-change-timeline` for recent or historical activity. Do not infer
baseline architecture from PR titles or issue status.

Represent capabilities as `Feature` entities. Link repositories or services to
features with `PROVIDES`, and use `IMPLEMENTED_IN` only when a source locates
the implementation.

## Source Rules

- Tickets and issues can record timeline events, bug patterns, decisions, and
  docs. They do not prove a fix unless tied to a merged PR, commit, deployment,
  or explicit shipped-resolution source.
- Documents can record preferences, decisions, runbook notes, service notes, and
  infra facts only when they explicitly say them.
- Logs and transcripts can record diagnostic signals, investigations, fixes, and
  verifications. Keep raw logs out of descriptions except for short distinctive
  error text.
