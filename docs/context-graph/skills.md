---
title: Skills and harness-led intelligence
description: How coding harnesses use Potpie skills to read and write the graph.
---

## Overview

> Status: reflects the capability-first layout, last reviewed 2026-08-24.

Potpie does not own the reasoning that turns a repo, PR, ticket, or bug into
durable memory. That intelligence lives in the **user's coding harness** (Claude
Code, Codex, Cursor, OpenCode), running on the user's own model subscription, and
is taught to the harness through **Potpie skills**. Potpie validates, lowers,
commits, audits, and ranks; it does not infer rich facts from prose or scan a
repository for you. This is the product's central anti-goal made concrete: *no
Potpie-owned LLM/reconciliation agent as the canonical source of graph
intelligence.*

A skill is **pure instruction text** — a `SKILL.md` of markdown (no executable
code) that a harness loads into its context. Skills teach an agent *how to use the
`potpie` CLI*; they are never graph facts, and they add no new tools. The
`potpie-graph` skill states it plainly: "You are the intelligence that reads it
before acting and writes durable learnings after... It does **not** scan a
repository or infer rich facts from prose for you."

> **Two skill surfaces — never conflate them.** (1) The **user-installed bundle
> skills** that drive the `potpie graph …` CLI workbench, managed by
> `DefaultSkillManager`. (2) The **server-side reconciliation skills** loaded by
> the ingestion-server's deep agent (`pydantic_deep_agent.py`), with a different
> tool surface and install path, off by default. Everything from here to
> [§Server-side reconciliation skills](#server-side-reconciliation-skills-a-separate-surface)
> is surface (1); surface (2) is documented at the end and in
> [ingestion-nudge.md](./ingestion-nudge.md).

---

## 1. The skill catalog & packaging

The single source of truth for skill content **and** metadata is the bundled set
of `SKILL.md` files under
`potpie/cli/templates/agent_bundle/.agents/skills/*/SKILL.md`.

`potpie/skills/catalog.py` scans those templates at runtime
(`lru_cache`d), parses each file's YAML front-matter (`name` / `version` /
`description`, optional `recommended: false`) into a `SkillInfo`, and exposes
`catalog_by_id()` and `RECOMMENDED_SKILL_IDS` (every recommended bundled skill).
Adding or editing a skill means editing the bundled markdown — nothing else.

There are **8 skills in the agent bundle**, and it is the only copy: every
harness installs its skills from it, remapped to the harness's own layout. The
compact instruction block merged into `AGENTS.md` / `CLAUDE.md` likewise has one
source, `templates/routing/POTPIE.md`.

## 2. Installation, targets & drift (`DefaultSkillManager`)

`potpie/skills/manager.py DefaultSkillManager` owns the catalog +
per-harness install/drift logic and delegates *where/how* to a registered
`AgentTargetPort` per harness. Operations: `list / install / update / remove /
status / nudge / add` (`add` is a TODO stub).

One table, `HARNESS_LAYOUTS` in `potpie/skills/harnesses.py`, says where each
harness reads Potpie's files at each scope. One target class,
`AgentTarget` in `potpie/skills/targets.py`, reads it for a harness at
`global` or `project` scope; `potpie/runtime/composition.py` registers one
global target per row, and a `--scope project` call builds a project target for
a registered harness.

| Harness (`--agent`) | Global skills | Global instructions | Project skills | Project instructions |
|---|---|---|---|---|
| `claude` | `~/.claude/skills` | `~/.claude/CLAUDE.md` | `.claude/skills` | `CLAUDE.md` |
| `codex` | `~/.agents/skills` | `~/.codex/AGENTS.md` | `.agents/skills` | `AGENTS.md` |
| `cursor` | `~/.cursor/skills` | — | `.cursor/skills` | `AGENTS.md` |
| `opencode` | `~/.config/opencode/skills` | — | `.opencode/skills` | — |

`default` is an alias of `codex` for the repository bundle an embedding host
installs with `install_agent_bundle()`; it is not a harness the skills CLI
manages, and `potpie setup --agent default` skips the skills step.

Global roots hang off `POTPIE_HARNESS_HOME` when it is set (the test suite pins
it), otherwise the real home directory; `CONTEXT_ENGINE_HOME` deliberately does
not move them.

Install mechanics (`potpie/skills/installer.py`, one `install_bundle` and one
`uninstall_bundle`; both read the bundle through `potpie/skills/bundle.py`):

- Every harness installs the same skill files, placed under its skills
  directory from the table.
- `AGENTS.md` / `CLAUDE.md` are **merged**, not overwritten — managed content
  lives between `<!-- potpie-start -->` / `<!-- potpie-end -->` markers
  (`_merge_managed_markdown`), preserving the user's own instructions.
- **The routing block belongs to the sweep.** It is written only by a bundle
  install or update (no skill id), and the instruction file it went into is
  named in the result's `metadata.support_files`; naming one skill id installs
  that skill alone. `skills remove --all` takes the same block back out — the
  managed section only, so a user's own `CLAUDE.md` text survives.
- **Retired Claude files.** Earlier releases also wrote two slash commands into
  a repository's `.claude/commands/` and offered a Claude Code plugin directory
  under `.claude/`; neither ships now. A project-scope Claude sweep (install or
  update without an id, or `remove --all`) deletes a leftover command file only
  when its bytes match a version Potpie shipped
  (`metadata.retired_files_removed`). An edited command file, or the old plugin
  directory (recognised by a manifest naming `potpie`), is never deleted: it is
  listed under `metadata.leftovers` with the step that clears it.
- **Drift tracking:** each target keeps one JSON manifest in the Potpie home
  (`skill_manifest_<agent>_<scope>.json`, plus a per-repository suffix at
  project scope) recording, per skill, the installed version, a content hash
  and the disabled flag. The three per-target files earlier releases wrote
  (`skills_…`, `skill_hashes_…`, `skill_disabled_…`) are folded into it on
  first read and then deleted; an unreadable one contributes nothing and
  nothing crashes. A skill whose files no longer match the bundle is
  **drifted** — reported inside `outdated` and fixed by the same reinstall —
  while a hand-edited one is left alone by a sweep
  (`metadata.preserved_user_edits`). A skill removed by id is **disabled**:
  bundle installs skip it until it is installed by id again. `status()`
  partitions skills into installed / missing / outdated / disabled, and
  `potpie --json skills status` also lists the `drifted` subset; `nudge()` emits
  the single advisory command
  `potpie skills install --agent <agent>`. That advisory is the *only* skill
  signal agents ever see — it rides on `context_status` (see
  [querying.md](./querying.md)). Globally-installed skills can stale (an old
  `potpie-graph` that still teaches the legacy `graph mutate` predates v5);
  `potpie skills update --agent <a>` is the fix the drift nudge points at.

### The correctness gate (this is real, not aspirational)

`validate_packaged_skill_command_snippets` (`potpie/skills/snippets.py`)
extracts every `potpie …` command a packaged template teaches — lines in
```` ```bash ```` fences and inline `` `potpie …` `` spans — and validates each
command + option against a command table. `tests/unit/test_agent_skill_templates.py`
runs it at build time with the **live Typer specs introspected from
`potpie.cli.main.app`**, so a skill cannot ship a `potpie` command or flag that
does not exist; installs no longer import the CLI to re-check the bundle the
build already checked. (Only `potpie` commands are checked; other shell commands
depend on the user's repo.)

### CLI surface

```bash
potpie skills list   [--agent claude|codex|cursor|opencode] [--scope global|project] [--path]
potpie skills install [<id>] [--agent …] [--scope …] [--path]
potpie skills update  [<id>|--all] [--agent …]
potpie skills status  [--agent …]     # installed, missing, outdated, drifted, disabled
potpie skills remove  [<id>|--all] [--agent …]
potpie skills add     <source>        # TODO stub
```

`--scope` flips to `project` automatically when `--path` is given with `global`;
`--path` is resolved against the caller's working directory and must already
exist. Skill commands are filesystem-only and never contact the daemon
(`install` still accepts a hidden, no-op `--no-daemon` for older installers).
`potpie setup --agent <harness>` installs the recommended bundle during first
run. **There is no top-level `potpie install`** — skills install only via
`potpie skills install` (and `setup`). Full flags live in
[cli-flow.md](./cli-flow.md).

---

## 3. The 8 core bundle skills

| Skill | Ver | Role |
|---|---|---|
| `potpie-cli` | v4 | The `potpie` command itself: pot-scope resolution order, harness-led boundaries. |
| **`potpie-graph`** | **v7** | **THE contract skill** — one shared discovery pass, the read → resolve → record or propose/commit → inbox → quality loop, ontology selection, truth classes, retrieval-grade descriptions, reporting the commands behind an answer, and "Responding To Nudges". Teaches `potpie record` for one fix or note and **propose/commit** for everything else (never the legacy `graph mutate`). |
| `potpie-repo-baseline` | v2 | Deep repo-baseline mode: source priority, evidence matrix, canonical entity families with `PROVIDES` / `IMPLEMENTED_IN`. |
| `potpie-source-ingestion` | v3 | Todo-driven, phased (0–8) ingestion of a repo/PR/ticket/doc; parallel read-only subagents; GitHub/Linear/Jira hydrated via the agent's **own** integration tools (explicitly *not* Potpie connector queueing) → evidence matrix → identity resolution → propose/commit `--verify` → quality gate. |
| `potpie-project-preferences` | v2 | Use-case read+record skill (preferences). |
| `potpie-infra-architecture` | v3 | Use-case read+record skill (infra/topology). |
| `potpie-change-timeline` | v2 | Use-case read+record skill (recent changes). |
| `potpie-debug-memory` | v3 | Use-case read+record skill (prior bugs/fixes). |

Three per-format skills handle document payloads: `potpie-resource-pdf`,
`potpie-resource-spreadsheet` and `potpie-resource-markdown` (v1 each). Each
teaches the agent to write an extraction script that emits a chunk directory,
import it with `potpie resource import`, summarize its sections, and link the
document to what it covers; see [`resources.md`](./resources.md).

The four use-case skills share one shape: a **Fast Path** read, an **Apply
Results** step, a **Report Back** step (the exact commands behind the answer,
and a diagram only when the answer is a shape), and a **Record** flow over the
CLI.

---

## 4. `potpie-graph` v8 — the taught read/write loop

This is the contract skill: it points the agent at the *live* catalog rather than
baking the ontology into prose. The discipline it teaches (full read mechanics in
[querying.md](./querying.md), full write mechanics in [writing.md](./writing.md)):

1. **Discover once, in parallel.** One shared discovery pass across skills:
   `potpie resolve` (the intent is inferred from the task text unless `--intent`
   names one),
   scope-only `preferences_for_scope --repo current` for code work, and an
   untyped `graph search-entities` for a named entity with an unknown key.
   Before ingestion, `graph catalog --profile full` supplies the live ontology
   (entity types, identity policies, predicates, allowed endpoints) — derived
   at runtime, so no docs are needed.
2. **Read** over the fixed view table: `graph read --subgraph <s> --view <v>`.
   **Query expansion is the agent's job** — the bundled local embedder is small,
   so the agent broadens the user's words ("add retry to payments client" → also
   carry "timeout, flaky, backoff, external call") in-session, not the daemon.
   Always inspect `coverage` / `freshness` / `quality` before relying on results.
3. **Resolve identity *before* writing.** `graph search-entities "<name>" --type …
   --source-ref …`, then reuse the returned canonical `key`. Inventing a
   near-duplicate (`service:payments` vs `service:local:payments-api`) fragments
   the graph and breaks future reads.
4. **Write through the canonical two-phase door.** One fix or free-form note
   is one `potpie record` call (`--type`, `--summary`, `--scope`); everything
   structured is `graph propose --file mutation.json` →
   `graph commit <plan_id> --verify` (`--approved-by` for a `review_required`
   plan).
   `graph mutation-template --kind <…>` gives a schema-only skeleton to fill from
   sources actually read. **Never hard-delete** — use validity / retraction /
   supersession / merge. Pick the truth class honestly (it feeds the ranker).
   `graph mutate` exists only as a **legacy wrapper** over propose+commit and the
   skill steers away from it.
5. **Capture uncertainty** that isn't yet a safe fact: `graph inbox add`. Inbox
   items are pending work, never returned as graph facts until processed.
6. **Inspect quality (read-only):** `graph quality {summary | duplicate-candidates
   | stale-facts | conflicting-claims | orphan-entities | low-confidence |
   projection-drift}`. Repair through propose/commit or park in the inbox; quality
   never writes.

> **The one rule the skill emphasizes most:** every entity and claim carries a
> `description` written as a **retrieval card** — the symptoms, synonyms, and
> scope a *future searcher* would type, not display text. Validation only *warns*
> on a weak card, but a vague description means the fact never resurfaces. Weak:
> `"deadlock fix"`. Strong: `"Concurrent refund + settle deadlocks payments DB
> under load; seen as 'refund race timeout'; fixed by ordering lock acquisition
> in services/payments/settle.py"`.

---

## 5. Nudges: the agent's half of the loop

Potpie ships no hook adapter. A harness that wants nudges wires its own
lifecycle hooks to `potpie --json graph nudge --event <e> --session <id>`
and injects the result; the trigger model, the executor, dedup, and the event
mapping a harness should use are owned by
[ingestion-nudge.md](./ingestion-nudge.md).

`potpie-graph` → "Responding To Nudges" teaches what to do with an injected
result:

- **`inject_context`** → ranked graph truth already scoped to the task; use it
  directly rather than re-fetching.
- **`instruction`** (e.g. "you resolved `<error>` after editing `<files>` —
  record the bug+fix if non-obvious") → a *prompt to decide*, **not** an
  auto-write. The agent picks the truth class, resolves identity, writes a
  retrieval-grade description, then propose/commit or inbox. Writes are
  idempotent by `idempotency_key`, so a repeat capture never duplicates.

---

## Server-side reconciliation skills (a separate surface)

`adapters/outbound/reconciliation/skills/` holds **`backfill-enumerate-drain`**
and **`graph-mutation-plan`**. These are **not** part of the user-installed
bundle, are **not** managed by `DefaultSkillManager`, and are **never installed
into a harness**. They belong to the server-side reconciliation deep agent
(`adapters/outbound/reconciliation/pydantic_deep_agent.py`), which loads them via
its own `list_skills` / `load_skill` toolset from `_SKILLS_DIR` and writes
through a **different tool surface**: `apply_graph_mutations(plan, event_id,
summary)`, `mark_event_processed`, and `finish_batch`.

Same philosophy (a model authoring semantic mutations, downgrading to
`RELATED_TO` / `Document` / `Observation` when uncertain, with `domain/ontology.py`
as the entity-key source of truth) — but a distinct agent, tool surface, and
install path.

> **Roadmap (not yet wired):** this reconciliation agent runs only on the
> separate HTTP ingestion-server composition root and is **off by default**
> (`domain/reconciliation_flags.py agent_planner_enabled()` returns False; opt-in
> `CONTEXT_ENGINE_AGENT_PLANNER_ENABLED=1`). The canonical write path remains
> harness-authored semantic mutations + the deterministic record bridge. See
> [ingestion-nudge.md](./ingestion-nudge.md).

---

## See also

- [ingestion-nudge.md](./ingestion-nudge.md) — the zero-token nudge trigger model a harness calls through `graph nudge`, and the server-side reconciliation pipeline.
- [querying.md](./querying.md) — the read mechanics (catalog/read/search-entities and the AgentEnvelope) the skills drive.
- [writing.md](./writing.md) — the propose → commit `--verify` write door, the semantic DSL, inbox, and quality scoring.
- [cli-flow.md](./cli-flow.md) — the full `potpie skills` and `potpie graph …` command/flag reference.
