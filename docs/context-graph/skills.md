# Skills & Harness-Led Intelligence

> Status: reflects code on `main` @ `8dd175bc`, last reviewed 2026-06-29.

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

`adapters/outbound/skills/bundle_catalog.py` scans those templates at runtime
(`lru_cache`d), parses each file's YAML front-matter (`name` / `version` /
`description`, optional `recommended: false`) into a `SkillInfo`, and exposes
`catalog_by_id()` and `RECOMMENDED_SKILL_IDS` (every recommended bundled skill).
Adding or editing a skill means editing the bundled markdown — nothing else.

There are **8 skills in the agent bundle**; the Claude Code plugin ships **7** of
them (everything except `potpie-cli`).

## 2. Installation, targets & drift (`DefaultSkillManager`)

`application/services/skill_manager.py DefaultSkillManager` owns the catalog +
per-harness install/drift logic and delegates *where/how* to a registered
`AgentTargetPort` per harness. Operations: `list / install / update / remove /
status / nudge / add` (`add` is a TODO stub).

Targets are wired in `bootstrap/host_wiring.py`; each `FileBackedAgentTarget`
installs into a harness-specific **global** skills root, and a `--scope project`
install routes through `ProjectAgentTarget` instead:

| Harness (`--agent`) | Target | Global skills root |
|---|---|---|
| `claude` | `ClaudeAgentTarget` | `~/.claude/skills` (instructions in `~/.claude`) |
| `codex` | `CodexAgentTarget` | `~/.agents/skills` |
| `cursor` | `CursorAgentTarget` | `~/.cursor/skills` |
| `opencode` | `OpenCodeAgentTarget` | `~/.config/opencode/skills` |

Install mechanics (`adapters/outbound/skills/agent_installer.py`):

- Templates are copied/remapped per harness layout: `claude → .claude/skills`,
  `cursor → .cursor/skills`, `opencode → .opencode/skills`, and the Claude Code
  plugin (`claude-plugin`) installs as one self-contained directory under
  `.claude/potpie-plugin/` so its `.claude-plugin/plugin.json` stays the plugin
  root.
- `AGENTS.md` / `CLAUDE.md` are **merged**, not overwritten — managed content
  lives between `<!-- potpie-start -->` / `<!-- potpie-end -->` markers
  (`_merge_managed_markdown`), preserving the user's own instructions.
- **Drift tracking:** each target writes a JSON manifest (`skills_<agent>_<scope>.json`)
  recording the installed version. `status()` partitions skills into
  installed / missing / outdated; `nudge()` emits the single advisory command
  `potpie skills install --agent <agent>`. That advisory is the *only* skill
  signal agents ever see — it rides on `context_status` (see
  [querying.md](./querying.md)). Globally-installed skills can stale (an old
  `potpie-graph` that still teaches the legacy `graph mutate` predates v5);
  `potpie skills update --agent <a>` is the fix the drift nudge points at.

### The correctness gate (this is real, not aspirational)

Before **every** install/update, `validate_packaged_skill_command_snippets`
extracts every `potpie …` line from the skill's ```` ```bash ```` fences and
validates each command + option against the **live Typer specs introspected from
`potpie.cli.main.app`**. A skill physically cannot ship a `potpie`
command or flag that does not exist — the install raises `ValueError` first.
(Only `potpie` lines are checked; other shell commands depend on the user's repo.)

### CLI surface

```bash
potpie skills list   [--agent claude|codex|cursor|opencode] [--scope global|project] [--path]
potpie skills install [<id>] [--agent …] [--scope …] [--path]
potpie skills update  [--all] [--agent …]
potpie skills status  [--agent …]
potpie skills remove  [<id>|--all] [--agent …]
potpie skills add     <source>        # TODO stub
```

`--scope` flips to `project` automatically when `--path` is given with `global`.
`potpie setup --agent <harness>` installs the recommended bundle during first
run. **There is no top-level `potpie install`** — skills install only via
`potpie skills install` (and `setup`). Full flags live in
[cli-flow.md](./cli-flow.md).

---

## 3. The 8 core bundle skills

| Skill | Ver | Role |
|---|---|---|
| `potpie-cli` | v2 | The `potpie` command itself: pot-scope resolution order, harness-led boundaries. (Not shipped in the plugin.) |
| **`potpie-graph`** | **v5** | **THE contract skill** — the read → resolve → propose/commit → inbox → quality loop, truth classes, retrieval-grade descriptions, and "Responding To Nudges". Teaches **propose/commit only** (never the legacy `graph mutate`). Bundle and plugin copies are byte-identical. |
| `potpie-repo-baseline` | — | Deep repo-baseline mode: source priority, evidence matrix, canonical entity families with `PROVIDES` / `IMPLEMENTED_IN`. |
| `potpie-source-ingestion` | — | Todo-driven, phased (0–8) ingestion of a repo/PR/ticket/doc; parallel read-only subagents; GitHub/Linear/Jira hydrated via the agent's **own** integration tools (explicitly *not* Potpie connector queueing) → evidence matrix → identity resolution → propose/commit `--verify` → quality gate. |
| `potpie-project-preferences` | — | Use-case read+record skill (preferences). |
| `potpie-infra-architecture` | — | Use-case read+record skill (infra/topology). |
| `potpie-change-timeline` | — | Use-case read+record skill (recent changes). |
| `potpie-debug-memory` | — | Use-case read+record skill (prior bugs/fixes). |

The four use-case skills share one shape: a **Fast Path** read, an **Apply
Results** step, and a **Record** flow over the CLI.

---

## 4. `potpie-graph` v5 — the taught read/write loop

This is the contract skill: it points the agent at the *live* catalog rather than
baking the ontology into prose. The discipline it teaches (full read mechanics in
[querying.md](./querying.md), full write mechanics in [writing.md](./writing.md)):

1. **Discover the live contract.** `graph status`, then
   `graph catalog --task "<task>" --profile read`, then
   `graph describe <subgraph> --view <view> --examples`. Governing rule:
   *"Trust the catalog's current operation partition over any example in a skill
   file."* The contract (versions, views, ops, truth classes, match_mode) is
   derived from the ontology at runtime — no docs needed.
2. **Read** over the fixed view table: `graph read --subgraph <s> --view <v>`.
   **Query expansion is the agent's job** — the bundled local embedder is small,
   so the agent broadens the user's words ("add retry to payments client" → also
   carry "timeout, flaky, backoff, external call") in-session, not the daemon.
   Always inspect `coverage` / `freshness` / `quality` before relying on results.
3. **Resolve identity *before* writing.** `graph search-entities "<name>" --type …
   --source-ref …`, then reuse the returned canonical `key`. Inventing a
   near-duplicate (`service:payments` vs `service:local:payments-api`) fragments
   the graph and breaks future reads.
4. **Write through the canonical two-phase door.**
   `graph propose --file mutation.json` → `graph commit <plan_id> --verify`.
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

## 5. The Claude Code plugin (hooks + slash commands)

`potpie/cli/templates/claude_plugin/` is a self-contained Claude Code
plugin: `.claude-plugin/plugin.json` + `.claude-plugin/marketplace.json` declare
it, and it carries its own copy of the 7 plugin skills under `skills/`.

### Five model-free lifecycle hooks

`hooks/hooks.json` wires five lifecycle hook entries to one thin, fail-safe
adapter, `hooks/potpie_nudge.py`:

| Harness event | Matcher | Adapter event |
|---|---|---|
| `SessionStart` | — | `session_start` |
| `PreToolUse` | `Write\|Edit\|MultiEdit\|NotebookEdit` | `pre_edit` |
| `PreToolUse` | `Bash` | `bash_pre` → `pre_deploy` (only on deploy markers) |
| `PostToolUse` | `Bash` | `bash_post` → `test_failed` / `test_passed` (only on test markers) |
| `Stop` | — | `stop` |

The hook **never reasons.** It maps the harness event to a `NudgeEvent`, shells
`potpie --json graph nudge`, and renders the result as Claude
`hookSpecificOutput.additionalContext` (or a `systemMessage` at `Stop`). Any
error, missing binary, or unparseable payload → exit 0 with no output. The nudge
trigger model, the executor, and dedup are owned by
[ingestion-nudge.md](./ingestion-nudge.md).

### The agent's half of the loop

`potpie-graph` → "Responding To Nudges" teaches what to do with an injected
result:

- **`inject_context`** → ranked graph truth already scoped to the task; use it
  directly rather than re-fetching.
- **`instruction`** (e.g. "you resolved `<error>` after editing `<files>` —
  record the bug+fix if non-obvious") → a *prompt to decide*, **not** an
  auto-write. The agent picks the truth class, resolves identity, writes a
  retrieval-grade description, then propose/commit or inbox. Writes are
  idempotent by `idempotency_key`, so a repeat capture never duplicates.

### Two slash commands

| Command | Purpose |
|---|---|
| `/potpie-feature` | Load Potpie context **before** feature work (reads preferences, decisions, and the infra neighborhood). |
| `/potpie-record` | Record durable learnings **after** useful work (resolve identity → propose/commit `--verify`). |

> **Roadmap (not yet wired):** the Claude Code plugin has no first-class CLI
> install path. The bundle `claude-plugin` agent type exists
> (`install_agent_bundle(agent="claude-plugin")` lays it under
> `.claude/potpie-plugin/`), but no `potpie` command invokes it and the plugin's
> own README still references a non-existent `potpie install`. Install today is
> manual (`/plugin marketplace add`); folding it into the managed install/drift
> path is pending.

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

- [ingestion-nudge.md](./ingestion-nudge.md) — the zero-token nudge trigger model the plugin hook fires, and the server-side reconciliation pipeline.
- [querying.md](./querying.md) — the read mechanics (catalog/read/search-entities and the AgentEnvelope) the skills drive.
- [writing.md](./writing.md) — the propose → commit `--verify` write door, the semantic DSL, inbox, and quality scoring.
- [cli-flow.md](./cli-flow.md) — the full `potpie skills` and `potpie graph …` command/flag reference.
