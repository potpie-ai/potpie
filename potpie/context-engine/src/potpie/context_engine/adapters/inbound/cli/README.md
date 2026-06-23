# Potpie CLI (`potpie`)

The host-routed command-line entrypoint for the context graph. Every command
routes `CLI → HostShell → service(s) → ports`; the in-process `HostShell` is the
single composition root for the agent surface (shared with the MCP server).

- **Entrypoint:** `potpie.context_engine.adapters.inbound.cli.host_cli`
  (registered as the `potpie`
  console script in `pyproject.toml` → `[project.scripts]`).
- **Command groups:** `potpie.context_engine.adapters.inbound.cli.commands` — one module per
  `cli-flow.md` section (`bootstrap`, `query`, `pots`/`source`, `daemon`,
  `ledger`, `graph`, `timeline`, `backend`, `skills`, `cloud`).
- **Cross-cutting contract:** `commands/_common.py` owns `--json` output, the
  exit-code map (0 ok / 1 validation / 2 unavailable / 3 degraded / 4 auth), the
  structured error shape (`code`/`message`/`detail`/`recommended_next_action`),
  and active-pot resolution. An unbuilt capability surfaces as the structured
  not-implemented contract (`CapabilityNotImplemented`), never a traceback.

## The four-tool agent surface

The public agent contract is exactly four tools, each returning an
`AgentEnvelope` (no server-side synthesis):

| Tool / CLI | Use |
|------------|-----|
| `context_resolve` / `potpie resolve` | Primary bounded-context wrap for a task. |
| `context_search` / `potpie search` | Narrow follow-up lookup. |
| `context_record` / `potpie record` | Record a durable learning (decision, fix, preference, …). |
| `context_status` / `potpie status` | Cheap pot/scope readiness + recommended recipe. |

## Authoritative reference

The full command catalog, flags, profiles (local vs managed), and the output
contract live in **[`docs/context-graph/cli-flow.md`](../../../../../../docs/context-graph/cli-flow.md)**.
The end-state architecture (services, ports, composition roots) is in
**[`docs/context-graph/architecture.md`](../../../../../../docs/context-graph/architecture.md)**.
The in-progress Graph V1.5 handover plan is
**[`docs/context-graph/graphv1-5-implementation-plan.md`](../../../../../../docs/context-graph/graphv1-5-implementation-plan.md)**.

Run `potpie --help` (or
`python -m potpie.context_engine.adapters.inbound.cli.host_cli --help`) to list
the live commands.

## Agent harness install

`potpie skills install [<id>] --agent claude` materializes the packaged skill
bundle into an agent harness (`commands/skills.py` → `HostShell.skills`). The
default scope is global, so skills are installed once into the selected
harness's user-level skills directory:

| Harness | Global path |
|---------|-------------|
| Cursor | `~/.cursor/skills/<skill>/SKILL.md` |
| Claude Code | `~/.claude/skills/<skill>/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/<skill>/SKILL.md` |
| Codex | `$HOME/.agents/skills/<skill>/SKILL.md` |

For harnesses with documented file-backed global instructions, install/update
also refreshes a compact Potpie managed block in `~/.claude/CLAUDE.md` and
`~/.codex/AGENTS.md`.

Remove one global skill with `potpie skills remove <id> --agent claude`, or
delete every globally installed Potpie skill for a harness with
`potpie skills remove --all --agent claude`. Use `--scope project --path .` for
repo-local cleanup.

Use `--scope project --path .` for repo-local installs. The bundle keeps the
agent surface to the four tools above and encodes feature / debugging / review /
operations / docs / onboarding workflows as `context_resolve` recipes. Agents
see only an advisory `skills` block in `context_status` with missing/outdated
skills and the exact install command.

## MCP

The MCP server (`potpie-mcp`, stdio) binds to the same in-process `HostShell`
and exposes exactly the four tools above
(`potpie.context_engine.adapters.inbound.mcp.server`).
