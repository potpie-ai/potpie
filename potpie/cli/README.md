# Potpie CLI (`potpie`)

The command-line entrypoint for the context graph. Context operations route
through the typed engine client; product operations use explicit Potpie-owned
services from one local runtime composition.

- **Entrypoint:** `potpie/cli/main.py` (registered as the `potpie`
  console script in `pyproject.toml` → `[project.scripts]`).
- **Command groups:** `potpie/cli/commands/` — one module per
  `cli-flow.md` section (`bootstrap`, `query`, `pots`/`source`, `host`,
  `daemon`, `ledger`, `graph`, `timeline`, `backend`, `skills`, `cloud`).
- **Cross-cutting contract:** `commands/_common.py` owns `--json` output, the
  exit-code map (0 ok / 1 validation / 2 unavailable / 3 degraded / 4 auth), the
  structured error shape (`code`/`message`/`detail`/`recommended_next_action`),
  origin routing (local or managed host) and active-pot resolution. An unbuilt
  capability surfaces as the structured not-implemented contract
  (`CapabilityNotImplemented`), never a traceback.

## Agent compatibility commands

The CLI exposes four compatibility commands with distinct response contracts:

- `potpie resolve` and `potpie search` return an `AgentEnvelope` with no
  server-side synthesis.
- `potpie record` returns a record receipt containing `status`, `record_id`,
  and `mutations_applied`.
- `potpie status` returns readiness information for the selected pot and scope,
  including the recommended recipe.

| Command | Use |
|------------|-----|
| `potpie resolve` | Primary bounded-context wrap for a task. |
| `potpie search` | Narrow follow-up lookup. |
| `potpie record` | Record a durable learning (decision, fix, preference, …). |
| `potpie status` | Cheap pot/scope readiness + recommended recipe. |

Integration credential status is intentionally outside this four-tool surface:
use `potpie auth status [--verify]` for local provider auth state.

## Managed hosts

The same commands run against a hosted context graph. The local daemon needs no
configuration; a managed host needs its base URL and an API key, stored in
`<home>/cli_hosts.json` (`0600`) by `potpie host set`:

```bash
printf '%s\n' "$KEY" | potpie host set https://your-host.example --token -
potpie host use managed          # make it the active host
potpie pot list --managed
potpie pot use managed:<id> --also-default-for-current-repo
potpie host use local            # back to this machine
```

- The key never travels on argv: `--token -` reads it from stdin, and a literal
  `--token <key>` is refused. `POTPIE_MANAGED_URL` / `POTPIE_MANAGED_TOKEN`
  override the stored host for one run.
- One command's host is the prefix of a qualified ref (`managed:<ref>`,
  `local:<ref>`), else the global `--host local|managed`, else the active host.
  A bare ref resolves on that host only; when it is not there, the error names
  the other host's match.
- Engine commands on a managed host go through `RemoteEngineClient`
  (`potpie.runtime.remote`); pot, source, grant, `whoami` and `status` commands
  go through the host's `PotDirectory` (`potpie.pots.directory`). `daemon`,
  `skills`, `setup`, integration `auth` and harness configuration always stay
  on this machine.
- `potpie ui --pot managed:<id>` opens the explorer served by the local daemon
  with `?host=managed`; the daemon holds the key, the browser never sees it.
- A host older than this client answers `not_implemented` (exit `2`) for what
  it lacks; everything else keeps working.
- Python callers use `potpie.runtime.connect()`: `HostConnection.engine(selector)`,
  `.pots` (synchronous, any thread) and `.extension(name, request, result_type)`;
  synchronous code runs them with `.run(awaitable)`.

The full contract is in
[`docs/context-graph/cli-flow.md`](../../docs/context-graph/cli-flow.md#hosts-commandshostpy).

## Authoritative reference

The full command catalog, flags, profiles (local vs managed), and the output
contract live in **[`docs/context-graph/cli-flow.md`](../../docs/context-graph/cli-flow.md)**.
The end-state architecture (services, ports, composition roots) is in
**[`docs/context-graph/architecture.md`](../../docs/context-graph/architecture.md)**.

Run `potpie --help` (or `python -m potpie.cli.main --help`) to list
the live commands.

## Local install (this repo)

Repo-local development installs the CLI with:

```bash
make cli-install
make cli-status
```

That path builds the graph-explorer UI, stops any old daemon, and installs the
editable package. Published-package users should use `uv tool install potpie` or
`pip install potpie` instead.

## Agent harness install

`potpie skills install [<id>] --agent claude` materializes the packaged skill
bundle into an agent harness through the explicit root skill service. The
default scope is global, so skills are installed once into the selected
harness's user-level skills directory.

The shipped templates live under `potpie/cli/templates/`, with one source for
each kind of file:

- `agent_bundle/.agents/skills/` — every skill. All harnesses install their
  skills from here, remapped to the harness's own skills directory.
- `routing/POTPIE.md` — the compact instruction block merged into a harness's
  `AGENTS.md` or `CLAUDE.md`, globally and per repository.

| Harness | Global path |
|---------|-------------|
| Cursor | `~/.cursor/skills/<skill>/SKILL.md` |
| Claude Code | `~/.claude/skills/<skill>/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/<skill>/SKILL.md` |
| Codex | `$HOME/.agents/skills/<skill>/SKILL.md` |

`POTPIE_HARNESS_HOME` moves these roots away from the real home directory (the
test suite pins it); `CONTEXT_ENGINE_HOME` deliberately does not, because a
harness keeps reading its own home whatever Potpie's state directory is.

For harnesses with documented file-backed global instructions, a bundle install
or update (no skill id) also refreshes the compact Potpie managed block in
`~/.claude/CLAUDE.md` and `~/.codex/AGENTS.md`; naming one skill id installs
only that skill. Existing user-authored content is preserved; Potpie only
appends or updates the `<!-- potpie-start -->` / `<!-- potpie-end -->` managed
section, and `skills remove --all` takes that section back out.

Remove one global skill with `potpie skills remove <id> --agent claude`, or
delete every globally installed Potpie skill for a harness with
`potpie skills remove --all --agent claude`. Use `--scope project --path .` for
repo-local cleanup. A skill removed by id is remembered as disabled, so a later
bundle install skips it until it is installed again by id.

`potpie --json skills status --agent <harness>` lists `installed`, `missing`,
`outdated`, `drifted` (installed, but its files no longer match the bundle — a
subset of `outdated` with the same repair) and `disabled`. `--path` is resolved
against the caller's working directory and must already exist.

Use `--scope project --path .` for repo-local installs. The bundle teaches
feature / debugging / review / operations / docs / onboarding workflows over
the CLI graph surface. Agents see an advisory `skills` block in `potpie status` with missing/outdated
skills and the exact install command. Repo-local `AGENTS.md` and `CLAUDE.md`
files are merged the same way as global instruction files, so setup does not
replace existing agent instructions.
