# Setup and commands: choose where agents and graph memory run

| Status | Reviewed | Code |
|---|---|---|
| As-built command reference; setup not executed for this documentation | 2026-09-10 | [Potpie commands](../../potpie/cli/commands/), [Pie commands](../../../pie/apps/cli/src/pie_cli/) |

## Choose a topology

| Use case | Install/run locally | Run on a server |
|---|---|---|
| Terminal agent with local memory | `potpie[all]`, setup, daemon, your agent harness | Nothing required for graph storage |
| Terminal agent with shared memory | Bare `potpie`, managed URL/key, your harness | Managed context graph and its stores |
| Pie VS Code + Copilot + shared memory | VSIX, bundled Pie runtime, CLI tools, Copilot login | Managed context graph; Copilot provider handles inference |
| Pie web application | Browser; optional local development stack | Platform API, worker, Postgres and execution runtime |

Python package metadata requires Python `>=3.12,<3.15`. The local backend's wheel/platform constraints differ from the remote-only client; the [root README](../../README.md#install-and-setup-potpie) documents the supported install variants. The commands below use a POSIX shell; the packaged extension handles platform-specific installation.

## Local Potpie

```bash
# Published-package route; choose this OR the editable route below.
uv tool install 'potpie[all]'
potpie setup --repo . --agent claude
potpie --host local daemon status
potpie --host local status
potpie --host local doctor
potpie ui
```

`setup` provisions config, local storage, daemon, default pot and agent guidance, and can register the repository. It does not by itself complete a repository baseline scan. Open the configured harness and ask it to build or refresh context from repository evidence.

For development from this Potpie checkout, the editable install builds the explorer frontend and stops the previous daemon. It modifies the global tool installation and needs `uv` plus Node/npm:

```bash
make cli-install
make cli-status
potpie setup
```

## Connect to a managed graph

```bash
# For a machine that only needs the remote graph client.
uv tool install potpie
potpie setup --remote https://context.example.com --token '<managed-key>'
potpie host list
potpie --host managed pot list
potpie --host managed graph catalog

potpie pot use managed:team-project
potpie --json graph read --subgraph decisions --view active_decisions --pot managed:team-project
```

The URL and key above are placeholders. For an existing installation, use `potpie host set URL --token KEY` followed by `potpie host use managed`; `host set` probes before persisting by default. Use `host list` to verify the effective origin and environment overrides. Account `potpie login` does not configure this connection. `potpie host use local` returns to the local origin without graph synchronization.

To create and scope a new shared project, once connected:

```bash
potpie --host managed pot create team-project --use
potpie --host managed source add repo .
potpie --host managed status
```

## Run the managed backend from the sibling Pie checkout

In a terminal whose current directory is `../pie` relative to this repository:

```bash
task py:sync
task cg:run
```

This starts `127.0.0.1:8090` with development defaults unless overridden: in-memory graph, memory metadata stores, and disabled authentication. In another terminal, connect using the host registry:

```bash
potpie host set http://127.0.0.1:8090
potpie host use managed
potpie --host managed pot create demo --use
potpie --host managed graph catalog
```

The no-token example is only for this auth-disabled development profile. For persistence and shared access configure the service before starting it:

| Setting | Required role |
|---|---|
| `PIE_CONTEXT_GRAPH_BACKEND_PROFILE=falkordb` | Server graph backend; `task cg:run:falkordb` can start a local development instance |
| `PIE_CONTEXT_GRAPH_FALKORDB_URL` | Address of that server |
| `PIE_CONTEXT_GRAPH_STORE_PROFILE=postgres` | Durable tenancy, plans, inbox, audit and usage |
| `PIE_CONTEXT_GRAPH_DATABASE_URL` | Postgres DSN; apply the service migration before serving |
| `PIE_CONTEXT_GRAPH_AUTH_MODE=static_api_key` | Authenticate callers |
| `PIE_CONTEXT_GRAPH_API_KEYS` | Configured key-to-actor/org mappings in `secret:actor_id:org_id` format |
| `PIE_CONTEXT_GRAPH_RESOURCES_ROOT` | Persistent document volume, default `/data/resources` |

The [managed service configuration](../../../pie/services/context-graph/src/pie_context_graph/config.py) and [migration](../../../pie/services/context-graph/src/pie_context_graph/adapters/outbound/postgres/migrations/0001_init.sql) are the implementation references. Graph and metadata persistence are separate settings. `task` loads the repository env files; bare `uv run python -m pie_context_graph` reads the process environment. Align the server and client revisions before relying on document upload; see [the observed mismatch](hosts-and-storage.md#compatibility-and-unfinished-surfaces).

## Pie VS Code and Copilot

1. Install the intended packaged VSIX with `code --install-extension /path/to/potpie.vsix`, then open a workspace folder.
2. Complete the Potpie extension onboarding. Its packaged build supplies runtime/tool artifacts and configures the agent CLIs; check the installed build in **Potpie: Show Logs**.
3. Set `pie.managedHost` to the graph service and supply its key through the extension's managed-host setup. Select an accessible pot. `pie.copilotHost` independently selects the Copilot GitHub host, if a non-default host is required.
4. Install/update Copilot through the extension's harness controls and complete its login. The extension pins its expected CLI build; prefer that build over an arbitrary latest version.
5. Submit a task. The extension starts its workspace `pie serve`, opens gRPC, and starts the selected harness session. Inspect the displayed graph actions or run `potpie host list` in its integrated terminal to confirm the graph destination.

For a **source-development** Pie terminal, after dependency sync and Copilot installation/login (the extension normally manages `pie serve` itself):

```bash
uv run pie harness list
uv run pie harness setup copilot --path .
uv run pie serve --root .

# Alternative: drive an agent directly from the terminal.
uv run pie run "Explain the service architecture" --harness copilot --interactive
```

Run these in the sibling Pie repo; `--path`/`--root` can name another workspace. `harness setup` can write guidance; `--no-bundle` probes without that write. Packaged VS Code usually uses the per-machine bundle. `pie run` drives an agent task; `potpie resolve` only retrieves context.

## Everyday command map

| Intent | Command |
|---|---|
| Check binary and routing | `potpie --version`; `potpie host list` |
| Choose a project | `potpie pot list`; `potpie pot use local:project` |
| Check context readiness | `potpie status`; `potpie doctor` |
| Connect source access | `potpie github login`; `potpie linear login`; `potpie auth status --verify` |
| Register repository provenance | `potpie source add repo .` |
| Retrieve before a task | `potpie --json resolve "Explain authentication"` |
| Search project memory | `potpie --json search "authentication flow"` |
| Discover precise graph reads | `potpie --json graph catalog`; `potpie graph describe knowledge --examples` |
| Inspect an entity's links | `potpie --json graph neighborhood --entity service:example --depth 2 --detail full` |
| Write a durable learning | `potpie record --type decision --summary "Use the existing API boundary"` |
| Review and apply semantic writes | `potpie graph propose --file mutation.json`; `potpie graph commit PLAN_ID --verify` |
| Find graph maintenance work | `potpie graph inbox list`; `potpie graph quality summary` |
| Import extracted document chunks locally | `potpie --host local resource import ./chunks --doc architecture --source-ref docs/architecture.md` |
| Inspect document retrieval readiness | `potpie resource index status`; `potpie resource index build --wait` |
| Read document passages | `potpie --json search "deployment" --include resources` |
| Open the browser graph explorer | `potpie ui` (requires local daemon/UI assets) |

Replace example keys and plan IDs with returned identities. Use `--pot` qualifiers on reads and writes in scripts. The mutation and record examples change graph state; the document import expects an already extracted chunk directory with metadata, not an arbitrary PDF or Markdown file.

| Location / check | Meaning |
|---|---|
| `~/.potpie/cli_hosts.json` | Local/managed origin and managed credential; do not paste its contents into reports |
| `~/.potpie/discovery.json`, `daemon.pid`, `logs/potpied.log` | Local daemon endpoint, lifecycle and logs |
| `<potpie-home>/workspaces/<workspace-key>/` | Pie `serve.endpoint`, `serve.info`, `pie.db`, and `runs/`; overridden by `PIE_STATE_DIR` |
| `~/.pie/copilot/` | Extension-managed Copilot instructions and skills |
| `potpie daemon status` reports stale build | Installed CLI and running daemon differ; restart through daemon lifecycle commands |
| Managed graph reads work, resource upload fails | Check client/server import signatures and pinned revisions, not only credentials |
| Editor works but graph is wrong/empty | Check origin, pot, source registration and Pie's cached/direct context path |

See [Windows packaging and runtime](windows-packaging-and-runtime.md) for the reviewed Windows executables, `.pie` installation directories, user PATH and repair flow; [runtime anatomy](../../../pie/docs/vscode-vsix-runtime-anatomy.md) retains earlier build details.
