# Potpie CLI Flow and Command Contract

> Updated on 2026-07-13 for the accepted version 2.0.0 command vocabulary.
> Commit-scoped verification is pending. This document describes the live
> workflow-first CLI established by
> [SPEC-PACKAGE-BOUNDARY](../../spec/modules/package-boundary.md).

## Execution flow

```text
argv
  → Typer binding in potpie/cli/commands
  → get_runtime()
  → runtime product service or runtime.engine capability
  → root output/error mapping
  → Rich human output or one JSON envelope
```

Handlers bind arguments and invoke the runtime. Product auth, setup, config,
skills, install, status, and daemon commands call sibling services on
`PotpieRuntime`. Context, pot, source, graph, ledger, and timeline commands call
`runtime.engine.*`.

The CLI never constructs `ContextEngine`, a daemon transport, a keyring, or an
installer directly.

## Global runtime selection

Daemon mode is the default. Precedence is:

1. global `--runtime daemon|in-process`;
2. `POTPIE_RUNTIME_MODE`;
3. persisted product setting;
4. `daemon`.

If daemon mode is selected and the daemon is unavailable, engine commands exit
with code 3 and `RUNTIME_DAEMON_UNAVAILABLE`; the recommended action is
`potpie daemon start`. There is no implicit in-process fallback.

## Command tree

```text
potpie resolve
potpie search
potpie record
potpie status

potpie setup
potpie doctor
potpie login
potpie logout
potpie whoami
potpie ui

potpie pot list|info|create|use|linked|rename|reset|archive|default
potpie source add|list|status|remove
potpie timeline recent

potpie graph catalog|read|search-entities|mutation-template|nudge|status
potpie graph propose|commit|history|inspect|export|import|repair
potpie graph inbox add|claim|close|list|mark-applied|mark-rejected|show
potpie graph quality conflicting-claims|duplicate-candidates|low-confidence
potpie graph quality orphan-entities|projection-drift|stale-facts|summary
potpie graph bulk apply
potpie graph store list|status|use|doctor

potpie ledger status|query|use|disconnect|pull|sources

potpie integration list
potpie integration status [PROVIDER]
potpie integration github login|logout|list
potpie integration linear login|logout|list|select
potpie integration jira login|logout|list|select
potpie integration confluence login|logout|list|select

potpie skills list|install|update|remove|status|add

potpie daemon start|status|logs|restart|stop
potpie daemon service up|down|status|logs

potpie config list|get|set
potpie telemetry status|enable|disable
```

`tests/unit/test_cli_v1_contract.py` and the checked-in skill command manifest
pin this tree exactly.

## Canonical workflows

### First setup

```bash
potpie setup --repo . --agent codex
potpie status
potpie source list
```

Setup resolves product settings, checks auth/integrations, reconciles the
daemon, provisions the engine, installs root-owned skills, and reports each
step. It is idempotent. Use `--dry-run` to inspect the planned steps and `--yes`
for noninteractive confirmation.

### Context read and record

```bash
potpie resolve "understand the login flow"
potpie search "OAuth callback validation"
potpie record --type decision --summary "MCP belongs to root Potpie"
```

These map to `runtime.engine.context.resolve/search/record` with typed DTOs.

### Pot and source scope

```bash
potpie pot list
potpie pot use <id-or-name>
potpie pot default --set <pot-id>
potpie source add repo .
potpie source status --pot <pot-id> <source-id>
```

Commands resolve or require a pot before graph work. Product data locations and
existing pot/source formats are unchanged.

### Graph reads and writes

```bash
potpie graph catalog --subgraph infra_topology
potpie graph read --subgraph infra_topology --view service_neighborhood
potpie graph search-entities "auth service"

potpie graph propose --file mutation.json
potpie graph commit <plan-id> --verify
potpie graph history
```

Every graph write uses propose then commit. Contract discovery uses catalog;
semantic neighborhoods use named read views and the existing depth/direction
controls.

### Integrations

```bash
potpie login
potpie whoami
potpie integration list
potpie integration github login
potpie integration github list
potpie integration status github
```

Provider authentication and selection are root product operations. Credentials
do not enter engine configuration or daemon RPC.

### Skills

```bash
potpie skills list
potpie skills install --agent codex
potpie skills update --all --agent codex
potpie skills status --agent codex
```

Skill installation uses resources packaged in the root wheel and a static
command manifest validated during tests/build.

## JSON contract

Global `--json` writes exactly one JSON object to stdout. Logs and diagnostics
go to stderr. JSON mode never prompts.

Success:

```json
{
  "ok": true,
  "data": {},
  "meta": {
    "schema_version": "1",
    "command": "graph.read",
    "runtime_mode": "daemon",
    "request_id": "uuid-or-null"
  }
}
```

Failure:

```json
{
  "ok": false,
  "error": {
    "code": "RUNTIME_DAEMON_UNAVAILABLE",
    "message": "The Potpie daemon is not reachable.",
    "details": {},
    "retryable": true,
    "recommended_next_action": {
      "command": "potpie daemon start",
      "reason": "The configured runtime mode is daemon."
    }
  },
  "meta": {
    "schema_version": "1",
    "command": "status",
    "runtime_mode": "daemon",
    "request_id": null
  }
}
```

List data is always:

```json
{"items": [], "count": 0, "next_cursor": null}
```

`ok` describes command execution. A health command can execute successfully,
return degraded data, and still use a nonzero health exit.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | success or healthy |
| 1 | expected operation failure |
| 2 | usage or validation failure |
| 3 | selected runtime unavailable |
| 4 | authentication or permission failure |
| 5 | degraded, conflict, or not ready |
| 70 | unexpected internal error |
| 130 | interruption |

## Clean-break migration

| Removed path | Current path or behavior |
|---|---|
| `potpie use` | `potpie pot use` |
| top-level `github`, `linear`, `jira`, `confluence` | `potpie integration <provider>` |
| `potpie auth` | `potpie integration status`, provider groups, `login`, `whoami` |
| `github repos`, provider `ls` | provider `list` |
| `potpie backend` | `potpie graph store` |
| `potpie service` | `potpie daemon service` |
| `potpie cloud` | removed; unfinished push/pull/skills operations have no replacement |
| `potpie graph mutate` | `graph propose` then `graph commit` |
| `potpie graph describe` | `graph catalog --subgraph ...` |
| `potpie graph neighborhood` | named `graph read` view |

No deprecated alias is registered. Removed paths fail as unknown commands.

## MCP relationship

The root `potpie-mcp` process exposes exactly `context_resolve`,
`context_search`, `context_record`, and `context_status`. Resolve/search/record
call the same runtime engine methods as the CLI. Status calls the same root
product status service and returns the same flat data. MCP uses its native
result protocol, not the CLI JSON envelope.
