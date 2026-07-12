# Potpie CLI

The root `potpie` distribution owns the CLI. The accepted boundary is
[SPEC-PACKAGE-BOUNDARY](../../spec/modules/package-boundary.md); the full command
contract is [cli-flow.md](../../docs/context-graph/cli-flow.md).

## Entrypoint and flow

`pyproject.toml` registers `potpie = potpie.cli.main:main`.

```text
Typer arguments
  → get_runtime()
  → product service or runtime.engine.*
  → root renderer/error mapper
  → Rich output or one JSON object
```

`commands/` binds arguments. It does not construct engines, daemon transports,
auth stores, skill catalogs, or installers.

## Runtime modes

Daemon mode is the product default. `--runtime`, `POTPIE_RUNTIME_MODE`, and the
persisted product setting can select `daemon` or `in-process` in that order. An
unavailable selected daemon produces `RUNTIME_DAEMON_UNAVAILABLE` and recommends
`potpie daemon start`; it never falls back locally.

## Agent context surface

| CLI | MCP equivalent | Runtime call |
|---|---|---|
| `potpie resolve` | `context_resolve` | `runtime.engine.context.resolve` |
| `potpie search` | `context_search` | `runtime.engine.context.search` |
| `potpie record` | `context_record` | `runtime.engine.context.record` |
| `potpie status` | `context_status` data | root `runtime.status` composition |

MCP is root-owned under `potpie/mcp`; it is not implemented in this package or
the engine package as a CLI adapter.

## Command ownership

- `query.py`, `pots.py`, `graph*.py`, `ledger.py`: typed engine workflows.
- `bootstrap.py`: setup, doctor, status, config, and identity product workflows.
- root auth modules: account and provider integration workflows.
- `daemon.py`, `service.py`: product daemon lifecycle.
- `skills.py`: root resource installation and drift.
- `telemetry.py`: root product telemetry controls.
- `output/contracts.py`: JSON success/error envelopes and exit codes.

Run `potpie --help`; the exact hierarchy is pinned by
`tests/unit/test_cli_v1_contract.py` and the skill command manifest.

## JSON

Global `--json` emits exactly one schema-version-1 object to stdout. JSON mode
does not prompt. Logs and tracebacks stay on stderr. Lists use `items`, `count`,
and `next_cursor`; failures use stable error codes and recommended actions.

Exit codes are 0 success, 1 operation failure, 2 validation, 3 runtime
unavailable, 4 auth/permission, 5 degraded/conflict/not-ready, 70 internal, and
130 interruption.

## Skills

`potpie skills ...` materializes resources from
`potpie/skills/resources/templates`. Runtime installation uses a static command
manifest, not Typer introspection. Managed blocks preserve all user-authored
content outside `<!-- potpie-start -->` and `<!-- potpie-end -->`.
