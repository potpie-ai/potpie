# Context-Engine CLI Sentry Telemetry

Sentry is used only for unexpected Potpie context-engine CLI failures. It is a
small CLI-owned integration under `potpie/context-engine`, independent of
the context-engine telemetry path and `legacy/deploy/observability`.

## Configuration

The CLI root callback resolves runtime settings through
`bootstrap.runtime_settings`. A project `.env` file is read only when the
bootstrap environment is `dev`; non-dev environments read the process
environment and packaged distribution defaults only. `.env` values fill missing
process env keys and cannot set or change `POTPIE_ENVIRONMENT`.

Environment precedence:

- process environment
- `.env` values, only when `POTPIE_ENVIRONMENT` resolves to `dev` and only for
  missing keys
- distribution defaults packaged into the wheel
- code defaults

Canonical Sentry configuration:

- `POTPIE_ENVIRONMENT`: Sentry environment and telemetry event environment.
- `POTPIE_SENTRY_DSN`: enables Sentry when present.
- `POTPIE_TELEMETRY_DISABLED=1`: disables all outbound telemetry.
- `POTPIE_SENTRY_ENABLED=0`: disables Sentry only.
- `POTPIE_SENTRY_RELEASE`: optional release override; otherwise
  `potpie-cli@<potpie-context-engine version>`.
- `POTPIE_SENTRY_DIST`: optional dist override; otherwise the generated build
  Git SHA when available.

Generic `SENTRY_*` aliases are not read. Distribution defaults are packaged
public defaults for installed wheels, not production environment variables or
secrets.

Sentry initializes directly through `sentry-sdk` with:

- `send_default_pii=False`
- `include_local_variables=False`
- `max_request_body_size="never"`
- `before_send` event scrubbing
- `before_breadcrumb` breadcrumb scrubbing

The CLI never calls `sentry_sdk.set_user()`.

## Identity State

Reusable non-secret telemetry identity is stored globally:

```text
$XDG_CONFIG_HOME/potpie/telemetry/identity.json
~/.config/potpie/telemetry/identity.json
```

The file is written with `0600` permissions through an atomic temp-file replace.
It contains:

- `schema_version`
- `anonymous_install_id`
- `created_at`
- `last_seen_at`

The identity file never stores DSNs, auth tokens, API keys, user IDs, org IDs,
repo names, prompts, code, paths, headers, or request bodies. The
`anonymous_install_id` is stable until an explicit reset flow exists. Each CLI
run also gets an in-memory `invocation_id`; each process gets an in-memory
`daemon_session_id`.

## Captured Errors

Captured:

- unexpected exceptions crossing `commands/_common.py::contract()`
- unexpected auth implementation failures in login/logout flows that do not use
  `contract()`
- unexpected daemon command failures through the normal CLI command boundary

Not captured:

- `typer.Exit`
- `KeyboardInterrupt`, EOF, or user cancellation
- validation errors
- expected domain errors such as `pot_not_found`, `no_active_pot`,
  `context_engine_unavailable`, and `not_implemented`
- auth denied, expired, or missing credentials
- missing local config
- missing DSN or disabled telemetry

Captured events use stable metadata only:

- `service`
- `command`
- `subcommand` when available
- `output_mode`
- `cli_version`
- `python_version`
- `os`
- `arch`
- `error.code`
- `error.kind`
- `is_expected=false`

Allowed event context:

- `anonymous_install_id`
- `invocation_id`
- `daemon_session_id`

## Privacy Scrubbing

Telemetry code avoids attaching sensitive data in the first place. Sentry SDK
privacy settings and hooks are the final guard before transport.

`before_send` removes:

- request data
- headers
- cookies
- env
- extra payloads
- module lists
- server name
- stack frame locals
- full frame paths
- raw exception messages and values

`before_breadcrumb` drops HTTP/subprocess breadcrumbs and strips breadcrumb
`data` and `message` from kept breadcrumbs.

Sentry events must not contain:

- email, name, user ID, or org ID
- prompts or episode bodies
- source code or file contents
- terminal output
- repository names
- git remotes
- full local paths
- URLs with query strings
- request or response bodies
- headers
- environment variables
- API keys, GitHub tokens, bearer tokens, or secrets
- frame locals
- raw exception messages or values
- raw CLI command arguments

## Local Verification

Default tests use fake or in-memory Sentry surfaces and do not call the network.

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --package potpie-context-engine pytest potpie/context-engine/tests/unit/test_sentry_*.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run --package potpie-context-engine pytest potpie/context-engine/tests/unit/test_telemetry_*.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run --package potpie-context-engine ruff check adapters/inbound/cli tests/unit/test_sentry_*.py tests/unit/test_telemetry_*.py
```

Manual CLI smoke:

```bash
XDG_CONFIG_HOME=/tmp/potpie-xdg \
UV_CACHE_DIR=/private/tmp/uv-cache \
uv run --package potpie-context-engine potpie --json daemon status
```

Expected result: command exits `0`, prints daemon status JSON, and creates
`/tmp/potpie-xdg/potpie/telemetry/identity.json` without requiring a Sentry DSN.
