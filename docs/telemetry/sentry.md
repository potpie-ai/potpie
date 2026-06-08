# Sentry Crash Telemetry

Sentry is used for unexpected Potpie CLI and current in-process daemon failures.
Product analytics stays separate, and backend route capture plus worker/Celery
capture are out of scope for this ticket set.

## Configuration

Sentry is env-driven. No DSN is checked in.

- `SENTRY_DSN`: enables CLI Sentry when present.
- `POTPIE_SENTRY_ENABLED=0`: hard-disable kill switch, even with a DSN.
- `SENTRY_ENVIRONMENT`: `dev`, `staging`, or `prod`; local defaults to `dev`.
- `SENTRY_RELEASE`: overrides release naming.
- `SENTRY_DIST`: optional Sentry dist.

Default CLI release naming is `potpie-cli@<potpie-context-engine version>`.
The SDK is configured with `send_default_pii=False`.

## Captured Errors

Captured:

- uncaught exceptions crossing the CLI command boundary
- unexpected in-process daemon command failures
- unexpected auth implementation failures
- serialization, SDK/client response, and credential-store implementation bugs

Not captured:

- validation errors
- `pot_not_found`
- `no_active_pot`
- `auth_required`
- `auth_denied`
- `auth_expired`
- `context_engine_unavailable`
- `not_implemented`
- missing local config, DSN, or token
- user cancellation
- `typer.Exit`
- `KeyboardInterrupt`

Captured events use stable metadata:

- `error.code`
- `error.kind`
- `is_expected=false`

Raw exception messages are not used as Sentry tags or grouping data.

## Allowed Tags And Context

Allowed tags:

- `service`
- `environment`
- `release`
- `cli_version`
- `python_version`
- `os`
- `arch`
- `command`
- `subcommand`
- `output_mode`
- `exit_code`
- `error.code`
- `error.kind`
- `is_expected`

Allowed context:

- `anonymous_install_id`
- `invocation_id`
- `daemon_session_id`

The CLI does not call `sentry_sdk.set_user()`.

## Data Never Collected

Sentry events must not include:

- email, name, user id, or org id
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
- raw exception values
- CLI command arguments

`before_send` and `before_breadcrumb` enforce this before transport.

## Local Tests

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --package potpie-observability --extra sentry pytest potpie/observability/tests -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run --package potpie-context-engine pytest potpie/context-engine/tests/unit/test_sentry_*.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest potpie/observability/tests potpie/context-engine/tests/unit/test_sentry_*.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check potpie/observability potpie/context-engine
```

The automated tests use an in-memory Sentry `Transport`; default tests do not
call the network.

## Staging Smoke

The staging smoke is opt-in and must refuse all non-staging environments.

Required environment:

```bash
RUN_SENTRY_STAGING_SMOKE=1
SENTRY_DSN=<staging dsn>
SENTRY_ENVIRONMENT=staging
SENTRY_RELEASE=potpie-cli@<version-or-test>
```

Smoke event contract:

- `error.code=sentry.smoke_test`
- `error.kind=unexpected`
- `is_expected=false`
- no sensitive payload
- environment is `staging`
- release is the supplied `SENTRY_RELEASE`
- telemetry context has anonymous install, invocation, and daemon session IDs

## Triage

Group by release, command, `error.code`, and `cli_version`.
Route staging events to engineering validation. Route prod regressions by
command owner and release window. If privacy scrubbers drop or redact a value,
debug using local logs and reproduction steps rather than expanding Sentry
payloads.
