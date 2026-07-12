# Product Telemetry and Engine Observability

> Verified at `f435fb4` on 2026-07-13. Product telemetry ownership is defined by
> [PKG-OBS-001](../../spec/modules/package-boundary.md).

Sentry and PostHog are product integrations owned by root `potpie`. The
standalone context engine exposes generic observability ports/events and an
optional OpenTelemetry adapter; it does not import Sentry, PostHog, product
telemetry settings, or product build defaults.

## Ownership

| Concern | Owner | Primary code |
|---|---|---|
| CLI crash capture | root | `potpie/cli/telemetry/sentry_runtime.py` |
| Daemon crash capture | root | `potpie/daemon/telemetry/sentry_runtime.py` |
| Metrics and privacy filtering | root | `potpie/runtime/telemetry/` |
| Product usage/onboarding events | root | `potpie/cli/telemetry/product_analytics.py`, `onboarding_events.py` |
| Telemetry enable/disable/status | root | `potpie/cli/commands/telemetry.py` |
| Generic spans/events | engine | `potpie_context_engine.domain.ports.observability` |
| Console/OTel adapters | engine | `potpie_context_engine.adapters.outbound.observability` |

Product telemetry never becomes an engine configuration field or daemon RPC
method.

## Product settings

Root runtime settings resolve environment, packaged public defaults, and code
defaults. Common controls are:

```text
POTPIE_TELEMETRY_DISABLED
POTPIE_SENTRY_ENABLED
POTPIE_SENTRY_DSN
POTPIE_SENTRY_ENVIRONMENT
POTPIE_POSTHOG_ENABLED
POTPIE_PRODUCT_ANALYTICS_ENABLED
POTPIE_POSTHOG_API_KEY
POTPIE_POSTHOG_HOST
```

Use the product commands instead of editing files:

```bash
potpie telemetry status
potpie telemetry disable
potpie telemetry enable
```

The preference affects root product telemetry. It does not disable engine-local
generic observability explicitly injected by a library embedder.

## CLI behavior

The CLI configures product telemetry during root startup. Expected operational
errors—validation, unavailable daemon, auth, conflict, degraded health—are
mapped to stable CLI errors and are not treated as unexpected crashes.
Unexpected exceptions can be captured after privacy scrubbing, while JSON mode
still writes exactly one response to stdout.

Diagnostics, tracebacks, and telemetry logging use stderr/log sinks; telemetry
must never corrupt the JSON envelope.

## Daemon behavior

The daemon is a root product process and configures its own Sentry lifecycle.
Transport and process failures can be reported there. Engine methods running in
the daemon emit only generic engine observations; root decides whether and how
those observations are exported.

## Privacy boundary

Root privacy filters strip or reject sensitive fields before product telemetry
leaves the process. Credentials, keyring values, provider tokens, raw source
payloads, document bodies, graph claim content, and arbitrary CLI arguments are
not product telemetry payloads.

Stable operational attributes may include product/engine version, runtime mode,
command family, backend profile, success/failure category, timing, and anonymous
installation/invocation identifiers according to the telemetry settings.

## Product analytics

PostHog-style product analytics is implemented with root-owned HTTP/settings
code. It covers bounded product events such as setup progress, activation, CLI
invocation, and installation lifecycle. The engine is not aware of product
funnels or user identity.

## Engine observability

Engine embedders can inject `EngineDependencies.observability`. The default is a
no-op implementation. Optional engine observability can be installed with:

```bash
python -m pip install "potpie-context-engine[observability]==0.2.0"
```

When explicitly configured, the engine can use console or OpenTelemetry
adapters. Missing optional telemetry dependencies fail dark to the no-op path;
they do not activate product Sentry/PostHog.

## Build defaults

Product build-time telemetry defaults are root packaging concerns. The engine
wheel contains no Sentry/PostHog build hook or product OAuth/telemetry defaults.
Artifact tests inspect both distributions to enforce this boundary.

## Verification

```bash
uv run pytest tests/unit/test_sentry_*.py tests/unit/test_telemetry_*.py -q
uv run pytest tests/unit/test_product_analytics.py tests/unit/test_usage_analytics.py -q
cd potpie/context-engine
uv run pytest tests/unit/test_observability*.py tests/unit/test_telemetry_port.py -q
```

Boundary scans also require zero `sentry_sdk`, PostHog, and root `potpie`
imports in the engine source.
