# PR #792 Context Graph Test Overview

This document explains the test work added around PR #792 in plain English.
It is meant for reviewers who need the context quickly: what we wanted to
protect, what tests were added, and how CI runs them.

## What Changed In This PR

PR #792 gives context graph its own test lane and adds behaviour tests around
the risky seams where bugs would hurt most:

- the Potpie host bridge that builds context-engine containers;
- repo attach and source attach paths that create ingestion events;
- ingestion dedupe, batching, queueing, retries, and stale-batch recovery;
- the reconciliation agent's allowed tools and graph mutation boundary;
- user-scoped pot lookup and repo host validation;
- read/query routes, record routes, and NDJSON streams.

The tests live with the code they are protecting:

- engine tests live under `app/src/context-engine/tests/`;
- Potpie host bridge tests live under `tests/unit/context_graph/`;
- HTTP / integration-style host tests live under
  `tests/integration-tests/context_graph/`.

## The Goal

The context graph has many moving parts: Potpie host routes, pot/repo/source
tables, the standalone context engine, ingestion batches, a reconciliation
agent, sandbox/GitHub/Linear tools, and graph writes.

The test goal was not to prove each small helper in isolation. The goal was to
protect the contracts between those pieces:

- a repo attach must create the right ingestion event;
- events must dedupe, batch, enqueue, and process once;
- the reconciliation agent must not escape its pot/repo/tool boundaries;
- user-scoped reads must not reveal another user's pot;
- crashed or stuck batches must become recoverable;
- read and record routes must return stable envelopes;
- the whole context-graph suite must run independently in CI.

The guiding rule: tests should describe required behaviour. If a behaviour test
breaks the current code, that is useful signal.

## How To Run The Tests

Main command:

```bash
uv run python scripts/run_tests.py --context-graph-only
```

Split commands:

```bash
uv run python scripts/run_tests.py --context-graph-engine-only
uv run python scripts/run_tests.py --context-graph-host-only
```

Make targets:

```bash
make test-context-graph
make test-context-graph-engine
make test-context-graph-host
```

CI runs the context-graph suite as its own GitHub Actions job:

```bash
uv run python scripts/run_tests.py --context-graph-only
```

That CI job has Postgres and Redis services because the host-bridge tests import
monolith database and Celery wiring during collection.

## Test Groups

### CGT-1: Dedicated Test Runner And CI Wiring

What we wanted to test:

- context-engine tests under `app/src/context-engine/tests/` are discovered by
  the root test runner;
- host bridge tests under `tests/unit/context_graph/` and
  `tests/integration-tests/context_graph/` run in the same dedicated suite;
- developers can run engine-only or host-only slices;
- invalid mixed modes, like `--unit-only --context-graph-only`, fail clearly.

Why it matters:

Before this, context graph tests were scattered and could silently fall out of
CI. The new runner makes the suite explicit.

### CGT-2: Agent Tool And Repo Boundaries

What we wanted to test:

- the reconciliation agent only receives tools allowed by the event playbook;
- fallback/default playbook hints are not treated as an authorization boundary;
- a prompt-injected agent cannot ask GitHub tools to read a repo that is not
  attached to the pot;
- blocked foreign repo calls return `unknown_repo` before shared GitHub
  credentials are touched.

Important tests:

- `test_deep_agent_containment.py`
- `test_github_agent_tools_repo_binding.py`
- `test_wiring_github_repo_binding.py`

Why it matters:

Event payloads and tool outputs are untrusted. These tests make sure the agent
cannot expand its own tool surface or exfiltrate a foreign private repo.

### CGT-3: Repo Attach SSRF And User-Scoped Pot Access

What we wanted to test:

- only allowed provider hosts can be attached;
- `remote_url` cannot point at internal metadata, localhost, arbitrary domains,
  unsafe schemes, local files, or confusable hosts;
- `remote_url` must match the declared `provider_host`;
- GitHub Enterprise hosts work only when explicitly allowed;
- a user-scoped pot resolver returns only pots the user can access;
- archived pots and stranger access are hidden.

Important tests:

- `test_attach_repo_provider_host_guard.py`
- `test_user_scoped_pot_resolution.py`

Why it matters:

Repo attach eventually leads to clone/fetch operations. Without strict host
checks, an attacker could turn repo attach into SSRF or token exfiltration.
Without user-scoped resolution tests, pot tenancy could regress quietly.

### CGT-4: Host Container And Queue Wiring

What we wanted to test:

- the host builds a context-engine container with jobs, connectors, source
  listing, and the right pot resolver;
- user-scoped containers use actor-scoped pot resolution;
- one failed optional tool surface does not prevent other tool surfaces from
  being attached;
- the Celery queue adapter dispatches `context_graph_process_batch` by batch id.

Important tests:

- `test_wiring_container_contract.py`
- `test_celery_job_queue.py`
- `test_wiring_sandbox_tools.py`

Why it matters:

The context engine is standalone, but Potpie wires it into the monolith. These
tests protect that bridge.

### CGT-5: `apply_graph_mutations` Tool Contract

What we wanted to test:

- unknown event ids do not call graph writes;
- invalid plans do not call graph writes;
- conversion failures do not call graph writes;
- graph errors are surfaced as tool failures;
- successful calls pass the expected pot id and provenance;
- runaway repeated apply calls are capped.

Important test:

- `test_apply_graph_mutations_contract.py`

Why it matters:

`apply_graph_mutations` is the agent's write tool. It is the boundary between
model output and graph mutation. Bad input must fail before it reaches the
graph.

### CGT-6: Fake End-To-End Ingestion

What we wanted to test:

- submit an event through the real ingestion submission service;
- append/dedupe the event;
- create/open a batch;
- enqueue the batch;
- process the batch through the worker use case;
- run a fake agent;
- apply a real reconciliation plan through a fake graph port;
- ensure duplicates and redundant worker dispatches do not double-apply.

Important test:

- `test_fake_ingestion_e2e.py`

Why it matters:

This is the strongest behavioural test in the suite. It uses in-memory fake
ports with real state instead of patching private helpers, so it exercises the
pipeline as a system.

### CGT-7: Failure, Resume, And Stale Batch Recovery

What we wanted to test:

- if the agent crashes, resume checkpoints are not cleared;
- if the agent returns `ok=False`, resume checkpoints are not cleared;
- terminal failure events are written after the previous resume sequence;
- a retry skips events already credited before the crash;
- the stale-batch reaper lease always exceeds the Celery hard time limit.

Important tests:

- `test_process_batch_failure_resume.py`
- `test_stale_batch_reaper_lease.py`

Why it matters:

Context graph work runs in background workers. If a worker dies mid-batch, the
system must recover without double-applying completed events or leaving the pot
stuck forever.

### CGT-8: Read Dispatch And Async Contract

What we wanted to test:

- sync `query()` rejects agentic `ANSWER` / `INVESTIGATE` calls inside a live
  event loop;
- async `query_async()` is the correct entrypoint for those goals;
- non-agentic reads still use reader registry paths;
- `INVESTIGATE` uses the query agent when available and falls back when not;
- reconciliation read tools return a stable `ContextGraphResult` envelope.

Important test:

- `test_context_graph_query_dispatch.py`

Why it matters:

This prevents event-loop misuse and protects the public shape consumed by agent
read tools.

### CGT-9: Record And Ingestion Config Routes

What we wanted to test:

- `POST /record` turns durable context records into ingestion submissions;
- idempotency, scope, and `occurred_at` are passed through correctly;
- duplicate record submissions surface duplicate status;
- ingestion config GET/PUT returns stable config shapes;
- force flush enqueues open windowed batches and tolerates enqueue failure.

Important test:

- `test_context_record_and_ingestion_routes.py`

Why it matters:

These routes are user-facing control points for durable memory and ingestion
mode.

### CGT-10: NDJSON Stream Consumers

What we wanted to test:

- event streams return NDJSON;
- queued events that are not yet batched produce a terminal queued end event;
- execution log replay/tail events are emitted in order;
- iterator failures produce a terminal error event;
- pot status streams replay/tail from the publisher and end cleanly.

Important test:

- `test_event_stream_ndjson_consumer.py`

Why it matters:

Streaming endpoints must fail visibly and predictably. A UI should not hang
forever when a stream backend fails.

### CGT-11: Cleanup And Suite Hygiene

What we wanted to test or clean up:

- stale duplicate host tests were deleted;
- engine-owned tests were moved into `app/src/context-engine/tests/`;
- runner deselect workarounds were removed where tests were fixed;
- a few known broken legacy engine tests remain explicitly ignored with comments
  in `scripts/run_tests.py`.

Why it matters:

A dedicated suite is only useful if it is honest. We avoided hiding broad test
trees and kept the remaining ignores visible as follow-up work.

## Behaviour-First Notes For Reviewers

Some tests still patch private helpers because those helpers are today's seam
between the Potpie host and the context engine. That is acceptable when the
assertion is still a real boundary contract, for example:

- foreign repo never reaches the shared GitHub credential resolver;
- sandbox tools can be disabled by env flag;
- a bootstrap event is submitted async and does not block repo attach.

The best tests in this PR are the ones that drive public seams and durable
state:

- fake E2E ingestion;
- retry after partial failure;
- foreign repo blocked before credentials;
- remote URL SSRF rejection;
- unknown or invalid mutation plan never reaches graph writes.

Future cleanup should move more host wiring tests toward route/use-case level
fakes and away from patching private helper names.

## Current CI Shape

GitHub Actions has two jobs in `.github/workflows/test.yml`:

- `context-graph`: runs only the context graph suite.
- `regression`: runs the broader regression suite with coverage.

Both jobs run for PRs and pushes targeting:

- `main`
- `master`
- `staging`

The context-graph job uses Postgres and Redis services because host bridge tests
collect monolith database and Celery modules.

## Remaining Follow-Ups

Not everything is solved by this PR.

- Some known legacy engine tests are still ignored by `scripts/run_tests.py`.
- Some tests are still implementation-adjacent and should be refactored later.
- SonarCloud / broader regression failures, if any, need separate review from
  the context-graph suite itself.

The important thing this PR establishes is the safety net: context graph now has
a dedicated, behaviour-focused suite that can fail independently when a core
contract is broken.
