# Contributing to Potpie

This guide covers development of the Potpie CLI, local daemon, and workspace
packages. Run the commands from the repository root unless a step changes
directories.

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [How to contribute](#how-to-contribute)
3. [Getting started](#getting-started)
4. [Development workflow](#development-workflow)
5. [Testing](#testing)
6. [Submitting pull requests](#submitting-pull-requests)
7. [Documentation](#documentation)
8. [Community and support](#community-and-support)

## Code of conduct

Follow the [code of conduct](./CODE_OF_CONDUCT.md) when participating in the
project.

## How to contribute

Open an issue for a bug or proposed feature, improve documentation, review a pull
request, or contribute a fix. Include reproduction steps and the affected command
when reporting a bug. Discuss changes that affect multiple packages or public
behavior in an issue before starting a large implementation.

## Getting started

### Prerequisites

Install these tools for a full development checkout:

- [Git](https://git-scm.com/).
- Python 3.12 or 3.13. The root package supports Python 3.12 through 3.14, but
  the global CLI installation target currently defaults to 3.12 or 3.13.
- [uv](https://docs.astral.sh/uv/) to manage Python environments and dependencies.
- Node.js 24 and npm to build and test the graph explorer UI.
- Stable Rust, Cargo, and system C/C++ build tools to compile the native parsing
  package.

Docker is optional for local CLI development and root unit tests. Context Engine
recovery tests use database containers. See
[database recovery tests](#database-recovery-tests).

### Fork and clone the repository

Fork the repository on GitHub, then clone your fork and add the upstream remote:

```bash
git clone https://github.com/your-username/potpie.git
cd potpie
git remote add upstream https://github.com/potpie-ai/potpie.git
```

### Set up your environment

Install the workspace packages and development dependencies using the same sync
command as the regression workflow. This includes optional features and their ML
dependencies.

```bash
uv sync --frozen --all-packages --all-extras
uv run pre-commit install
```

Build the graph explorer assets:

```bash
(
  cd potpie/daemon/http/ui/frontend
  npm ci
  npm run build
)
```

Run the CLI from the checkout:

```bash
uv run potpie --help
uv run potpie setup
uv run potpie doctor
```

Local state defaults to `~/.potpie`. Set `CONTEXT_ENGINE_HOME` to use a separate
directory for development. You do not need a `.env` file or a Potpie account for
the local CLI. Operations that call external providers need that provider's
credentials.

The default local graph backend uses FalkorDBLite without a Docker server.
FalkorDBLite has no Windows build, and its Linux wheels require glibc 2.39 or
newer. On those unsupported systems, set `CONTEXT_ENGINE_BACKEND=embedded` to use
the file-backed local backend, or configure a FalkorDB server.

### Install the development CLI globally

To use this checkout's `potpie` command outside the repository, use the optional
Make targets:

```bash
make cli-install
make cli-status
```

`make cli-install` builds the UI, stops an existing detached Potpie daemon, and
replaces the global `uv` tool installation with an editable install of this
checkout. It includes all optional features. Use `make cli-install CLI_EXTRAS=`
for a base installation without those extras. The root package is editable; use
the checkout environment above when changing workspace packages. Rerun the target
after dependency or entrypoint changes.

## Development workflow

Create a branch for your change:

```bash
git switch -c feature/your-feature-name
```

Restart a running daemon after changing daemon or runtime code:

```bash
uv run potpie daemon restart
```

Keep related behavior and its tests in the same change. The main development
locations are:

| Area | Location |
| --- | --- |
| CLI and local daemon | [`potpie/cli/`](../potpie/cli/) and [`potpie/daemon/`](../potpie/daemon/) |
| Context Engine | [`potpie/context-engine/`](../potpie/context-engine/) |
| Native parser | [`potpie/parsing/`](../potpie/parsing/) |
| Graph explorer UI | [`potpie/daemon/http/ui/frontend/`](../potpie/daemon/http/ui/frontend/) |
| Root Python tests | [`tests/`](../tests/) |

Run the relevant tests below, then run the repository checks:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

The hooks run Ruff linting and formatting, validate the lockfile and configuration
files, and check workflows. Rust changes also trigger `cargo fmt`. If a hook edits
a file, review the edit and rerun the checks.

Update the lockfile when changing dependency manifests. For changes to specified
behavior, follow the [specification change process](../spec/process.md).

## Testing

Tests have two owners: the root product and the context engine. The root suite
includes CLI, authentication, daemon, UI, documentation, parsing, and
integration-package checks. The engine suite owns graph semantics, ingestion,
retrieval, persistence, and recovery.

```bash
make test-root
make test-context-engine
```

Each Python package runs in a separate process so its `tests` package cannot
shadow another suite. Parsing also runs its Rust tests; docs and UI use their
Node test runners. Focused package commands remain available:

```bash
make test-product TEST_CHECKS=python
make test-parsing
make test-integrations
make test-context-engine TEST_TIER=graph
make test-context-engine TEST_TIER=restart
```

Use shared fixtures for stable interfaces and named parameter rows for input,
provider, platform, or backend variations. Keep recovery scenarios explicit when
they require ordered actions and intermediate assertions. Source-layout and
packaging assertions use the `source_contract` marker; runtime behavior belongs
in tests that can run against installed distributions.

CI builds all four product distributions before test jobs begin: product,
engine, integrations, and native parsing wheels and source archives, plus the
repaired Linux ARM64 FalkorDB Lite wheel. A manifest records the commit, package
version, wheel tags, and checksum. Smoke jobs install those exact artifacts in
clean environments outside the checkout. Root and engine wheels are installed
together when testing the product, and the engine is also installed independently.

Deeper suites run on selected targets, with separate graph-acceptance and serial
restart profiles. Required native acceptance cases fail if skipped. Source
contracts run separately from installed runtime tests so source imports cannot
mask missing files or dependency errors in a wheel.

Live authentication tests are opt-in and need `RUN_CLI_AUTH_E2E=1` plus the
provider credentials described in their test files. Backend and recovery tests
need the isolated services configured in the [regression workflow](workflows/test.yml).

### Database recovery tests

Context Engine recovery tests kill and restart disposable FalkorDB and Neo4j
containers. Install Docker and start its daemon before running these tests:

```bash
docker info
```

The [regression workflow](workflows/test.yml) creates the containers and passes
their names and endpoints to the tests. Follow that workflow for the service
configuration and environment variables. The [journal crash tests](../potpie/context-engine/tests/conformance/durability/test_journal_crash.py)
exercise the crash-recovery path.

### Graph explorer UI

```bash
(
  cd potpie/daemon/http/ui/frontend
  npm ci
  npm test
  npm run build
)
```

### Documentation tooling

```bash
(
  cd tests/docs
  npm ci
  npm test
)
```

The documentation content checks cover the published `docs/` tree and its
configuration. Check links and commands in other Markdown files yourself.

## Submitting pull requests

Commit your changes and push your branch to your fork:

```bash
git add path/to/changed-file
git commit -m "Describe the change"
git push origin feature/your-feature-name
```

Open a pull request against `potpie-ai/potpie` with `main` as the base branch.
Describe the problem, the resulting behavior, and the tests you ran. Note any
relevant checks you could not run, including tests skipped for missing services
or credentials. Link related issues with text such as `Closes #123`.

Keep unrelated edits out of the pull request. Address review feedback and wait
for a maintainer to approve and merge it.

## Documentation

User-facing docs live under [`docs/`](../docs/). Update the matching page when a
change affects a command, setup step, or integration. Keep examples consistent
with the implemented behavior and check any commands you change.

## Community and support

Open a [GitHub issue](https://github.com/potpie-ai/potpie/issues) to ask questions,
report bugs, or propose changes.
