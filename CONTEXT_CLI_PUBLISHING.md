# Publishing `potpie-context-cli` to PyPI

## Purpose

This document proposes the publishing model for the Potpie context CLI as a standalone Python package. It is written for internal maintainers who need to approve the package registry, ownership model, authentication model, versioning model, and release flow before the package is published.

The package lives inside the open-source Potpie repository, but it should be released and versioned independently from the main Potpie application.

## Decision Requested

Maintainers are asked to approve the following proposal:

- Publish the CLI to PyPI as `potpie-context-cli`.
- Expose the installed command as `potpie`.
- Own the PyPI project through the Potpie organization account, not an individual maintainer account.
- Keep the CLI package standalone, with no runtime imports from the rest of the Potpie repository.
- Version the CLI independently from the main Potpie repo/application.
- Publish through GitHub Actions using PyPI Trusted Publishing.
- Use tag-triggered releases as the source of truth, with optional GitHub Releases for release notes.

## Why PyPI

PyPI is the standard distribution channel for Python packages and command-line tools. Publishing there gives users the expected install path:

```bash
pip install potpie-context-cli
```

The package already has Python packaging metadata through `pyproject.toml`, declares a console entry point, and can be distributed as a wheel and source distribution. PyPI also supports Trusted Publishing, which lets GitHub Actions publish without storing a long-lived PyPI API token in GitHub secrets.

Alternatives exist, but they are less suitable for the first public distribution path:

- GitHub Releases alone can host artifacts, but users do not get the standard `pip install` experience.
- A private package index adds operational overhead and is not appropriate for an open-source package.
- Manual wheel sharing is not auditable enough for a repeatable release process.

## Current Package

The package root is:

```text
app/src/context-engine/adapters/inbound/context-cli
```

Current packaging metadata:

- Package name: `potpie-context-cli`
- Installed command: `potpie`
- Python support: `>=3.10,<3.14`
- Build backend: `hatchling`
- Runtime dependencies: `httpx`, `rich`, `typer`
- Current version source: `[project].version` in `pyproject.toml`
- Current version: `0.1.0`

The package module is:

```text
potpie_context_cli/
```

The command entry point is:

```toml
[project.scripts]
potpie = "potpie_context_cli.main:main"
```

## CLI Features

The CLI provides terminal access to Potpie context graph workflows.

Core commands:

- `potpie login`: store Potpie API key and API base URL locally.
- `potpie logout`: remove stored local credentials.
- `potpie doctor`: inspect local configuration and API connectivity.
- `potpie --version`: print the installed CLI package version.

Context pot commands:

- `potpie pot pots`: list accessible context pots.
- `potpie pot create <slug>`: create a user-owned context pot.
- `potpie pot use <slug-or-id>`: set the active local pot.
- `potpie pot unset`: clear the active local pot.
- `potpie pot list`: show local pot aliases and active pot.
- `potpie pot alias <name> <pot-id>`: create a local alias.
- `potpie pot slug-available <slug>`: check slug availability.
- `potpie pot clear-local`: remove local active pot and aliases while keeping API login.
- `potpie pot repo list`: list repositories attached to a pot.
- `potpie pot repo add <owner/repo>`: attach a GitHub repository to a pot.
- `potpie pot hard-reset`: destructive operator reset for a pot.

Context graph commands:

- `potpie search`: semantic search over the context graph.
- `potpie resolve`: answer a query with synthesized summary and evidence.
- `potpie overview`: fetch graph-wide readiness/activity snapshot.
- `potpie status`: fetch readiness and capability envelope.
- `potpie ingest`: add raw context episodes from inline text or a UTF-8 file.
- `potpie record`: record a durable context fact.
- `potpie event list/show/wait`: inspect and wait for ingestion events.
- `potpie conflict list/resolve`: inspect and resolve predicate-family conflicts.

Local helper commands:

- `potpie add`: inspect a git remote and print provider-scoped repository identity.
- `potpie init-agent`: install packaged agent instructions and skills into a repository.

Packaged agent bundles:

- Default/Codex bundle: `AGENTS.md` and `.agents/skills/*`
- Claude bundle: `CLAUDE.md` and `.claude/commands/*`

The `.claude` template directory is hidden and may also match repository-level ignore rules. The packaging configuration must force-include it so `potpie init-agent claude` works after installation from PyPI.

## Package Boundary

The CLI package must remain standalone.

That means:

- Runtime code must live under `app/src/context-engine/adapters/inbound/context-cli/potpie_context_cli`.
- Runtime templates must live under `potpie_context_cli/templates`.
- Runtime dependencies must be declared in `context-cli/pyproject.toml`.
- The CLI must not import Python modules from the rest of the Potpie repository.
- Tests may live elsewhere, but packaging and runtime behavior must not require the main app source tree.

This is important because users installing from PyPI will receive only the built package, not the full Potpie monorepo.

Before each release, maintainers should verify that the wheel works in a clean environment and that import resolution does not depend on the surrounding repo checkout.

## User CLI Authentication

This section describes how users authenticate the installed CLI to a Potpie API server. It is separate from PyPI publishing authentication.

The implemented API configuration resolution order is:

1. Environment variables.
2. Stored credentials written by `potpie login`.

Supported environment variables:

- `POTPIE_API_KEY`
- `POTPIE_API_URL`
- `POTPIE_BASE_URL`
- `POTPIE_PORT`
- `POTPIE_API_PORT`

Resolution behavior:

- `POTPIE_API_KEY` wins over any stored API key.
- `POTPIE_API_URL` wins over `POTPIE_BASE_URL`.
- If no URL is set and `POTPIE_PORT` or `POTPIE_API_PORT` is set, the CLI builds `http://127.0.0.1:<port>`.
- If environment values are missing, the CLI falls back to stored credentials.
- The packaged CLI does not load `.env` files.

Stored credential behavior:

```bash
potpie login <api-key> --url https://your-potpie-host
```

`potpie login` writes credentials to the user config directory, normally:

```text
~/.config/potpie/credentials.json
```

The credentials file is written with mode `600`. It can also store local CLI state such as `active_pot_id` and `pot_aliases`.

Operational guidance:

- For local interactive use, prefer `potpie login`.
- For automation, CI, or ephemeral environments, prefer environment variables.
- Do not ask users to place secrets in `.env` files for the packaged CLI, because the package does not load `.env` files.

## PyPI Publishing Authentication

Publishing authentication should use PyPI Trusted Publishing through GitHub Actions.

Trusted Publishing uses OpenID Connect between GitHub Actions and PyPI. In practice, this means the release workflow asks GitHub for a short-lived identity token, PyPI verifies that the workflow matches the trusted publisher configuration, and PyPI mints a short-lived project-scoped API token for that publish.

This is preferred because:

- No long-lived PyPI token is stored in GitHub repository secrets.
- The trusted publisher can be restricted to one GitHub repository and one workflow file.
- A GitHub Environment, such as `pypi`, can add a manual approval gate.
- The publishing credential is short-lived and scoped to the PyPI project.

Initial PyPI setup:

1. Create or claim the `potpie-context-cli` project under the Potpie PyPI organization account.
2. Add a GitHub Actions trusted publisher for the Potpie repository.
3. Restrict the trusted publisher to the release workflow file, for example `.github/workflows/publish-context-cli.yml`.
4. Use the GitHub Environment name `pypi` in both PyPI's trusted publisher configuration and the GitHub Actions job.
5. Configure the `pypi` environment in GitHub with required reviewers if maintainers want an explicit approval gate before publishing.

The release job must include:

```yaml
permissions:
  id-token: write
```

The PyPA publishing action should be used without username/password inputs:

```yaml
- name: Publish package distributions to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
```

## Versioning Model

The CLI must have its own version, independent of the main Potpie application.

The version source is:

```text
app/src/context-engine/adapters/inbound/context-cli/pyproject.toml
```

Specifically:

```toml
[project]
version = "0.1.0"
```

The installed CLI reads that package version through Python package metadata and exposes it through:

```bash
potpie --version
```

Recommended versioning policy:

- Use SemVer: `MAJOR.MINOR.PATCH`.
- Increment `PATCH` for backwards-compatible bug fixes.
- Increment `MINOR` for backwards-compatible new commands, options, or behavior.
- Increment `MAJOR` for breaking CLI behavior, removed commands/options, or incompatible API contract changes.
- Pre-release versions may use PEP 440 forms when needed, for example `0.2.0rc1`.

Version ownership:

- The CLI version is not tied to the main repo version.
- The CLI version is not inferred from git tags.
- The CLI version in `pyproject.toml` is the canonical package version.
- Release tags must match the version in `pyproject.toml`.

Recommended tag format:

```text
context-cli-v0.1.0
```

This avoids conflict with any future repo-wide application tags such as `v1.2.3`.

Release rule:

- A tag `context-cli-vX.Y.Z` may only publish a package whose `pyproject.toml` version is `X.Y.Z`.
- If the tag and package version do not match, the workflow should fail before publishing.
- Once a version is published to PyPI, it cannot be overwritten. A broken release must be fixed by publishing a new version.

## Release Trigger: Tags vs GitHub Releases

There are two reasonable GitHub Actions trigger models.

### Option A: Tag-triggered publishing

The workflow runs when a maintainer pushes a matching tag:

```yaml
on:
  push:
    tags:
      - "context-cli-v*"
```

Flow:

1. Update `context-cli/pyproject.toml` with the next CLI version.
2. Update the package README or changelog if needed.
3. Open and review a PR.
4. Merge the PR.
5. Create and push a tag such as `context-cli-v0.1.0`.
6. GitHub Actions verifies the package version matches the tag.
7. GitHub Actions builds the wheel and sdist from the CLI package directory.
8. GitHub Actions publishes to PyPI through Trusted Publishing.
9. Maintainers optionally create a GitHub Release using the same tag for release notes.

Pros:

- Simple source of truth: the git tag is the release event.
- Easy to automate and audit.
- Works well with independent package versioning.
- Avoids coupling publication to manually written release notes.

Cons:

- A pushed tag can publish immediately unless the workflow uses a protected GitHub Environment.
- Release notes are optional unless the team enforces them separately.

### Option B: GitHub Release-triggered publishing

The workflow runs when a GitHub Release is published:

```yaml
on:
  release:
    types: [published]
```

Flow:

1. Update the CLI version and docs in a PR.
2. Merge the PR.
3. Create a GitHub Release for tag `context-cli-vX.Y.Z`.
4. Publishing starts when the release is published.

Pros:

- More explicit human step.
- Release notes are naturally part of the flow.
- Easier for maintainers to see what is being shipped.

Cons:

- Release drafts, pre-releases, and tag creation behavior can create confusion.
- The release object becomes the trigger, but the package version still lives in `pyproject.toml`.
- Maintainers must be disciplined about matching the release tag and package version.

### Recommendation

Use tag-triggered publishing with a protected GitHub Environment named `pypi`.

This gives a clean release source of truth while still allowing a manual approval gate. GitHub Releases can be created after publishing or as part of release preparation, but the tag remains the event that decides which source is published.

## Proposed GitHub Actions Workflow

Proposed file:

```text
.github/workflows/publish-context-cli.yml
```

Proposed workflow:

```yaml
name: Publish context CLI

on:
  push:
    tags:
      - "context-cli-v*"

jobs:
  publish:
    name: Build and publish potpie-context-cli
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      contents: read
      id-token: write

    defaults:
      run:
        working-directory: app/src/context-engine/adapters/inbound/context-cli

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: python -m pip install --upgrade build twine

      - name: Verify tag matches package version
        run: |
          PACKAGE_VERSION="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
          TAG_VERSION="${GITHUB_REF_NAME#context-cli-v}"
          test "$PACKAGE_VERSION" = "$TAG_VERSION"

      - name: Build distributions
        run: python -m build

      - name: Check distributions
        run: python -m twine check dist/*

      - name: Publish package distributions to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: app/src/context-engine/adapters/inbound/context-cli/dist
```

Notes:

- The `environment: pypi` name should match the environment configured on the PyPI trusted publisher.
- `id-token: write` is required for Trusted Publishing.
- No PyPI username, password, or API token should be configured in the workflow.
- The workflow builds from the CLI package directory, not the repo root.
- A version mismatch fails before upload.

## Release Checklist

Before tagging:

- Confirm `context-cli/pyproject.toml` has the intended version.
- Confirm `potpie --version` reports that version after local installation.
- Confirm the CLI README reflects current auth behavior.
- Confirm runtime imports stay inside `potpie_context_cli` or declared dependencies.
- Confirm templates are included in the wheel, including hidden template directories such as `.agents` and `.claude`.
- Run package tests if available.
- Build locally:

```bash
cd app/src/context-engine/adapters/inbound/context-cli
python -m build
python -m twine check dist/*
```

Optional clean install check:

```bash
python -m venv /tmp/potpie-context-cli-release-check
/tmp/potpie-context-cli-release-check/bin/python -m pip install dist/*.whl
/tmp/potpie-context-cli-release-check/bin/potpie --version
```

Tag and publish:

```bash
git tag context-cli-v0.1.0
git push origin context-cli-v0.1.0
```

After publishing:

- Verify the release exists on PyPI.
- Verify install from PyPI:

```bash
python -m pip install --upgrade potpie-context-cli
potpie --version
```

- Create or update the GitHub Release notes for the same tag.

## Maintainer Responsibilities

Potpie organization account owners:

- Own the PyPI project.
- Configure Trusted Publishing.
- Manage PyPI project roles.
- Ensure at least two maintainers have recovery access.

Repository maintainers:

- Review version bumps and release PRs.
- Ensure package boundary is preserved.
- Approve the `pypi` GitHub Environment deployment if approval is enabled.
- Verify releases after publication.

Release owner:

- Prepares the release PR.
- Confirms version and tag match.
- Pushes the release tag.
- Checks GitHub Actions and PyPI after publishing.

## Open Questions

- Should the first public version remain `0.1.0`, or should it move to a higher version before the first PyPI publish?
- Should TestPyPI be used for the first dry run?
- Should a changelog file be added under the CLI package directory or at the repository root?
- Should the package add CI that installs the built wheel and runs smoke tests on every PR touching `context-cli`?

## References

- PyPI Trusted Publishing: [https://docs.pypi.org/trusted-publishers/](https://docs.pypi.org/trusted-publishers/)
- Publishing with a Trusted Publisher: [https://docs.pypi.org/trusted-publishers/using-a-publisher/](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- Adding a Trusted Publisher to an existing PyPI project: [https://docs.pypi.org/trusted-publishers/adding-a-publisher/](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- Python Packaging User Guide, build and publish: [https://packaging.python.org/en/latest/guides/section-build-and-publish/](https://packaging.python.org/en/latest/guides/section-build-and-publish/)
- GitHub Actions workflow syntax: [https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax](https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax)
- GitHub Actions release events: [https://docs.github.com/en/actions/reference/events-that-trigger-workflows](https://docs.github.com/en/actions/reference/events-that-trigger-workflows)

