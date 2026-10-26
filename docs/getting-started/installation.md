---
title: Installation
description: Install the Potpie CLI from PyPI with uv or pip.
---

## Step 1: install the CLI through [PyPI](https://pypi.org/project/potpie/) with `uv` or `pip`

```bash
uv tool install potpie
```

or:

```bash
python3 -m pip install --user potpie
```

> [!NOTE]
> `uv tool install potpie` is recommended for CLI installs because global
> mutation of Python packages is generally not recommended.

## Step 2: start Qdrant for resource semantic search

The default `qdrant_hybrid` resource index combines SQLite text search with
vectors stored in a separate Qdrant service. Start a local service with Docker:

```bash
docker run -d --name potpie-qdrant -p 127.0.0.1:6333:6333 -v potpie-qdrant-data:/qdrant/storage qdrant/qdrant:v1.19.2
```

The same command works with `podman` in place of `docker`. The named volume
preserves vectors across container replacements; the port is bound to this
machine only. Start an existing stopped container with
`docker start potpie-qdrant` (or `podman start potpie-qdrant`).

Potpie connects to `http://127.0.0.1:6333` by default. To use another service,
set its URL before starting the Potpie daemon:

```bash
export CONTEXT_ENGINE_QDRANT_URL="https://your-qdrant-host.example"
export CONTEXT_ENGINE_QDRANT_API_KEY="your-service-api-key"
```

In PowerShell:

```powershell
$env:CONTEXT_ENGINE_QDRANT_URL = "https://your-qdrant-host.example"
$env:CONTEXT_ENGINE_QDRANT_API_KEY = "your-service-api-key"
```

Set the API key only when the Qdrant service requires one. These commands apply
to the current shell; restart an already running daemon from that shell with
`potpie daemon restart` to pick up changes.

Qdrant stores vectors and pot/document routing identifiers, including document
slugs. Resource text stays in SQLite, along with cached embeddings and pending
vector updates and deletions. If Qdrant is unavailable, text search continues
and index status reports the reason. The background worker retries pending work after the
service returns, reusing cached embeddings. Check it with
`potpie resource index status`.

For text search without a Qdrant service, select `sqlite_fts`:

```bash
potpie config set resource_index sqlite_fts
potpie daemon restart
```

`CONTEXT_ENGINE_RESOURCE_INDEX=sqlite_fts` is the equivalent environment
override and takes precedence over the config key. Existing `sqlite_hybrid`
settings remain accepted as an alias for `qdrant_hybrid`; semantic search with
that alias also uses Qdrant.

## Linux ARM64

The Linux ARM64 default graph backend requires
`potpie-falkordblite==0.10.0.post1`, which repairs the native module packaging.
The prebuilt wheel supports glibc 2.36 or newer and has been validated on Debian
12 (glibc 2.36). This corrected distribution must be published
alongside the Potpie release before a normal PyPI installation of that release
can succeed. Until then, use wheels built from this checkout; the native build
recipe is in `scripts/packaging/falkordblite/README.md`. macOS and other Linux
architectures continue using the upstream `falkordblite` distribution.

Fresh isolated environments install only the corrected distribution. For an
existing environment, follow the [migration steps below](#upgrading-an-existing-linux-arm64-environment)
before restarting Potpie.

### Upgrading an existing Linux ARM64 environment

Stop the daemon with `potpie daemon stop` and download or build the corrected
wheel first. In the Python environment that runs Potpie, remove the old package
before installing the repair:

```bash
python -m pip uninstall falkordblite
python -m pip install --force-reinstall --no-deps /path/to/potpie_falkordblite-0.10.0.post1-<platform>.whl
```

Replace the path with the actual wheel filename. Both distributions own the
`redislite` import, so they must not coexist. If both are already installed,
remove `falkordblite` and reinstall the corrected wheel in the same order. This
package migration preserves Potpie's data directory. For a `uv tool` install,
reinstall into a fresh isolated tool environment instead of mixing packages.

Continue with [First steps](./first-steps.md) to run the setup wizard.
