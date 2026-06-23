# Legacy demo host (optional)

This folder contains the original Potpie monolith: FastAPI app, agents, Celery workers, seed CLI, deployment assets, and regression tests.

Core libraries live at the repo root under `potpie/` (`context-engine`, `parsing`, `sandbox`, `integrations`). You can delete this entire `legacy/` directory if you only need those packages.

## Quick start

From the repository root:

```bash
cp legacy/.env.template legacy/.env
# edit legacy/.env

uv sync --all-packages
chmod +x legacy/scripts/start.sh
./legacy/scripts/start.sh
```

Or use the Makefile from this directory:

```bash
cd legacy
cp .env.template .env
make dev
```

## Imports

The Python import root remains `app` (not `legacy.app`). Run commands with `legacy/` on the working directory or install the `potpie-legacy` project via `uv sync --project legacy`.

## Tests

```bash
cd ..
SKIP_REAL_PARSE=1 uv run --project legacy python legacy/scripts/run_tests.py --coverage
```
