# Potpie developer Makefile
#
# Quick start:
#   make dev                  # start infra + API + worker (Daytona by default)
#   make dev SANDBOX=local    # override sandbox to local subprocess
#   make daytona-up           # boot the Daytona compose stack for local dev
#   make help                 # list all targets
#
# Conventions:
#   - All Python work goes through `uv` so we share one venv (.venv).
#   - Recipes that need env vars source .env via $(LOAD_ENV).
#   - Long-running targets stay in the foreground; Ctrl+C stops them cleanly.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Sandbox selection. Override .env at the command line: `make dev SANDBOX=local`.
# When unset, whatever's in .env / .env.daytona.local wins (Daytona by default per
# .env.template). Keep this in sync with sandbox/bootstrap/settings.py.
SANDBOX ?=
ifeq ($(SANDBOX),daytona)
  SANDBOX_OVERRIDE := SANDBOX_WORKSPACE_PROVIDER=daytona SANDBOX_RUNTIME_PROVIDER=daytona
else ifeq ($(SANDBOX),local)
  SANDBOX_OVERRIDE := SANDBOX_WORKSPACE_PROVIDER=local SANDBOX_RUNTIME_PROVIDER=local_subprocess
else ifeq ($(SANDBOX),docker)
  SANDBOX_OVERRIDE := SANDBOX_WORKSPACE_PROVIDER=local SANDBOX_RUNTIME_PROVIDER=docker
else ifneq ($(SANDBOX),)
  $(error Unknown SANDBOX=$(SANDBOX). Use one of: daytona, local, docker)
else
  SANDBOX_OVERRIDE :=
endif

# Load .env (and .env.daytona.local if present, written by `make daytona-up`).
# Inline assignments after this line override what was sourced.
LOAD_ENV := set -a; \
            [ -f .env ] && . ./.env; \
            [ -f app/src/sandbox/.env.daytona.local ] && . ./app/src/sandbox/.env.daytona.local; \
            set +a;

# Daytona helper scripts shell out to python (daytona_local.py,
# build_agent_snapshot.py). The latter imports the daytona SDK from the
# project's uv-managed .venv, so we point PYTHON at `uv run python`. Single
# source of truth — every target invoking setup-daytona-local.sh exports this.
DAYTONA_PYTHON := PYTHON='uv run --project $(CURDIR) python'

# Shell snippet that exports CELERY_Q with the resolved queue list.
# Inlined into recipes that need it (mirrors scripts/start.sh).
# Built as a plain shell var (not $(...)) so case-patterns don't collide
# with command-substitution parens inside double quotes.
SET_CELERY_Q := CELERY_Q="$${CELERY_QUEUE_NAME}_process_repository,$${CELERY_QUEUE_NAME}_agent_tasks,external-event"; \
  cg_lc=$$(printf '%s' "$${CONTEXT_GRAPH_ENABLED:-true}" | tr '[:upper:]' '[:lower:]'); \
  if [ "$$cg_lc" != "false" ] && [ "$$cg_lc" != "0" ] && [ "$$cg_lc" != "no" ] && [ "$$cg_lc" != "off" ]; then \
    CELERY_Q="$$CELERY_Q,context-graph-etl"; \
  fi;

.DEFAULT_GOAL := help
.PHONY: help dev infra-up infra-down infra-logs infra-reset \
        sync deps env-check sandbox-prep \
        migrate migration downgrade migration-history \
        api worker stop \
        daytona-up daytona-down sandbox-status \
        test test-unit test-integration test-real-parse test-stress test-cov \
        lint format fix precommit \
        clean

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
	     /^##@ / {sub(/^##@ /, ""); printf "\n\033[1m%s\033[0m\n", $$0; next} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)

##@ Dev (one-shot)

dev: env-check infra-up sync migrate sandbox-prep ## Start infra, sync deps, migrate, boot Daytona if needed, then run API + worker. Use SANDBOX=local|docker|daytona to override.
	@$(LOAD_ENV) \
	source .venv/bin/activate; \
	$(SET_CELERY_Q) \
	echo "─── starting API (gunicorn) and Celery worker ───"; \
	$(SANDBOX_OVERRIDE) gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 \
	  --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app & \
	GUNICORN_PID=$$!; \
	$(SANDBOX_OVERRIDE) celery -A app.celery.celery_app worker --loglevel=debug \
	  -Q "$$CELERY_Q" -E --concurrency=1 --pool=solo & \
	CELERY_PID=$$!; \
	trap 'kill -TERM $$GUNICORN_PID $$CELERY_PID 2>/dev/null || true; \
	      wait $$GUNICORN_PID $$CELERY_PID 2>/dev/null || true' INT TERM EXIT; \
	echo "API on http://localhost:8001 (gunicorn $$GUNICORN_PID, celery $$CELERY_PID). Ctrl+C to stop."; \
	wait $$GUNICORN_PID $$CELERY_PID

env-check: ## Warn if .env is missing
	@if [ ! -f .env ]; then \
	  echo "⚠️  No .env found. Copy .env.template to .env and fill in secrets first."; \
	  exit 1; \
	fi
	@if ! command -v uv >/dev/null 2>&1; then \
	  echo "❌ uv not installed. See https://docs.astral.sh/uv/getting-started/"; \
	  exit 1; \
	fi

##@ Infrastructure (Postgres / Neo4j / Redis)

infra-up: ## Start Postgres, Neo4j, Redis via docker compose, wait for Postgres
	@docker compose up -d
	@echo "Waiting for Postgres..."
	@until docker exec potpie_postgres pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
	@echo "✓ Postgres ready · Neo4j on :7474 · Redis on :6379"

infra-down: ## Stop infra containers (keeps volumes)
	docker compose down

infra-logs: ## Tail logs from all infra containers
	docker compose logs -f

infra-reset: ## Stop infra and DELETE all data volumes (destructive)
	docker compose down -v

##@ Python environment

sync: ## uv sync — install/update Python deps into .venv
	uv sync

deps: sync ## Alias for sync

##@ Database migrations (Alembic)

migrate: ## Apply all pending migrations (alembic upgrade heads)
	@$(LOAD_ENV) uv run alembic upgrade heads

migration: ## Create a new migration. Usage: make migration name="add users table"
	@if [ -z "$(name)" ]; then echo "Usage: make migration name=\"description\""; exit 1; fi
	@$(LOAD_ENV) uv run alembic revision --autogenerate -m "$(name)"

downgrade: ## Roll back one migration (alembic downgrade -1)
	@$(LOAD_ENV) uv run alembic downgrade -1

migration-history: ## Show alembic history
	@$(LOAD_ENV) uv run alembic history

##@ Run individual processes (for debugging)

api: ## Run only the FastAPI server (assumes infra is up). Honors SANDBOX=...
	@$(LOAD_ENV) $(SANDBOX_OVERRIDE) uv run gunicorn --worker-class uvicorn.workers.UvicornWorker \
	  --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app

worker: ## Run only the Celery worker (assumes infra is up). Honors SANDBOX=...
	@$(LOAD_ENV) $(SET_CELERY_Q) $(SANDBOX_OVERRIDE) uv run celery -A app.celery.celery_app worker --loglevel=debug \
	  -Q "$$CELERY_Q" -E --concurrency=1 --pool=solo

stop: ## Kill API + worker processes and stop infra
	-pkill -f "gunicorn" || true
	-pkill -f "celery" || true
	docker compose down

##@ Sandbox

# Internal: ensure Daytona is up if it's the selected sandbox. Skipped silently
# for local/docker. If DAYTONA_API_KEY is already set and there's no
# .env.daytona.local from a previous local run, we assume the user is pointing
# at a remote Daytona and don't try to boot a local stack.
sandbox-prep:
	@$(LOAD_ENV) $(SANDBOX_OVERRIDE) bash -c '\
	  ws=$${SANDBOX_WORKSPACE_PROVIDER:-local}; rt=$${SANDBOX_RUNTIME_PROVIDER:-local_subprocess}; \
	  echo "─── sandbox: $$ws/$$rt ───"; \
	  if [ "$$ws" != "daytona" ] && [ "$$rt" != "daytona" ]; then exit 0; fi; \
	  if [ -n "$${DAYTONA_API_KEY:-}" ] && [ ! -f app/src/sandbox/.env.daytona.local ]; then \
	    echo "    using configured Daytona at $${DAYTONA_API_URL:-(unset)}"; \
	    exit 0; \
	  fi; \
	  echo "─── booting local Daytona stack ───"; \
	  $(DAYTONA_PYTHON) bash app/src/sandbox/scripts/setup-daytona-local.sh'

sandbox-status: ## Print which sandbox provider+runtime are currently selected
	@$(LOAD_ENV) $(SANDBOX_OVERRIDE) bash -c 'echo "workspace: $${SANDBOX_WORKSPACE_PROVIDER:-local (default)}"; \
	  echo "runtime:   $${SANDBOX_RUNTIME_PROVIDER:-local_subprocess (default)}"; \
	  if [ "$${SANDBOX_WORKSPACE_PROVIDER:-}" = "daytona" ] || [ "$${SANDBOX_RUNTIME_PROVIDER:-}" = "daytona" ]; then \
	    echo "daytona url: $${DAYTONA_API_URL:-(unset)}"; \
	    echo "daytona key: $${DAYTONA_API_KEY:+set}$${DAYTONA_API_KEY:-MISSING — run \`make daytona-up\`}"; \
	  fi'

daytona-up: ## Start the local Daytona compose stack and write .env.daytona.local
	$(DAYTONA_PYTHON) bash app/src/sandbox/scripts/setup-daytona-local.sh

daytona-down: ## Tear down the local Daytona stack (only potpie-managed sandboxes)
	bash app/src/sandbox/scripts/teardown-daytona-local.sh

##@ Tests

test: ## Run full test suite (unit → integration → real_parse)
	@$(LOAD_ENV) uv run python scripts/run_tests.py

test-unit: ## Run unit tests only (no infra required)
	@uv run python scripts/run_tests.py --unit-only

test-integration: ## Run integration tests (requires infra up)
	@$(LOAD_ENV) uv run python scripts/run_tests.py --integration-only

test-real-parse: ## Run real_parse tests (requires infra up; slow)
	@$(LOAD_ENV) uv run python scripts/run_tests.py --real-parse-only

test-stress: ## Run stress tests
	@$(LOAD_ENV) RUN_STRESS=1 uv run python scripts/run_tests.py --stress-only

test-cov: ## Run full suite with coverage report (htmlcov/)
	@$(LOAD_ENV) uv run python scripts/run_tests.py --coverage

##@ Code quality

lint: ## Run ruff check
	uv run ruff check app/ tests/

format: ## Run ruff format
	uv run ruff format app/ tests/

fix: ## Run ruff check --fix and format
	uv run ruff check --fix app/ tests/
	uv run ruff format app/ tests/

precommit: ## Run all pre-commit hooks against all files
	uv run pre-commit run --all-files

##@ Cleanup

clean: ## Remove caches, coverage, and build artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
