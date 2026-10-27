# Potpie root Makefile
#
# Builds and installs the local `potpie` CLI and daemon globally. Package-level
# development commands live in the workspaces under `potpie/`.
#
# Quick start:
#   make cli-install   # build graph-explorer UI + install potpie on your PATH (editable)
#   make cli-status    # confirm the install is healthy
#   make help          # list all targets

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help
.PHONY: help ui-build ui-build-ci cli-install cli-update cli-uninstall cli-status test-root test-product test-context-engine test-parsing test-integrations

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
	     /^##@ / {sub(/^##@ /, ""); printf "\n\033[1m%s\033[0m\n", $$0; next} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)

##@ Potpie CLI

# The `potpie` CLI installs from the root distribution in EDITABLE mode:
# the global `potpie` / `potpie-daemon` on your PATH run straight from the repo
# source, so code changes — including the in-process daemon — are live on the
# next invocation with no reinstall. The CLI also auto-loads this repo's .env
# (Neo4j/Postgres creds) regardless of which directory you run it from, so a
# global `potpie` reaches your local backends out of the box. Re-run
# `make cli-install` only when dependencies or entry points change.
CE_DIR := potpie/context-engine
UI_FRONTEND_DIR := potpie/daemon/http/ui/frontend
CLI_TOOL := potpie
CLI_PYTHON ?= >=3.12,<3.14
# Extras for the editable install. The base package leaves out embeddings
# (torch), ingestion, telemetry and the CLI logo; a dev checkout wants them all.
# `make cli-install CLI_EXTRAS=` installs the lean base instead.
CLI_EXTRAS ?= all
CLI_TARGET := $(if $(strip $(CLI_EXTRAS)),.[$(strip $(CLI_EXTRAS))],.)

##@ Tests

TEST_PROFILE ?= local
TEST_CHECKS ?= all
TEST_TIER ?=

test-root: test-product test-parsing test-integrations ## Test the complete root owner in separate component environments

test-product: ## Test root CLI/runtime behavior, artifact controls, docs and graph UI
	uv run --isolated --frozen --all-extras python scripts/test_suites.py root --profile $(TEST_PROFILE) --checks $(TEST_CHECKS)

test-context-engine: ## Test the engine independently; TEST_TIER selects a CI acceptance tier
	uv run --isolated --frozen --package potpie-context-engine --all-extras python scripts/test_suites.py engine --profile $(TEST_PROFILE) $(if $(TEST_TIER),--tier $(TEST_TIER),)

test-parsing: ## Test parsing's Python bindings and native Rust contracts
	uv run --isolated --frozen --package potpie-parsing python scripts/test_suites.py parsing --profile $(TEST_PROFILE) --checks $(TEST_CHECKS)

test-integrations: ## Test the independent integrations package
	uv run --isolated --frozen --package potpie-integrations python scripts/test_suites.py integrations --profile $(TEST_PROFILE)

##@ UI and local installation

ui-build: ## Build the graph-explorer SPA (npm install + vite) into frontend/dist
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not installed — see https://nodejs.org/"; exit 1; }
	cd $(UI_FRONTEND_DIR) && npm install && npm run build

ui-build-ci: ## Reproducibly build the graph-explorer SPA from package-lock.json
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not installed — see https://nodejs.org/"; exit 1; }
	cd $(UI_FRONTEND_DIR) && npm ci && npm run build

cli-install: ui-build ## Install potpie + potpie-daemon globally from the root package. Re-run after dep/entrypoint changes.
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv not installed — see https://docs.astral.sh/uv/"; exit 1; }
	@# Stop any old detached daemon before replacing the tool env; otherwise the
	@# fresh CLI can still talk to a daemon running the previous Python/backend.
	-@if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi
	uv tool install --python '$(CLI_PYTHON)' --force --editable "$(CLI_TARGET)"
	@case ":$$PATH:" in \
	  *":$$HOME/.local/bin:"*) echo "✓ potpie installed (editable). Try: potpie --help";; \
	  *) echo "✓ installed, but $$HOME/.local/bin is not on PATH — run: uv tool update-shell, then restart your shell";; \
	esac

cli-update: cli-install ## Refresh the global install (alias for cli-install; code is already live via editable mode)

cli-uninstall: ## Remove the global potpie CLI
	-@if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi
	-uv tool uninstall $(CLI_TOOL)

cli-status: ## Show the global potpie install + run a quick health check
	@uv tool list 2>/dev/null | grep -A2 '^$(CLI_TOOL)' || echo "$(CLI_TOOL): not installed (run: make cli-install)"
	@printf 'on PATH:  '; command -v potpie || echo "potpie NOT on PATH"
	@if command -v potpie >/dev/null 2>&1; then \
	  py="$$(head -n 1 "$$(command -v potpie)" | sed 's/^#!//')"; \
	  printf 'python:   '; "$$py" --version; \
	else \
	  echo "python:   unknown"; \
	fi
	@potpie --help >/dev/null 2>&1 && echo "health:   ✓ potpie runs current source" || echo "health:   ✗ potpie failed (run: make cli-install)"
	@if [ -f "$(UI_FRONTEND_DIR)/dist/index.html" ]; then \
	  echo "ui:       ✓ graph explorer built ($(UI_FRONTEND_DIR)/dist)"; \
	else \
	  echo "ui:       ✗ not built (run: make ui-build)"; \
	fi
