# Potpie root Makefile
#
# Installs the local `potpie` context-engine CLI globally. The day-to-day dev
# stack (infra, API, worker, migrations, tests) lives in legacy/Makefile — run
# those with `make -C legacy <target>`.
#
# Quick start:
#   make cli-install   # build graph-explorer UI + install potpie on your PATH (editable)
#   make cli-status    # confirm the install is healthy
#   make help          # list all targets

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help
.PHONY: help ui-build cli-install cli-update cli-uninstall cli-status

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
	     /^##@ / {sub(/^##@ /, ""); printf "\n\033[1m%s\033[0m\n", $$0; next} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)

##@ Context Engine CLI (potpie)

# The `potpie` CLI lives in $(CE_DIR) and installs as a uv tool in EDITABLE mode:
# the global `potpie` / `potpie-mcp` on your PATH run straight from the repo
# source, so code changes — including the in-process daemon — are live on the
# next invocation with no reinstall. The CLI also auto-loads this repo's .env
# (Neo4j/Postgres creds) regardless of which directory you run it from, so a
# global `potpie` reaches your local backends out of the box. Re-run
# `make cli-install` only when dependencies or entry points change.
CE_DIR := potpie/context-engine
UI_FRONTEND_DIR := $(CE_DIR)/adapters/inbound/http/ui/frontend
CLI_TOOL := potpie-context-engine
CLI_PYTHON ?= >=3.12,<3.14

ui-build: ## Build the graph-explorer SPA (npm install + vite) into frontend/dist
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not installed — see https://nodejs.org/"; exit 1; }
	cd $(UI_FRONTEND_DIR) && npm install && npm run build

cli-install: ui-build ## Install potpie + potpie-mcp globally (editable, all extras). Re-run after dep/entrypoint changes.
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv not installed — see https://docs.astral.sh/uv/"; exit 1; }
	@# Drop the pre-rename "context-engine" tool if a stale copy is lingering.
	-@uv tool uninstall context-engine >/dev/null 2>&1 || true
	@# Stop any old detached daemon before replacing the tool env; otherwise the
	@# fresh CLI can still talk to a daemon running the previous Python/backend.
	-@if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi
	uv tool install --python '$(CLI_PYTHON)' --force --editable "./$(CE_DIR)[all]"
	@case ":$$PATH:" in \
	  *":$$HOME/.local/bin:"*) echo "✓ potpie installed (editable). Try: potpie --help";; \
	  *) echo "✓ installed, but $$HOME/.local/bin is not on PATH — run: uv tool update-shell, then restart your shell";; \
	esac

cli-update: cli-install ## Refresh the global install (alias for cli-install; code is already live via editable mode)

cli-uninstall: ## Remove the global potpie CLI
	-@if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi
	-uv tool uninstall $(CLI_TOOL)
	-@uv tool uninstall context-engine >/dev/null 2>&1 || true

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
