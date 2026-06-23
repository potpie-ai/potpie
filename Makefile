# Potpie root Makefile
#
# Installs the local `potpie` CLI package globally. The day-to-day dev
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

# The user-facing `potpie` package depends on the context-engine implementation
# package and owns the public console scripts. `make cli-install` installs the
# root package as a uv tool in EDITABLE mode so the local behavior matches
# `pip install potpie`.
CE_DIR := potpie/context-engine
UI_FRONTEND_DIR := $(CE_DIR)/src/context_engine/adapters/inbound/http/ui/frontend
CLI_TOOL := potpie
CLI_PYTHON ?= >=3.12,<3.14

ui-build: ## Build the graph-explorer SPA (npm install + vite) into frontend/dist
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not installed — see https://nodejs.org/"; exit 1; }
	cd $(UI_FRONTEND_DIR) && npm install && npm run build

cli-install: ui-build ## Install potpie + potpie-mcp globally (editable, all extras). Re-run after dep/entrypoint changes.
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv not installed — see https://docs.astral.sh/uv/"; exit 1; }
	@# Stop any old detached daemon before replacing the tool env; otherwise the
	@# fresh CLI can still talk to a daemon running the previous Python/backend.
	@{ if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi; } || true
	@# Drop stale tool names that may still own the potpie script.
	-@uv tool uninstall context-engine >/dev/null 2>&1 || true
	-@uv tool uninstall potpie-context-engine >/dev/null 2>&1 || true
	uv tool install --python '$(CLI_PYTHON)' --force --editable "."
	@case ":$$PATH:" in \
	  *":$$HOME/.local/bin:"*) echo "✓ potpie installed (editable). Try: potpie --help";; \
	  *) echo "✓ installed, but $$HOME/.local/bin is not on PATH — run: uv tool update-shell, then restart your shell";; \
	esac

cli-update: cli-install ## Refresh the global install (alias for cli-install; code is already live via editable mode)

cli-uninstall: ## Remove the global potpie CLI
	@{ if command -v potpie >/dev/null 2>&1; then potpie daemon stop >/dev/null 2>&1; fi; } || true
	-uv tool uninstall $(CLI_TOOL)
	-@uv tool uninstall potpie-context-engine >/dev/null 2>&1 || true
	-@uv tool uninstall context-engine >/dev/null 2>&1 || true

cli-status: ## Show the global potpie install + run a quick health check
	@uv tool list 2>/dev/null | grep -A3 '^$(CLI_TOOL) ' || echo "$(CLI_TOOL): not installed (run: make cli-install)"
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
