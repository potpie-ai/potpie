#!/usr/bin/env bash
# One-command test runner. Delegates to scripts/run_tests.py.
# Usage: ./scripts/run_tests.sh [--coverage|-c] [--unit-only | --integration-only | ...] [pytest args...]
set -e
cd "$(dirname "$0")/.."
exec uv run python scripts/run_tests.py "$@"
