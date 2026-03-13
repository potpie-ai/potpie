#!/usr/bin/env bash
# Run pytest for the project. Use --coverage or -c to run with coverage report.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

COVERAGE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --coverage|-c)
      COVERAGE=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Ensure app is importable (auth tests don't add project root to path)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Run auth tests + top-level tests/ (integration, etc.)
TEST_PATHS=("app/modules/auth/tests" "tests")

if [[ "$COVERAGE" == true ]]; then
  uv run pytest "${TEST_PATHS[@]}" --cov=app --cov-report=term-missing --cov-report=html "${EXTRA_ARGS[@]}"
  echo ""
  echo "HTML report: file://$ROOT_DIR/htmlcov/index.html"
else
  uv run pytest "${TEST_PATHS[@]}" "${EXTRA_ARGS[@]}"
fi
