#!/usr/bin/env bash
# Run pytest for the project in phases: Unit (auth) → Integration.
#
# Usage:
#   ./scripts/run_tests.sh              run all tests (unit then integration)
#   ./scripts/run_tests.sh -c            run all tests with coverage (term + html)
#   ./scripts/run_tests.sh --coverage    same as -c
#   ./scripts/run_tests.sh -c -k "auth" run with coverage, only tests matching "auth"
#   ./scripts/run_tests.sh -h            show help
#
# Phases:
#   Phase 1: Unit tests (auth)  → app/modules/auth/tests
#   Phase 2: Integration tests  → tests/ (conversations, code_provider/github)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

BANNER_WIDTH=72
COVERAGE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --coverage|-c)
      COVERAGE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--coverage|-c] [pytest args...]"
      echo "  -c, --coverage  run with coverage (term + html report in htmlcov/)"
      echo "  -h, --help      show this help"
      echo "  Any other args are passed to pytest (e.g. -k 'auth', -v, path)."
      echo ""
      echo "Phases: Unit (auth) → Integration (conversations, github)."
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Ensure app is importable (auth tests don't add project root to path)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

print_phase_banner() {
  echo ""
  printf '=%.0s' $(seq 1 $BANNER_WIDTH); echo ""
  echo "  PHASE: $1"
  printf '=%.0s' $(seq 1 $BANNER_WIDTH); echo ""
  echo ""
}

run_phase() {
  local phase_name="$1"
  shift
  if [[ "$COVERAGE" == true ]]; then
    if [[ "$phase_name" == "Unit (auth)" ]]; then
      uv run pytest "$@" --cov=app --cov-report=term-missing "${EXTRA_ARGS[@]}"
    else
      uv run pytest "$@" --cov=app --cov-append --cov-report=term-missing --cov-report=html "${EXTRA_ARGS[@]}"
    fi
  else
    uv run pytest "$@" "${EXTRA_ARGS[@]}"
  fi
  local code=$?
  echo ""
  printf '─%.0s' $(seq 1 $BANNER_WIDTH); echo ""
  echo "  Phase «$phase_name» finished: $([ $code -eq 0 ] && echo 'OK' || echo 'FAILED') (exit code $code)"
  printf '─%.0s' $(seq 1 $BANNER_WIDTH); echo ""
  echo ""
  return $code
}

# Phase 1: Unit tests (auth)
print_phase_banner "Unit (auth)"
run_phase "Unit (auth)" "app/modules/auth/tests" || exit $?

# Phase 2: Integration tests
print_phase_banner "Integration"
run_phase "Integration" "tests" || exit $?

echo ""
printf '=%.0s' $(seq 1 $BANNER_WIDTH); echo ""
echo "  ALL PHASES PASSED"
printf '=%.0s' $(seq 1 $BANNER_WIDTH); echo ""
echo ""

if [[ "$COVERAGE" == true ]]; then
  echo "HTML report: file://$ROOT_DIR/htmlcov/index.html"
fi
