#!/usr/bin/env python3
"""
Single entry point to run the full test suite. Used by developers and CI.

- Runs tests by phase (unit → integration → real_parse → stress) so output is clear.
- Uses pytest discovery and markers only; no test file paths. New tests under
  tests/unit/ or tests/integration-tests/ are picked up automatically.
- Control via env or flags: SKIP_REAL_PARSE=1, RUN_STRESS=1, or --unit-only, etc.

Usage:
  uv run python scripts/run_tests.py
  uv run python scripts/run_tests.py --unit-only
  SKIP_REAL_PARSE=1 uv run python scripts/run_tests.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"


BANNER_WIDTH = 72
PHASE_REAL_PARSE = "Real parse"


def run_pytest(
    *pytest_args: str,
    extra_env: dict[str, str] | None = None,
    phase_name: str | None = None,
) -> int:
    """Run pytest with project root as cwd; return exit code.
    Uses addopts from pyproject.toml (-v -ra --durations=5) for clearer output."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
    )
    if phase_name is not None:
        status = "FAILED" if result.returncode != 0 else "OK"
        print(f"\n{'─' * BANNER_WIDTH}")
        print(f"  Phase «{phase_name}» finished: {status} (exit code {result.returncode})")
        print(f"{'─' * BANNER_WIDTH}\n")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test suite (unit → integration → real_parse → stress). "
        "Uses markers and testpaths; new tests are discovered automatically.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (tests/unit/, marker: unit).",
    )
    group.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests, excluding stress and real_parse.",
    )
    group.add_argument(
        "--real-parse-only",
        action="store_true",
        help="Run only real_parse tests (Postgres + Neo4j required).",
    )
    group.add_argument(
        "--stress-only",
        action="store_true",
        help="Run only stress tests.",
    )
    parser.add_argument(
        "pytest_extra",
        nargs="*",
        help="Extra arguments passed to pytest (e.g. -x, -k 'test_foo').",
    )
    args = parser.parse_args()

    skip_real_parse = os.environ.get("SKIP_REAL_PARSE", "").strip().lower() in ("1", "true", "yes")
    run_stress = os.environ.get("RUN_STRESS", "").strip().lower() in ("1", "true", "yes")

    def print_phase_banner(name: str) -> None:
        print()
        print("=" * BANNER_WIDTH)
        print(f"  PHASE: {name}")
        print("=" * BANNER_WIDTH)
        print()
        sys.stdout.flush()

    if args.unit_only:
        print_phase_banner("Unit")
        code = run_pytest(
            str(TESTS_DIR / "unit"), "-m", "unit",
            *args.pytest_extra,
            phase_name="Unit",
        )
        return code

    if args.integration_only:
        print_phase_banner("Integration")
        code = run_pytest(
            str(TESTS_DIR / "integration-tests"),
            "-m", "not stress and not real_parse and not github_live",
            *args.pytest_extra,
            phase_name="Integration",
        )
        return code

    if args.real_parse_only:
        print_phase_banner(PHASE_REAL_PARSE)
        code = run_pytest(
            TESTS_DIR, "-m", "real_parse",
            *args.pytest_extra,
            phase_name=PHASE_REAL_PARSE,
        )
        return code

    if args.stress_only:
        print_phase_banner("Stress")
        code = run_pytest(
            TESTS_DIR, "-m", "stress",
            *args.pytest_extra,
            phase_name="Stress",
        )
        return code

    # Full run: unit → integration (no stress/real_parse) → real_parse (optional) → stress (optional)
    phases = [
        ("Unit", [str(TESTS_DIR / "unit"), "-m", "unit"]),
        (
            "Integration",
            [
                str(TESTS_DIR / "integration-tests"),
                "-m", "not stress and not real_parse and not github_live",
            ],
        ),
    ]
    if not skip_real_parse:
        phases.append((PHASE_REAL_PARSE, [str(TESTS_DIR), "-m", "real_parse"]))
    if run_stress:
        phases.append(("Stress", [str(TESTS_DIR), "-m", "stress"]))

    for name, pytest_args in phases:
        print_phase_banner(name)
        code = run_pytest(*pytest_args, *args.pytest_extra, phase_name=name)
        if code != 0:
            return code

    print()
    print("=" * BANNER_WIDTH)
    print("  ALL PHASES PASSED")
    print("=" * BANNER_WIDTH)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
