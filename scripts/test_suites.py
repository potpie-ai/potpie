"""Run an owned test suite without mixing its Python test package with another."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "root": ROOT,
    "engine": ROOT / "potpie/context-engine",
    "parsing": ROOT / "potpie/parsing",
    "integrations": ROOT / "potpie/integrations",
}
WINDOWS_ROOT = (
    "test_daemon_liveness.py",
    "test_daemon_spawn.py",
    "test_daemon_recorded_pid.py",
    "test_daemon_logs.py",
    "test_cli_daemon_status_logs.py",
    "test_git_probe.py",
    "test_cli_install_status_path.py",
    "test_local_extra_windows_markers.py",
    "test_local_extra_next_action.py",
    "test_arm_wheels.py",
    "test_ladybug_runtime.py",
    "test_ladybug_registry.py",
    "test_cli_bootstrap_status.py",
    "test_runtime_composition.py::test_local_runtime_uses_platform_backend_default",
    "test_runtime_composition.py::test_default_backend_preserves_primary_env_precedence",
    "test_runtime_composition.py::test_default_backend_ignores_blank_primary_env",
)
WINDOWS_ENGINE = (
    "test_ladybug_roundtrip.py",
    "test_ladybug_windows_bootstrap.py",
    "test_settings_graph_backend.py",
    "test_graph_backend_conformance.py",
    "test_graph_reliability.py",
    "test_public_semantic_mutation_reliability.py",
    "test_record_recovery.py",
    "test_memory_embedded_atomicity.py",
    "test_plan_storage_reliability.py",
    "test_mutation_apply_once.py::test_embedded_concurrent_processes_share_one_durable_receipt",
    "test_graph_workbench_plans.py::test_embedded_competing_processes_atomically_compare_and_apply",
    "test_graph_workbench_plans.py::test_idempotency_key_different_bodies_race_atomically_across_processes",
    "test_graph_workbench_plans.py::test_local_json_plan_store_compare_and_set_is_process_safe",
)


def suite_environment(python: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    # Keep the venv path, rather than resolving its Python symlink to the base
    # interpreter. Product subprocesses must find this installation's CLI.
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    env["PYO3_PYTHON"] = python
    env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="commit.gpgsign",
        GIT_CONFIG_VALUE_0="false",
    )
    return env


def resolve_tests(package: Path, selectors: tuple[str, ...]) -> list[str]:
    """Target owned scenarios by unique filenames, independent of domain folders."""
    result = []
    for selector in selectors:
        name, separator, case = selector.partition("::")
        paths = list((package / "tests").rglob(name))
        if len(paths) != 1:
            raise ValueError(f"Expected one owned test file {name}, found {len(paths)}")
        result.append(str(paths[0]) + (separator + case if separator else ""))
    return result


def pytest_command(
    package: Path,
    *,
    python: str,
    profile: str,
    selectors: tuple[str, ...] = (),
    tier: str | None = None,
    strict: bool = False,
    keyword: str | None = None,
) -> list[str]:
    tests = package / "tests"
    # Seed only the test package, rather than putting the product checkout on
    # sys.path. This supports explicit tests.support imports in isolated runs.
    bootstrap = """
import importlib.util, pathlib, sys, types
tests = pathlib.Path(sys.argv.pop(1))
support = pathlib.Path(sys.argv.pop(1))
scripts = types.ModuleType('scripts')
scripts.__path__ = [str(support)]
sys.modules['scripts'] = scripts
if (tests / '__init__.py').exists():
    spec = importlib.util.spec_from_file_location('tests', tests / '__init__.py', submodule_search_locations=[str(tests)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['tests'] = module
    spec.loader.exec_module(module)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
    command = [
        python,
        "-I",
        "-c",
        bootstrap,
        str(tests),
        str(ROOT / "scripts"),
        "--import-mode=importlib",
        "-c",
        str(package / "pyproject.toml"),
    ]
    if selectors:
        command.extend(resolve_tests(package, selectors))
    else:
        command.append(str(tests))
    if profile == "source":
        command.extend(["-m", "source_contract"])
        if package == ROOT:
            command.append(str(ROOT / "scripts/tests"))
    elif profile != "local":
        command.extend(["-m", "not source_contract"])
    elif package == ROOT:
        command.append(str(ROOT / "scripts/tests"))
    if package.name == "context-engine" and (tier or strict):
        command.extend(["-p", "tests.required_acceptance"])
        if tier:
            command.extend(["--acceptance-tier", tier])
    if keyword:
        command.extend(["-k", keyword])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=PACKAGES)
    parser.add_argument(
        "--profile",
        choices=("local", "installed", "source", "windows"),
        default="local",
    )
    parser.add_argument("--tier", choices=("general", "graph", "restart"))
    parser.add_argument("--checks", choices=("all", "python"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    interpreter = shutil.which(args.python)
    if interpreter is None:
        parser.error(f"Python interpreter not found: {args.python}")
    args.python = os.path.abspath(interpreter)
    if args.tier and args.suite != "engine":
        parser.error("Acceptance tiers belong to the engine suite")
    if args.profile == "windows" and args.suite not in {"root", "engine"}:
        parser.error("The Windows profile belongs to root or engine")
    package = PACKAGES[args.suite]
    calls = []
    if args.profile == "windows":
        if args.suite == "root":
            calls.append(
                pytest_command(
                    package,
                    python=args.python,
                    profile=args.profile,
                    selectors=WINDOWS_ROOT,
                )
            )
        else:
            calls.append(
                pytest_command(
                    package,
                    python=args.python,
                    profile=args.profile,
                    selectors=WINDOWS_ENGINE[:5],
                    strict=True,
                    keyword="ladybug or settings_graph_backend",
                )
            )
            calls.append(
                pytest_command(
                    package,
                    python=args.python,
                    profile=args.profile,
                    selectors=WINDOWS_ENGINE[4:],
                    strict=True,
                )
            )
    elif args.suite == "engine" and args.profile == "installed":
        for tier in [args.tier] if args.tier else ["general", "graph", "restart"]:
            calls.append(
                pytest_command(
                    package, python=args.python, profile=args.profile, tier=tier
                )
            )
    else:
        calls.append(
            pytest_command(
                package, python=args.python, profile=args.profile, tier=args.tier
            )
        )
    if args.suite == "parsing" and args.profile == "installed":
        calls.append(pytest_command(package, python=args.python, profile="source"))
    if args.suite == "parsing" and args.checks == "all":
        calls.append(
            [
                "cargo",
                "test",
                "--manifest-path",
                str(package / "Cargo.toml"),
                "--no-default-features",
            ]
        )
    if (
        args.suite == "root"
        and args.checks == "all"
        and args.profile in {"local", "installed"}
    ):
        calls.extend(
            [
                [
                    "npm",
                    "--prefix",
                    str(ROOT / "tests/docs"),
                    "ci",
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                ],
                ["npm", "--prefix", str(ROOT / "tests/docs"), "test"],
                [
                    "npm",
                    "--prefix",
                    str(ROOT / "potpie/daemon/http/ui/frontend"),
                    "test",
                ],
            ]
        )
    if args.dry_run:
        print(json.dumps(calls, indent=2))
        return
    if args.profile in {"installed", "windows"}:
        module = {
            "root": "potpie",
            "engine": "potpie_context_engine",
            "parsing": "parsing",
            "integrations": "integrations",
        }[args.suite]
        code = """
import importlib.util, pathlib, sys, sysconfig
spec = importlib.util.find_spec(sys.argv[1])
sites = {pathlib.Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
if spec is None or not any(pathlib.Path(spec.origin).resolve().is_relative_to(site) for site in sites):
    raise RuntimeError('Installed profiles require wheel imports; prepare the environment with scripts/test_distributions.py')
"""
        subprocess.run([args.python, "-I", "-c", code, module], check=True)
    env = suite_environment(args.python)
    if args.profile == "windows":
        env["POTPIE_REQUIRE_LADYBUG"] = "1"
    with tempfile.TemporaryDirectory(prefix=f"potpie-{args.suite}-tests-") as temporary:
        for index, command in enumerate(calls):
            if args.profile == "windows" and args.suite == "engine":
                env["POTPIE_RELIABILITY_PROFILES"] = (
                    "ladybug" if index == 0 else "in_memory,embedded"
                )
            # Editable local installs remain usable; installed profiles never
            # resolve products from the checkout working directory.
            cwd = package if args.profile == "local" else Path(temporary)
            print(f"Running {args.suite} ({args.profile}): {command[0]}", flush=True)
            subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
