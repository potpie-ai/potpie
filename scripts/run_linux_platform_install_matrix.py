#!/usr/bin/env python3
"""Run Linux platform install smoke tests for the published Potpie CLI.

This script is intentionally separate from ``run_test_plan.py``. It tests the
package as a user would install it on Linux architectures, while the workbook
runner tests deterministic command coverage once an environment already exists.

Default matrix:
  - linux/amd64 with ``uv tool install`` and ``pip install``
  - linux/arm64 with ``uv tool install`` and ``pip install``

Requirements:
  - Docker with multi-platform support.
  - Network access from containers to package indexes.
  - QEMU/binfmt configured if running arm64 containers on an amd64 host.

Use ``--install-build-deps`` as a diagnostic when a dependency falls back to a
source build on clean Linux images. The default intentionally does not install
compilers, because that is the realistic end-user install smoke.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
DEFAULT_PLATFORMS = ("linux/amd64", "linux/arm64")
DEFAULT_INSTALLERS = ("uv", "pip")
DEFAULT_TIMEOUT = 900


@dataclass(frozen=True)
class Case:
    platform: str
    installer: str

    @property
    def name(self) -> str:
        return f"{self.platform.replace('/', '-')}-{self.installer}"


@dataclass
class Result:
    case: Case
    exit_code: int
    output: str
    command: list[str]

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = [
        Case(platform=platform, installer=installer)
        for platform in args.platform
        for installer in args.installer
    ]
    results: list[Result] = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}")
        result = run_case(case, args)
        results.append(result)
        status = "PASS" if result.passed else f"FAIL exit={result.exit_code}"
        print(f"[{case.name}] {status}\n")
        if args.fail_fast and not result.passed:
            break

    write_report(results, args.report)
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    print("=" * 72)
    print(f"Summary: PASS={passed} FAIL={failed} report={args.report}")
    print("=" * 72)
    return 0 if failed == 0 else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        default="potpie",
        help="Package spec to install, e.g. 'potpie' or 'potpie==2.0.0b3'.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Multi-arch Docker image with Python and uv available.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        choices=DEFAULT_PLATFORMS,
        help="Docker platform to test. Repeatable. Defaults to both Linux platforms.",
    )
    parser.add_argument(
        "--installer",
        action="append",
        choices=DEFAULT_INSTALLERS,
        help="Installer to test. Repeatable. Defaults to uv and pip.",
    )
    parser.add_argument(
        "--smoke",
        choices=("install", "cli"),
        default="cli",
        help=(
            "'install' runs version/help only; 'cli' also runs isolated setup, "
            "pot/source/graph/status/skills smoke."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-container timeout in seconds.",
    )
    parser.add_argument(
        "--report",
        default=str(Path("linux-platform-install-results.json")),
        help="JSON report path.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pass --pull always to docker run.",
    )
    parser.add_argument(
        "--install-build-deps",
        action="store_true",
        help=(
            "Install Debian build tools in the container before installing "
            "Potpie. Use this to distinguish clean install failures from "
            "runtime CLI failures."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing case.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print full container output after each case.",
    )
    args = parser.parse_args(argv)
    args.platform = tuple(args.platform or DEFAULT_PLATFORMS)
    args.installer = tuple(args.installer or DEFAULT_INSTALLERS)
    return args


def run_case(case: Case, args: argparse.Namespace) -> Result:
    script = container_script(
        package=args.package,
        installer=case.installer,
        smoke=args.smoke,
        install_build_deps=args.install_build_deps,
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        case.platform,
        "--name",
        f"potpie-platform-smoke-{case.name}",
    ]
    if args.pull:
        command.extend(["--pull", "always"])
    command.extend([args.image, "bash", "-lc", script])

    try:
        proc = subprocess.run(  # noqa: S603 - command is constructed from fixed argv + CLI args.
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        result = Result(
            case=case,
            exit_code=proc.returncode,
            output=output,
            command=command,
        )
    except subprocess.TimeoutExpired as exc:
        partial = ""
        if isinstance(exc.stdout, str):
            partial += exc.stdout
        if isinstance(exc.stderr, str):
            partial += exc.stderr
        result = Result(
            case=case,
            exit_code=124,
            output=f"TIMEOUT after {args.timeout}s\n{partial}".strip(),
            command=command,
        )

    if args.print_output or not result.passed:
        print(result.output)
    return result


def container_script(
    *,
    package: str,
    installer: str,
    smoke: str,
    install_build_deps: bool,
) -> str:
    package_q = shlex.quote(package)
    build_deps = ""
    if install_build_deps:
        build_deps = (
            "apt-get update\n"
            "apt-get install -y --no-install-recommends gcc g++ make\n"
            "rm -rf /var/lib/apt/lists/*\n"
        )

    if installer == "uv":
        install = (
            f"uv tool install {package_q} --force\n"
            "export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "POTPIE_BIN=\"potpie\"\n"
        )
    elif installer == "pip":
        install = (
            "PIP_VENV=\"$(mktemp -d)/venv\"\n"
            "python -m venv \"$PIP_VENV\"\n"
            "\"$PIP_VENV/bin/python\" -m pip install --upgrade pip\n"
            f"\"$PIP_VENV/bin/python\" -m pip install {package_q}\n"
            "POTPIE_BIN=\"$PIP_VENV/bin/potpie\"\n"
        )
    else:  # pragma: no cover - argparse choices guard this.
        raise ValueError(f"unknown installer: {installer}")

    cli_smoke = ""
    if smoke == "cli":
        cli_smoke = r"""
export CONTEXT_ENGINE_HOME="$(mktemp -d /tmp/potpie-context-engine-home.XXXXXX)"
export XDG_CONFIG_HOME="$(mktemp -d /tmp/potpie-xdg-config.XXXXXX)"
export XDG_CACHE_HOME="$(mktemp -d /tmp/potpie-xdg-cache.XXXXXX)"
export XDG_DATA_HOME="$(mktemp -d /tmp/potpie-xdg-data.XXXXXX)"
export XDG_STATE_HOME="$(mktemp -d /tmp/potpie-xdg-state.XXXXXX)"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export CONTEXT_ENGINE_HOST_MODE="in_process"
export CONTEXT_ENGINE_BACKEND="embedded"
export CONTEXT_ENGINE_EMBEDDER="none"
export PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring"

mkdir -p /tmp/potpie-smoke-repo
cd /tmp/potpie-smoke-repo
mkdir -p .git
printf '# Potpie platform smoke\n' > README.md
printf 'def hello():\n    return "world"\n' > app.py

"$POTPIE_BIN" setup --repo . --agent claude --backend embedded --yes --in-process
"$POTPIE_BIN" pot info
"$POTPIE_BIN" source add repo .
"$POTPIE_BIN" source list
"$POTPIE_BIN" graph status
"$POTPIE_BIN" graph catalog --task "platform smoke"
"$POTPIE_BIN" skills install --agent claude --scope project --path .
test -f CLAUDE.md
"$POTPIE_BIN" doctor
test -f "$CONTEXT_ENGINE_HOME/config.json"
test -f "$CONTEXT_ENGINE_HOME/pots.json"
test -f "$CONTEXT_ENGINE_HOME/graph.json"
"""

    return (
        "set -euo pipefail\n"
        "python --version\n"
        "uv --version\n"
        f"{build_deps}"
        f"{install}"
        '"$POTPIE_BIN" --version\n'
        '"$POTPIE_BIN" --help >/tmp/potpie-help.txt\n'
        f"{cli_smoke}"
    )


def write_report(results: list[Result], path: str) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "summary": {
            "pass": sum(1 for result in results if result.passed),
            "fail": sum(1 for result in results if not result.passed),
            "total": len(results),
        },
        "results": [
            {
                "platform": result.case.platform,
                "installer": result.case.installer,
                "name": result.case.name,
                "exit_code": result.exit_code,
                "passed": result.passed,
                "command": result.command,
                "output": result.output,
            }
            for result in results
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
