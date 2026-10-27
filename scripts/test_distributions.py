"""Install verified CI artifacts and exercise the product outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from distribution_manifest import load_artifacts, require_product_pair, select_wheel


PROFILES = {
    "base": {"potpie": "", "potpie-context-engine": ""},
    "engine": {"potpie-context-engine": ""},
    "full": {"potpie": "all", "potpie-context-engine": "all"},
    "engine-full": {"potpie-context-engine": "all"},
    "native": {"potpie": "auth,telemetry", "potpie-context-engine": ""},
    "parsing": {"potpie-parsing": ""},
    "integrations": {"potpie-integrations": ""},
}
MODULES = {
    "potpie": "potpie",
    "potpie-context-engine": "potpie_context_engine",
    "potpie-parsing": "parsing.parsing_rs",
    "potpie-integrations": "integrations",
}


def install_targets(artifacts: list[dict], profile: str, python: str, target: str):
    selected = {
        package: select_wheel(artifacts, package, python, target)
        for package in PROFILES[profile]
    }
    if "potpie" in selected:
        require_product_pair(selected["potpie"], selected["potpie-context-engine"])
    # Root/local extras select this native dependency on Linux ARM. Supplying
    # its artifact explicitly prevents silently validating a published wheel.
    if target == "linux-aarch64" and "potpie" in selected:
        selected["potpie-falkordblite"] = select_wheel(
            artifacts, "potpie-falkordblite", python, target
        )
    requirements = []
    for package, artifact in selected.items():
        extra = PROFILES[profile].get(package, "")
        requirements.append(str(artifact["file"]) + (f"[{extra}]" if extra else ""))
    return selected, requirements


def check_installation(selected: dict[str, dict], *, profile: str) -> None:
    expected = {
        package: {"version": artifact["version"], "module": MODULES.get(package)}
        for package, artifact in selected.items()
    }
    # -I ignores the checkout, PYTHONPATH and user site-packages. The script
    # checks both installed versions and the actual origin of imported code.
    code = """
import importlib, importlib.metadata, importlib.util, json, pathlib, sys, sysconfig
expected, profile = json.loads(sys.argv[1]), sys.argv[2]
sites = {pathlib.Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
for package, row in expected.items():
    actual = importlib.metadata.version(package)
    if actual != row['version']:
        raise RuntimeError(f'{package}: expected {row["version"]}, installed {actual}')
    if row['module']:
        module = importlib.import_module(row['module'])
        origin = pathlib.Path(module.__file__).resolve()
        if not any(origin.is_relative_to(site) for site in sites):
            raise RuntimeError(f'Source import rather than installed artifact: {origin}')
        print(f'{package}=={actual}: {origin}')
if profile in ('engine', 'engine-full') and importlib.util.find_spec('potpie') is not None:
    raise RuntimeError('The independent engine installation unexpectedly requires the root package')
if profile == 'base':
    for name in ('torch', 'sentence_transformers', 'transformers'):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f'{name} reached a base installation')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, json.dumps(expected), profile], check=True
    )


def check_product_contents(artifact: dict) -> None:
    with ZipFile(artifact["file"]) as archive:
        names = archive.namelist()
    ui = "potpie/daemon/http/ui/frontend/"
    if f"{ui}dist/index.html" not in names:
        raise ValueError("The root wheel is missing its compiled graph explorer")
    if any("/tests/" in name or "node_modules" in name for name in names):
        raise ValueError("The root wheel includes tests or node_modules")
    if any(name.startswith(ui) and not name.startswith(ui + "dist/") for name in names):
        raise ValueError("The root wheel includes frontend build inputs")
    if any(name.startswith("potpie/context-engine/") for name in names):
        raise ValueError("The root wheel includes the engine source project")


def _json_cli(executable: Path, cwd: Path, env: dict, *args: str) -> dict:
    result = subprocess.run(
        [str(executable), "--json", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    value = json.loads(result.stdout)
    if value.get("ok") is False:
        raise RuntimeError(f"CLI {' '.join(args)} failed: {value}")
    detail = value.get("result", value)
    if not isinstance(detail, dict):
        raise ValueError(f"CLI {' '.join(args)} returned no structured result: {value}")
    return detail


def product_journey(target: str) -> None:
    executable = Path(sys.executable).parent / (
        "potpie.exe" if sys.platform == "win32" else "potpie"
    )
    with tempfile.TemporaryDirectory(prefix="potpie-artifact-") as temporary:
        work = Path(temporary)
        repo = work / "repo"
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.email=ci@potpie.ai",
                "-c",
                "user.name=Potpie CI",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "Artifact acceptance",
            ],
            check=True,
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        env.update(
            CONTEXT_ENGINE_HOME=str(work / "home"),
            POTPIE_HARNESS_HOME=str(work / "harness"),
            XDG_CONFIG_HOME=str(work / "config"),
            POTPIE_TELEMETRY_DISABLED="1",
            CONTEXT_ENGINE_EMBEDDER="hashing",
            PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
            _TYPER_FORCE_DISABLE_TERMINAL="1",
        )
        backend = "ladybug" if target.startswith("windows-") else "falkordb_lite"
        _json_cli(
            executable,
            repo,
            {**env, "CONTEXT_ENGINE_HOST_MODE": "in_process"},
            "setup",
            "--repo",
            ".",
            "--pot",
            "artifact-smoke",
            "--agent",
            "claude",
            "--backend",
            backend,
            "--yes",
            "--in-process",
        )
        mutation = work / "mutation.json"
        mutation.write_text(
            json.dumps(
                {
                    "operations": [
                        {
                            "op": "link_entities",
                            "subgraph": "infra_topology",
                            "subject": {
                                "key": "service:artifact-web",
                                "type": "Service",
                            },
                            "predicate": "DEPENDS_ON",
                            "object": {
                                "key": "service:artifact-api",
                                "type": "Service",
                            },
                            "truth": "source_observation",
                            "description": "artifact web depends on artifact api",
                            "evidence": [
                                {
                                    "source_ref": "test:installed-artifacts",
                                    "authority": "repository_metadata",
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            _json_cli(executable, repo, env, "daemon", "start")
            proposal = _json_cli(
                executable, repo, env, "graph", "propose", "--file", str(mutation)
            )
            committed = _json_cli(
                executable,
                repo,
                env,
                "graph",
                "commit",
                proposal["plan_id"],
                "--verify",
            )
            if committed.get("status") != "committed":
                raise RuntimeError(f"Artifact mutation was not committed: {committed}")
            result = _json_cli(
                executable,
                repo,
                env,
                "graph",
                "read",
                "--subgraph",
                "infra_topology",
                "--view",
                "service_neighborhood",
                "--scope",
                "service:artifact-web",
            )
            if not result.get("items"):
                raise RuntimeError(
                    "The installed daemon could not read its committed graph data"
                )
        finally:
            subprocess.run(
                [str(executable), "daemon", "stop"],
                cwd=repo,
                env=env,
                check=False,
                timeout=30,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run artifact acceptance inside a clean virtual environment")
    operating_system = {"win32": "windows", "darwin": "macos"}.get(
        sys.platform, sys.platform
    )
    architecture = platform.machine().lower()
    if operating_system == "windows" and architecture == "x86_64":
        architecture = "amd64"
    elif operating_system == "linux" and architecture == "arm64":
        architecture = "aarch64"
    if args.target != f"{operating_system}-{architecture}":
        raise ValueError(
            f"Expected native {args.target}, found {operating_system}-{architecture}"
        )
    artifacts = load_artifacts(args.artifacts, args.commit)
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    selected, requirements = install_targets(
        artifacts, args.profile, python, args.target
    )
    binary_only = ["--only-binary=:all:"] if args.profile in {"base", "engine"} else []
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *binary_only, *requirements],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    with tempfile.TemporaryDirectory(prefix="potpie-install-origin-") as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            check_installation(selected, profile=args.profile)
        finally:
            os.chdir(previous)
    if "potpie" in selected:
        check_product_contents(selected["potpie"])
        if not args.install_only:
            product_journey(args.target)


if __name__ == "__main__":
    main()
