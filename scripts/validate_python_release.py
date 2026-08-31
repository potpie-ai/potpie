#!/usr/bin/env python3
"""Validate Potpie Python release inputs and emit release metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError as exc:  # pragma: no cover - exercised in CI bootstrap failures.
    raise SystemExit(
        "Missing dependency: packaging. Install it with `python -m pip install packaging`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
PYPI_BASE = "https://pypi.org/pypi"
CONTEXT_ENGINE_NAME = "potpie-context-engine"
PACKAGE_SOURCES = {
    "context-engine": (
        "potpie/context-engine/pyproject.toml",
        CONTEXT_ENGINE_NAME,
    ),
    "potpie": ("pyproject.toml", "potpie"),
}
PACKAGE_KEYS_BY_SCOPE = {
    "all": ("context-engine", "potpie"),
    "context-engine": ("context-engine",),
    "potpie": ("potpie",),
}


@dataclass(frozen=True)
class PackageInfo:
    key: str
    name: str
    version: str
    source: str

    @property
    def parsed_version(self) -> Version:
        return Version(self.version)


def fail(message: str) -> None:
    print(f"release validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def pyproject_package(key: str) -> PackageInfo:
    path, expected_name = PACKAGE_SOURCES[key]
    project = load_toml(path).get("project", {})
    name = project.get("name")
    version = project.get("version")
    if name != expected_name:
        fail(f"{path} has project.name={name!r}; expected {expected_name!r}")
    if not isinstance(version, str) or not version:
        fail(f"{path} must define a static project.version")
    return PackageInfo(key=key, name=name, version=version, source=path)


def channel_for_version(package: PackageInfo) -> str:
    try:
        version = package.parsed_version
    except InvalidVersion as exc:
        fail(f"{package.name} version {package.version!r} is not PEP 440: {exc}")

    if version.local is not None:
        fail(
            f"{package.name} version {package.version!r} must not use a local '+...' segment"
        )
    if version.is_devrelease:
        fail(f"{package.name} version {package.version!r} must not be a dev release")
    if version.pre is None:
        return "final"
    if version.pre[0] == "b":
        if "b" not in package.version.lower() or "beta" in package.version.lower():
            fail(
                f"{package.name} beta version must use compact bN syntax, not {package.version!r}"
            )
        return "beta"
    if version.pre[0] == "rc":
        return "rc"
    fail(f"{package.name} version {package.version!r} must be final, beta bN, or rcN")


def infer_channel(packages: list[PackageInfo]) -> str:
    channels = {channel_for_version(package) for package in packages}
    if len(channels) != 1:
        rendered = ", ".join(
            f"{package.name}=={package.version} ({channel_for_version(package)})"
            for package in packages
        )
        fail(f"selected packages must use one release channel; got {rendered}")
    return next(iter(channels))


def package_requirement(package_key: str, dependency_name: str) -> Requirement:
    source, _ = PACKAGE_SOURCES[package_key]
    dependencies = load_toml(source).get("project", {}).get("dependencies", [])
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as exc:
            fail(f"invalid dependency in {source}: {dependency!r}: {exc}")
        if canonicalize_name(requirement.name) == canonicalize_name(dependency_name):
            return requirement
    fail(f"{source} must depend on {dependency_name}")


def exact_requirement_version(package: PackageInfo, dependency_name: str) -> str:
    requirement = package_requirement(package.key, dependency_name)
    specifiers = list(requirement.specifier)
    if (
        requirement.marker is not None
        or len(specifiers) != 1
        or specifiers[0].operator != "=="
    ):
        fail(
            f"{package.name} must unconditionally pin {dependency_name} to exactly "
            "one version; "
            f"found {requirement.specifier or '<unversioned>'}"
        )
    pinned_version = specifiers[0].version
    try:
        Version(pinned_version)
    except InvalidVersion:
        fail(
            f"{package.name} must pin {dependency_name} to a concrete PEP 440 "
            f"version; found =={pinned_version}"
        )
    return pinned_version


def resolve_context_engine_version(
    selected_keys: set[str],
    packages: dict[str, PackageInfo],
) -> str:
    if "potpie" not in selected_keys:
        return packages["context-engine"].version

    potpie = packages["potpie"]
    pinned_version = exact_requirement_version(potpie, CONTEXT_ENGINE_NAME)
    if "context-engine" not in selected_keys:
        return pinned_version

    engine = packages["context-engine"]
    if pinned_version != engine.version:
        fail(
            f"{potpie.name} must pin {engine.name} exactly as =={engine.version} "
            f"for a combined release; found =={pinned_version}"
        )
    return pinned_version


def package_version_exists(package_name: str, version: str) -> bool:
    name = urllib.parse.quote(package_name)
    release = urllib.parse.quote(version)
    url = f"{PYPI_BASE}/{name}/{release}/json"
    # The scheme and host come only from the fixed HTTPS constants above.
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": "potpie-release-validator/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        fail(f"could not check pypi for {package_name}=={version}: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        fail(f"could not check pypi for {package_name}=={version}: {exc.reason}")
    return False


def validate_index_availability(selected: list[PackageInfo]) -> None:
    for package in selected:
        if package_version_exists(package.name, package.version):
            fail(f"{package.name}=={package.version} already exists on pypi")


def validate_external_dependencies(
    selected_keys: set[str],
    engine_name: str,
    engine_version: str,
) -> None:
    if "potpie" not in selected_keys or "context-engine" in selected_keys:
        return
    if not package_version_exists(engine_name, engine_version):
        fail(
            f"{engine_name}=={engine_version} must already exist on pypi "
            "for a potpie-only release"
        )


def validate_release_source() -> None:
    ref_type = os.getenv("GITHUB_REF_TYPE", "")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    default_branch = os.getenv("REPOSITORY_DEFAULT_BRANCH", "")
    if not default_branch:
        fail("release validation requires REPOSITORY_DEFAULT_BRANCH")
    if ref_type != "branch" or ref_name != default_branch:
        fail(
            "release workflow must run from the repository default branch "
            f"{default_branch!r}; got ref_type={ref_type!r}, ref_name={ref_name!r}"
        )

    commit_sha = os.getenv("GITHUB_SHA", "")
    if not commit_sha:
        fail("release validation requires GITHUB_SHA")
    git = shutil.which("git")
    if git is None:
        fail("could not verify release commit ancestry: git is unavailable")

    result = subprocess.run(  # noqa: S603
        [
            git,
            "merge-base",
            "--is-ancestor",
            commit_sha,
            f"origin/{default_branch}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        fail(
            f"release commit {commit_sha} is not reachable from origin/{default_branch}"
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"git exited {result.returncode}"
        fail(f"could not verify release commit ancestry: {detail}")


def github_run_url() -> str:
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server_url}/{repository}/actions/runs/{run_id}"
    return ""


def emit_metadata(
    output_dir: str,
    scope: str,
    selected: list[PackageInfo],
    channel: str,
) -> Path:
    output_path_dir = ROOT / output_dir
    output_path_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_dir / "release-metadata.json"
    metadata = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "release_scope": scope,
        "channel": channel,
        "commit_sha": os.getenv("GITHUB_SHA", ""),
        "ref": os.getenv("GITHUB_REF", ""),
        "ref_type": os.getenv("GITHUB_REF_TYPE", ""),
        "ref_name": os.getenv("GITHUB_REF_NAME", ""),
        "repository": os.getenv("GITHUB_REPOSITORY", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "run_url": github_run_url(),
        "release_tags": {
            package.name: f"{package.name}-v{package.version}" for package in selected
        },
        "packages": {
            package.name: {
                "version": package.version,
                "source": package.source,
            }
            for package in selected
        },
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def emit_github_outputs(
    scope: str,
    packages: dict[str, PackageInfo],
    context_engine_version: str,
    metadata_path: Path,
) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    try:
        metadata_output = str(metadata_path.relative_to(ROOT))
    except ValueError:
        metadata_output = str(metadata_path)
    lines = [
        f"release_scope={scope}",
        f"metadata_path={metadata_output}",
        f"context_engine_version={context_engine_version}",
    ]
    if engine := packages.get("context-engine"):
        lines.append(f"context_engine_tag={engine.name}-v{engine.version}")
    if potpie := packages.get("potpie"):
        lines.extend(
            [
                f"potpie_version={potpie.version}",
                f"potpie_tag={potpie.name}-v{potpie.version}",
            ]
        )
    with Path(github_output).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-scope",
        required=True,
        choices=sorted(PACKAGE_KEYS_BY_SCOPE),
    )
    parser.add_argument("--output-dir", default="release-metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope = args.release_scope
    selected_keys = PACKAGE_KEYS_BY_SCOPE[scope]
    packages = {key: pyproject_package(key) for key in selected_keys}
    selected = [packages[key] for key in selected_keys]
    selected_key_set = set(selected_keys)
    channel = infer_channel(selected)
    context_engine_version = resolve_context_engine_version(
        selected_key_set,
        packages,
    )

    validate_release_source()
    validate_index_availability(selected)
    validate_external_dependencies(
        selected_key_set,
        CONTEXT_ENGINE_NAME,
        context_engine_version,
    )
    metadata_path = emit_metadata(args.output_dir, scope, selected, channel)
    emit_github_outputs(scope, packages, context_engine_version, metadata_path)

    print("release validation passed")
    print(f"- scope: {scope}")
    print(f"- channel: {channel}")
    for package in selected:
        print(f"- {package.name}=={package.version} ({package.source})")
    try:
        metadata_display = str(metadata_path.relative_to(ROOT))
    except ValueError:
        metadata_display = str(metadata_path)
    print(f"- metadata: {metadata_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
