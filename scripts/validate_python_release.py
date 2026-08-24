#!/usr/bin/env python3
"""Validate Potpie Python release inputs and emit release metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
PYPI_BASE = {
    "pypi": "https://pypi.org/pypi",
    "testpypi": "https://test.pypi.org/pypi",
}
PYPI_REPOSITORY_URL = {
    "pypi": "https://upload.pypi.org/legacy/",
    "testpypi": "https://test.pypi.org/legacy/",
}
PACKAGE_SOURCES = {
    "context-engine": (
        "potpie/context-engine/pyproject.toml",
        "potpie-context-engine",
    ),
    "potpie": ("pyproject.toml", "potpie"),
}
PACKAGE_ALIASES = {
    "context-engine": "context-engine",
    "potpie-context-engine": "context-engine",
    "potpie": "potpie",
}
PACKAGE_KEYS_BY_SCOPE = {
    "all": ("context-engine", "potpie"),
    "context-engine": ("context-engine",),
    "potpie": ("potpie",),
}
TAG_PREFIXES = {
    "context-engine": "potpie-context-engine",
    "potpie": "potpie",
}
DEPENDENCY_KEYS = {
    "potpie": ("context-engine",),
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


def normalize_scope(value: str) -> str:
    scope = PACKAGE_ALIASES.get(value, value)
    if scope not in PACKAGE_KEYS_BY_SCOPE:
        allowed = ", ".join(sorted(PACKAGE_KEYS_BY_SCOPE))
        fail(f"unsupported release scope {value!r}; allowed: {allowed}")
    return scope


def package_keys_for_scope(scope: str) -> tuple[str, ...]:
    return PACKAGE_KEYS_BY_SCOPE[normalize_scope(scope)]


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


def validate_exact_requirement(package: PackageInfo, dependency: PackageInfo) -> None:
    requirement = package_requirement(package.key, dependency.name)
    specifiers = list(requirement.specifier)
    if (
        len(specifiers) != 1
        or specifiers[0].operator != "=="
        or specifiers[0].version != dependency.version
    ):
        fail(
            f"{package.name} must pin {dependency.name} exactly as "
            f"=={dependency.version}; found {requirement.specifier or '<unversioned>'}"
        )


def validate_dependency_chain(packages: dict[str, PackageInfo]) -> None:
    validate_exact_requirement(packages["potpie"], packages["context-engine"])


def package_version_exists(index: str, package_name: str, version: str) -> bool:
    base_url = PYPI_BASE[index]
    name = urllib.parse.quote(package_name)
    release = urllib.parse.quote(version)
    url = f"{base_url}/{name}/{release}/json"
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
        fail(f"could not check {index} for {package_name}=={version}: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        fail(f"could not check {index} for {package_name}=={version}: {exc.reason}")
    return False


def index_for_target(publish_target: str) -> str:
    return "testpypi" if publish_target == "testpypi" else "pypi"


def validate_index_availability(
    publish_target: str,
    selected: list[PackageInfo],
) -> None:
    if publish_target == "build-only":
        return
    index = index_for_target(publish_target)
    for package in selected:
        if package_version_exists(index, package.name, package.version):
            fail(f"{package.name}=={package.version} already exists on {index}")


def validate_external_dependencies(
    publish_target: str,
    selected_keys: set[str],
    packages: dict[str, PackageInfo],
) -> None:
    if publish_target == "build-only":
        return
    index = index_for_target(publish_target)
    required_keys = {
        dependency_key
        for package_key in selected_keys
        for dependency_key in DEPENDENCY_KEYS.get(package_key, ())
        if dependency_key not in selected_keys
    }
    for dependency_key in sorted(required_keys):
        dependency = packages[dependency_key]
        if not package_version_exists(index, dependency.name, dependency.version):
            fail(
                f"{dependency.name}=={dependency.version} must already exist on {index} "
                "when it is outside the selected release scope"
            )


def validate_publish_policy(
    args: argparse.Namespace,
    scope: str,
    selected: list[PackageInfo],
    packages: dict[str, PackageInfo],
) -> None:
    if args.publish_target != "pypi":
        return
    if args.confirm_publish != "publish":
        fail("publish_target=pypi requires confirm_publish=publish")

    if scope == "all":
        allowed_tags = {f"python-v{packages['potpie'].version}"}
    else:
        package = selected[0]
        allowed_tags = {f"{TAG_PREFIXES[package.key]}-v{package.version}"}

    ref_type = os.getenv("GITHUB_REF_TYPE", "")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_type != "tag" or ref_name not in allowed_tags:
        rendered = ", ".join(sorted(allowed_tags))
        fail(
            "publish_target=pypi must run from an allowed release tag "
            f"({rendered}); got ref_type={ref_type!r}, ref_name={ref_name!r}"
        )


def github_run_url() -> str:
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server_url}/{repository}/actions/runs/{run_id}"
    return ""


def emit_metadata(
    args: argparse.Namespace,
    scope: str,
    selected: list[PackageInfo],
    channel: str,
) -> Path:
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "release-metadata.json"
    metadata = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "release_scope": scope,
        "publish_target": args.publish_target,
        "channel": channel,
        "commit_sha": os.getenv("GITHUB_SHA", ""),
        "ref": os.getenv("GITHUB_REF", ""),
        "ref_type": os.getenv("GITHUB_REF_TYPE", ""),
        "ref_name": os.getenv("GITHUB_REF_NAME", ""),
        "repository": os.getenv("GITHUB_REPOSITORY", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "run_url": github_run_url(),
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
    publish_target: str,
    packages: dict[str, PackageInfo],
    metadata_path: Path,
    channel: str,
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
        f"publish_target={publish_target}",
        f"channel={channel}",
        f"metadata_path={metadata_output}",
        f"context_engine_version={packages['context-engine'].version}",
        f"potpie_version={packages['potpie'].version}",
    ]
    if publish_target != "build-only":
        lines.append(f"repository_url={PYPI_REPOSITORY_URL[publish_target]}")
    with Path(github_output).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-scope",
        required=True,
        choices=sorted(PACKAGE_KEYS_BY_SCOPE),
    )
    parser.add_argument(
        "--publish-target",
        required=True,
        choices=["build-only", "testpypi", "pypi"],
    )
    parser.add_argument("--confirm-publish", default="")
    parser.add_argument("--output-dir", default="release-metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope = normalize_scope(args.release_scope)
    packages = {key: pyproject_package(key) for key in PACKAGE_SOURCES}
    selected_keys = package_keys_for_scope(scope)
    selected = [packages[key] for key in selected_keys]
    channel = infer_channel(selected)

    validate_dependency_chain(packages)
    validate_publish_policy(args, scope, selected, packages)
    validate_index_availability(args.publish_target, selected)
    validate_external_dependencies(args.publish_target, set(selected_keys), packages)
    metadata_path = emit_metadata(args, scope, selected, channel)
    emit_github_outputs(scope, args.publish_target, packages, metadata_path, channel)

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
