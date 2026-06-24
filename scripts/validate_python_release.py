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
PACKAGE_SOURCES = {
    "potpie": ("pyproject.toml", "potpie"),
    "context-engine": ("potpie/context-engine/pyproject.toml", "potpie-context-engine"),
}
PACKAGE_ALIASES = {
    "potpie": "potpie",
    "context-engine": "context-engine",
    "potpie-context-engine": "context-engine",
}
TAG_PREFIXES = {
    "potpie": "potpie",
    "context-engine": "potpie-context-engine",
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


def normalize_package(value: str) -> str:
    package = PACKAGE_ALIASES.get(value)
    if package is None:
        allowed = ", ".join(sorted(PACKAGE_ALIASES))
        fail(f"unsupported package {value!r}; allowed: {allowed}")
    return package


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
        fail(f"{package.name} version {package.version!r} must not use a local '+...' segment")
    if version.is_devrelease:
        fail(f"{package.name} version {package.version!r} must not be a dev release")
    if version.pre is None:
        return "final"
    if version.pre[0] == "b":
        if "b" not in package.version.lower() or "beta" in package.version.lower():
            fail(f"{package.name} beta version must use compact bN syntax, not {package.version!r}")
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


def context_engine_requirement() -> Requirement:
    dependencies = load_toml("pyproject.toml").get("project", {}).get("dependencies", [])
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as exc:
            fail(f"invalid root dependency {dependency!r}: {exc}")
        if canonicalize_name(requirement.name) == "potpie-context-engine":
            return requirement
    fail("root potpie package must depend on potpie-context-engine")


def validate_root_dependency(channel: str, context_version: str) -> None:
    requirement = context_engine_requirement()
    specifiers = list(requirement.specifier)
    if not specifiers:
        fail("root potpie dependency on potpie-context-engine must include a version specifier")

    if channel in {"beta", "rc"}:
        if len(specifiers) != 1 or specifiers[0].operator != "==" or specifiers[0].version != context_version:
            fail(
                "beta/rc root potpie must pin potpie-context-engine exactly "
                f"as =={context_version}; found {requirement.specifier}"
            )
        return

    if not requirement.specifier.contains(Version(context_version), prereleases=True):
        fail(
            "final root potpie dependency range must include "
            f"potpie-context-engine {context_version}; found {requirement.specifier}"
        )


def package_version_exists(index: str, package_name: str, version: str) -> bool:
    base_url = PYPI_BASE[index]
    name = urllib.parse.quote(package_name)
    release = urllib.parse.quote(version)
    url = f"{base_url}/{name}/{release}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "potpie-release-validator/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        fail(f"could not check {index} for {package_name}=={version}: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        fail(f"could not check {index} for {package_name}=={version}: {exc.reason}")
    return False


def validate_index_availability(publish_target: str, packages: list[PackageInfo]) -> None:
    if publish_target == "build-only":
        return
    index = "testpypi" if publish_target == "testpypi" else "pypi"
    for package in packages:
        if package_version_exists(index, package.name, package.version):
            fail(f"{package.name}=={package.version} already exists on {index}")


def validate_publish_policy(args: argparse.Namespace, packages: list[PackageInfo]) -> None:
    if args.publish_target != "pypi":
        return
    if args.confirm_publish != "publish":
        fail("publish_target=pypi requires confirm_publish=publish")

    ref_type = os.getenv("GITHUB_REF_TYPE", "")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    allowed_tags = {
        f"{TAG_PREFIXES[package.key]}-v{package.version}"
        for package in packages
    }
    if args.allow_python_release_tag:
        root_potpie = pyproject_package("potpie")
        allowed_tags.add(f"python-v{root_potpie.version}")

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
    packages: list[PackageInfo],
    channel: str,
) -> Path:
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "release-metadata.json"
    metadata = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
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
            for package in packages
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def emit_github_outputs(packages: dict[str, PackageInfo], metadata_path: Path, channel: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    try:
        metadata_output = str(metadata_path.relative_to(ROOT))
    except ValueError:
        metadata_output = str(metadata_path)
    lines = [
        f"channel={channel}",
        f"metadata_path={metadata_output}",
    ]
    if "potpie" in packages:
        lines.append(f"potpie_version={packages['potpie'].version}")
    if "context-engine" in packages:
        lines.append(f"context_engine_version={packages['context-engine'].version}")
    with Path(github_output).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, choices=sorted(PACKAGE_ALIASES))
    parser.add_argument("--publish-target", required=True, choices=["build-only", "testpypi", "pypi"])
    parser.add_argument("--confirm-publish", default="")
    parser.add_argument("--allow-python-release-tag", action="store_true")
    parser.add_argument("--output-dir", default="release-metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_key = normalize_package(args.package)
    selected = {package_key: pyproject_package(package_key)}
    package_list = list(selected.values())
    channel = infer_channel(package_list)

    if package_key == "potpie":
        context_engine = pyproject_package("context-engine")
        validate_root_dependency(channel, context_engine.version)

    validate_publish_policy(args, package_list)
    validate_index_availability(args.publish_target, package_list)
    metadata_path = emit_metadata(args, package_list, channel)
    emit_github_outputs(selected, metadata_path, channel)

    print("release validation passed")
    print(f"- channel: {channel}")
    for package in package_list:
        print(f"- {package.name}=={package.version} ({package.source})")
    try:
        metadata_display = str(metadata_path.relative_to(ROOT))
    except ValueError:
        metadata_display = str(metadata_path)
    print(f"- metadata: {metadata_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
