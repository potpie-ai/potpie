#!/usr/bin/env python3
"""Verify that PyPI contains the exact distributions from a release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

PYPI_BASE = "https://pypi.org/pypi"
PACKAGE_DIRS = {
    "potpie-context-engine": "context-engine",
    "potpie": "potpie",
}
PACKAGE_NAMES_BY_SCOPE = {
    "all": frozenset(PACKAGE_DIRS),
    "context-engine": frozenset({"potpie-context-engine"}),
    "potpie": frozenset({"potpie"}),
}
EXPECTED_ARTIFACT_COUNTS = {
    "potpie-context-engine": {"wheel": 1, "sdist": 0},
    "potpie": {"wheel": 1, "sdist": 1},
}


@dataclass(frozen=True)
class PackageRelease:
    name: str
    version: str
    artifacts: Mapping[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_plan(metadata_path: Path, dist_dir: Path) -> list[PackageRelease]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scope = metadata.get("release_scope")
    if scope not in PACKAGE_NAMES_BY_SCOPE:
        raise ValueError(f"unsupported release scope in metadata: {scope!r}")

    raw_packages = metadata.get("packages")
    if not isinstance(raw_packages, dict):
        raise ValueError("release metadata must contain a packages object")
    expected_names = PACKAGE_NAMES_BY_SCOPE[scope]
    actual_names = frozenset(raw_packages)
    if actual_names != expected_names:
        raise ValueError(
            "release metadata package set does not match its scope: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )

    plan: list[PackageRelease] = []
    for name in sorted(expected_names):
        package = raw_packages[name]
        if not isinstance(package, dict) or not isinstance(package.get("version"), str):
            raise ValueError(f"release metadata has no version for {name}")
        package_dir = dist_dir / PACKAGE_DIRS[name]
        artifacts = sorted(
            path
            for path in package_dir.iterdir()
            if path.is_file()
            and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
        )
        wheels = [path for path in artifacts if path.name.endswith(".whl")]
        sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
        expected_counts = EXPECTED_ARTIFACT_COUNTS[name]
        if (
            len(wheels) != expected_counts["wheel"]
            or len(sdists) != expected_counts["sdist"]
        ):
            raise ValueError(
                f"{name} must have exactly {expected_counts['wheel']} wheel(s) and "
                f"{expected_counts['sdist']} sdist(s) in {package_dir}"
            )
        plan.append(
            PackageRelease(
                name=name,
                version=package["version"],
                artifacts={path.name: sha256(path) for path in artifacts},
            )
        )
    return plan


def pypi_release_files(name: str, version: str) -> dict[str, str]:
    package = urllib.parse.quote(name)
    release = urllib.parse.quote(version)
    request = urllib.request.Request(  # noqa: S310 - host is fixed above.
        f"{PYPI_BASE}/{package}/{release}/json",
        headers={"User-Agent": "potpie-release-verifier/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        payload = json.load(response)
    return {
        item["filename"]: item["digests"]["sha256"] for item in payload.get("urls", [])
    }


def mismatch(expected: Mapping[str, str], actual: Mapping[str, str]) -> str | None:
    if set(actual) != set(expected):
        return f"expected files {sorted(expected)}, got {sorted(actual)}"
    bad_hashes = [
        filename
        for filename, expected_hash in expected.items()
        if actual.get(filename) != expected_hash
    ]
    if bad_hashes:
        return "SHA256 mismatch for " + ", ".join(sorted(bad_hashes))
    return None


def verify_release(
    plan: list[PackageRelease],
    *,
    attempts: int,
    delay_seconds: float,
    fetch: Callable[[str, str], Mapping[str, str]] = pypi_release_files,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_errors: list[str] = []
    for attempt in range(1, attempts + 1):
        last_errors = []
        for package in plan:
            try:
                actual = fetch(package.name, package.version)
                problem = mismatch(package.artifacts, actual)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                problem = str(exc)
            if problem:
                last_errors.append(f"{package.name}=={package.version}: {problem}")
            else:
                print(f"verified {package.name}=={package.version} on PyPI")
        if not last_errors:
            return
        if attempt < attempts:
            print(
                f"PyPI readback attempt {attempt}/{attempts} is not complete; "
                f"retrying in {delay_seconds:g}s"
            )
            sleep(delay_seconds)

    raise RuntimeError("; ".join(last_errors))


def select_packages(
    plan: list[PackageRelease],
    requested_names: list[str],
) -> list[PackageRelease]:
    if not requested_names:
        return plan
    requested = set(requested_names)
    available = {package.name for package in plan}
    unknown = requested - available
    if unknown:
        raise ValueError(
            "requested package is not present in release metadata: "
            + ", ".join(sorted(unknown))
        )
    return [package for package in plan if package.name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=18)
    parser.add_argument("--delay-seconds", type=float, default=10)
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(PACKAGE_DIRS),
        default=[],
        help="Verify only this package from the release bundle; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = select_packages(
            release_plan(args.metadata, args.dist_dir),
            args.package,
        )
        verify_release(
            plan,
            attempts=args.attempts,
            delay_seconds=args.delay_seconds,
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"PyPI release verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
