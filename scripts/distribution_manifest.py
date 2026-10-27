"""Describe and verify the exact distributions passed between CI jobs."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def describe(path: Path) -> dict:
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            metadata_paths = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"Expected one wheel metadata record: {path}")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            wheel = BytesParser().parsebytes(
                archive.read(metadata_paths[0].removesuffix("METADATA") + "WHEEL")
            )
            tags = wheel.get_all("Tag", [])
            if not tags:
                raise ValueError(f"Wheel has no compatibility tags: {path}")
        kind = "wheel"
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            metadata_paths = [
                member
                for member in archive.getmembers()
                if len(Path(member.name).parts) == 2
                and member.name.endswith("/PKG-INFO")
                and member.isfile()
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"Expected one source metadata record: {path}")
            stream = archive.extractfile(metadata_paths[0])
            if stream is None:
                raise ValueError(f"Source metadata is unreadable: {path}")
            metadata = BytesParser().parsebytes(stream.read())
        tags, kind = [], "sdist"
    else:
        raise ValueError(f"Unsupported distribution: {path}")
    if not metadata.get("Name") or not metadata.get("Version"):
        raise ValueError(f"Distribution lacks its name or version: {path}")
    if kind == "wheel":
        fields = path.name.removesuffix(".whl").split("-")
        if (
            len(fields) not in {5, 6}
            or normalized_name(fields[0]) != normalized_name(metadata["Name"])
            or fields[1] != metadata["Version"]
            or set(tags)
            != {
                "-".join(tag)
                for tag in product(*(field.split(".") for field in fields[-3:]))
            }
        ):
            raise ValueError(
                f"Wheel filename disagrees with its package/version/tags: {path}"
            )
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "package": normalized_name(metadata["Name"]),
        "version": metadata["Version"],
        "kind": kind,
        "tags": sorted(tags),
        "requires_dist": sorted(metadata.get_all("Requires-Dist", [])),
        "sha256": checksum,
    }


def _commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("A full lowercase Git commit SHA is required")
    return value


def write_manifest(directory: Path, commit: str) -> Path:
    directory = directory.resolve()
    distributions = []
    for path in sorted(directory.rglob("*")):
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            distributions.append(
                {"path": path.relative_to(directory).as_posix(), **describe(path)}
            )
    if not distributions:
        raise ValueError(f"No distributions found: {directory}")
    output = directory / "manifest.json"
    output.write_text(
        json.dumps(
            {"schema": 1, "commit": _commit(commit), "distributions": distributions},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def load_artifacts(directory: Path, commit: str) -> list[dict]:
    expected_commit = _commit(commit)
    manifests = sorted(directory.resolve().rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"No distribution manifests found: {directory}")
    artifacts = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("commit") != expected_commit:
            raise ValueError(f"Distribution manifest schema/commit mismatch: {path}")
        records = manifest.get("distributions")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Empty distribution manifest: {path}")
        seen = set()
        for record in records:
            relative = Path(record["path"])
            resolved = (path.parent / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not resolved.is_relative_to(path.parent)
                or relative in seen
            ):
                raise ValueError(f"Invalid or repeated artifact path: {relative}")
            seen.add(relative)
            actual = describe(resolved)
            if actual != {key: value for key, value in record.items() if key != "path"}:
                raise ValueError(f"Distribution metadata/checksum mismatch: {resolved}")
            artifacts.append({**record, "file": resolved})
    return artifacts


def select_wheel(artifacts: list[dict], package: str, python: str, target: str) -> dict:
    """Select the wheel for an explicit tested interpreter/platform tuple."""
    interpreter = "cp" + python.replace(".", "")
    os_name, architecture = target.split("-", 1)

    def compatible(tag: str) -> bool:
        implementation, abi, platform = tag.split("-")
        if implementation not in {"py3", interpreter}:
            return False
        if abi not in {"none", interpreter}:
            return False
        if platform == "any":
            return True
        if not platform.endswith("_" + architecture):
            return False
        return {
            "linux": platform.startswith(("linux_", "manylinux")),
            "windows": platform.startswith("win_"),
            "macos": platform.startswith("macosx_"),
        }.get(os_name, False)

    selected = [
        artifact
        for artifact in artifacts
        if artifact["package"] == normalized_name(package)
        and artifact["kind"] == "wheel"
        and any(compatible(tag) for tag in artifact["tags"])
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {package} wheel for Python {python}/{target}, found {len(selected)}"
        )
    return selected[0]


def require_product_pair(root: dict, engine: dict) -> None:
    pins = [
        re.fullmatch(
            r"potpie-context-engine(?:\[[^]]+\])?\s*==\s*([^; ]+)(?:;.*)?", row
        )
        for row in root["requires_dist"]
    ]
    versions = {match[1] for match in pins if match is not None}
    if versions != {engine["version"]}:
        raise ValueError(
            "The root wheel must pin the exact locally built engine version"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if args.action == "create":
        print(write_manifest(args.directory, args.commit))
    else:
        print(
            f"Verified {len(load_artifacts(args.directory, args.commit))} distributions"
        )


if __name__ == "__main__":
    main()
