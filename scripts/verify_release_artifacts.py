"""Verify that the three public distribution artifacts form one release set."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import sys
import tomllib
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "potpie-context-core": ROOT / "potpie/context-core/pyproject.toml",
    "potpie-context-engine": ROOT / "potpie/context-engine/pyproject.toml",
    "potpie": ROOT / "pyproject.toml",
}


def _project_versions() -> dict[str, str]:
    return {
        name: tomllib.loads(path.read_text(encoding="utf-8"))["project"]["version"]
        for name, path in PROJECTS.items()
    }


def _wheel_metadata(path: Path):
    with ZipFile(path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return BytesParser().parsebytes(wheel.read(metadata_name)), set(
            wheel.namelist()
        )


def verify(dist: Path, *, tag: str | None) -> None:
    versions = _project_versions()
    wheels: dict[str, tuple[Path, object, set[str]]] = {}
    for path in sorted(dist.glob("*.whl")):
        metadata, members = _wheel_metadata(path)
        name = metadata["Name"]
        if name in PROJECTS:
            if name in wheels:
                raise ValueError(f"duplicate wheel for {name}")
            wheels[name] = (path, metadata, members)

    missing = sorted(set(PROJECTS) - set(wheels))
    if missing:
        raise ValueError(f"missing wheel(s): {', '.join(missing)}")

    for name, (_, metadata, _) in wheels.items():
        actual = metadata["Version"]
        if actual != versions[name]:
            raise ValueError(
                f"{name} wheel version {actual!r} != project version {versions[name]!r}"
            )

    core_members = wheels["potpie-context-core"][2]
    engine_members = wheels["potpie-context-engine"][2]
    if "potpie_context_core/py.typed" not in core_members:
        raise ValueError("potpie-context-core wheel does not contain py.typed")
    if "potpie_context_engine/py.typed" not in engine_members:
        raise ValueError("potpie-context-engine wheel does not contain py.typed")

    engine_requires = wheels["potpie-context-engine"][1].get_all(
        "Requires-Dist", failobj=[]
    )
    root_requires = wheels["potpie"][1].get_all("Requires-Dist", failobj=[])
    expected_core = f"potpie-context-core=={versions['potpie-context-core']}"
    expected_engine = f"potpie-context-engine[all]=={versions['potpie-context-engine']}"
    if expected_core not in engine_requires:
        raise ValueError(f"engine wheel must require {expected_core}")
    if expected_core not in root_requires or expected_engine not in root_requires:
        raise ValueError("root wheel must exactly pin both context distributions")

    root_members = wheels["potpie"][2]
    forbidden = (
        "potpie/context-core/",
        "potpie/context-engine/",
        "potpie_context_core/",
        "potpie_context_engine/",
    )
    leaked = sorted(member for member in root_members if member.startswith(forbidden))
    if leaked:
        raise ValueError(f"root wheel contains context package files: {leaked[:3]}")

    if tag is not None:
        expected_tag = f"v{versions['potpie']}"
        if tag != expected_tag:
            raise ValueError(f"release tag {tag!r} must equal {expected_tag!r}")

    print(
        "verified release set: "
        + ", ".join(f"{name}=={version}" for name, version in versions.items())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    try:
        verify(args.dist, tag=args.tag)
    except (OSError, StopIteration, ValueError) as exc:
        print(f"release artifact verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
