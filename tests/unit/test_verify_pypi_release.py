from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_pypi_release as verify


def write_bundle(
    tmp_path: Path,
    *,
    scope: str = "potpie",
) -> tuple[Path, Path, dict[str, str]]:
    dist_dir = tmp_path / "dist"
    package_dir = dist_dir / "potpie"
    package_dir.mkdir(parents=True)
    files = {
        "potpie-2.1.0-py3-none-any.whl": b"wheel",
        "potpie-2.1.0.tar.gz": b"sdist",
    }
    for filename, content in files.items():
        (package_dir / filename).write_bytes(content)

    metadata_path = tmp_path / "release-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "release_scope": scope,
                "packages": {
                    "potpie": {
                        "version": "2.1.0",
                        "source": "pyproject.toml",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    hashes = {
        filename: hashlib.sha256(content).hexdigest()
        for filename, content in files.items()
    }
    return metadata_path, dist_dir, hashes


def write_engine_bundle(
    tmp_path: Path, *, include_sdist: bool = False
) -> tuple[Path, Path]:
    dist_dir = tmp_path / "dist"
    package_dir = dist_dir / "context-engine"
    package_dir.mkdir(parents=True)
    (package_dir / "potpie_context_engine-0.2.0-py3-none-any.whl").write_bytes(b"wheel")
    if include_sdist:
        (package_dir / "potpie_context_engine-0.2.0.tar.gz").write_bytes(b"sdist")

    metadata_path = tmp_path / "release-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "release_scope": "context-engine",
                "packages": {
                    "potpie-context-engine": {
                        "version": "0.2.0",
                        "source": "potpie/context-engine/pyproject.toml",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_path, dist_dir


def test_release_plan_requires_package_set_to_match_scope(tmp_path: Path) -> None:
    metadata_path, dist_dir, _ = write_bundle(tmp_path, scope="all")

    with pytest.raises(ValueError, match="package set does not match"):
        verify.release_plan(metadata_path, dist_dir)


def test_context_engine_release_contains_only_one_wheel(tmp_path: Path) -> None:
    metadata_path, dist_dir = write_engine_bundle(tmp_path)

    plan = verify.release_plan(metadata_path, dist_dir)

    assert list(plan[0].artifacts) == ["potpie_context_engine-0.2.0-py3-none-any.whl"]


def test_context_engine_release_rejects_source_distribution(tmp_path: Path) -> None:
    metadata_path, dist_dir = write_engine_bundle(tmp_path, include_sdist=True)

    with pytest.raises(ValueError, match=r"1 wheel\(s\) and 0 sdist\(s\)"):
        verify.release_plan(metadata_path, dist_dir)


def test_select_packages_limits_exact_readback_to_requested_package(
    tmp_path: Path,
) -> None:
    metadata_path, dist_dir, _ = write_bundle(tmp_path)
    plan = verify.release_plan(metadata_path, dist_dir)

    selected = verify.select_packages(plan, ["potpie"])

    assert [package.name for package in selected] == ["potpie"]


def test_select_packages_rejects_package_outside_release_scope(tmp_path: Path) -> None:
    metadata_path, dist_dir, _ = write_bundle(tmp_path)
    plan = verify.release_plan(metadata_path, dist_dir)

    with pytest.raises(ValueError, match="not present in release metadata"):
        verify.select_packages(plan, ["potpie-context-engine"])


def test_verify_release_retries_until_exact_files_are_visible(tmp_path: Path) -> None:
    metadata_path, dist_dir, hashes = write_bundle(tmp_path)
    plan = verify.release_plan(metadata_path, dist_dir)
    responses = iter([{}, hashes])
    sleeps: list[float] = []

    verify.verify_release(
        plan,
        attempts=2,
        delay_seconds=0.25,
        fetch=lambda _name, _version: next(responses),
        sleep=sleeps.append,
    )

    assert sleeps == [0.25]


def test_verify_release_rejects_hash_mismatch(tmp_path: Path) -> None:
    metadata_path, dist_dir, hashes = write_bundle(tmp_path)
    plan = verify.release_plan(metadata_path, dist_dir)
    hashes["potpie-2.1.0.tar.gz"] = "0" * 64

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify.verify_release(
            plan,
            attempts=1,
            delay_seconds=0,
            fetch=lambda _name, _version: hashes,
        )
