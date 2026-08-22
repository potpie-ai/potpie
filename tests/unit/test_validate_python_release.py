from __future__ import annotations

import argparse
import json
import sys

import pytest

from scripts import validate_python_release as release


def package(key: str, name: str, version: str) -> release.PackageInfo:
    return release.PackageInfo(
        key=key,
        name=name,
        version=version,
        source=f"{key}/pyproject.toml",
    )


def test_build_only_all_accepts_current_dependency_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_python_release.py",
            "--release-scope",
            "all",
            "--publish-target",
            "build-only",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert release.main() == 0
    metadata = json.loads((tmp_path / "release-metadata.json").read_text())
    assert metadata["release_scope"] == "all"
    assert set(metadata["packages"]) == {
        "potpie",
        "potpie-context-core",
        "potpie-context-engine",
    }


def test_engine_only_publish_requires_core_on_target_index(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "context-core": package("context-core", "potpie-context-core", "0.1.0"),
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    monkeypatch.setattr(release, "package_version_exists", lambda *_args: False)

    with pytest.raises(SystemExit):
        release.validate_external_dependencies(
            "pypi",
            {"context-engine"},
            packages,
        )

    assert (
        "potpie-context-core==0.1.0 must already exist on pypi"
        in capsys.readouterr().err
    )


def test_all_scope_uses_aggregate_release_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        "context-core": package("context-core", "potpie-context-core", "0.1.0"),
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    args = argparse.Namespace(publish_target="pypi", confirm_publish="publish")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "python-v2.1.0")

    release.validate_publish_policy(
        args,
        "all",
        list(packages.values()),
        packages,
    )


def test_package_scope_rejects_wrong_release_tag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "context-core": package("context-core", "potpie-context-core", "0.1.0"),
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    args = argparse.Namespace(publish_target="pypi", confirm_publish="publish")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "python-v2.1.0")

    with pytest.raises(SystemExit):
        release.validate_publish_policy(
            args,
            "context-engine",
            [packages["context-engine"]],
            packages,
        )

    assert "potpie-context-engine-v0.2.0" in capsys.readouterr().err


def test_dev_release_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    dev = package("potpie", "potpie", "2.1.0rc1.dev1")

    with pytest.raises(SystemExit):
        release.channel_for_version(dev)

    assert "must not be a dev release" in capsys.readouterr().err
