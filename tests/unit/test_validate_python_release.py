from __future__ import annotations

import argparse
import json
import subprocess
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
        "potpie-context-engine",
    }


def test_potpie_only_publish_requires_engine_on_pypi(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    checked: list[tuple[str, str]] = []

    def version_exists(name: str, version: str) -> bool:
        checked.append((name, version))
        return False

    monkeypatch.setattr(release, "package_version_exists", version_exists)

    with pytest.raises(SystemExit):
        release.validate_external_dependencies(
            "pypi",
            {"potpie"},
            packages,
        )

    assert (
        "potpie-context-engine==0.2.0 must already exist on pypi"
        in capsys.readouterr().err
    )
    assert checked == [("potpie-context-engine", "0.2.0")]


def test_all_scope_uses_aggregate_release_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    args = argparse.Namespace(publish_target="pypi", confirm_publish="publish")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "python-v2.1.0")
    monkeypatch.setenv("GITHUB_SHA", "release-sha")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    release.validate_publish_policy(
        args,
        "all",
        list(packages.values()),
        packages,
    )


def test_publish_rejects_tag_not_reachable_from_default_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    args = argparse.Namespace(publish_target="pypi", confirm_publish="publish")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "python-v2.1.0")
    monkeypatch.setenv("GITHUB_SHA", "release-sha")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    with pytest.raises(SystemExit):
        release.validate_publish_policy(
            args,
            "all",
            list(packages.values()),
            packages,
        )

    assert "not reachable from origin/main" in capsys.readouterr().err


def test_package_scope_rejects_wrong_release_tag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
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
