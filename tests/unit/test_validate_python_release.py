from __future__ import annotations

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


def test_all_release_accepts_matching_dependency_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_SHA", "release-sha")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")
    monkeypatch.setattr(release, "pyproject_package", packages.__getitem__)
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: release.Requirement("potpie-context-engine[all]==0.2.0"),
    )
    monkeypatch.setattr(release, "package_version_exists", lambda *_args: False)
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_python_release.py",
            "--release-scope",
            "all",
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
    assert metadata["release_tags"] == {
        "potpie": "potpie-v2.1.0",
        "potpie-context-engine": "potpie-context-engine-v0.2.0",
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
            {"potpie"},
            packages["context-engine"].name,
            packages["context-engine"].version,
        )

    assert (
        "potpie-context-engine==0.2.0 must already exist on pypi "
        "for a potpie-only release" in capsys.readouterr().err
    )
    assert checked == [("potpie-context-engine", "0.2.0")]


def test_engine_only_release_uses_repository_engine_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.3.0"),
    }
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: pytest.fail("engine-only release must not inspect Potpie"),
    )

    assert (
        release.resolve_context_engine_version({"context-engine"}, packages) == "0.3.0"
    )


def test_potpie_only_release_uses_its_pinned_engine_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: release.Requirement("potpie-context-engine[all]==0.2.0"),
    )

    assert release.resolve_context_engine_version({"potpie"}, packages) == "0.2.0"


def test_combined_release_rejects_mismatched_engine_pin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.3.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: release.Requirement("potpie-context-engine[all]==0.2.0"),
    )

    with pytest.raises(SystemExit):
        release.resolve_context_engine_version(
            {"context-engine", "potpie"},
            packages,
        )

    assert "for a combined release; found ==0.2.0" in capsys.readouterr().err


def test_potpie_pin_must_name_one_concrete_engine_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packages = {
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: release.Requirement("potpie-context-engine[all]==0.2.*"),
    )

    with pytest.raises(SystemExit):
        release.resolve_context_engine_version({"potpie"}, packages)

    assert "concrete PEP 440 version" in capsys.readouterr().err


def test_release_accepts_commit_from_default_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_SHA", "release-sha")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(release.subprocess, "run", run)

    release.validate_release_source()

    assert commands == [
        [
            release.shutil.which("git"),
            "merge-base",
            "--is-ancestor",
            "release-sha",
            "origin/main",
        ]
    ]


def test_release_rejects_tag_ref(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "potpie-v2.1.0")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")

    with pytest.raises(SystemExit):
        release.validate_release_source()

    assert (
        "must run from the repository default branch 'main'" in capsys.readouterr().err
    )


def test_release_rejects_non_default_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "release-candidate")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")

    with pytest.raises(SystemExit):
        release.validate_release_source()

    assert "ref_name='release-candidate'" in capsys.readouterr().err


def test_release_rejects_commit_not_reachable_from_default_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_SHA", "release-sha")
    monkeypatch.setenv("REPOSITORY_DEFAULT_BRANCH", "main")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1),
    )

    with pytest.raises(SystemExit):
        release.validate_release_source()

    assert "not reachable from origin/main" in capsys.readouterr().err


def test_github_outputs_include_package_specific_tags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    output = tmp_path / "github-output"
    metadata = tmp_path / "release-metadata.json"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    release.emit_github_outputs("all", packages, "0.2.0", metadata)

    assert "context_engine_tag=potpie-context-engine-v0.2.0" in output.read_text()
    assert "potpie_tag=potpie-v2.1.0" in output.read_text()


def test_dev_release_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    dev = package("potpie", "potpie", "2.1.0rc1.dev1")

    with pytest.raises(SystemExit):
        release.channel_for_version(dev)

    assert "must not be a dev release" in capsys.readouterr().err
