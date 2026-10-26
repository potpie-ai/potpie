from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error

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
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(release, "pyproject_package", packages.__getitem__)
    monkeypatch.setattr(
        release,
        "package_requirement",
        lambda *_args: release.Requirement("potpie-context-engine[all]==0.2.0"),
    )
    monkeypatch.setattr(
        release,
        "package_version_exists",
        lambda name, _version: name == release.NATIVE_FALKORDBLITE_NAME,
    )
    _pypi_native_files(monkeypatch, _native_wheels())
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
    assert metadata["preexisting_packages"] == []
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


def _native_dependency(monkeypatch: pytest.MonkeyPatch, requirement: str) -> None:
    monkeypatch.setattr(
        release,
        "load_toml",
        lambda _source: {
            "project": {"optional-dependencies": {"local": [requirement]}}
        },
    )


def _native_wheel(
    interpreter: str,
    abi: str | None = None,
    platform: str = "manylinux_2_36_aarch64",
) -> dict:
    return {
        "filename": (
            f"potpie_falkordblite-0.10.0.post1-{interpreter}-{abi or interpreter}-"
            f"{platform}.whl"
        ),
        "packagetype": "bdist_wheel",
        "requires_python": ">=3.12,<3.15",
        "yanked": False,
    }


def _native_wheels() -> list[dict]:
    return [_native_wheel(interpreter) for interpreter in ("cp312", "cp313", "cp314")]


def _pypi_native_files(monkeypatch: pytest.MonkeyPatch, files: list[dict]) -> list:
    checked = []

    def urlopen(request, timeout):
        checked.append((request.full_url, timeout))
        return io.BytesIO(json.dumps({"urls": files}).encode())

    monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
    return checked


@pytest.mark.parametrize("selected", [{"context-engine"}, {"context-engine", "potpie"}])
def test_engine_publish_requires_native_pin_even_on_other_platforms(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selected: set[str],
) -> None:
    _native_dependency(
        monkeypatch,
        'potpie-falkordblite==0.10.0.post1; sys_platform == "linux" '
        'and platform_machine == "aarch64"',
    )
    checked = _pypi_native_files(monkeypatch, [])
    with pytest.raises(SystemExit):
        release.validate_native_dependency_availability(selected, set())

    assert checked == [
        ("https://pypi.org/pypi/potpie-falkordblite/0.10.0.post1/json", 20)
    ]
    assert "publish the corrected native dependency first" in capsys.readouterr().err


def test_engine_publish_accepts_complete_published_native_wheels(monkeypatch) -> None:
    _native_dependency(monkeypatch, "potpie-falkordblite==0.10.0.post1")
    checked = _pypi_native_files(monkeypatch, _native_wheels())
    release.validate_native_dependency_availability({"context-engine"}, set())
    assert checked == [
        ("https://pypi.org/pypi/potpie-falkordblite/0.10.0.post1/json", 20)
    ]


@pytest.mark.parametrize(
    "files", [[], [_native_wheel("cp312")]], ids=["empty", "partial"]
)
def test_incomplete_native_release_stops_preflight_before_emitting_metadata(
    monkeypatch, tmp_path, files, capsys
) -> None:
    _native_dependency(monkeypatch, "potpie-falkordblite==0.10.0.post1")
    _pypi_native_files(monkeypatch, files)
    monkeypatch.setattr(
        release,
        "pyproject_package",
        lambda key: package(key, "potpie-context-engine", "0.3.0"),
    )
    monkeypatch.setattr(release, "validate_release_source", lambda: None)
    monkeypatch.setattr(release, "package_version_exists", lambda *_args: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_python_release.py",
            "--release-scope",
            "context-engine",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        release.main()
    assert not (tmp_path / "release-metadata.json").exists()
    assert "3.13, 3.14" in capsys.readouterr().err


@pytest.mark.parametrize(
    "files",
    [
        [
            {
                "filename": "potpie_falkordblite-0.10.0.post1.tar.gz",
                "packagetype": "sdist",
            }
        ],
        [_native_wheel("cp312", platform="manylinux_2_36_x86_64")],
        [_native_wheel("cp312", platform="manylinux_2_37_aarch64")],
        [_native_wheel("cp312", platform="linux_aarch64")],
        [_native_wheel("cp312", abi="cp313")],
        [_native_wheel("cp313", abi="cp313t")],
        [_native_wheel("py3", abi="none", platform="any")],
        [dict(wheel, yanked=True) for wheel in _native_wheels()],
        [dict(wheel, requires_python=">=3.15") for wheel in _native_wheels()],
        [dict(_native_wheel("cp312"), filename="invalid.whl")],
        [dict(_native_wheel("cp312"), requires_python="invalid")],
        [
            dict(
                wheel,
                filename=wheel["filename"].replace("0.10.0.post1", "0.10.0"),
            )
            for wheel in _native_wheels()
        ],
    ],
    ids=[
        "sdist-only",
        "wrong-architecture",
        "newer-glibc",
        "unverified-glibc",
        "wrong-abi",
        "free-threaded-abi",
        "platform-independent",
        "yanked",
        "requires-newer-python",
        "malformed-filename",
        "malformed-requires-python",
        "wrong-version",
    ],
)
def test_native_release_rejects_unusable_wheels(monkeypatch, capsys, files):
    _pypi_native_files(monkeypatch, files)
    with pytest.raises(SystemExit):
        release.validate_native_wheel_availability("0.10.0.post1")
    assert "missing CPython 3.12, 3.13, 3.14" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["cp312", "cp313", "cp314"])
def test_native_release_requires_every_supported_python(monkeypatch, capsys, missing):
    _pypi_native_files(
        monkeypatch,
        [
            wheel
            for wheel in _native_wheels()
            if f"-{missing}-" not in wheel["filename"]
        ],
    )
    with pytest.raises(SystemExit):
        release.validate_native_wheel_availability("0.10.0.post1")
    expected = f"3.{missing[3:]}"
    assert f"missing CPython {expected} before" in capsys.readouterr().err


@pytest.mark.parametrize(
    "platform",
    ["manylinux_2_36_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64"],
)
def test_native_release_accepts_compatible_abi3_wheel(monkeypatch, platform):
    _pypi_native_files(
        monkeypatch, [_native_wheel("cp312", abi="abi3", platform=platform)]
    )
    release.validate_native_wheel_availability("0.10.0.post1")


def test_native_release_checks_python_requirement_even_when_tags_match(
    monkeypatch, capsys
):
    wheel = dict(_native_wheel("cp312", abi="abi3"), requires_python=">=3.13")
    _pypi_native_files(monkeypatch, [wheel])
    with pytest.raises(SystemExit):
        release.validate_native_wheel_availability("0.10.0.post1")
    assert "missing CPython 3.12 before" in capsys.readouterr().err


def test_missing_native_pypi_release_fails_validation(monkeypatch, capsys):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
    with pytest.raises(SystemExit):
        release.validate_native_wheel_availability("0.10.0.post1")
    assert "missing CPython 3.12, 3.13, 3.14" in capsys.readouterr().err


@pytest.mark.parametrize("payload", [b"not json", b"{}", b'{"urls": null}'])
def test_native_release_fails_closed_on_invalid_pypi_metadata(
    monkeypatch, capsys, payload
):
    monkeypatch.setattr(
        release.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )
    with pytest.raises(SystemExit):
        release.validate_native_wheel_availability("0.10.0.post1")
    assert "invalid pypi file metadata" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("selected", "preexisting"),
    [
        ({"potpie"}, set()),
        ({"context-engine", "potpie"}, {"potpie-context-engine"}),
    ],
)
def test_existing_engine_release_does_not_read_checkout_native_dependencies(
    monkeypatch, selected, preexisting
) -> None:
    monkeypatch.setattr(
        release,
        "load_toml",
        lambda *_args: pytest.fail("this release does not publish the checkout engine"),
    )
    monkeypatch.setattr(
        release,
        "package_release_files",
        lambda *_args: pytest.fail("no native dependency lookup is needed"),
    )
    release.validate_native_dependency_availability(selected, preexisting)


@pytest.mark.parametrize(
    "requirement",
    [
        "potpie-falkordblite",
        "potpie-falkordblite>=0.10.0.post1",
        "potpie-falkordblite==0.10.*",
        "potpie-falkordblite==0.10.0+local",
        "potpie-falkordblite @ https://example.com/native.whl",
    ],
)
def test_native_release_dependency_requires_concrete_index_pin(
    monkeypatch, capsys, requirement
) -> None:
    _native_dependency(monkeypatch, requirement)
    monkeypatch.setattr(
        release,
        "package_release_files",
        lambda *_args: pytest.fail("invalid requirement must fail before index lookup"),
    )
    with pytest.raises(SystemExit):
        release.validate_native_dependency_availability({"context-engine"}, set())
    assert "PyPI" in capsys.readouterr().err


def test_engine_without_native_fork_has_no_fork_release_prerequisite(monkeypatch):
    _native_dependency(monkeypatch, "falkordblite>=0.10.0")
    monkeypatch.setattr(
        release,
        "package_release_files",
        lambda *_args: pytest.fail("removed native fork has no release prerequisite"),
    )
    release.validate_native_dependency_availability({"context-engine"}, set())


def test_combined_release_allows_preexisting_engine_for_exact_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        package("context-engine", "potpie-context-engine", "0.2.0"),
        package("potpie", "potpie", "2.1.0"),
    ]
    monkeypatch.setattr(
        release,
        "package_version_exists",
        lambda name, _version: name == "potpie-context-engine",
    )

    assert release.validate_index_availability("all", selected) == {
        "potpie-context-engine"
    }


def test_combined_release_rejects_preexisting_potpie(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = [
        package("context-engine", "potpie-context-engine", "0.2.0"),
        package("potpie", "potpie", "2.1.0"),
    ]
    monkeypatch.setattr(
        release,
        "package_version_exists",
        lambda name, _version: name == "potpie",
    )

    with pytest.raises(SystemExit):
        release.validate_index_availability("all", selected)

    assert "potpie==2.1.0 already exists on pypi" in capsys.readouterr().err


def test_engine_only_release_rejects_preexisting_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = [
        package("context-engine", "potpie-context-engine", "0.2.0"),
    ]
    monkeypatch.setattr(release, "package_version_exists", lambda *_args: True)

    with pytest.raises(SystemExit):
        release.validate_index_availability("context-engine", selected)

    assert (
        "potpie-context-engine==0.2.0 already exists on pypi" in capsys.readouterr().err
    )


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

    release.emit_github_outputs("all", packages, "0.2.0", metadata, set())

    assert "context_engine_tag=potpie-context-engine-v0.2.0" in output.read_text()
    assert "potpie_tag=potpie-v2.1.0" in output.read_text()
    assert "context_engine_preexisting=false" in output.read_text()


def test_github_outputs_mark_preexisting_context_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    packages = {
        "context-engine": package("context-engine", "potpie-context-engine", "0.2.0"),
        "potpie": package("potpie", "potpie", "2.1.0"),
    }
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    release.emit_github_outputs(
        "all",
        packages,
        "0.2.0",
        tmp_path / "release-metadata.json",
        {"potpie-context-engine"},
    )

    assert "context_engine_preexisting=true" in output.read_text()


def test_dev_release_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    dev = package("potpie", "potpie", "2.1.0rc1.dev1")

    with pytest.raises(SystemExit):
        release.channel_for_version(dev)

    assert "must not be a dev release" in capsys.readouterr().err


def test_non_normalized_version_spelling_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    non_normalized = package("potpie", "potpie", "2.1.0-rc.1")

    with pytest.raises(SystemExit):
        release.channel_for_version(non_normalized)

    assert "normalized PEP 440 spelling '2.1.0rc1'" in capsys.readouterr().err
