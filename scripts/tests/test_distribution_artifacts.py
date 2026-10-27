"""Exercise artifact substitution, native selection and isolated suite execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


pytestmark = pytest.mark.source_contract
SCRIPTS = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = _load("distribution_manifest")
suites = _load("test_suites")


def _wheel(
    directory, package="potpie", version="2.0.1", tag="py3-none-any", requires=()
):
    normalized = package.replace("-", "_")
    path = directory / f"{normalized}-{version}-{tag}.whl"
    metadata = f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {row}\n" for row in requires)
    with ZipFile(path, "w") as archive:
        archive.writestr(f"{normalized}-{version}.dist-info/METADATA", metadata)
        archive.writestr(
            f"{normalized}-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nTag: {tag}\n",
        )
    return path


def test_verified_artifact_cannot_be_changed_after_manifest_creation(tmp_path):
    wheel = _wheel(tmp_path)
    manifest.write_manifest(tmp_path, COMMIT)
    assert manifest.load_artifacts(tmp_path, COMMIT)[0]["file"] == wheel
    with ZipFile(wheel, "a") as archive:
        archive.writestr("unreviewed.py", "raise RuntimeError('substitution')\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        manifest.load_artifacts(tmp_path, COMMIT)


@pytest.mark.parametrize("change", ["commit", "traversal", "duplicate"])
def test_foreign_commit_and_escaped_or_repeated_paths_are_refused(tmp_path, change):
    _wheel(tmp_path)
    path = manifest.write_manifest(tmp_path, COMMIT)
    value = json.loads(path.read_text())
    if change == "commit":
        value["commit"] = "b" * 40
    elif change == "traversal":
        value["distributions"][0]["path"] = "../elsewhere.whl"
    else:
        value["distributions"].append(value["distributions"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        manifest.load_artifacts(tmp_path, COMMIT)


def test_native_selection_requires_matching_python_and_platform(tmp_path):
    _wheel(tmp_path, "potpie-parsing", "0.1.0", "cp313-cp313-linux_x86_64")
    _wheel(tmp_path, "potpie-parsing", "0.1.0", "cp314-cp314-linux_x86_64")
    manifest.write_manifest(tmp_path, COMMIT)
    artifacts = manifest.load_artifacts(tmp_path, COMMIT)
    assert manifest.select_wheel(artifacts, "potpie-parsing", "3.13", "linux-x86_64")[
        "tags"
    ] == ["cp313-cp313-linux_x86_64"]
    with pytest.raises(ValueError, match="found 0"):
        manifest.select_wheel(artifacts, "potpie-parsing", "3.13", "linux-aarch64")
    with pytest.raises(ValueError, match="found 0"):
        manifest.select_wheel(artifacts, "potpie-parsing", "3.12", "linux-x86_64")


def test_root_install_requires_its_local_engine_pin(tmp_path):
    _wheel(tmp_path, requires=("potpie-context-engine[http,local]==0.2.0",))
    _wheel(tmp_path, "potpie-context-engine", "0.3.0")
    manifest.write_manifest(tmp_path, COMMIT)
    artifacts = manifest.load_artifacts(tmp_path, COMMIT)
    root = manifest.select_wheel(artifacts, "potpie", "3.13", "linux-x86_64")
    engine = manifest.select_wheel(
        artifacts, "potpie-context-engine", "3.13", "linux-x86_64"
    )
    with pytest.raises(ValueError, match="exact locally built engine"):
        manifest.require_product_pair(root, engine)


def test_wheel_filename_cannot_claim_different_metadata(tmp_path):
    wheel = _wheel(tmp_path)
    renamed = wheel.with_name(wheel.name.replace("2.0.1", "9.0.0"))
    wheel.rename(renamed)
    with pytest.raises(ValueError, match="filename disagrees"):
        manifest.write_manifest(tmp_path, COMMIT)


def test_owned_test_helpers_do_not_put_checkout_products_on_import_path(tmp_path):
    package = tmp_path / "package"
    tests = package / "tests"
    tests.mkdir(parents=True)
    (tests / "__init__.py").touch()
    (tests / "support.py").write_text("VALUE = 42\n")
    (package / "checkout_product_fixture").mkdir()
    (package / "checkout_product_fixture/__init__.py").touch()
    (package / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nmarkers = ['source_contract: source inspection']\n"
    )
    (tests / "test_imports.py").write_text(
        "import importlib.util\nfrom tests.support import VALUE\n"
        "def test_isolation():\n"
        "    assert VALUE == 42\n"
        "    assert importlib.util.find_spec('checkout_product_fixture') is None\n"
    )
    command = suites.pytest_command(package, python=sys.executable, profile="installed")
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_tiers_cannot_be_applied_to_a_different_product_suite():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "test_suites.py"),
            "root",
            "--tier",
            "restart",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Acceptance tiers belong to the engine suite" in result.stderr


def test_suite_children_find_the_same_virtual_environment_cli(tmp_path):
    binary = tmp_path / "venv" / "bin"
    binary.mkdir(parents=True)
    interpreter = binary / "python"
    interpreter.symlink_to(sys.executable)
    cli = binary / "potpie"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    result = subprocess.run(
        [sys.executable, "-c", "import shutil; print(shutil.which('potpie'))"],
        env=suites.suite_environment(str(interpreter)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(cli)
