"""Built-artifact contract for the product and standalone engine distributions."""

from __future__ import annotations

import configparser
import shutil
import subprocess
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

from tests.boundary.isolated_import import run_isolated_python

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "potpie" / "context-engine"
ENGINE_EXTRAS = {
    "embedded",
    "http",
    "postgres",
    "neo4j",
    "embeddings",
    "github",
    "reconciliation",
    "hatchet",
    "observability",
}
ROOT_SCRIPTS = {
    "potpie": "potpie.cli.main:main",
    "potpie-daemon": "potpie.daemon.main:main",
    "potpie-mcp": "potpie.mcp.main:main",
}
REMOVED_ROOT_MODULES = {
    "potpie/cli/commands/service.py",
    "potpie/daemon/http/errors.py",
    "potpie/daemon/http/transport.py",
    "potpie/daemon/process/ipc_client.py",
}
REMOVED_ROOT_PREFIXES = {
    "potpie/daemon/managed_services/",
    "potpie/daemon/ports/",
    "potpie/daemon/runtime/",
}


def _build(project: Path, output: Path) -> tuple[Path, Path]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for distribution artifact tests")
    result = subprocess.run(
        [uv, "build", "--out-dir", str(output)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def _metadata(archive: zipfile.ZipFile) -> object:
    metadata_name = next(
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    )
    parsed = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_name))
    assert not parsed.defects
    return parsed


def _entrypoints(archive: zipfile.ZipFile) -> dict[str, str]:
    entrypoint_names = [
        name
        for name in archive.namelist()
        if name.endswith(".dist-info/entry_points.txt")
    ]
    if not entrypoint_names:
        return {}
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(entrypoint_names[0]).decode("utf-8"))
    return dict(parser["console_scripts"])


def test_built_distributions_have_exact_package_ownership(tmp_path: Path) -> None:
    root_wheel, root_sdist = _build(ROOT, tmp_path / "root")
    engine_wheel, engine_sdist = _build(ENGINE, tmp_path / "engine")

    with zipfile.ZipFile(root_wheel) as archive:
        names = archive.namelist()
        metadata = _metadata(archive)
        assert metadata["Name"] == "potpie"
        assert metadata["Version"] == "2.0.0"
        assert _entrypoints(archive) == ROOT_SCRIPTS
        assert any(name.startswith("potpie/mcp/") for name in names)
        assert any(name.startswith("potpie/skills/resources/") for name in names)
        assert not any(name.startswith("potpie_context_engine/") for name in names)
        assert "potpie/runtime/sync_view.py" not in names
        assert not REMOVED_ROOT_MODULES.intersection(names)
        assert not any(
            name.startswith(prefix)
            for name in names
            for prefix in REMOVED_ROOT_PREFIXES
        )
        requirements = metadata.get_all("Requires-Dist") or []
        assert "potpie-context-engine[embedded]==0.2.0" in requirements
        assert not any("[all]" in requirement for requirement in requirements)

    with zipfile.ZipFile(engine_wheel) as archive:
        names = archive.namelist()
        metadata = _metadata(archive)
        assert metadata["Name"] == "potpie-context-engine"
        assert metadata["Version"] == "0.2.0"
        assert _entrypoints(archive) == {}
        runtime_roots = {
            name.split("/", 1)[0]
            for name in names
            if "/" in name and ".dist-info/" not in name
        }
        assert runtime_roots == {"potpie_context_engine"}
        assert "pydantic>=2.0" in (metadata.get_all("Requires-Dist") or [])
        assert set(metadata.get_all("Provides-Extra") or []) == ENGINE_EXTRAS
        assert any(
            name.endswith("domain/playbooks/repo_one_shot_ingestion.md")
            for name in names
        )
        assert not any(name.startswith("benchmarks/") for name in names)

    with tarfile.open(root_sdist) as archive:
        root_names = archive.getnames()
        for owned in ("auth", "config", "install", "mcp", "setup", "skills"):
            assert any(f"/potpie/{owned}/" in name for name in root_names)
        assert not any(
            name.endswith(module) for name in root_names for module in REMOVED_ROOT_MODULES
        )
        assert not any(
            f"/{prefix}" in name
            for name in root_names
            for prefix in REMOVED_ROOT_PREFIXES
        )

    with tarfile.open(engine_sdist) as archive:
        engine_names = archive.getnames()
        assert any("/src/potpie_context_engine/" in name for name in engine_names)
        assert any("/benchmarks/" in name for name in engine_names)
        assert not any(
            name.endswith("distribution_defaults_hook.py") for name in engine_names
        )
        assert not any(name.endswith("build_config_values.py") for name in engine_names)

    probe = run_isolated_python(
        "import importlib.util; "
        "import potpie.runtime as runtime; "
        "from potpie.runtime import PotpieRuntime; "
        "assert not hasattr(runtime, 'ProductShell'); "
        "assert not hasattr(runtime, 'build_product_shell'); "
        "assert importlib.util.find_spec('potpie.runtime.sync_view') is None; "
        "assert importlib.util.find_spec('potpie.cli.commands.service') is None; "
        "assert importlib.util.find_spec('potpie.daemon.runtime') is None; "
        "assert importlib.util.find_spec('potpie.daemon.ports') is None; "
        "assert importlib.util.find_spec('potpie.daemon.managed_services') is None; "
        "assert importlib.util.find_spec('potpie.daemon.http.transport') is None; "
        "assert importlib.util.find_spec('potpie.daemon.process.ipc_client') is None; "
        "print(PotpieRuntime.__name__)",
        import_roots=(root_wheel, engine_wheel),
        cwd=tmp_path,
    )
    assert probe.stdout.strip() == "PotpieRuntime"
