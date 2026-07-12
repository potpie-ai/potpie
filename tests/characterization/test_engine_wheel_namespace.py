from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.boundary.isolated_import import run_isolated_python

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "potpie" / "context-engine"
GENERIC_TOP_LEVEL_PACKAGES = {
    "adapters",
    "application",
    "bootstrap",
    "domain",
    "host",
}


def test_engine_wheel_uses_only_the_distribution_namespace(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the isolated engine-wheel smoke test")

    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ENGINE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    wheel = next(tmp_path.glob("potpie_context_engine-*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        runtime_roots = {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and not name.endswith("/")
        }

    assert "potpie_context_engine" in runtime_roots
    assert GENERIC_TOP_LEVEL_PACKAGES.isdisjoint(runtime_roots)
    assert "benchmarks" not in runtime_roots

    probe = run_isolated_python(
        "import potpie_context_engine; "
        "from potpie_context_engine.domain.actor import Actor; "
        "print(potpie_context_engine.__name__, Actor.__name__)",
        import_roots=(wheel,),
        cwd=tmp_path,
    )
    assert probe.stdout.strip() == "potpie_context_engine Actor"
