from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


def _archive_text(path: Path, member_suffix: str) -> str:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.read(member_suffix).decode("utf-8")
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            item
            for item in archive.getmembers()
            if item.name.endswith(f"/{member_suffix}")
        )
        extracted = archive.extractfile(member)
        assert extracted is not None
        return extracted.read().decode("utf-8")


@pytest.mark.integration
def test_engine_build_packages_only_standalone_sentry_defaults(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the packaging smoke test")
    engine_root = Path(__file__).resolve().parents[2]
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR", "UV_CACHE_DIR", "SSL_CERT_FILE")
        if name in os.environ
    }
    env.update(
        {
            "POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS": "1",
            "POTPIE_ENVIRONMENT": "prod_oss",
            "POTPIE_SENTRY_DSN": "https://sentry.example.invalid/1",
            "POTPIE_BUILD_GIT_SHA": "smoke-sha",
            "POTPIE_BUILD_TIME": "2026-06-28T00:00:00Z",
        }
    )

    result = subprocess.run(
        [uv, "build", "--out-dir", str(tmp_path)],
        cwd=engine_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifacts = list(tmp_path.iterdir())
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    sdist = next(path for path in artifacts if path.name.endswith(".tar.gz"))
    for artifact in (wheel, sdist):
        defaults = _archive_text(
            artifact, "potpie_context_engine/bootstrap/_distribution_defaults.py"
        )
        build_info = _archive_text(
            artifact, "potpie_context_engine/bootstrap/_build_info.py"
        )
        assert "'environment': 'prod_oss'" in defaults
        assert "'sentry_dsn': 'https://sentry.example.invalid/1'" in defaults
        assert "posthog" not in defaults
        assert "client_id" not in defaults
        assert "GIT_SHA = 'smoke-sha'" in build_info

    generated_dir = engine_root / "src" / "potpie_context_engine" / "bootstrap"
    assert not (generated_dir / "_distribution_defaults.py").exists()
    assert not (generated_dir / "_build_info.py").exists()
