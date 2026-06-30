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
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            member = next(
                item
                for item in archive.getmembers()
                if item.name.endswith(f"/{member_suffix}")
            )
            extracted = archive.extractfile(member)
            assert extracted is not None
            return extracted.read().decode("utf-8")
    raise AssertionError(f"Unsupported build artifact: {path}")


def _build_smoke_env() -> dict[str, str]:
    env = {
        name: os.environ[name]
        for name in (
            "PATH",
            "HOME",
            "TMPDIR",
            "TEMP",
            "TMP",
            "UV_CACHE_DIR",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
        )
        if name in os.environ
    }
    env.update(
        {
            "POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS": "1",
            "POTPIE_ENVIRONMENT": "prod_oss",
            "POTPIE_SENTRY_DSN": "https://sentry.example.invalid/1",
            "POTPIE_POSTHOG_API_KEY": "phc_public_smoke",
            "POTPIE_POSTHOG_HOST": "https://posthog.example.invalid",
            "LINEAR_CLIENT_ID": "linear-smoke-client",
            "POTPIE_GITHUB_CLIENT_ID": "github-smoke-client",
            "POTPIE_BUILD_GIT_SHA": "smoke-sha",
            "POTPIE_BUILD_TIME": "2026-06-28T00:00:00Z",
        }
    )
    return env


@pytest.mark.integration
def test_distribution_defaults_build_includes_generated_modules(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the packaging smoke test")
    context_engine = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [uv, "build", "--out-dir", str(tmp_path)],
        cwd=context_engine,
        env=_build_smoke_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifacts = list(tmp_path.iterdir())
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    sdist = next(path for path in artifacts if path.name.endswith(".tar.gz"))
    for artifact in (wheel, sdist):
        distribution_defaults = _archive_text(
            artifact, "bootstrap/_distribution_defaults.py"
        )
        build_info = _archive_text(artifact, "bootstrap/_build_info.py")
        assert "DISTRIBUTION_DEFAULTS = {" in distribution_defaults
        assert "'environment': 'prod_oss'" in distribution_defaults
        assert "'sentry_dsn': 'https://sentry.example.invalid/1'" in (
            distribution_defaults
        )
        assert "'posthog_api_key': 'phc_public_smoke'" in distribution_defaults
        assert "'posthog_host': 'https://posthog.example.invalid'" in (
            distribution_defaults
        )
        assert "'linear_client_id': 'linear-smoke-client'" in distribution_defaults
        assert "'github_client_id': 'github-smoke-client'" in distribution_defaults
        assert "GIT_SHA = 'smoke-sha'" in build_info
        assert "BUILD_TIME = '2026-06-28T00:00:00Z'" in build_info
    assert not (context_engine / "bootstrap" / "_distribution_defaults.py").exists()
    assert not (context_engine / "bootstrap" / "_build_info.py").exists()
