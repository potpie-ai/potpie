from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from tests.boundary.isolated_import import run_isolated_python
from tests.boundary.normalization import normalize_engine_result


@dataclass(frozen=True, slots=True)
class _LocalResult:
    status: str
    items: tuple[str, ...]
    observed_at: datetime


class _DaemonResult(BaseModel):
    status: str
    items: list[str]
    observed_at: datetime
    request_id: str
    protocol_version: str


def test_result_normalization_ignores_only_transport_metadata() -> None:
    observed_at = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    local = _LocalResult(
        status="ready",
        items=("resolve", "search"),
        observed_at=observed_at,
    )
    daemon = _DaemonResult(
        status="ready",
        items=["resolve", "search"],
        observed_at=observed_at,
        request_id="req:daemon-only",
        protocol_version="legacy",
    )

    assert normalize_engine_result(local) == normalize_engine_result(daemon)


def test_result_normalization_is_deterministic_for_sets_and_paths(
    tmp_path: Path,
) -> None:
    value = {
        "paths": {tmp_path / "b", tmp_path / "a"},
        "transport": "domain-value-must-remain",
    }

    assert normalize_engine_result(value) == {
        "paths": [str(tmp_path / "a"), str(tmp_path / "b")],
        "transport": "domain-value-must-remain",
    }


def test_isolated_import_harness_uses_only_explicit_roots(tmp_path: Path) -> None:
    package = tmp_path / "probe_package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'isolated-ok'\n", encoding="utf-8")
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()

    success = run_isolated_python(
        "import probe_package; print(probe_package.VALUE)",
        import_roots=(tmp_path,),
        cwd=unrelated_cwd,
    )
    missing = run_isolated_python(
        "import probe_package",
        cwd=unrelated_cwd,
        check=False,
    )

    assert success.stdout.strip() == "isolated-ok"
    assert success.stderr == ""
    assert missing.returncode != 0
    assert "ModuleNotFoundError" in missing.stderr
