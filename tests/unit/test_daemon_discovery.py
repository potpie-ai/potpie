from __future__ import annotations

import json
from pathlib import Path

from potpie.daemon.process.discovery import load_discovery, parse_discovery


def test_load_discovery_returns_none_for_missing_or_invalid_file(tmp_path: Path) -> None:
    assert load_discovery(tmp_path) is None

    (tmp_path / "discovery.json").write_text("{not-json", encoding="utf-8")

    assert load_discovery(tmp_path) is None


def test_load_discovery_preserves_active_typed_fields(tmp_path: Path) -> None:
    (tmp_path / "discovery.json").write_text(
        json.dumps(
            {
                "transport": "http",
                "base_url": "http://127.0.0.1:12345",
                "token": "unit-test-token",
                "log_file": "/tmp/potpie.log",
                "backend": "embedded",
                "pid": 123,
            }
        ),
        encoding="utf-8",
    )

    assert load_discovery(tmp_path) == {
        "transport": "http",
        "base_url": "http://127.0.0.1:12345",
        "token": "unit-test-token",
        "log_file": "/tmp/potpie.log",
        "backend": "embedded",
        "pid": 123,
    }


def test_parse_discovery_ignores_legacy_bind_and_coerces_string_pid() -> None:
    assert parse_discovery({"pid": "123", "bind": "unix:/tmp/legacy.sock"}) == {
        "pid": 123
    }
