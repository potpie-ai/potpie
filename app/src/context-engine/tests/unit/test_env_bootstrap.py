"""env_bootstrap .env parsing."""

from pathlib import Path

import pytest

from adapters.inbound.cli import env_bootstrap as eb


def test_parse_env_line() -> None:
    assert eb._parse_env_line("FOO=bar") == ("FOO", "bar")
    assert eb._parse_env_line('  export BAZ="x y"  ') == ("BAZ", "x y")
    assert eb._parse_env_line("# comment") is None


def test_load_env_file_respects_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / ".env"
    p.write_text("A=from_file\nB=2\n", encoding="utf-8")
    monkeypatch.setenv("A", "keep")
    eb._load_env_file(p)
    import os

    assert os.environ["A"] == "keep"
    assert os.environ["B"] == "2"
    del os.environ["B"]
