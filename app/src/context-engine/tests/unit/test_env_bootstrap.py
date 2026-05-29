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


def test_load_cli_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_root = tmp_path / "context-engine"
    cli_root.mkdir()
    (cli_root / ".env.cli").write_text(
        "POTPIE_CLI_API_BASE_URL=https://stage-api.potpie.ai\n"
        "POTPIE_CLI_UI_BASE_URL=https://stage.potpie.ai\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("POTPIE_CLI_API_BASE_URL", raising=False)
    monkeypatch.delenv("POTPIE_CLI_UI_BASE_URL", raising=False)
    monkeypatch.setattr(eb, "_cli_package_root", lambda: cli_root)

    eb._load_cli_env_defaults()

    import os

    assert os.environ["POTPIE_CLI_API_BASE_URL"] == "https://stage-api.potpie.ai"
    assert os.environ["POTPIE_CLI_UI_BASE_URL"] == "https://stage.potpie.ai"
