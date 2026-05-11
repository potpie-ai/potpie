from __future__ import annotations

from pathlib import Path

import potpie_cli as cli


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.body


def test_start_runs_start_script(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "start.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["start", "--root", str(tmp_path)]) == 0
    assert calls == [(["bash", str(script)], tmp_path.resolve())]


def test_stop_reports_missing_script(tmp_path: Path, capsys) -> None:
    assert cli.main(["stop", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "scripts/stop.sh" in captured.err


def test_status_runs_docker_compose_ps(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["status", "--root", str(tmp_path)]) == 0
    assert calls == [(["docker", "compose", "ps"], tmp_path.resolve())]


def test_health_prints_json_response(monkeypatch, capsys) -> None:
    def fake_urlopen(url, *, timeout):
        assert url == "http://localhost:8001/health"
        assert timeout == 5.0
        return FakeResponse(b'{"status":"ok"}')

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["health"]) == 0
    captured = capsys.readouterr()
    assert '"status": "ok"' in captured.out


def test_health_reports_unreachable_api(monkeypatch, capsys) -> None:
    def fake_urlopen(_url, *, timeout):
        assert timeout == 5.0
        raise cli.URLError("connection refused")

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["health", "--api-url", "http://127.0.0.1:9999"]) == 1
    captured = capsys.readouterr()
    assert "not reachable" in captured.err


def test_health_reports_timeout(monkeypatch, capsys) -> None:
    def fake_urlopen(_url, *, timeout):
        assert timeout == 5.0
        raise cli.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["health"]) == 1
    captured = capsys.readouterr()
    assert "timed out" in captured.err


def test_health_rejects_non_positive_timeout(capsys) -> None:
    assert cli.main(["health", "--timeout", "0"]) == 1
    captured = capsys.readouterr()
    assert "timeout must be greater than 0" in captured.err


def test_health_rejects_invalid_api_url(capsys) -> None:
    assert cli.main(["health", "--api-url", "localhost:8001"]) == 1
    captured = capsys.readouterr()
    assert "Invalid API URL" in captured.err
