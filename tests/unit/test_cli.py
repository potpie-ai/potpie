from pathlib import Path
import json

from typer.testing import CliRunner

from potpie.cli import app


runner = CliRunner()


def test_status_reports_not_running_in_empty_workspace():
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Potpie is not running." in result.output


def test_parse_rejects_non_git_directory(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    result = runner.invoke(app, ["parse", str(repo_path)])

    assert result.exit_code != 0
    assert "not a Git repository" in result.output


def test_start_writes_pid_and_log_path(monkeypatch):
    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 12345

    monkeypatch.setattr("potpie.cli.subprocess.Popen", FakePopen)

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["start", "--sandbox", "local"])
        pid_file = Path(".potpie/potpie.pid")
        log_file = Path(".potpie/potpie.log")

        assert result.exit_code == 0
        assert json.loads(pid_file.read_text()) == {
            "pid": 12345,
            "command": ["make", "dev", "SANDBOX=local"],
        }
        assert log_file.exists()
        assert "Started Potpie with PID 12345." in result.output
