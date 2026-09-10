"""The CLI owns the encoding of its output, including when stdout is a pipe."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap

import pytest

from potpie.cli import main as host_cli


def _run_python(
    args: list[str], *, encoding: str
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, *args],
        env={**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUTF8": "0"},
        capture_output=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_help_emits_utf8_with_a_legacy_pipe_encoding(encoding: str) -> None:
    result = _run_python(["-m", "potpie.cli.main", "--help"], encoding=encoding)

    assert result.returncode == 0, result.stderr.decode("utf-8")
    assert "CLI → HostShell" in result.stdout.decode("utf-8")
    assert result.stderr == b""


@pytest.mark.parametrize("as_json", [False, True])
def test_command_output_preserves_unicode_on_both_streams(as_json: bool) -> None:
    # Substitute only dispatch: exercise run_cli and the real result/error
    # presenters without contacting a host or relying on a particular graph.
    script = textwrap.dedent(
        r"""
        import sys
        from potpie.cli import main as cli
        from potpie.cli.commands import _common
        from potpie.cli.ui.output import configure_error_output, emit_error

        sample = "Prüfung/Montage/Service \u2060 → ↳"
        as_json = sys.argv[1] == "json"

        def command(*args, **kwargs):
            _common.set_json(as_json)
            configure_error_output(as_json=as_json)
            _common.emit({"text": sample}, human=sample)
            emit_error("Probe", sample)

        cli.app = command
        cli.run_cli([])
        """
    )
    result = _run_python(
        ["-c", script, "json" if as_json else "human"], encoding="cp1252"
    )

    assert result.returncode == 0, result.stderr.decode("utf-8")
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    sample = "Prüfung/Montage/Service \u2060 → ↳"
    if as_json:
        success, error = [json.loads(line) for line in stdout.splitlines()]
        assert success["text"] == sample
        assert error["message"] == sample
        assert stderr == ""
    else:
        assert sample in stdout
        assert sample in stderr


def test_help_preserves_caller_owned_text_streams(monkeypatch) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    host_cli.run_cli(["--help"])

    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert "CLI → HostShell" in stdout.getvalue()
    assert stderr.getvalue() == ""
