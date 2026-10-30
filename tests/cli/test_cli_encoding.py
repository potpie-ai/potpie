"""The CLI owns the encoding of its output, including when stdout is a pipe."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap

import pytest

from potpie.cli import main as cli_main

pytestmark = pytest.mark.unit

_SAMPLE = "Grüße/Straße/Café ⁠ → ↳"


def _run_python(
    args: list[str], *, encoding: str
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, *args],
        env={**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUTF8": "0"},
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_help_emits_utf8_with_a_legacy_pipe_encoding(encoding: str) -> None:
    result = _run_python(["-m", "potpie.cli.main", "--help"], encoding=encoding)

    assert result.returncode == 0, result.stderr.decode("utf-8")
    assert "context_resolve — a bounded context wrap" in result.stdout.decode("utf-8")
    assert result.stderr == b""


@pytest.mark.parametrize("as_json", [False, True])
def test_command_output_preserves_unicode_on_both_streams(as_json: bool) -> None:
    # Substitute only dispatch: exercise run_cli and the real result/error
    # presenters without contacting an engine or relying on a particular graph.
    script = textwrap.dedent(
        r"""
        import sys
        from potpie.cli import main as cli
        from potpie.cli.commands import _common
        from potpie.cli.ui.output import configure_error_output, emit_error

        sample = "Grüße/Straße/Café ⁠ → ↳"
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
    if as_json:
        # The result is the one JSON value on stdout; the structured error
        # stays on stderr, and both survive the legacy code page.
        assert json.loads(stdout)["text"] == _SAMPLE
        assert json.loads(stderr)["message"] == _SAMPLE
    else:
        assert _SAMPLE in stdout
        assert _SAMPLE in stderr


def test_help_preserves_caller_owned_text_streams(monkeypatch) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    cli_main.run_cli(["--help"])

    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert "context_resolve — a bounded context wrap" in stdout.getvalue()
    assert stderr.getvalue() == ""
