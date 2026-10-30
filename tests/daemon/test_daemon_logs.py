"""Daemon logs are bounded, filterable by time, and followable."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from potpie.daemon.lifecycle import Daemon
from potpie.daemon.log_reader import parse_since


def _write_log(home: Path, lines: list[str]) -> Path:
    log_file = home / "logs" / "potpied.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_file


def _daemon(home: Path) -> Daemon:
    return Daemon(home=home, in_process=False)


def test_logs_are_capped_by_default(tmp_path: Path) -> None:
    _write_log(tmp_path, [f"line {n}" for n in range(5000)])

    lines = _daemon(tmp_path).logs()

    assert len(lines) == 200
    assert lines[0] == "line 4800"
    assert lines[-1] == "line 4999"


def test_logs_honour_an_explicit_tail_and_read_everything_on_request(
    tmp_path: Path,
) -> None:
    _write_log(tmp_path, [f"line {n}" for n in range(5000)])
    daemon = _daemon(tmp_path)

    assert daemon.logs(tail=3) == ["line 4997", "line 4998", "line 4999"]
    assert len(daemon.logs(tail=None)) == 5000


def test_logs_tail_spans_more_than_one_read_chunk(tmp_path: Path) -> None:
    padding = "x" * 512
    _write_log(tmp_path, [f"{n} {padding}" for n in range(1000)])

    lines = _daemon(tmp_path).logs(tail=400)

    assert len(lines) == 400
    assert lines[0].startswith("600 ")


def test_logs_fall_back_to_the_legacy_log_file(tmp_path: Path) -> None:
    (tmp_path / "daemon.log").write_text("legacy line\n", encoding="utf-8")
    daemon = _daemon(tmp_path)

    assert daemon.log_path() == tmp_path / "daemon.log"
    assert daemon.logs() == ["legacy line"]


def test_logs_without_a_log_file_are_empty(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path)

    assert daemon.log_path() is None
    assert daemon.logs() == []


def test_logs_since_keeps_the_traceback_under_its_header(tmp_path: Path) -> None:
    _write_log(
        tmp_path,
        [
            "2026-08-12 09:00:00,000 INFO potpie [] old news",
            "2026-08-12 11:00:00,000 ERROR potpie [] boom",
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "RuntimeError: boom",
        ],
    )

    lines = _daemon(tmp_path).logs(since=datetime(2026, 8, 12, 10, 0, 0))

    assert lines[0].endswith("boom")
    assert "old news" not in "\n".join(lines)
    assert lines[-1] == "RuntimeError: boom"


def test_logs_since_reads_the_json_formatter_too(tmp_path: Path) -> None:
    _write_log(
        tmp_path,
        [
            json.dumps({"ts": "2026-08-12T09:00:00+0000", "msg": "old"}),
            json.dumps({"ts": "2026-08-12T23:00:00+0000", "msg": "new"}),
        ],
    )

    # Through ``parse_since`` so the cutoff lands in the same local frame the
    # offsets above are converted into; otherwise this only passes in UTC.
    lines = _daemon(tmp_path).logs(since=parse_since("2026-08-12T12:00:00+00:00"))

    assert len(lines) == 1
    assert "new" in lines[0]


def test_follow_yields_lines_appended_after_the_backlog(tmp_path: Path) -> None:
    log_file = _write_log(tmp_path, ["first"])
    stream = _daemon(tmp_path).follow_logs(tail=10, poll_interval=0.01)

    assert next(stream) == "first"

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("second\n")
    assert next(stream) == "second"

    # A partial write is held back until the line is whole.
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("thi")
        handle.flush()
        handle.write("rd\n")
    assert next(stream) == "third"
    stream.close()


def test_follow_strips_windows_line_endings(tmp_path: Path) -> None:
    log_file = _write_log(tmp_path, ["first"])
    stream = _daemon(tmp_path).follow_logs(tail=10, poll_interval=0.01)
    assert next(stream) == "first"

    with log_file.open("ab") as handle:
        handle.write(b"second\r\n")

    assert next(stream) == "second"
    stream.close()


def test_follow_waits_for_a_log_that_does_not_exist_yet(tmp_path: Path) -> None:
    stream = _daemon(tmp_path).follow_logs(poll_interval=0.02)
    produced: list[str] = []

    def _consume() -> None:
        produced.append(next(stream))

    # The reader stops after its first line and leaves the generator suspended,
    # so nothing outlives the test whether or not the assertion holds.
    reader = threading.Thread(target=_consume, daemon=True)
    reader.start()
    time.sleep(0.1)
    _write_log(tmp_path, ["late arrival"])
    reader.join(timeout=10)

    assert produced == ["late arrival"]


@pytest.mark.parametrize(
    ("text", "delta"),
    [
        ("30s", timedelta(seconds=30)),
        ("15m", timedelta(minutes=15)),
        ("2h", timedelta(hours=2)),
        ("1d", timedelta(days=1)),
    ],
)
def test_parse_since_accepts_relative_ages(text: str, delta: timedelta) -> None:
    parsed = parse_since(text)

    assert abs((datetime.now() - delta) - parsed) < timedelta(seconds=5)


def test_parse_since_accepts_iso_and_refuses_nonsense() -> None:
    assert parse_since("2026-08-12T09:00:00") == datetime(2026, 8, 12, 9, 0, 0)

    with pytest.raises(ValueError, match="cannot read 'yesterday' as a time"):
        parse_since("yesterday")
