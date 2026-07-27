import os
import pathlib
import stat
import threading
from unittest.mock import patch

import pytest
from potpie.daemon.process.pidfile import (
    AlreadyRunning,
    read_discovery,
    read_pid_file,
    remove_pid_file,
    write_discovery,
    write_pid_file,
)


def test_pid_roundtrip(tmp_path: pathlib.Path):
    p = tmp_path / "potpied.pid"
    write_pid_file(p, 12345)
    assert read_pid_file(p) == 12345
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    remove_pid_file(p)
    assert not p.exists()


def test_pid_already_running_when_live_pid(tmp_path: pathlib.Path):
    p = tmp_path / "potpied.pid"
    write_pid_file(p, os.getpid())  # use our own PID — guaranteed alive
    with pytest.raises(AlreadyRunning):
        write_pid_file(p, 99999)


def test_pid_overwrites_stale(tmp_path: pathlib.Path):
    p = tmp_path / "potpied.pid"
    # write a PID we know isn't running
    p.write_text("999999\n")
    write_pid_file(p, 12345)
    assert read_pid_file(p) == 12345


def test_pid_file_is_created_exclusively(tmp_path: pathlib.Path):
    p = tmp_path / "potpied.pid"

    with patch("potpie.daemon.process.pidfile.os.open", wraps=os.open) as open_file:
        write_pid_file(p, 12345)

    flags = open_file.call_args.args[1]
    assert flags & os.O_EXCL


def test_stale_pid_replacement_is_serialized(tmp_path: pathlib.Path, monkeypatch):
    p = tmp_path / "potpied.pid"
    p.write_text("999999\n", encoding="utf-8")
    stale_removal_started = threading.Event()
    allow_stale_removal = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    second_errors: list[Exception] = []
    original_remove = remove_pid_file

    def blocking_remove(path: pathlib.Path) -> None:
        stale_removal_started.set()
        assert allow_stale_removal.wait(timeout=2)
        original_remove(path)

    def first_writer() -> None:
        write_pid_file(p, 111)

    def second_writer() -> None:
        second_started.set()
        try:
            write_pid_file(p, 222)
        except Exception as exc:
            second_errors.append(exc)
        finally:
            second_done.set()

    monkeypatch.setattr(
        "potpie.daemon.process.pidfile._pid_alive", lambda pid: pid == 111
    )
    monkeypatch.setattr(
        "potpie.daemon.process.pidfile.remove_pid_file", blocking_remove
    )

    first = threading.Thread(target=first_writer)
    second = threading.Thread(target=second_writer)
    first.start()
    assert stale_removal_started.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    assert not second_done.wait(timeout=0.1)

    allow_stale_removal.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert read_pid_file(p) == 111
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], AlreadyRunning)


def test_discovery_roundtrip(tmp_path: pathlib.Path):
    f = tmp_path / "discovery.json"
    write_discovery(f, transport="http", bind="unix:/tmp/x.sock")
    d = read_discovery(f)
    assert d["transport"] == "http"
    assert d["bind"] == "unix:/tmp/x.sock"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_discovery_overwrites_permissive_file_as_private(tmp_path: pathlib.Path):
    f = tmp_path / "discovery.json"
    f.write_text("{}", encoding="utf-8")
    f.chmod(0o644)

    write_discovery(f, token="secret")

    assert read_discovery(f)["token"] == "secret"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
