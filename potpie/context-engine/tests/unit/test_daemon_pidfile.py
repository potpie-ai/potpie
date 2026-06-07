import os
import pathlib
import pytest
from adapters.outbound.daemon_process.pidfile import (
    write_pid_file,
    read_pid_file,
    remove_pid_file,
    write_discovery,
    read_discovery,
    AlreadyRunning,
)


def test_pid_roundtrip(tmp_path: pathlib.Path):
    p = tmp_path / "potpied.pid"
    write_pid_file(p, 12345)
    assert read_pid_file(p) == 12345
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


def test_discovery_roundtrip(tmp_path: pathlib.Path):
    f = tmp_path / "discovery.json"
    write_discovery(f, transport="http", bind="unix:/tmp/x.sock")
    d = read_discovery(f)
    assert d["transport"] == "http"
    assert d["bind"] == "unix:/tmp/x.sock"
