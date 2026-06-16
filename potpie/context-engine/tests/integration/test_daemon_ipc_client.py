"""Additional client.py coverage: tcp bind path, missing discovery, unknown bind."""

from __future__ import annotations
import json
import pathlib
import pytest
from adapters.outbound.daemon_process.ipc_client import client_for, load_discovery


def test_load_discovery_missing(tmp_path: pathlib.Path):
    assert load_discovery(tmp_path) is None


def test_load_discovery_present(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text(json.dumps({"bind": "unix:/tmp/x.sock"}))
    d = load_discovery(tmp_path)
    assert d is not None
    assert d["bind"] == "unix:/tmp/x.sock"


def test_load_discovery_invalid_json_returns_none(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text("{not-json")

    assert load_discovery(tmp_path) is None


def test_client_for_raises_when_no_discovery(tmp_path: pathlib.Path):
    with pytest.raises(RuntimeError, match="no discovery file"):
        client_for(tmp_path)


def test_client_for_raises_when_discovery_has_no_bind(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text(json.dumps({"transport": "http"}))

    with pytest.raises(RuntimeError, match="missing 'bind'"):
        client_for(tmp_path)


def test_client_for_tcp_bind(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text(json.dumps({"bind": "tcp:127.0.0.1:9999"}))
    c = client_for(tmp_path)
    # Should return an httpx.Client without raising
    c.close()


def test_client_for_ipv6_tcp_bind(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text(json.dumps({"bind": "tcp:[::1]:9999"}))
    c = client_for(tmp_path)

    assert str(c.base_url) == "http://[::1]:9999"
    c.close()


def test_client_for_unknown_bind_raises(tmp_path: pathlib.Path):
    (tmp_path / "discovery.json").write_text(json.dumps({"bind": "ftp://something"}))
    with pytest.raises(RuntimeError, match="unknown bind"):
        client_for(tmp_path)
