"""Hatchet client env defaults for local Docker (insecure gRPC)."""

import os

from adapters.outbound.hatchet.env_bootstrap import prepare_hatchet_client_env


def test_prepare_sets_tls_none_for_http_server_url(monkeypatch) -> None:
    monkeypatch.delenv("HATCHET_CLIENT_TLS_STRATEGY", raising=False)
    monkeypatch.setenv("HATCHET_CLIENT_SERVER_URL", "http://localhost:8888")
    prepare_hatchet_client_env()
    assert os.environ.get("HATCHET_CLIENT_TLS_STRATEGY") == "none"


def test_prepare_respects_explicit_tls(monkeypatch) -> None:
    monkeypatch.setenv("HATCHET_CLIENT_TLS_STRATEGY", "tls")
    monkeypatch.setenv("HATCHET_CLIENT_SERVER_URL", "http://localhost:8888")
    prepare_hatchet_client_env()
    assert os.environ.get("HATCHET_CLIENT_TLS_STRATEGY") == "tls"
