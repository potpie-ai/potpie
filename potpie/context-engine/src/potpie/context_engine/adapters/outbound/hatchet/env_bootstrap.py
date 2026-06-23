"""Align process env with Hatchet self-hosted (Docker) defaults before ``Hatchet()`` / ``ClientConfig()``."""

from __future__ import annotations

import os


def prepare_hatchet_client_env() -> None:
    """
    The Python SDK defaults to TLS for gRPC; Hatchet Lite in Docker uses insecure gRPC.

    If ``HATCHET_CLIENT_TLS_STRATEGY`` is unset and ``HATCHET_CLIENT_SERVER_URL`` uses plain
    ``http://``, set ``HATCHET_CLIENT_TLS_STRATEGY=none`` so ``grpc.insecure_channel`` is used
    (see ``hatchet_sdk.connection.new_conn``).

    Call this before constructing ``Hatchet()`` or ``ClientConfig()`` so pydantic-settings picks
    up the values.
    """
    if os.getenv("HATCHET_CLIENT_TLS_STRATEGY"):
        return
    url = (os.getenv("HATCHET_CLIENT_SERVER_URL") or "").strip()
    if url.startswith("http://"):
        os.environ["HATCHET_CLIENT_TLS_STRATEGY"] = "none"
