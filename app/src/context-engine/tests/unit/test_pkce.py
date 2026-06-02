"""Tests for PKCE helper generation."""

from __future__ import annotations

import base64
import hashlib

from adapters.inbound.cli.pkce import generate_pkce_pair


def test_generate_pkce_pair_returns_s256_challenge() -> None:
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) == 64
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected


def test_generate_pkce_pair_produces_unique_values() -> None:
    first = generate_pkce_pair()
    second = generate_pkce_pair()
    assert first != second
