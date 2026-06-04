"""PKCE helpers for OAuth flows that require code challenge verification."""

from __future__ import annotations

import base64
import hashlib
import secrets


def generate_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)[:64]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge
