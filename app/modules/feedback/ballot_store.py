"""Opaque ballot tokens for blind A/B presentation.

The voter is given an opaque `ballot_id` instead of the underlying randomness
seed. The token contains only an issued-at timestamp, random nonce, and HMAC;
the seed is derived server-side from the token plus comparison id. This keeps
the ordering hidden without relying on process-local state, so multiple API
workers can validate the same ballot as long as they share the same secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional, Tuple

from fastapi import HTTPException

# How long a ballot remains valid after it is issued. Long enough that voters
# can spend several minutes reading both responses without timing out, short
# enough that abandoned ballots don't accumulate forever.
BALLOT_TTL_SECONDS = 60 * 60  # 60 minutes
LOCAL_ENV_NAMES = {"local", "dev", "development", "test", "testing"}
LOCAL_DEV_BALLOT_SECRET = "potpie-feedback-local-dev-secret"

_TIMESTAMP_BYTES = 4
_NONCE_BYTES = 16
_SIGNATURE_BYTES = 16
_RAW_BALLOT_BYTES = _TIMESTAMP_BYTES + _NONCE_BYTES + _SIGNATURE_BYTES


def _is_local_environment() -> bool:
    if os.getenv("isDevelopmentMode", "").strip().lower() == "enabled":
        return True
    env_name = (
        os.getenv("ENVIRONMENT")
        or os.getenv("ENV")
        or os.getenv("LOGFIRE_ENVIRONMENT")
        or ""
    ).strip().lower()
    return env_name in LOCAL_ENV_NAMES


def _ballot_secret() -> bytes:
    """Shared secret used to sign opaque ballots and derive presentation seeds."""
    secret = os.getenv("FEEDBACK_BALLOT_SECRET") or os.getenv(
        "FEEDBACK_ADMIN_PASSWORD"
    )
    if not secret and _is_local_environment():
        secret = LOCAL_DEV_BALLOT_SECRET
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="FEEDBACK_BALLOT_SECRET is required for this environment",
        )
    return secret.encode("utf-8")


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> Optional[bytes]:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return None


def _signature(secret: bytes, comparison_id: str, issued_and_nonce: bytes) -> bytes:
    return hmac.new(
        secret,
        b"feedback-ballot:" + comparison_id.encode("utf-8") + b":" + issued_and_nonce,
        hashlib.sha256,
    ).digest()[:_SIGNATURE_BYTES]


def _seed(secret: bytes, comparison_id: str, issued_and_nonce: bytes) -> int:
    digest = hmac.new(
        secret,
        b"feedback-seed:" + comparison_id.encode("utf-8") + b":" + issued_and_nonce,
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class BallotStore:
    """Stateless issuer/verifier for opaque ballots."""

    def __init__(self, ttl_seconds: int = BALLOT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds

    def issue(self, comparison_id: str) -> Tuple[str, int]:
        """Mint a fresh ballot for the given comparison and return (ballot_id, seed)."""
        secret = _ballot_secret()
        issued_at = int(time.time()).to_bytes(_TIMESTAMP_BYTES, "big")
        issued_and_nonce = issued_at + secrets.token_bytes(_NONCE_BYTES)
        signature = _signature(secret, comparison_id, issued_and_nonce)
        ballot_id = _urlsafe_b64encode(issued_and_nonce + signature)
        return ballot_id, _seed(secret, comparison_id, issued_and_nonce)

    def redeem(self, comparison_id: str, ballot_id: str) -> Optional[int]:
        """Resolve a ballot to its seed, or None if malformed, forged, or expired."""
        raw = _urlsafe_b64decode(ballot_id)
        if raw is None or len(raw) != _RAW_BALLOT_BYTES:
            return None

        issued_and_nonce = raw[: _TIMESTAMP_BYTES + _NONCE_BYTES]
        provided_signature = raw[_TIMESTAMP_BYTES + _NONCE_BYTES :]
        secret = _ballot_secret()
        expected_signature = _signature(secret, comparison_id, issued_and_nonce)
        if not hmac.compare_digest(provided_signature, expected_signature):
            return None

        issued_at = int.from_bytes(raw[:_TIMESTAMP_BYTES], "big")
        now = int(time.time())
        if issued_at > now + 60 or now - issued_at > self._ttl:
            return None

        return _seed(secret, comparison_id, issued_and_nonce)


# Module-level singleton. Tests can substitute their own instance via the
# `set_ballot_store` helper below.
_ballot_store = BallotStore()


def get_ballot_store() -> BallotStore:
    return _ballot_store


def set_ballot_store(store: BallotStore) -> None:
    """Replace the singleton (test hook)."""
    global _ballot_store
    _ballot_store = store
