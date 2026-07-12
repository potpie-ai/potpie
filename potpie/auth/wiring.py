"""Composition root for CLI auth credential persistence."""

from __future__ import annotations

from potpie.auth.credentials import CredentialStore, FileCredentialStore


def build_credential_store() -> CredentialStore:
    """Construct the production file-backed credential store."""
    return FileCredentialStore()


__all__ = ["build_credential_store"]
