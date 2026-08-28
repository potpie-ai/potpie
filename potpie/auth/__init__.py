"""Potpie-owned authentication contracts, adapters, and composition."""

from potpie.auth.ports.credentials import CredentialStore
from potpie.auth.wiring import build_credential_store

__all__ = ["CredentialStore", "build_credential_store"]
