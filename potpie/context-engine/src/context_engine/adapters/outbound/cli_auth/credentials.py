"""Keychain-backed implementation of the ``CredentialStore`` port.

`KeyringCredentialStore` implements the core-owned
:class:`~domain.ports.cli_auth.credentials.CredentialStore` port, delegating to the
existing :mod:`adapters.outbound.cli_auth.credentials_store` module (the keyring-backed
store is *wrapped, not rewritten*). It is constructed at the composition root
(:func:`bootstrap.cli_auth_wiring.build_credential_store`); inbound code depends on
the port, never on this class. Tests inject an in-memory fake satisfying the same
port instead of monkeypatching the module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from context_engine.domain.ports.cli_auth.credentials import CredentialStore
from context_engine.adapters.outbound.cli_auth import credentials_store as _store


class KeyringCredentialStore(CredentialStore):
    """Production `CredentialStore` backed by the system keychain + config file.

    Thin delegation to the `credentials_store` module so the existing
    implementation stays the single source of truth.
    """

    # --- paths / Potpie account ------------------------------------------
    def credentials_path(self) -> Path:
        return _store.credentials_path()

    def get_stored_api_key(self) -> str:
        return _store.get_stored_api_key()

    def get_stored_api_base_url(self) -> str:
        return _store.get_stored_api_base_url()

    def write_api_base_url(self, api_base_url: Optional[str]) -> None:
        _store.write_api_base_url(api_base_url)

    def store_potpie_api_key(self, api_key: str, *, created_at: str) -> None:
        _store.store_potpie_api_key(api_key, created_at=created_at)

    def store_potpie_firebase_refresh_token(
        self,
        refresh_token: str,
        *,
        created_at: str,
        firebase_api_key: str | None = None,
    ) -> None:
        _store.store_potpie_firebase_refresh_token(
            refresh_token, created_at=created_at, firebase_api_key=firebase_api_key
        )

    def store_potpie_firebase_id_token(self, id_token: str) -> None:
        _store.store_potpie_firebase_id_token(id_token)

    def update_potpie_firebase_refresh_token(self, refresh_token: str) -> None:
        _store.update_potpie_firebase_refresh_token(refresh_token)

    def get_potpie_auth_type(self) -> str:
        return _store.get_potpie_auth_type()

    def get_potpie_firebase_refresh_token(self) -> str:
        return _store.get_potpie_firebase_refresh_token()

    def get_potpie_firebase_id_token(self) -> str:
        return _store.get_potpie_firebase_id_token()

    def get_potpie_firebase_api_key(self) -> str:
        return _store.get_potpie_firebase_api_key()

    def clear_potpie_auth(self, *, clear_api_key: bool = False) -> None:
        _store.clear_potpie_auth(clear_api_key=clear_api_key)

    # --- generic provider credentials (github) ---------------------------
    def get_provider_credentials(self, provider: str) -> dict[str, Any]:
        return _store.get_provider_credentials(provider)

    def write_provider_credentials(
        self, provider: str, payload: dict[str, Any]
    ) -> None:
        _store.write_provider_credentials(provider, payload)

    def clear_provider_credentials(self, provider: str) -> None:
        _store.clear_provider_credentials(provider)

    # --- integrations (linear / atlassian) -------------------------------
    def get_integration_tokens(self, provider: str) -> dict[str, Any]:
        return _store.get_integration_tokens(provider)

    def save_integration_tokens(self, provider: str, tokens: dict[str, Any]) -> None:
        _store.save_integration_tokens(provider, tokens)

    def clear_integration_tokens(self, provider: str) -> None:
        _store.clear_integration_tokens(provider)

    def get_integration_status(self, provider: str) -> dict[str, Any]:
        return _store.get_integration_status(provider)

    def list_integration_providers(self) -> list[str]:
        return _store.list_integration_providers()

    # --- Atlassian product credentials + workspace prefs -----------------
    def get_jira_credentials(self) -> dict[str, Any]:
        return _store.get_jira_credentials()

    def save_jira_credentials(self, credentials: dict[str, Any]) -> None:
        _store.save_jira_credentials(credentials)

    def clear_jira_credentials(self) -> None:
        _store.clear_jira_credentials()

    def get_confluence_credentials(self) -> dict[str, Any]:
        return _store.get_confluence_credentials()

    def save_confluence_credentials(self, credentials: dict[str, Any]) -> None:
        _store.save_confluence_credentials(credentials)

    def clear_confluence_credentials(self) -> None:
        _store.clear_confluence_credentials()

    def get_atlassian_credentials(self) -> dict[str, Any]:
        return _store.get_atlassian_credentials()

    def save_atlassian_credentials(self, credentials: dict[str, Any]) -> None:
        _store.save_atlassian_credentials(credentials)

    def clear_atlassian_credentials(self) -> None:
        _store.clear_atlassian_credentials()

    def save_jira_workspace_prefs(self, *, project_key: str) -> None:
        _store.save_jira_workspace_prefs(project_key=project_key)

    def save_confluence_workspace_prefs(self, *, space_key: str) -> None:
        _store.save_confluence_workspace_prefs(space_key=space_key)


__all__ = ["CredentialStore", "KeyringCredentialStore"]
