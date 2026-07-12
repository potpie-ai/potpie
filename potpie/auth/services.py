"""Root-owned account and provider authentication services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from potpie.auth.credentials import CredentialStore


@dataclass(frozen=True, slots=True)
class AccountIdentity:
    subject: str
    authenticated: bool
    auth_type: str


@dataclass(slots=True)
class AccountAuthService:
    credentials: CredentialStore

    def login_api_key(
        self, api_key: str, *, api_url: str | None = None
    ) -> AccountIdentity:
        token = api_key.strip()
        if not token:
            raise ValueError("API key cannot be empty")
        self.credentials.store_potpie_api_key(
            token, created_at=datetime.now(timezone.utc).isoformat()
        )
        self.credentials.write_api_base_url(api_url)
        return self.whoami()

    def logout(self) -> None:
        self.credentials.clear_potpie_auth(clear_api_key=True)

    def whoami(self) -> AccountIdentity:
        auth_type = self.credentials.get_potpie_auth_type() or "none"
        authenticated = auth_type in {"api_key", "firebase", "potpie"}
        return AccountIdentity(
            subject="potpie-account" if authenticated else "anonymous",
            authenticated=authenticated,
            auth_type=auth_type,
        )


@dataclass(frozen=True, slots=True)
class IntegrationStatus:
    provider: str
    connected: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntegrationAuthService:
    credentials: CredentialStore

    def list(self) -> tuple[IntegrationStatus, ...]:
        providers = {"github", "linear", "jira", "confluence"}
        providers.update(self.credentials.list_integration_providers())
        return tuple(self.status(provider) for provider in sorted(providers))

    def status(self, provider: str) -> IntegrationStatus:
        name = provider.strip().lower()
        if name == "github":
            details = self.credentials.get_provider_credentials(name)
            connected = bool(details.get("access_token") or details.get("token"))
        elif name == "jira":
            details = self.credentials.get_jira_credentials()
            connected = bool(details)
        elif name == "confluence":
            details = self.credentials.get_confluence_credentials()
            connected = bool(details)
        else:
            details = self.credentials.get_integration_status(name)
            connected = bool(details.get("authenticated") or details.get("connected"))
        return IntegrationStatus(
            provider=name,
            connected=connected,
            details={
                key: value
                for key, value in details.items()
                if key not in {"access_token", "refresh_token", "token", "api_token"}
            },
        )

    def logout(self, provider: str) -> None:
        name = provider.strip().lower()
        if name == "github":
            self.credentials.clear_provider_credentials(name)
        elif name == "jira":
            self.credentials.clear_jira_credentials()
        elif name == "confluence":
            self.credentials.clear_confluence_credentials()
        else:
            self.credentials.clear_integration_tokens(name)


__all__ = [
    "AccountAuthService",
    "AccountIdentity",
    "IntegrationAuthService",
    "IntegrationStatus",
]
