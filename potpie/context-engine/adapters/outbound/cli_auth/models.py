from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class AccessToken:
    access_token: str
    token_type: str
    scopes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GitHubAccount:
    login: str
    id: int
    name: str | None
    email: str | None


@dataclass(frozen=True)
class ProviderCredentials:
    provider: str
    provider_host: str
    access_token: str
    token_type: str
    scopes: list[str]
    account: dict[str, Any]
    created_at: str
    updated_at: str
    expires_at: str | None
    metadata: dict[str, Any]
    token_storage: str = "file"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_host": self.provider_host,
            "access_token": self.access_token,
            "token_type": self.token_type,
            "scopes": list(self.scopes),
            "account": dict(self.account),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
            "token_storage": self.token_storage,
        }
