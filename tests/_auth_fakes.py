"""In-memory test doubles for the CLI auth seams (HttpClient + CredentialStore).

Let auth flow/impl tests inject fakes instead of monkeypatching ``httpx`` or the
``credentials_store`` module. ``FakeAuthHttpClient`` records calls and returns
scripted responses; ``InMemoryCredentialStore`` keeps credentials in dicts and
satisfies the :class:`~domain.ports.cli_auth.credentials.CredentialStore`
Protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from adapters.outbound.cli_auth.credentials_store import ProviderCredentialError


class FakeAuthHttpClient:
    """Scriptable :class:`~adapters.outbound.cli_auth.http.HttpClient` for tests.

    Provide either ``responses`` (returned in order) or a ``handler``
    ``(method, url, **kwargs) -> httpx.Response``. Every call is recorded on
    ``.calls`` as ``(method, url, kwargs)``.
    """

    def __init__(
        self,
        responses: Optional[list[httpx.Response]] = None,
        *,
        handler: Optional[Callable[..., httpx.Response]] = None,
    ) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._responses = list(responses or [])
        self._handler = handler
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        if self._handler is not None:
            return self._handler(method, url, **kwargs)
        if self._responses:
            return self._responses.pop(0)
        raise AssertionError(
            f"FakeAuthHttpClient: no scripted response for {method} {url}"
        )

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FakeAuthHttpClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class InMemoryCredentialStore:
    """In-memory :class:`CredentialStore` — no keychain, no filesystem."""

    def __init__(self) -> None:
        self.api_key: str = ""
        self.api_base_url: str = ""
        self.auth_type: str = ""
        self.firebase_refresh_token: str = ""
        self.firebase_id_token: str = ""
        self.firebase_api_key: str = ""
        self.providers: dict[str, dict[str, Any]] = {}
        self.integrations: dict[str, dict[str, Any]] = {}
        self.atlassian: dict[str, dict[str, Any]] = {}
        self.workspace_prefs: dict[str, dict[str, Any]] = {}

    # --- paths / Potpie account ------------------------------------------
    def credentials_path(self) -> Path:
        return Path("/tmp/fake/credentials.json")

    def get_stored_api_key(self) -> str:
        return self.api_key

    def get_stored_api_base_url(self) -> str:
        return self.api_base_url

    def write_api_base_url(self, api_base_url: Optional[str]) -> None:
        self.api_base_url = (api_base_url or "").strip()

    def store_potpie_api_key(self, api_key: str, *, created_at: str) -> None:
        self.api_key = api_key
        self.auth_type = "api_key"

    def store_potpie_firebase_refresh_token(
        self,
        refresh_token: str,
        *,
        created_at: str,
        firebase_api_key: str | None = None,
    ) -> None:
        self.firebase_refresh_token = refresh_token
        if firebase_api_key:
            self.firebase_api_key = firebase_api_key
        self.auth_type = "potpie"

    def store_potpie_firebase_id_token(self, id_token: str) -> None:
        self.firebase_id_token = id_token

    def update_potpie_firebase_refresh_token(self, refresh_token: str) -> None:
        self.firebase_refresh_token = refresh_token

    def get_potpie_auth_type(self) -> str:
        return self.auth_type

    def get_potpie_firebase_refresh_token(self) -> str:
        return self.firebase_refresh_token

    def get_potpie_firebase_id_token(self) -> str:
        return self.firebase_id_token

    def get_potpie_firebase_api_key(self) -> str:
        return self.firebase_api_key

    def clear_potpie_auth(self, *, clear_api_key: bool = False) -> None:
        self.firebase_refresh_token = ""
        self.firebase_id_token = ""
        self.firebase_api_key = ""
        self.auth_type = ""
        if clear_api_key:
            self.api_key = ""

    # --- generic provider credentials (github) ---------------------------
    def get_provider_credentials(self, provider: str) -> dict[str, Any]:
        return dict(self.providers.get(provider, {}))

    def write_provider_credentials(
        self, provider: str, payload: dict[str, Any]
    ) -> None:
        self.providers[provider] = dict(payload)

    def clear_provider_credentials(self, provider: str) -> None:
        self.providers.pop(provider, None)

    # --- integrations (linear / atlassian) -------------------------------
    def get_integration_tokens(self, provider: str) -> dict[str, Any]:
        return dict(self.integrations.get(provider, {}))

    def save_integration_tokens(self, provider: str, tokens: dict[str, Any]) -> None:
        self.integrations[provider] = dict(tokens)

    def clear_integration_tokens(self, provider: str) -> None:
        self.integrations.pop(provider, None)

    def get_integration_status(self, provider: str) -> dict[str, Any]:
        if provider == "github":
            credentials = self.providers.get(provider, {})
            account = credentials.get("account")
            account_dict = dict(account) if isinstance(account, dict) else {}
            return {
                "provider": provider,
                "authenticated": bool(credentials.get("access_token")),
                "login": account_dict.get("login"),
                "email": account_dict.get("email"),
                "expires_at": credentials.get("expires_at"),
            }
        tokens = self.integrations.get(provider, {})
        authenticated = bool(tokens.get("access_token") or tokens.get("api_token"))
        return {
            "provider": provider,
            "authenticated": authenticated,
            "expires_at": tokens.get("expires_at"),
        }

    def list_integration_providers(self) -> list[str]:
        return sorted(self.integrations)

    # --- Atlassian product credentials + workspace prefs -----------------
    def get_jira_credentials(self) -> dict[str, Any]:
        return dict(self.atlassian.get("jira", {}))

    def save_jira_credentials(self, credentials: dict[str, Any]) -> None:
        self.atlassian["jira"] = dict(credentials)

    def clear_jira_credentials(self) -> None:
        self.atlassian.pop("jira", None)

    def get_confluence_credentials(self) -> dict[str, Any]:
        return dict(self.atlassian.get("confluence", {}))

    def save_confluence_credentials(self, credentials: dict[str, Any]) -> None:
        self.atlassian["confluence"] = dict(credentials)

    def clear_confluence_credentials(self) -> None:
        self.atlassian.pop("confluence", None)

    def get_atlassian_credentials(self) -> dict[str, Any]:
        return dict(self.atlassian.get("atlassian", {}))

    def save_atlassian_credentials(self, credentials: dict[str, Any]) -> None:
        self.atlassian["atlassian"] = dict(credentials)

    def clear_atlassian_credentials(self) -> None:
        self.atlassian.pop("atlassian", None)

    def save_jira_workspace_prefs(self, *, project_key: str) -> None:
        self.workspace_prefs["jira"] = {"project_key": project_key}

    def save_confluence_workspace_prefs(self, *, space_key: str) -> None:
        self.workspace_prefs["confluence"] = {"space_key": space_key}

    # --- GitLab credentials + workspace prefs ----------------------------
    def get_gitlab_credentials(
        self,
        instance_host: str | None = None,
    ) -> dict[str, Any]:
        gitlab = self.providers.get("gitlab", {})
        if not gitlab:
            return {}
        instances = gitlab.get("instances", {})
        host = instance_host or gitlab.get("active_instance_host")
        if host and isinstance(instances, dict):
            return dict(instances.get(host, {}))
        return {}

    def save_gitlab_credentials(
        self,
        credentials: dict[str, Any],
        *,
        account: dict[str, Any] | None = None,
    ) -> None:
        pat = str(credentials.get("personal_access_token") or "").strip()
        if not pat:
            raise ProviderCredentialError("GitLab personal access token is required.")
        gitlab = dict(self.providers.get("gitlab", {}))
        instances = dict(gitlab.get("instances", {}))
        host = credentials.get("instance_host") or "gitlab.com"
        entry = dict(credentials)
        entry.pop("personal_access_token", None)
        if account:
            entry["account"] = account
        existing = instances.get(host) or {}
        if "workspaces" not in entry and isinstance(existing.get("workspaces"), dict):
            entry["workspaces"] = existing["workspaces"]
        if "created_at" not in entry and existing.get("created_at"):
            entry["created_at"] = existing["created_at"]
        instances[host] = entry
        gitlab["instances"] = instances
        gitlab["active_instance_host"] = host
        self.providers["gitlab"] = gitlab

    def clear_gitlab_credentials(
        self,
        instance_host: str | None = None,
    ) -> None:
        if instance_host:
            gitlab = dict(self.providers.get("gitlab", {}))
            instances = dict(gitlab.get("instances", {}))
            instances.pop(instance_host, None)
            if instances:
                gitlab["instances"] = instances
                self.providers["gitlab"] = gitlab
            else:
                self.providers.pop("gitlab", None)
        else:
            self.providers.pop("gitlab", None)

    def list_gitlab_instances(self) -> list[dict[str, Any]]:
        gitlab = self.providers.get("gitlab", {})
        instances = gitlab.get("instances", {})
        if not isinstance(instances, dict):
            return []
        active = str(gitlab.get("active_instance_host") or "")
        return [
            {**entry, "instance_host": host, "active": host == active}
            for host, entry in instances.items()
            if isinstance(entry, dict)
        ]

    def save_gitlab_workspace_prefs(
        self,
        *,
        instance_host: str | None = None,
        default_project: str,
    ) -> None:
        gitlab = dict(self.providers.get("gitlab", {}))
        instances = dict(gitlab.get("instances", {}))
        host = instance_host or gitlab.get("active_instance_host")
        if not host or host not in instances:
            raise ProviderCredentialError(
                "GitLab is not connected. Run: potpie gitlab login"
            )
        entry = dict(instances.get(host) or {})
        workspaces = dict(entry.get("workspaces") or {})
        workspaces["default_project"] = default_project.strip()
        entry["workspaces"] = workspaces
        instances[host] = entry
        gitlab["instances"] = instances
        self.providers["gitlab"] = gitlab


__all__ = ["FakeAuthHttpClient", "InMemoryCredentialStore"]
