from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from adapters.outbound.cli_auth.env_bootstrap import load_cli_env
from adapters.outbound.cli_auth.errors import CliAuthError
from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.models import (
    AccessToken,
    DeviceCode,
    GitHubAccount,
    ProviderCredentials,
)

GITHUB_CLIENT_ID_ENV = "POTPIE_GITHUB_CLIENT_ID"
GITHUB_SCOPES = ("repo", "read:org", "read:user", "user:email")
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_USER_EMAILS_URL = "https://api.github.com/user/emails"
GITHUB_USER_REPOS_URL = "https://api.github.com/user/repos"
GITHUB_PROVIDER = "github"
GITHUB_PROVIDER_HOST = "github.com"
GITHUB_VERIFICATION_URI = "https://github.com/login/device"


class GitHubDeviceFlowError(CliAuthError):
    """Expected GitHub auth flow failure."""


def get_github_client_id() -> str:
    """Resolve GitHub OAuth app client ID from environment (.env via load_cli_env)."""
    load_cli_env()
    client_id = os.getenv(GITHUB_CLIENT_ID_ENV, "").strip()
    if not client_id:
        raise GitHubDeviceFlowError(
            f"{GITHUB_CLIENT_ID_ENV} is not set. "
            "Add it to potpie/.env (see .env.template)."
        )
    return client_id


def _parse_scopes(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _json_or_error(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception as exc:
        raise GitHubDeviceFlowError("GitHub returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise GitHubDeviceFlowError("GitHub returned an unexpected response body.")
    return data


def _json_list_or_error(response: httpx.Response) -> list[Any]:
    try:
        data = response.json()
    except Exception as exc:
        raise GitHubDeviceFlowError("GitHub returned a non-JSON response.") from exc
    if not isinstance(data, list):
        raise GitHubDeviceFlowError("GitHub returned an unexpected response body.")
    return data


def request_device_code(
    *,
    client_id: str | None = None,
    scopes: tuple[str, ...] = GITHUB_SCOPES,
    http: HttpClient | None = None,
) -> DeviceCode:
    client_id = client_id or get_github_client_id()
    owns = http is None
    http = http or AuthHttpClient()
    try:
        response = http.post(
            GITHUB_DEVICE_CODE_URL,
            headers={"Accept": "application/json"},
            data={"client_id": client_id, "scope": " ".join(scopes)},
        )
    except AuthHttpError as exc:
        raise GitHubDeviceFlowError(str(exc)) from exc
    finally:
        if owns:
            http.close()
    data = _json_or_error(response)
    if response.status_code >= 400:
        raise GitHubDeviceFlowError(
            str(
                data.get("error_description")
                or data.get("error")
                or "GitHub device-code request failed."
            )
        )
    return DeviceCode(
        device_code=str(data["device_code"]),
        user_code=str(data["user_code"]),
        verification_uri=str(data.get("verification_uri") or GITHUB_VERIFICATION_URI),
        expires_in=int(data["expires_in"]),
        interval=max(int(data.get("interval") or 5), 1),
    )


def poll_for_access_token(
    device_code: DeviceCode,
    *,
    client_id: str | None = None,
    http: HttpClient | None = None,
    sleep_fn: Any = time.sleep,
) -> AccessToken:
    client_id = client_id or get_github_client_id()
    owns = http is None
    http = http or AuthHttpClient()
    interval = device_code.interval
    deadline = time.monotonic() + max(device_code.expires_in, 1)
    try:
        sleep_fn(interval)
        while True:
            if time.monotonic() >= deadline:
                raise GitHubDeviceFlowError(
                    "GitHub device code expired before authorization completed."
                )
            response = http.post(
                GITHUB_TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "device_code": device_code.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            data = _json_or_error(response)
            error = str(data.get("error") or "").strip()
            if response.status_code < 400 and data.get("access_token"):
                return AccessToken(
                    access_token=str(data["access_token"]),
                    token_type=str(data.get("token_type") or "bearer"),
                    scopes=_parse_scopes(data.get("scope") or data.get("scopes")),
                )
            if error == "authorization_pending":
                sleep_fn(interval)
                continue
            if error == "slow_down":
                interval += 5
                sleep_fn(interval)
                continue
            if error == "expired_token":
                raise GitHubDeviceFlowError(
                    "GitHub device code expired before authorization completed."
                )
            if error == "access_denied":
                raise GitHubDeviceFlowError("GitHub authorization was denied.")
            detail = str(
                data.get("error_description") or error or "GitHub token exchange failed."
            )
            raise GitHubDeviceFlowError(detail)
    except AuthHttpError as exc:
        raise GitHubDeviceFlowError(str(exc)) from exc
    finally:
        if owns:
            http.close()


def verify_account(
    access_token: str,
    *,
    http: HttpClient | None = None,
) -> GitHubAccount:
    owns = http is None
    http = http or AuthHttpClient()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}",
    }
    try:
        response = http.get(
            GITHUB_USER_URL,
            headers=headers,
        )
        data = _json_or_error(response)
        if response.status_code >= 400:
            raise GitHubDeviceFlowError(
                str(data.get("message") or "GitHub account verification failed.")
            )
        login = str(data.get("login") or "").strip()
        account_id = data.get("id")
        if not login or not isinstance(account_id, int):
            raise GitHubDeviceFlowError(
                "GitHub account verification returned incomplete account data."
            )
        email = str(data["email"]) if data.get("email") is not None else None
        if email is None:
            emails_response = http.get(GITHUB_USER_EMAILS_URL, headers=headers)
            if emails_response.status_code >= 400:
                emails_data = _json_or_error(emails_response)
                raise GitHubDeviceFlowError(
                    str(emails_data.get("message") or "GitHub email lookup failed.")
                )
            for item in _json_list_or_error(emails_response):
                if (
                    isinstance(item, dict)
                    and item.get("primary") is True
                    and item.get("verified") is True
                    and item.get("email") is not None
                ):
                    email = str(item["email"])
                    break
        return GitHubAccount(
            login=login,
            id=account_id,
            name=str(data["name"]) if data.get("name") is not None else None,
            email=email,
        )
    except AuthHttpError as exc:
        raise GitHubDeviceFlowError(str(exc)) from exc
    finally:
        if owns:
            http.close()


def _has_next_page(response: httpx.Response) -> bool:
    link = response.headers.get("Link", "")
    return 'rel="next"' in link


def list_user_owned_repositories(
    access_token: str,
    *,
    http: HttpClient | None = None,
) -> list[dict[str, Any]]:
    owns = http is None
    http = http or AuthHttpClient()
    repos: list[dict[str, Any]] = []
    page = 1
    try:
        while True:
            response = http.get(
                GITHUB_USER_REPOS_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {access_token}",
                },
                params={
                    "affiliation": "owner",
                    "per_page": 100,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
            )
            if response.status_code >= 400:
                data = _json_or_error(response)
                raise GitHubDeviceFlowError(
                    str(data.get("message") or "GitHub repository listing failed.")
                )
            data = _json_list_or_error(response)
            for item in data:
                if isinstance(item, dict):
                    repos.append(
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "full_name": item.get("full_name"),
                            "private": bool(item.get("private")),
                            "html_url": item.get("html_url"),
                            "default_branch": item.get("default_branch"),
                            "updated_at": item.get("updated_at"),
                        }
                    )
            if not _has_next_page(response):
                return repos
            page += 1
    except AuthHttpError as exc:
        raise GitHubDeviceFlowError(str(exc)) from exc
    finally:
        if owns:
            http.close()


def build_provider_credentials(
    token: AccessToken,
    account: GitHubAccount,
    *,
    now: datetime | None = None,
    verification_uri: str = GITHUB_VERIFICATION_URI,
) -> ProviderCredentials:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    scopes = token.scopes or list(GITHUB_SCOPES)
    return ProviderCredentials(
        provider=GITHUB_PROVIDER,
        provider_host=GITHUB_PROVIDER_HOST,
        access_token=token.access_token,
        token_type=token.token_type,
        scopes=scopes,
        account={
            "login": account.login,
            "id": account.id,
            "name": account.name,
            "email": account.email,
        },
        created_at=stamp,
        updated_at=stamp,
        expires_at=None,
        metadata={
            "auth_flow": "device",
            "verification_uri": verification_uri,
        },
        token_storage="keychain",
    )
