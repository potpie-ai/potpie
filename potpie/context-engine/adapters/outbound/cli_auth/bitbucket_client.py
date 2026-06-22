"""Read-only Bitbucket Cloud token verification helpers."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Literal

from adapters.outbound.cli_auth.http import AuthHttpClient, AuthHttpError, HttpClient
from adapters.outbound.cli_auth.provider_config import (
    BITBUCKET_API_BASE,
    BITBUCKET_USER_WORKSPACES_PATH,
)

BitbucketAuthErrorKind = Literal[
    "invalid_credentials",
    "insufficient_scopes",
    "unknown",
]


@dataclass(frozen=True)
class BitbucketVerifyResult:
    ok: bool
    email: str
    display_name: str = ""
    error_kind: BitbucketAuthErrorKind | None = None


def _basic_auth_header(email: str, api_token: str) -> str:
    raw = f"{email.strip()}:{api_token.strip()}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _bitbucket_api_error_message(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    return str(error.get("message") or "").strip()


def workspace_slugs_from_list_payload(payload: dict[str, Any]) -> list[str]:
    """Extract workspace slugs from GET /user/workspaces (or legacy /workspaces)."""
    values = payload.get("values")
    if not isinstance(values, list):
        return []
    slugs: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        workspace = item.get("workspace")
        if isinstance(workspace, dict):
            record = workspace
        else:
            record = item
        slug = str(record.get("slug") or record.get("username") or "").strip()
        if slug:
            slugs.append(slug)
    return slugs


def _classify_bitbucket_failure(
    status_code: int,
    response: Any,
) -> BitbucketAuthErrorKind:
    message = _bitbucket_api_error_message(response)
    if "CHANGE-2770" in message:
        return "unknown"
    if status_code == 401:
        return "invalid_credentials"
    if status_code in {400, 403}:
        return "insufficient_scopes"
    return "unknown"


def verify_bitbucket_api_token(
    email: str,
    api_token: str,
    *,
    http: HttpClient | None = None,
) -> BitbucketVerifyResult:
    email_value = email.strip()
    token_value = api_token.strip()
    if not email_value or not token_value:
        return BitbucketVerifyResult(
            ok=False,
            email=email_value,
            error_kind="invalid_credentials",
        )

    headers = {
        "Authorization": _basic_auth_header(email_value, token_value),
        "Accept": "application/json",
    }
    owns = http is None
    http = http or AuthHttpClient(timeout=15.0)
    try:
        user_response = http.get(f"{BITBUCKET_API_BASE}/user", headers=headers)
        if user_response.status_code != 200:
            return BitbucketVerifyResult(
                ok=False,
                email=email_value,
                error_kind=_classify_bitbucket_failure(
                    user_response.status_code,
                    user_response,
                ),
            )
        try:
            user_data = user_response.json()
        except ValueError:
            return BitbucketVerifyResult(
                ok=False,
                email=email_value,
                error_kind="unknown",
            )

        workspaces_response = http.get(
            f"{BITBUCKET_API_BASE}{BITBUCKET_USER_WORKSPACES_PATH}",
            headers=headers,
            params={"pagelen": 1},
        )
        if workspaces_response.status_code != 200:
            return BitbucketVerifyResult(
                ok=False,
                email=email_value,
                error_kind=_classify_bitbucket_failure(
                    workspaces_response.status_code,
                    workspaces_response,
                ),
            )
        try:
            workspaces_data = workspaces_response.json()
        except ValueError:
            return BitbucketVerifyResult(
                ok=False,
                email=email_value,
                error_kind="unknown",
            )
        if not isinstance(workspaces_data, dict):
            return BitbucketVerifyResult(
                ok=False,
                email=email_value,
                error_kind="unknown",
            )
        workspace_slug = ""
        slugs = workspace_slugs_from_list_payload(workspaces_data)
        if slugs:
            workspace_slug = slugs[0]
        if workspace_slug:
            repositories_response = http.get(
                f"{BITBUCKET_API_BASE}/repositories/{workspace_slug}",
                headers=headers,
                params={"pagelen": 1},
            )
            if repositories_response.status_code != 200:
                return BitbucketVerifyResult(
                    ok=False,
                    email=email_value,
                    error_kind=_classify_bitbucket_failure(
                        repositories_response.status_code,
                        repositories_response,
                    ),
                )
            try:
                repositories_response.json()
            except ValueError:
                return BitbucketVerifyResult(
                    ok=False,
                    email=email_value,
                    error_kind="unknown",
                )
    except AuthHttpError:
        return BitbucketVerifyResult(
            ok=False,
            email=email_value,
            error_kind="unknown",
        )
    finally:
        if owns:
            http.close()

    display_name = ""
    if isinstance(user_data, dict):
        display_name = str(
            user_data.get("display_name")
            or user_data.get("nickname")
            or user_data.get("username")
            or ""
        ).strip()
    return BitbucketVerifyResult(
        ok=True,
        email=email_value,
        display_name=display_name,
    )
