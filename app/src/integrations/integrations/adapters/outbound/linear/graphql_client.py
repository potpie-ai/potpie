"""Minimal synchronous Linear GraphQL client (Celery-safe)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LINEAR_API = "https://api.linear.app/graphql"


class LinearGraphQLError(RuntimeError):
    """Raised when the Linear GraphQL API returns errors in the response body."""


def linear_graphql(
    access_token: str, query: str, variables: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not access_token or not isinstance(access_token, str):
        raise ValueError("access_token must be a non-empty string")
    token = access_token.strip()
    if not token:
        raise ValueError("access_token is blank after stripping whitespace")
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                LINEAR_API,
                json={"query": query, "variables": variables or {}},
                headers={
                    "Authorization": token,
                    "Content-Type": "application/json",
                },
            )
            if r.status_code >= 400:
                # Linear often returns GraphQL errors in JSON even for 4xx; surface them.
                detail = r.text[:4000]
                try:
                    err_body = r.json()
                    errs = err_body.get("errors")
                    if isinstance(errs, list) and errs:
                        first = errs[0]
                        msg = (
                            first.get("message")
                            if isinstance(first, dict)
                            else str(first)
                        )
                        raise LinearGraphQLError(
                            f"Linear GraphQL HTTP {r.status_code}: {msg}"
                        ) from None
                except LinearGraphQLError:
                    raise
                except Exception:
                    pass
                raise LinearGraphQLError(
                    f"Linear GraphQL HTTP {r.status_code}: {detail}"
                ) from None
            body = r.json()
    except LinearGraphQLError:
        raise
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Linear API request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Linear API returned unexpected response type: {type(body).__name__}")
    errs = body.get("errors")
    if errs:
        first_msg = errs[0].get("message", str(errs[0])) if errs else "unknown"
        raise LinearGraphQLError(
            f"Linear GraphQL error ({len(errs)} error(s)): {first_msg}"
        )
    return body.get("data") or {}
