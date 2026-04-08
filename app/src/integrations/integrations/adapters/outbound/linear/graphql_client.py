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
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPStatusError:
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
